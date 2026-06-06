import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'FedMGP',
                      "vision_depth": cfg.TRAINER.FEDMGP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.FEDMGP.PROMPT_DEPTH_TEXT, 
                      "vision_ctx": cfg.TRAINER.FEDMGP.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.FEDMGP.N_CTX_TEXT,
                      "vision_num_prompts": cfg.TRAINER.FEDMGP.NUM_PROMPTS_VISION,
                      "language_num_prompts": cfg.TRAINER.FEDMGP.NUM_PROMPTS_TEXT}
    
    # Ensure vision prompt depth is at least 1 when using vision prompts
    if design_details["vision_num_prompts"] > 0 and design_details["vision_depth"] <= 0:
        design_details["vision_depth"] = 1
        
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiPromptLearner(nn.Module):
    """Multi-prompt learner that manages multiple text prompts.

    Each prompt has its own context vectors while sharing token_prefix and token_suffix.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        assert cfg.TRAINER.FEDMGP.PROMPT_DEPTH_TEXT >= 1, "In FedMGP, Language prompt depth should be >=1"
        
        n_ctx = cfg.TRAINER.FEDMGP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.FEDMGP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.num_prompts_text = cfg.TRAINER.FEDMGP.NUM_PROMPTS_TEXT
        
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # Initialize context vectors using given words
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

            # Clone vectors for each prompt with small noise for diversity
            multi_ctx_vectors = ctx_vectors.unsqueeze(0).expand(self.num_prompts_text, -1, -1).clone()

            for i in range(self.num_prompts_text):
                torch.manual_seed(i + 123)
                noise = torch.randn_like(multi_ctx_vectors[i]) * 0.01
                multi_ctx_vectors[i] += noise

        else:
            # Random initialization with different seeds per prompt
            multi_ctx_vectors = torch.empty(self.num_prompts_text, n_ctx, ctx_dim, dtype=dtype)

            for i in range(self.num_prompts_text):
                torch.manual_seed(i + 42)
                nn.init.normal_(multi_ctx_vectors[i], std=0.02)

            prompt_prefix = " ".join(["X"] * n_ctx)
            
        print(f"FedMGP design with multiple prompts")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of language prompts: {self.num_prompts_text}")
        print(f"Number of vision prompts: {cfg.TRAINER.FEDMGP.NUM_PROMPTS_VISION}")
        print(f"Using enhanced random initialization for prompts")

        random_select = getattr(cfg.TRAINER.FEDMGP, 'RANDOM_SELECT', False)
        agg_highest = getattr(cfg.TRAINER.FEDMGP, 'AGGREGATE_HIGHEST_SIM', False)
        topk = getattr(cfg.TRAINER.FEDMGP, 'TOPK', 2)

        if random_select:
            agg_strategy = f"Random select Top{topk} prompts"
        else:
            if agg_highest:
                agg_strategy = f"Select Top{topk} highest similarity prompts"
            else:
                agg_strategy = f"Select Top{topk} lowest similarity prompts"

        print(f"Prompt aggregation strategy: {agg_strategy}")

        # Shape: [num_prompts_text, n_ctx, ctx_dim]
        self.ctx = nn.Parameter(multi_ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        """Construct prompt vectors by concatenating context with prefix and suffix."""
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        """Build input vectors for each prompt group."""
        ctx = self.ctx  # [num_prompts_text, n_ctx, ctx_dim]
        all_prompts = []

        for i in range(self.num_prompts_text):
            ctx_i = ctx[i]

            if ctx_i.dim() == 2:
                ctx_i = ctx_i.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts_i = self.construct_prompts(ctx_i, prefix, suffix)
            all_prompts.append(prompts_i)

        return all_prompts


class CustomCLIP(nn.Module):
    """CLIP model with multi-prompt support.

    Manages multiple text and vision prompts, computes multiple feature sets and logits,
    and aggregates results. Supports mismatched numbers of text/vision prompts.
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_prompts_vision = cfg.TRAINER.FEDMGP.NUM_PROMPTS_VISION
        self.num_prompts_text = cfg.TRAINER.FEDMGP.NUM_PROMPTS_TEXT
        self.classnames = [name.replace("_", " ") for name in classnames]

        # Divergent loss parameters
        self.use_divergent_loss = getattr(cfg.TRAINER.FEDMGP, 'USE_DIVERGENT_LOSS', False)
        self.divergent_loss_weight = getattr(cfg.TRAINER.FEDMGP, 'DIVERGENT_LOSS_WEIGHT', 0.1)
        self.divergent_loss_type = getattr(cfg.TRAINER.FEDMGP, 'DIVERGENT_LOSS_TYPE', "cos")

        # Inference mode parameters
        self.inference_mode = getattr(cfg.TRAINER.FEDMGP, 'INFERENCE_MODE', 'average')
        self.selected_prompt_group = getattr(cfg.TRAINER.FEDMGP, 'SELECTED_PROMPT_GROUP', 0)

    def forward(self, image, label=None, image_paths=None):
        """Forward pass with multiple text and vision prompts.

        Returns:
            Training: (avg_loss, avg_logits) or (total_loss, avg_logits, divergent_loss)
            Testing: avg_logits
        """
        text_prompts = self.prompt_learner()
        text_features_list = []

        for prompts in text_prompts:
            text_features = self.text_encoder(prompts, self.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features_list.append(text_features)

        has_vision_prompt = hasattr(self.image_encoder, 'VPT')
        image_features_list = []

        if has_vision_prompt and self.num_prompts_vision > 0:
            for i in range(self.num_prompts_vision):
                image_features = self.image_encoder(image.type(self.dtype), prompt_idx=i)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features_list.append(image_features)
        else:
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(image_features)

        num_text_prompts = len(text_features_list)
        num_vision_prompts = len(image_features_list)

        all_logits = []
        logit_scale = self.logit_scale.exp()

        # Pair corresponding prompts
        min_prompts = min(num_text_prompts, num_vision_prompts)
        for i in range(min_prompts):
            logits = logit_scale * image_features_list[i] @ text_features_list[i].t()
            all_logits.append(logits)

        # Extra text prompts pair with first vision prompt
        for i in range(min_prompts, num_text_prompts):
            if num_vision_prompts > 0:
                logits = logit_scale * image_features_list[0] @ text_features_list[i].t()
                all_logits.append(logits)
            else:
                continue

        # Extra vision prompts pair with first text prompt
        for i in range(min_prompts, num_vision_prompts):
            if num_text_prompts > 0:
                logits = logit_scale * image_features_list[i] @ text_features_list[0].t()
                all_logits.append(logits)
            else:
                continue

        if all_logits:
            avg_logits = torch.stack(all_logits).mean(dim=0)
        else:
            # Fallback for vision-only ablation
            if num_text_prompts == 0 and num_vision_prompts > 0:
                num_classes = len(self.classnames)
                random_text_features = torch.randn(num_classes, image_features_list[0].size(-1),
                                                  device=image.device, dtype=self.dtype)
                random_text_features = random_text_features / random_text_features.norm(dim=-1, keepdim=True)

                logits = logit_scale * image_features_list[0] @ random_text_features.t()
                avg_logits = logits
                all_logits.append(logits)
                print("Using vision features with random text features (ablation_vision_only)")
            else:
                print("Warning: No valid logits! Ensure at least one text or vision prompt exists.")
                avg_logits = torch.zeros((image.size(0), len(self.classnames)),
                                        device=image.device, dtype=self.dtype).requires_grad_(True)
        
        if self.training and label is not None:
            losses = []

            for logits in all_logits:
                loss = F.cross_entropy(logits, label)
                losses.append(loss)

            if losses:
                avg_loss = torch.stack(losses).mean()
            else:
                print("Warning: No valid loss! Using zero loss. Ensure at least one prompt exists.")
                avg_loss = torch.tensor(0.0, device=image.device, dtype=self.dtype, requires_grad=True)

            divergent_loss = torch.tensor(0.0, device=image.device, dtype=self.dtype)

            if self.use_divergent_loss:
                text_divergent_loss = torch.tensor(0.0, device=image.device, dtype=self.dtype)
                if num_text_prompts > 1:
                    text_divergent_loss = self._compute_divergent_loss(
                        text_features_list,
                        batch_dim_first=False
                    )

                vision_divergent_loss = torch.tensor(0.0, device=image.device, dtype=self.dtype)
                if num_vision_prompts > 1:
                    vision_divergent_loss = self._compute_divergent_loss(
                        image_features_list,
                        batch_dim_first=True
                    )

                components_count = ((num_text_prompts > 1) + (num_vision_prompts > 1))
                if components_count > 0:
                    divergent_loss = (text_divergent_loss + vision_divergent_loss) / components_count
                    divergent_loss *= self.divergent_loss_weight

                    if divergent_loss < 0:
                        divergent_loss = torch.abs(divergent_loss)

            total_loss = avg_loss

            if self.use_divergent_loss and divergent_loss > 0:
                total_loss += divergent_loss

            if self.use_divergent_loss:
                return total_loss, avg_logits, divergent_loss
            else:
                return avg_loss, avg_logits
        else:
            # Inference mode selection
            if self.inference_mode == 'average':
                final_logits = avg_logits
            elif self.inference_mode == 'selected_group':
                if 0 <= self.selected_prompt_group < len(all_logits):
                    final_logits = all_logits[self.selected_prompt_group]
                else:
                    print(f"Warning: Selected prompt group index {self.selected_prompt_group} out of range, using average mode")
                    final_logits = avg_logits
            elif self.inference_mode == 'max_logits':
                stacked_logits = torch.stack(all_logits)
                max_values, max_indices = torch.max(stacked_logits, dim=2)
                _, best_prompt_indices = torch.max(max_values, dim=0)
                batch_size = image.size(0)
                selected_logits = torch.zeros_like(avg_logits)
                for i in range(batch_size):
                    selected_logits[i] = stacked_logits[best_prompt_indices[i], i]
                final_logits = selected_logits
            elif self.inference_mode == 'feature_average':
                avg_text_features = torch.stack(text_features_list).mean(dim=0)
                avg_image_features = torch.stack(image_features_list).mean(dim=0)
                final_logits = logit_scale * avg_image_features @ avg_text_features.t()
            else:
                print(f"Warning: Unknown inference mode '{self.inference_mode}', using average mode")
                final_logits = avg_logits

            return final_logits

    def _compute_divergent_loss(self, features_list, batch_dim_first=False):
        """Compute divergent loss to encourage diversity among prompt features.

        Args:
            features_list: List of features [batch/classes, dim] from different prompts
            batch_dim_first: True for vision features, False for text features
        """
        device = features_list[0].device
        dtype = features_list[0].dtype
        num_prompts = len(features_list)

        if num_prompts <= 1:
            return torch.tensor(0.0, device=device, dtype=dtype)

        first_dim_size = features_list[0].size(0)
        total_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for sample_idx in range(first_dim_size):
            sample_features = []
            for prompt_idx in range(num_prompts):
                feature = features_list[prompt_idx][sample_idx]
                sample_features.append(feature)

            sample_loss = torch.tensor(0.0, device=device, dtype=dtype)
            num_pairs = 0

            for i in range(num_prompts):
                for j in range(i+1, num_prompts):
                    feat_i = sample_features[i]
                    feat_j = sample_features[j]

                    if self.divergent_loss_type == "cos":
                        sim = F.cosine_similarity(feat_i.unsqueeze(0), feat_j.unsqueeze(0), dim=1)[0]
                        pair_loss = 1.0 - sim
                    elif self.divergent_loss_type == "l1":
                        pair_loss = -torch.abs(feat_i - feat_j).mean()
                    elif self.divergent_loss_type == "l2":
                        pair_loss = -torch.sqrt(torch.sum((feat_i - feat_j) ** 2) + 1e-8)
                    else:
                        sim = F.cosine_similarity(feat_i.unsqueeze(0), feat_j.unsqueeze(0), dim=1)[0]
                        pair_loss = 1.0 - sim

                    sample_loss = sample_loss + pair_loss
                    num_pairs += 1

            if num_pairs > 0:
                sample_loss = sample_loss / num_pairs
                total_loss = total_loss + sample_loss

        return total_loss / first_dim_size


# @TRAINER_REGISTRY.register()
class FedMGP(TrainerX):
    """Federated Multi-Grained Prompting trainer.

    Supports multiple vision and text prompts with selective aggregation.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FEDMGP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.FEDMGP.PREC == "fp32" or cfg.TRAINER.FEDMGP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP with multiple prompts")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        for name, param in self.model.named_parameters():
            if "prompt_learner.ctx" in name:
                param.requires_grad_(True)
                print(f"Enabling gradient: {name}, shape={param.shape}")

            if "visual.VPT" in name or "image_encoder.VPT" in name:
                param.requires_grad_(True)
                print(f"Enabling gradient: {name}, shape={param.shape}")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.FEDMGP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        print("\nTrainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}, shape={param.shape}")
                
    def forward_backward(self, batch):
        """Forward and backward pass for training."""
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.FEDMGP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image, label)
                loss = output[0]  # Total loss is the first output
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image, label)
            loss = output[0]
            self.model_backward_and_update(loss)

        def get_loss_value(loss_tensor_or_scalar):
            if hasattr(loss_tensor_or_scalar, 'item'):
                return loss_tensor_or_scalar.item()
            return float(loss_tensor_or_scalar)

        loss_summary = {
            "loss": get_loss_value(loss),
            "acc": compute_accuracy(output[1], label)[0].item(),
        }

        if len(output) >= 3:
            divergent_loss = output[2]
            loss_summary["divergent_loss"] = get_loss_value(divergent_loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

            has_vpt_params = False
            for name, param in self.model.named_parameters():
                if ("visual.VPT" in name or "image_encoder.VPT" in name) and param.requires_grad:
                    has_vpt_params = True
                    param_norm = torch.norm(param.data)

                    if hasattr(self, 'old_vpt_params') and name in self.old_vpt_params:
                        param_diff = torch.norm(param.data - self.old_vpt_params[name])
                        change_percent = param_diff / torch.norm(self.old_vpt_params[name]) * 100 if torch.norm(self.old_vpt_params[name]) > 0 else 0
                        print(f"VPT param '{name}' change: {param_diff:.6f} ({change_percent:.2f}%)")

                    if not hasattr(self, 'old_vpt_params'):
                        self.old_vpt_params = {}
                    self.old_vpt_params[name] = param.data.clone().detach()

            if not has_vpt_params:
                print("Warning: No trainable VPT parameters found!")

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def model_inference(self, input):
        """Model inference, returns logits only."""
        output = self.model(input)
        if isinstance(output, tuple):
            return output[0]
        return output

    def load_model(self, directory, epoch=None):
        """Load pretrained model, ignoring fixed token vectors."""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # Load best model by default
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    def test(self, split=None, is_global=False, current_epoch=0, idx=-1, global_test=False):
        """Test pipeline with visualization support."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and hasattr(self, 'val_loader') and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"

            if global_test and not getattr(self, 'is_special_dataset', False):
                print(f"Global test mode: using global test set")
                if hasattr(self, 'test_loader') and self.test_loader is not None:
                    data_loader = self.test_loader
                elif hasattr(self, 'fed_test_loader_x_dict') and len(self.fed_test_loader_x_dict) > 0:
                    for client_id in sorted(self.fed_test_loader_x_dict.keys()):
                        if self.fed_test_loader_x_dict[client_id] is not None:
                            print(f"Warning: test_loader not found, using client {client_id}'s federated test set")
                            data_loader = self.fed_test_loader_x_dict[client_id]
                            break
                else:
                    raise ValueError("No available test dataset")
            elif idx != -1:
                print(f"Client test mode: using federated test set for client {idx}")
                data_loader = None
                if hasattr(self, 'fed_test_loader_dict') and idx in self.fed_test_loader_dict:
                    data_loader = self.fed_test_loader_dict[idx]
                elif hasattr(self, 'fed_test_loader_x_dict') and idx in self.fed_test_loader_x_dict:
                    data_loader = self.fed_test_loader_x_dict[idx]

                if data_loader is None:
                    print(f"Warning: Client {idx} has no dedicated test set, trying global test set")
                    if hasattr(self, 'test_loader') and self.test_loader is not None:
                        data_loader = self.test_loader
                    else:
                        raise ValueError(f"No test data available for client {idx}")
            else:
                print(f"Standard test mode: using global test set")
                if hasattr(self, 'test_loader') and self.test_loader is not None:
                    data_loader = self.test_loader
                elif hasattr(self, 'fed_test_loader_x_dict') and len(self.fed_test_loader_x_dict) > 0:
                    for client_id in sorted(self.fed_test_loader_x_dict.keys()):
                        if self.fed_test_loader_x_dict[client_id] is not None:
                            print(f"Warning: test_loader not found, using client {client_id}'s federated test set")
                            data_loader = self.fed_test_loader_x_dict[client_id]
                            break
                else:
                    raise ValueError("No available test dataset")

        test_type = "global" if global_test and not getattr(self, 'is_special_dataset', False) else f"client {idx}"
        print(f"Evaluating on {test_type} {split} set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)

            with torch.no_grad():
                output = self.model_inference(input)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            if not is_global and idx >= 0:
                tag = f"{tag}/{str(idx)}"
            elif global_test and not getattr(self, 'is_special_dataset', False):
                tag = f"{tag}/global"
            self.write_scalar(tag, v, current_epoch)

        return list(results.values()) 