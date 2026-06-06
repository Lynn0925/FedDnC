import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import GradScaler, autocast

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
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'FEDNC', "vision_depth": 0, "language_depth": 0, "vision_ctx": 0, "language_ctx": 0}
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
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class FedDnCPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.FEDNC.N_CTX
        ctx_init = cfg.TRAINER.FEDNC.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.N = cfg.TRAINER.FEDNC.N
        device = next(clip_model.parameters()).device

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt.to(next(clip_model.parameters()).device)).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        else:
            ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx_b = nn.Parameter(ctx_vectors.clone())  # discriminative stream

        ctx_n_init = ctx_vectors.clone()
        ablation_random_ctx_n = getattr(cfg.TRAINER.FEDNC, 'ABLATION_RANDOM_CTX_N', False)
        if not ctx_init and not ablation_random_ctx_n:
            zs_tok = clip.tokenize("a photo of a")
            with torch.no_grad():
                device = next(clip_model.parameters()).device
                zs_tok = zs_tok.to(device)
                zs_emb = clip_model.token_embedding(zs_tok).type(dtype)  # [1,77,D]
            n_fill = min(4, n_ctx)  # "a photo of a" has 4 tokens
            ctx_n_init[:, :n_fill, :] = zs_emb[0, 1:1+n_fill, :].unsqueeze(0).expand(self.N, -1, -1)
        self.ctx_n = nn.Parameter(ctx_n_init)  # generalization stream

        classnames = [name.replace("_", " ") for name in classnames]
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).repeat(self.N, 1)
        tokenized_prompts = tokenized_prompts.to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def forward(self, mode="base"):
        ctx = self.ctx_b if mode == "base" else self.ctx_n
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx = ctx.permute(1, 0, 2, 3).contiguous().view(self.N * self.n_cls, self.n_ctx, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts

class FedDnCModel(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = FedDnCPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.N = cfg.TRAINER.FEDNC.N
        self.OT = cfg.TRAINER.FEDNC.OT
        self.eps = cfg.TRAINER.FEDNC.EPS
        self.max_iter = cfg.TRAINER.FEDNC.MAX_ITER
        self.thresh = cfg.TRAINER.FEDNC.THRESH
        self.top_percent = cfg.TRAINER.FEDNC.TOP_PERCENT
        self.alpha_ensemble = getattr(cfg.TRAINER.FEDNC, 'ALPHA_ENSEMBLE', 0.6)
        self.num_shots = cfg.DATASET.NUM_SHOTS
        self.ablation_single_stream = getattr(cfg.TRAINER.FEDNC, 'ABLATION_SINGLE_STREAM', False)
        self.ablation_fixed_alpha = getattr(cfg.TRAINER.FEDNC, 'ABLATION_FIXED_ALPHA', -1.0)

        # zero-shot text features as anchor for the generalization stream (not persisted in checkpoint)
        device = next(clip_model.parameters()).device
        with torch.no_grad():
            zs_tokens = torch.cat([
                clip.tokenize(f"a photo of a {c.replace('_', ' ')}.")
                for c in classnames
            ]).to(device)
            x = clip_model.token_embedding(zs_tokens).type(self.dtype)
            x = x + clip_model.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
            x = clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = clip_model.ln_final(x).type(self.dtype)
            x = x[torch.arange(x.shape[0]), zs_tokens.argmax(dim=-1)]
            x = x @ clip_model.text_projection
            zs_feats = F.normalize(x.float(), dim=-1)
        self.register_buffer('zs_text_features', zs_feats, persistent=False)

    def Sinkhorn(self, K, u, v):
        r, c = torch.ones_like(u), torch.ones_like(v)
        for _ in range(self.max_iter):
            r0 = r
            r = u / (torch.matmul(K, c.unsqueeze(-1)).squeeze(-1) + 1e-8)
            c = v / (torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1) + 1e-8)
            if (r - r0).abs().mean().item() < self.thresh: break
        return torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

    def entropic_COT_fast(self, a, b, M, reg):
        dx, dy = torch.ones_like(a), torch.ones_like(b)
        Kp = torch.matmul(torch.diag_embed(1 / (a + 1e-8), dim1=1), M)
        Kq = torch.matmul(torch.diag_embed(1 / (b + 1e-8), dim1=1), M.permute(0, 2, 1))
        u, v = dx, dy
        for _ in range(self.max_iter):
            v0 = v
            u = torch.minimum(torch.div(dx, torch.matmul(Kp, v.unsqueeze(-1)).squeeze(-1) + 1e-8), dx)
            v = torch.div(dy, torch.matmul(Kq, u.unsqueeze(-1)).squeeze(-1) + 1e-8)
            if (v - v0).abs().mean().item() < self.thresh: break
        return torch.matmul(torch.diag_embed(u, dim1=1), M) @ torch.diag_embed(v, dim1=1)

    def run_ot_logic(self, image_features, text_features):
        M = image_features.shape[0]
        B_act = image_features.shape[1]
        N = text_features.shape[0]
        C = text_features.shape[1]

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        sim = torch.einsum('mbd,ncd->mbnc', image_features, text_features)
        sim = sim.permute(1, 3, 0, 2).contiguous()  # [B, C, M, N]
        sim_flattened = sim.reshape(-1, M, N)         # [B*C, M, N]

        wdist = 1.0 - sim_flattened
        total_samples = sim_flattened.shape[0]
        xx = torch.full((total_samples, M), 1./M, dtype=sim.dtype, device=sim.device)
        yy = torch.full((total_samples, N), 1./N, dtype=sim.dtype, device=sim.device)

        if self.OT == 'COT':
            yy = yy * min(1.0, self.top_percent)

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            if self.OT == 'Sinkhorn':
                T = self.Sinkhorn(KK, xx, yy)
            else:
                T = self.entropic_COT_fast(xx, yy, KK, 0.01)

        sim_op = torch.sum(T * sim_flattened, dim=(1, 2))  # [B*C]
        logits = sim_op.reshape(B_act, C)

        return self.logit_scale.exp() * logits

    def forward(self, image):
        if image.dim() == 5:
            image = image.squeeze(0)

        img_feats_full = self.image_encoder(image.type(self.dtype))

        if isinstance(img_feats_full, (list, tuple)):
            img_feats = img_feats_full[-1]  # [B, D]
        else:
            img_feats = img_feats_full      # [B, D]

        if img_feats.dim() == 3:
            img_feats = img_feats[0]

        img_feats = F.normalize(img_feats.float(), dim=-1)  # [B, D]

        # discriminative stream (ctx_b)
        prompts_b = self.prompt_learner(mode="base")
        text_feats_b = self.text_encoder(
            prompts_b, self.tokenized_prompts
        ).view(self.N, self.n_cls, -1)
        text_feats_b = F.normalize(text_feats_b.float(), dim=-1)
        text_feats_b_mean = text_feats_b.mean(0)              # [n_cls, D]
        logits_b = self.logit_scale.exp() * img_feats @ text_feats_b_mean.t()

        # generalization stream (ctx_n)
        prompts_n = self.prompt_learner(mode="novel")
        text_feats_n = self.text_encoder(
            prompts_n, self.tokenized_prompts
        ).view(self.N, self.n_cls, -1)
        text_feats_n = F.normalize(text_feats_n.float(), dim=-1)
        text_feats_n_mean = text_feats_n.mean(0)              # [n_cls, D]
        logits_n = self.logit_scale.exp() * img_feats @ text_feats_n_mean.t()

        if not self.training:
            if self.ablation_single_stream:
                return logits_b

            if self.ablation_fixed_alpha >= 0:
                alpha = self.ablation_fixed_alpha
                return alpha * logits_b + (1 - alpha) * logits_n

            # confidence-adaptive weighting: the more confident stream dominates
            shots = max(self.num_shots, 1)
            conf_T = max(1.0, 8.0 / shots)        # shots=1→T=8, 2→4, 4→2, 8+→1
            alpha_ceil = min(0.9, 0.5 + 0.05 * shots)
            conf_b = F.softmax(logits_b / conf_T, dim=-1).max(dim=-1).values
            conf_n = F.softmax(logits_n / conf_T, dim=-1).max(dim=-1).values
            alpha = (conf_b / (conf_b + conf_n + 1e-8)).clamp(0.3, alpha_ceil)
            return alpha.unsqueeze(-1) * logits_b + (1 - alpha).unsqueeze(-1) * logits_n

        logits_zs = self.logit_scale.exp() * img_feats @ self.zs_text_features.t()
        return logits_b, logits_n, logits_zs, text_feats_b_mean, text_feats_n_mean

    @torch.no_grad()
    def compute_alpha_batch(self, image):
        """[ANALYSIS] Return per-sample alpha weights for a batch (does not affect training).
        alpha_i = clamp(conf_b / (conf_b + conf_n), 0.3, alpha_ceil)
        where conf = max(softmax(logits / conf_T)).
        """
        if image.dim() == 5:
            image = image.squeeze(0)

        img_feats = self.image_encoder(image.type(self.dtype))
        if isinstance(img_feats, (list, tuple)):
            img_feats = img_feats[-1]
        if img_feats.dim() == 3:
            img_feats = img_feats[0]
        img_feats = F.normalize(img_feats.float(), dim=-1)

        prompts_b = self.prompt_learner(mode="base")
        text_b = self.text_encoder(prompts_b, self.tokenized_prompts).view(self.N, self.n_cls, -1)
        text_b_mean = F.normalize(text_b.float(), dim=-1).mean(0)
        logits_b = self.logit_scale.exp() * img_feats @ text_b_mean.t()

        prompts_n = self.prompt_learner(mode="novel")
        text_n = self.text_encoder(prompts_n, self.tokenized_prompts).view(self.N, self.n_cls, -1)
        text_n_mean = F.normalize(text_n.float(), dim=-1).mean(0)
        logits_n = self.logit_scale.exp() * img_feats @ text_n_mean.t()

        shots = max(self.num_shots, 1)
        conf_T = max(1.0, 8.0 / shots)
        alpha_ceil = min(0.9, 0.5 + 0.05 * shots)
        conf_b = F.softmax(logits_b / conf_T, dim=-1).max(dim=-1).values
        conf_n = F.softmax(logits_n / conf_T, dim=-1).max(dim=-1).values
        alpha = (conf_b / (conf_b + conf_n + 1e-8)).clamp(0.3, alpha_ceil)
        return alpha.cpu().float()


class FedDnC(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.FEDNC.PREC in ["fp32", "amp"]:
            clip_model.float()

        clip_model.to(self.device)

        self.model = FedDnCModel(cfg, classnames, clip_model)

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.FEDNC.PREC == "amp" else None

    def before_train(self, is_fed=False):
        super().before_train(is_fed)
        with torch.no_grad():
            self._ctx_b_ref = self.model.prompt_learner.ctx_b.detach().clone()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec      = self.cfg.TRAINER.FEDNC.PREC
        lambda_kd = self.cfg.TRAINER.FEDNC.LAMBDA_KD
        beta_div  = self.cfg.TRAINER.FEDNC.BETA_DIV
        mu_prox   = getattr(self.cfg.TRAINER.FEDNC, 'MU_PROX', 0.01)
        # [ABLATION] ABLATION_LAMBDA_ALIGN: separate weight for loss_n_align (feature cosine alignment).
        # Default -1 → use lambda_kd (identical to original behaviour).
        # Set to 0 in ablation script to isolate "w/o L_align"; set lambda_kd=0 here to isolate "w/o L_feat".
        _abl_lambda_align = getattr(self.cfg.TRAINER.FEDNC, 'ABLATION_LAMBDA_ALIGN', -1.0)
        lambda_align = lambda_kd if _abl_lambda_align < 0 else _abl_lambda_align

        def compute_loss(image, label):
            logits_b, logits_n, logits_zs, text_feats_b_mean, text_feats_n_mean = self.model(image)

            # discriminative stream: CE + FedProx only
            loss_ce = F.cross_entropy(logits_b, label)
            loss_prox = ((self.model.prompt_learner.ctx_b - self._ctx_b_ref) ** 2).mean()

            # generalization stream: KD (logit-level) + feature alignment toward zero-shot anchor
            T = getattr(self.cfg.TRAINER.FEDNC, 'KD_TEMP', 4.0)
            loss_kd_n = F.kl_div(
                F.log_softmax(logits_n / T, dim=-1),
                F.softmax(logits_zs.detach() / T, dim=-1),
                reduction='batchmean',
            ) * (T ** 2)
            loss_n_align = (1.0 - F.cosine_similarity(
                text_feats_n_mean,
                self.model.zs_text_features.detach(),
                dim=-1
            )).mean()

            # diversity: push ctx_n away from ctx_b (one-sided; ctx_b gradient is detached)
            loss_div_bn = F.cosine_similarity(
                text_feats_b_mean.detach(),
                text_feats_n_mean,
                dim=-1
            ).mean()

            # [ABLATION] lambda_kd controls L_feat (logit-level KD, loss_kd_n).
            # lambda_align controls L_align (feature cosine alignment, loss_n_align).
            # When ABLATION_LAMBDA_ALIGN < 0, lambda_align == lambda_kd → original behaviour.
            loss = (loss_ce
                    + mu_prox    * loss_prox
                    + lambda_kd  * loss_kd_n
                    + lambda_align * loss_n_align
                    + beta_div   * loss_div_bn)
            return loss, logits_b

        if prec == "amp":
            with autocast('cuda'):
                loss, logits_b = compute_loss(image, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss, logits_b = compute_loss(image, label)
            self.model_backward_and_update(loss)

        return {
            "loss": loss.item(),
            "acc": compute_accuracy(logits_b, label)[0].item()
        }

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)
