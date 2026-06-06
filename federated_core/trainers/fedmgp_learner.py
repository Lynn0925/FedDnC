# federated_core/trainers/fedmgp_learner.py

from typing import Dict, List, Any, Optional
import torch
import copy
import os
from ..base_federated_learner import BaseFederatedLearner
from ..fed_utils import average_weights

class FedMGPLearner(BaseFederatedLearner):
    """
    FedMGP implementation with similarity-based prompt ranking and aggregation.
    Handles text and vision prompt groups with complex similarity computation.
    """

    def _get_trainer_specific_attributes(self) -> Dict[str, Any]:
        """Get FedMGP-specific attributes."""
        attrs = {}
        attrs['global_text_prompts'] = None
        attrs['global_vision_prompts'] = None
        attrs['text_prompt_ranks'] = {}
        attrs['vision_prompt_ranks'] = {}
        attrs['text_contributions_by_rank'] = {}
        attrs['vision_contributions_by_rank'] = {}
        return attrs

    def _initialize_components_storage(self) -> None:
        """Initialize FedMGP component storage."""
        self.client_weights['components'] = {
            'text_prompts': [{} for _ in range(self.cfg.DATASET.USERS)],
            'vision_prompts': [{} for _ in range(self.cfg.DATASET.USERS)]
        }

    def aggregate_models(self, client_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Aggregate prompts based on similarity ranking."""
        text_prompts_list = self._extract_text_prompts(client_ids)
        vision_prompts_list = self._extract_vision_prompts(client_ids)

        topk = getattr(self.cfg.TRAINER.FEDMGP, 'TOPK', 2)
        global_component = {}
        is_first_round = self.current_round == 1

        if is_first_round:
            text_prompt_ranks, vision_prompt_ranks = self._first_round_ranking(
                text_prompts_list, vision_prompts_list, client_ids, topk)
        else:
            text_prompt_ranks = self._rank_prompts_by_similarity(
                text_prompts_list, client_ids, is_vision=False)
            vision_prompt_ranks = self._rank_prompts_by_similarity(
                vision_prompts_list, client_ids, is_vision=True)

        global_component.update(self._aggregate_text_prompts(
            text_prompts_list, client_ids, text_prompt_ranks, topk))
        global_component.update(self._aggregate_vision_prompts(
            vision_prompts_list, client_ids, vision_prompt_ranks, topk))

        self._update_global_prompts(global_component, topk)

        self.text_prompt_ranks = text_prompt_ranks
        self.vision_prompt_ranks = vision_prompt_ranks

        return global_component

    def update_client_models(self, client_ids: List[int],
                           global_component: Dict[str, torch.Tensor],
                           all_clients: List[int] = None) -> None:
        """Update client prompts based on ranking."""
        topk = getattr(self.cfg.TRAINER.FEDMGP, 'TOPK', 2)

        print("\nStarting client prompt update:")
        print(f"Updating selected prompt indices for {len(client_ids)} participating clients")

        for client_id in client_ids:
            if (client_id not in self.text_prompt_ranks and
                client_id not in self.vision_prompt_ranks):
                continue

            client_weight = self.client_weights['local'][client_id]

            self._update_client_text_prompts(client_id, client_weight, global_component, topk)
            self._update_client_vision_prompts(client_id, client_weight, global_component, topk)

        # Compute pre-training similarities after parameter update
        if self.current_round > 1:
            self._compute_pre_training_similarities(client_ids, global_component)

    def _store_components(self, client_id: int, trained_state_dict: Dict[str, torch.Tensor]) -> None:
        """Store trained text and vision prompts."""
        text_prompts = {}
        vision_prompts = {}

        # Extract text prompts
        for name, param in trained_state_dict.items():
            if "prompt_learner.ctx" in name:
                clean_name = name.replace('module.', '')
                text_prompts[clean_name] = copy.deepcopy(param)

        # Extract vision prompts
        vision_prompts = self._extract_visual_prompt_with_fallback(client_id, trained_state_dict)

        self.client_weights['components']['text_prompts'][client_id] = text_prompts
        self.client_weights['components']['vision_prompts'][client_id] = vision_prompts

        # Update local model weights
        for name, param in text_prompts.items():
            self.client_weights['local'][client_id][name] = copy.deepcopy(param)
        for name, param in vision_prompts.items():
            self.client_weights['local'][client_id][name] = copy.deepcopy(param)

    # === Private helper methods ===

    def _extract_text_prompts(self, client_ids: List[int]) -> List[torch.Tensor]:
        """Extract text prompts from all clients."""
        text_prompts_list = []

        for client_id in client_ids:
            local_weights = self.client_weights['local'][client_id]
            if 'prompt_learner.ctx' in local_weights:
                ctx = local_weights['prompt_learner.ctx']
                text_prompts_list.append(ctx)

        return text_prompts_list

    def _extract_vision_prompts(self, client_ids: List[int]) -> List[torch.Tensor]:
        """Extract vision prompts from all clients."""
        vision_prompts_list = []

        for client_id in client_ids:
            local_weights = self.client_weights['local'][client_id]
            vision_key = 'visual.VPT' if 'visual.VPT' in local_weights else 'image_encoder.VPT'
            if vision_key in local_weights:
                vpt = local_weights[vision_key]
                vision_prompts_list.append(vpt)

        return vision_prompts_list

    def _first_round_ranking(self, text_prompts_list: List[torch.Tensor],
                           vision_prompts_list: List[torch.Tensor],
                           client_ids: List[int], topk: int) -> tuple:
        """Random ranking for first round."""
        text_prompt_ranks = {}
        vision_prompt_ranks = {}

        for client_idx, client_id in enumerate(client_ids):
            # Random text prompt selection
            if client_idx < len(text_prompts_list):
                client_prompt = text_prompts_list[client_idx]
                num_prompts = client_prompt.size(0) if client_prompt.dim() == 3 else 1

                seed = 42 + client_id
                orig_state = torch.random.get_rng_state()
                torch.manual_seed(seed)

                if num_prompts <= topk:
                    selected_indices = list(range(num_prompts))
                else:
                    selected_indices = torch.randperm(num_prompts)[:topk].tolist()

                torch.random.set_rng_state(orig_state)
                text_prompt_ranks[client_id] = selected_indices
                print(f"Client {client_id}: First round random text prompt selection: {selected_indices}")

            # Random vision prompt selection
            if client_idx < len(vision_prompts_list):
                client_prompt = vision_prompts_list[client_idx]
                num_prompts = client_prompt.size(0) if client_prompt.dim() == 3 else 1

                seed = 42 + client_id
                orig_state = torch.random.get_rng_state()
                torch.manual_seed(seed)

                if num_prompts <= topk:
                    selected_indices = list(range(num_prompts))
                else:
                    selected_indices = torch.randperm(num_prompts)[:topk].tolist()

                torch.random.set_rng_state(orig_state)
                vision_prompt_ranks[client_id] = selected_indices
                print(f"Client {client_id}: First round random vision prompt selection: {selected_indices}")

        return text_prompt_ranks, vision_prompt_ranks

    def _rank_prompts_by_similarity(self, prompts_list: List[torch.Tensor],
                                   client_ids: List[int], is_vision: bool = False) -> Dict[int, List[int]]:
        """Rank prompts by similarity to global prompts."""
        if not prompts_list or len(prompts_list) < 1:
            topk = getattr(self.cfg.TRAINER.FEDMGP, 'TOPK', 2)
            default_indices = list(range(min(topk, 2)))
            default_ranks = {client_id: default_indices for client_id in client_ids if len(client_ids) > 0}
            prompt_type = "vision" if is_vision else "text"
            print(f"Warning: Insufficient {prompt_type} prompts, using default ranking {default_indices}")
            return default_ranks

        prompt_type = "vision" if is_vision else "text"
        client_prompt_ranks = {}

        topk = getattr(self.cfg.TRAINER.FEDMGP, 'TOPK', 2)
        aggregate_highest = getattr(self.cfg.TRAINER.FEDMGP, 'AGGREGATE_HIGHEST_SIM', False)
        select_mode = getattr(self.cfg.TRAINER.FEDMGP, 'SELECT_MODE', 'similarity')
        probabilistic_select = getattr(self.cfg.TRAINER.FEDMGP, 'PROBABILISTIC_SELECTION', False)
        temperature = getattr(self.cfg.TRAINER.FEDMGP, 'TEMPERATURE', 1.0)

        global_prompts = self.global_vision_prompts if is_vision else self.global_text_prompts

        # Determine selection mode description
        if select_mode == "random":
            selection_mode = "random"
        elif select_mode == "top_fixed":
            selection_mode = f"fixed top-{topk}"
        elif select_mode == "all":
            selection_mode = "all"
        elif probabilistic_select:
            selection_mode = f"probabilistic (temp={temperature})"
        else:
            sort_direction = "highest" if aggregate_highest else "lowest"
            selection_mode = f"{sort_direction} similarity"

        print(f"\nRanking {prompt_type} prompts for {len(client_ids)} clients, {selection_mode} top-{topk}...")

        for client_idx, client_id in enumerate(client_ids):
            if client_idx >= len(prompts_list):
                default_indices = list(range(min(topk, 2)))
                client_prompt_ranks[client_id] = default_indices
                continue

            client_prompt = prompts_list[client_idx]
            if client_prompt.dim() < 2:
                default_indices = list(range(min(topk, 2)))
                client_prompt_ranks[client_id] = default_indices
                continue

            num_prompts = client_prompt.size(0) if client_prompt.dim() == 3 else 1
            if num_prompts <= 1:
                client_prompt_ranks[client_id] = [0]
                continue

            if select_mode == "random":
                seed = 42 + client_id + self.current_round
                orig_state = torch.random.get_rng_state()
                torch.manual_seed(seed)

                selected_indices = torch.randperm(num_prompts)[:min(topk, num_prompts)].tolist()
                print(f"Client {client_id}: Random {prompt_type} prompt selection: {selected_indices}")

                torch.random.set_rng_state(orig_state)
                client_prompt_ranks[client_id] = selected_indices

            elif select_mode == "top_fixed":
                selected_indices = list(range(min(topk, num_prompts)))
                print(f"Client {client_id}: Fixed top-{len(selected_indices)} {prompt_type} prompts: {selected_indices}")
                client_prompt_ranks[client_id] = selected_indices

            elif select_mode == "all":
                selected_indices = list(range(num_prompts))
                print(f"Client {client_id}: Selected all {num_prompts} {prompt_type} prompts: {selected_indices}")
                client_prompt_ranks[client_id] = selected_indices

            else:  # similarity mode
                similarities = self._compute_prompt_similarities(client_prompt, global_prompts)

                if probabilistic_select:
                    selected_indices = self._probabilistic_prompt_selection(
                        similarities, topk, client_id, aggregate_highest, temperature)

                    similarities_str = ", ".join([f"{idx}:{sim:.4f}" for idx, sim in similarities])
                    print(f"Client {client_id}: {prompt_type} similarities [{similarities_str}], probabilistic selection: {selected_indices}")

                    client_prompt_ranks[client_id] = selected_indices
                else:
                    if aggregate_highest:
                        similarities.sort(key=lambda x: x[1], reverse=True)
                    else:
                        similarities.sort(key=lambda x: x[1])

                    selected_indices = [idx for idx, _ in similarities[:min(topk, num_prompts)]]

                    similarities_str = ", ".join([f"{idx}:{sim:.4f}" for idx, sim in similarities])
                    print(f"Client {client_id}: {prompt_type} similarities [{similarities_str}], selected: {selected_indices}")

                    client_prompt_ranks[client_id] = selected_indices

        print(f"Completed {prompt_type} prompt ranking for all clients, {selection_mode} top-{topk}")
        return client_prompt_ranks

    def _compute_prompt_similarities(self, client_prompt: torch.Tensor,
                                   global_prompts: List[torch.Tensor]) -> List[tuple]:
        """Compute cosine similarity between client and global prompts."""
        similarities = []
        num_prompts = client_prompt.size(0) if client_prompt.dim() == 3 else 1

        for i in range(num_prompts):
            if client_prompt.dim() == 3:
                prompt_i = client_prompt[i].reshape(-1)

                prompt_sims = []
                for global_prompt in global_prompts:
                    global_vec = global_prompt.reshape(-1)

                    norm_i = torch.norm(prompt_i)
                    norm_g = torch.norm(global_vec)

                    if norm_i > 0 and norm_g > 0:
                        sim = torch.dot(prompt_i, global_vec) / (norm_i * norm_g)
                        prompt_sims.append(sim.item())
                    else:
                        prompt_sims.append(0.0)

                if prompt_sims:
                    similarities.append((i, max(prompt_sims)))
                else:
                    similarities.append((i, 0.0))
            else:
                similarities.append((i, 0.0))

        return similarities

    def _probabilistic_prompt_selection(self, similarities: List[tuple],
                                      topk: int, client_id: int,
                                      aggregate_highest: bool, temperature: float) -> List[int]:
        """Select prompts based on probability distribution."""
        sim_values = torch.tensor([sim for _, sim in similarities])

        if not aggregate_highest:
            sim_values = -sim_values

        probs = torch.nn.functional.softmax(sim_values / temperature, dim=0)

        seed = 42 + client_id + self.current_round
        orig_state = torch.random.get_rng_state()
        torch.manual_seed(seed)

        # Sample topk indices without replacement
        selected_indices = []
        remaining_indices = list(range(len(similarities)))

        for _ in range(min(topk, len(similarities))):
            if not remaining_indices:
                break

            curr_probs = probs[remaining_indices]
            curr_probs = curr_probs / curr_probs.sum()

            idx = torch.multinomial(curr_probs, 1).item()
            selected_idx = remaining_indices.pop(idx)
            selected_indices.append(selected_idx)

        torch.random.set_rng_state(orig_state)

        return selected_indices

    def _aggregate_text_prompts(self, text_prompts_list: List[torch.Tensor],
                              client_ids: List[int], text_prompt_ranks: Dict[int, List[int]],
                              topk: int) -> Dict[str, torch.Tensor]:
        """Aggregate text prompts by rank."""
        global_component = {}
        text_contributions_by_rank = {}

        select_mode = getattr(self.cfg.TRAINER.FEDMGP, 'SELECT_MODE', 'similarity')

        if select_mode == "all":
            max_prompts = 0
            for client_idx, client_id in enumerate(client_ids):
                if client_id in text_prompt_ranks and client_idx < len(text_prompts_list):
                    max_prompts = max(max_prompts, len(text_prompt_ranks[client_id]))

            for rank in range(max_prompts):
                rank_text_values = []
                rank_text_contributions = []

                for client_idx, client_id in enumerate(client_ids):
                    if client_id in text_prompt_ranks and client_idx < len(text_prompts_list):
                        if rank < len(text_prompt_ranks[client_id]):
                            prompt_idx = text_prompt_ranks[client_id][rank]
                            client_prompt = text_prompts_list[client_idx]

                            if client_prompt.dim() == 3 and prompt_idx < client_prompt.size(0):
                                selected_prompt = client_prompt[prompt_idx].unsqueeze(0)
                                rank_text_values.append(selected_prompt)
                                rank_text_contributions.append((client_id, prompt_idx))

                if rank_text_values:
                    try:
                        all_prompts = torch.cat(rank_text_values, dim=0)
                        avg_prompt = all_prompts.mean(dim=0, keepdim=True)

                        global_component[f'prompt_learner.ctx_{rank}'] = avg_prompt
                        text_contributions_by_rank[rank] = rank_text_contributions

                        contributors = [cid for cid, _ in rank_text_contributions]
                        prompt_indices = [idx for _, idx in rank_text_contributions]
                        print(f"Text prompt rank {rank+1} aggregated: {len(contributors)} clients {contributors} (prompt indices {prompt_indices})")
                    except Exception as e:
                        print(f"Text prompt rank {rank+1} aggregation failed: {str(e)}")
        else:
            for rank in range(topk):
                rank_text_values = []
                rank_text_contributions = []

                for client_idx, client_id in enumerate(client_ids):
                    if client_id in text_prompt_ranks and client_idx < len(text_prompts_list):
                        if rank < len(text_prompt_ranks[client_id]):
                            prompt_idx = text_prompt_ranks[client_id][rank]
                            client_prompt = text_prompts_list[client_idx]

                            if client_prompt.dim() == 3 and prompt_idx < client_prompt.size(0):
                                selected_prompt = client_prompt[prompt_idx].unsqueeze(0)
                                rank_text_values.append(selected_prompt)
                                rank_text_contributions.append((client_id, prompt_idx))

                if rank_text_values:
                    try:
                        all_prompts = torch.cat(rank_text_values, dim=0)
                        avg_prompt = all_prompts.mean(dim=0, keepdim=True)

                        global_component[f'prompt_learner.ctx_{rank}'] = avg_prompt
                        text_contributions_by_rank[rank] = rank_text_contributions

                        contributors = [cid for cid, _ in rank_text_contributions]
                        prompt_indices = [idx for _, idx in rank_text_contributions]
                        print(f"Text prompt rank {rank+1} aggregated: {len(contributors)} clients {contributors} (prompt indices {prompt_indices})")
                    except Exception as e:
                        print(f"Text prompt rank {rank+1} aggregation failed: {str(e)}")

        self.text_contributions_by_rank = text_contributions_by_rank
        return global_component

    def _aggregate_vision_prompts(self, vision_prompts_list: List[torch.Tensor],
                                client_ids: List[int], vision_prompt_ranks: Dict[int, List[int]],
                                topk: int) -> Dict[str, torch.Tensor]:
        """Aggregate vision prompts by rank."""
        global_component = {}
        vision_contributions_by_rank = {}

        select_mode = getattr(self.cfg.TRAINER.FEDMGP, 'SELECT_MODE', 'similarity')

        if select_mode == "all":
            max_prompts = 0
            for client_idx, client_id in enumerate(client_ids):
                if client_id in vision_prompt_ranks and client_idx < len(vision_prompts_list):
                    max_prompts = max(max_prompts, len(vision_prompt_ranks[client_id]))

            for rank in range(max_prompts):
                rank_vision_values = []
                rank_vision_contributions = []

                for client_idx, client_id in enumerate(client_ids):
                    if client_id in vision_prompt_ranks and client_idx < len(vision_prompts_list):
                        if rank < len(vision_prompt_ranks[client_id]):
                            prompt_idx = vision_prompt_ranks[client_id][rank]
                            client_prompt = vision_prompts_list[client_idx]

                            if client_prompt.dim() == 3 and prompt_idx < client_prompt.size(0):
                                selected_prompt = client_prompt[prompt_idx].unsqueeze(0)
                                rank_vision_values.append(selected_prompt)
                                rank_vision_contributions.append((client_id, prompt_idx))
                            elif client_prompt.dim() == 2 and rank == 0:
                                selected_prompt = client_prompt.unsqueeze(0)
                                rank_vision_values.append(selected_prompt)
                                rank_vision_contributions.append((client_id, 0))

                if rank_vision_values:
                    try:
                        all_prompts = torch.cat(rank_vision_values, dim=0)
                        avg_prompt = all_prompts.mean(dim=0, keepdim=True)

                        global_component[f'visual.VPT_{rank}'] = avg_prompt
                        vision_contributions_by_rank[rank] = rank_vision_contributions

                        contributors = [cid for cid, _ in rank_vision_contributions]
                        prompt_indices = [idx for _, idx in rank_vision_contributions]
                        print(f"Vision prompt rank {rank+1} aggregated: {len(contributors)} clients {contributors} (prompt indices {prompt_indices})")
                    except Exception as e:
                        print(f"Vision prompt rank {rank+1} aggregation failed: {str(e)}")
        else:
            for rank in range(topk):
                rank_vision_values = []
                rank_vision_contributions = []

                for client_idx, client_id in enumerate(client_ids):
                    if client_id in vision_prompt_ranks and client_idx < len(vision_prompts_list):
                        if rank < len(vision_prompt_ranks[client_id]):
                            prompt_idx = vision_prompt_ranks[client_id][rank]
                            client_prompt = vision_prompts_list[client_idx]

                            if client_prompt.dim() == 3 and prompt_idx < client_prompt.size(0):
                                selected_prompt = client_prompt[prompt_idx].unsqueeze(0)
                                rank_vision_values.append(selected_prompt)
                                rank_vision_contributions.append((client_id, prompt_idx))
                            elif client_prompt.dim() == 2 and rank == 0:
                                selected_prompt = client_prompt.unsqueeze(0)
                                rank_vision_values.append(selected_prompt)
                                rank_vision_contributions.append((client_id, 0))

                if rank_vision_values:
                    try:
                        all_prompts = torch.cat(rank_vision_values, dim=0)
                        avg_prompt = all_prompts.mean(dim=0, keepdim=True)

                        global_component[f'visual.VPT_{rank}'] = avg_prompt
                        vision_contributions_by_rank[rank] = rank_vision_contributions

                        contributors = [cid for cid, _ in rank_vision_contributions]
                        prompt_indices = [idx for _, idx in rank_vision_contributions]
                        print(f"Vision prompt rank {rank+1} aggregated: {len(contributors)} clients {contributors} (prompt indices {prompt_indices})")
                    except Exception as e:
                        print(f"Vision prompt rank {rank+1} aggregation failed: {str(e)}")

        self.vision_contributions_by_rank = vision_contributions_by_rank
        return global_component

    def _update_global_prompts(self, global_component: Dict[str, torch.Tensor], topk: int) -> None:
        """Update global prompts for next round."""
        global_text_prompts = []
        global_vision_prompts = []

        select_mode = getattr(self.cfg.TRAINER.FEDMGP, 'SELECT_MODE', 'similarity')

        if select_mode == "all":
            max_text_rank = -1
            max_vision_rank = -1

            for key in global_component.keys():
                if key.startswith('prompt_learner.ctx_'):
                    rank = int(key.split('_')[-1])
                    max_text_rank = max(max_text_rank, rank)
                elif key.startswith('visual.VPT_'):
                    rank = int(key.split('_')[-1])
                    max_vision_rank = max(max_vision_rank, rank)

            for rank in range(max_text_rank + 1):
                text_key = f'prompt_learner.ctx_{rank}'
                if text_key in global_component:
                    global_text_prompts.append(global_component[text_key][0])

            for rank in range(max_vision_rank + 1):
                vision_key = f'visual.VPT_{rank}'
                if vision_key in global_component:
                    global_vision_prompts.append(global_component[vision_key][0])
        else:
            for rank in range(topk):
                text_key = f'prompt_learner.ctx_{rank}'
                if text_key in global_component:
                    global_text_prompts.append(global_component[text_key][0])

                vision_key = f'visual.VPT_{rank}'
                if vision_key in global_component:
                    global_vision_prompts.append(global_component[vision_key][0])

        self.global_text_prompts = global_text_prompts
        self.global_vision_prompts = global_vision_prompts

    def _update_client_text_prompts(self, client_id: int, client_weight: Dict,
                                  global_component: Dict[str, torch.Tensor], topk: int) -> None:
        """Update client text prompts."""
        text_prompt_indices = []

        if client_id in self.text_prompt_ranks and 'prompt_learner.ctx' in client_weight:
            ctx = client_weight['prompt_learner.ctx']
            for rank, idx in enumerate(self.text_prompt_ranks[client_id]):
                key = f'prompt_learner.ctx_{rank}'
                if key in global_component and idx < ctx.size(0):
                    ctx[idx] = global_component[key][0]
                    text_prompt_indices.append(idx)

        if text_prompt_indices:
            print(f"Client {client_id}:")
            print(f"  - Text prompt update: indices {text_prompt_indices} (rank order {list(range(len(text_prompt_indices)))})")
            print(f"  - Text prompt index mapping: {dict(enumerate(self.text_prompt_ranks[client_id]))}")

    def _update_client_vision_prompts(self, client_id: int, client_weight: Dict,
                                    global_component: Dict[str, torch.Tensor], topk: int) -> None:
        """Update client vision prompts."""
        vision_prompt_indices = []

        vision_key = 'visual.VPT' if 'visual.VPT' in client_weight else 'image_encoder.VPT'
        if client_id in self.vision_prompt_ranks and vision_key in client_weight:
            vpt = client_weight[vision_key]
            if vpt.dim() == 3:
                for rank, idx in enumerate(self.vision_prompt_ranks[client_id]):
                    key = f'visual.VPT_{rank}'
                    if key in global_component and idx < vpt.size(0):
                        vpt[idx] = global_component[key][0]
                        vision_prompt_indices.append(idx)
            elif vpt.dim() == 2:
                key = 'visual.VPT_0'
                if key in global_component:
                    client_weight[vision_key] = global_component[key][0]
                    vision_prompt_indices.append(0)

        if vision_prompt_indices:
            print(f"  - Vision prompt update: indices {vision_prompt_indices} (rank order {list(range(len(vision_prompt_indices)))})")
            print(f"  - Vision prompt index mapping: {dict(enumerate(self.vision_prompt_ranks[client_id]))}")

    def _extract_visual_prompt_with_fallback(self, client_id: int,
                                           trained_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract vision prompt with fallback mechanism."""
        vision_prompts = {}
        found_visual_vpt = False

        # Try extracting from state_dict first
        for name, param in trained_state_dict.items():
            if any(pattern in name for pattern in ["visual.VPT", "image_encoder.VPT", "VPT"]):
                found_visual_vpt = True
                clean_name = "visual.VPT"
                vision_prompts[clean_name] = copy.deepcopy(param)
                break

        # Fallback: try extracting from model directly
        if not found_visual_vpt:
            possible_paths = [
                (self.trainer.model, 'image_encoder.VPT'),
                (self.trainer.model, 'visual.VPT'),
                (self.trainer.model, 'VPT'),
                (getattr(self.trainer.model, 'module', None), 'image_encoder.VPT'),
                (getattr(self.trainer.model, 'module', None), 'visual.VPT'),
                (getattr(self.trainer.model, 'module', None), 'VPT'),
                (getattr(self.trainer.model, 'image_encoder', None), 'VPT'),
                (getattr(self.trainer.model, 'visual', None), 'VPT'),
                (getattr(getattr(self.trainer.model, 'module', None), 'image_encoder', None), 'VPT'),
                (getattr(getattr(self.trainer.model, 'module', None), 'visual', None), 'VPT'),
            ]

            vpt_param = None
            for obj, attr_path in possible_paths:
                if obj is None:
                    continue

                parts = attr_path.split('.')
                current = obj

                try:
                    for part in parts:
                        current = getattr(current, part)

                    if isinstance(current, torch.nn.Parameter):
                        vpt_param = current
                        break
                except (AttributeError, KeyError):
                    continue

            if vpt_param is not None:
                vision_prompts['visual.VPT'] = copy.deepcopy(vpt_param.data)

        return vision_prompts

    def _compute_pre_training_similarities(self, client_ids: List[int],
                                         global_component: Dict[str, torch.Tensor]) -> None:
        """Compute pre-training similarities after parameter update."""
        print(f"\nComputing round {self.current_round} pre-training similarities:")

        text_prompts_list = self._extract_text_prompts(client_ids)
        vision_prompts_list = self._extract_vision_prompts(client_ids)

        current_global_text_prompts = []
        current_global_vision_prompts = []

        topk = getattr(self.cfg.TRAINER.FEDMGP, 'TOPK', 2)
        select_mode = getattr(self.cfg.TRAINER.FEDMGP, 'SELECT_MODE', 'similarity')

        if select_mode == "all":
            max_text_rank = -1
            max_vision_rank = -1

            for key in global_component.keys():
                if key.startswith('prompt_learner.ctx_'):
                    rank = int(key.split('_')[-1])
                    max_text_rank = max(max_text_rank, rank)
                elif key.startswith('visual.VPT_'):
                    rank = int(key.split('_')[-1])
                    max_vision_rank = max(max_vision_rank, rank)

            for rank in range(max_text_rank + 1):
                text_key = f'prompt_learner.ctx_{rank}'
                if text_key in global_component:
                    current_global_text_prompts.append(global_component[text_key][0])

            for rank in range(max_vision_rank + 1):
                vision_key = f'visual.VPT_{rank}'
                if vision_key in global_component:
                    current_global_vision_prompts.append(global_component[vision_key][0])
        else:
            for rank in range(topk):
                text_key = f'prompt_learner.ctx_{rank}'
                if text_key in global_component:
                    current_global_text_prompts.append(global_component[text_key][0])

                vision_key = f'visual.VPT_{rank}'
                if vision_key in global_component:
                    current_global_vision_prompts.append(global_component[vision_key][0])

        for client_idx, client_id in enumerate(client_ids):
            updated_text_indices = self.text_prompt_ranks.get(client_id, [])
            updated_vision_indices = self.vision_prompt_ranks.get(client_id, [])

            # Compute text prompt similarities
            if client_idx < len(text_prompts_list) and current_global_text_prompts:
                client_text_prompt = text_prompts_list[client_idx]
                text_similarities = self._compute_prompt_similarities(
                    client_text_prompt, current_global_text_prompts)

                text_sim_parts = []
                for idx, sim in text_similarities:
                    if idx in updated_text_indices:
                        text_sim_parts.append(f"{idx}:{sim:.4f}*")
                    else:
                        text_sim_parts.append(f"{idx}:{sim:.4f}")

                text_sim_str = ", ".join(text_sim_parts)
                print(f"Client {client_id}: Pre-training text similarities [{text_sim_str}] (updated: {updated_text_indices})")

            # Compute vision prompt similarities
            if client_idx < len(vision_prompts_list) and current_global_vision_prompts:
                client_vision_prompt = vision_prompts_list[client_idx]
                vision_similarities = self._compute_prompt_similarities(
                    client_vision_prompt, current_global_vision_prompts)

                vision_sim_parts = []
                for idx, sim in vision_similarities:
                    if idx in updated_vision_indices:
                        vision_sim_parts.append(f"{idx}:{sim:.4f}*")
                    else:
                        vision_sim_parts.append(f"{idx}:{sim:.4f}")

                vision_sim_str = ", ".join(vision_sim_parts)
                print(f"Client {client_id}: Pre-training vision similarities [{vision_sim_str}] (updated: {updated_vision_indices})")
