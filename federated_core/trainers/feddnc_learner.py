# federated_core/trainers/feddnc_learner.py

from typing import Dict, List, Any
import torch
import torch.nn.functional as F
from ..base_federated_learner import BaseFederatedLearner
from ..fed_utils import average_weights


class FedDnCLearner(BaseFederatedLearner):
    """
    Dual-stream federated aggregation:
      - ctx_b  -> FedAvg (consensus-seeking, discriminative stream)
      - ctx_n  -> EMA-weighted aggregation (generalization-preserving stream)
    """

    def _get_trainer_specific_attributes(self) -> Dict[str, Any]:
        return {'global_ctx_n_prev': None}

    def _initialize_components_storage(self) -> None:
        n = self.cfg.DATASET.USERS
        self.client_weights['components'] = {
            'ctx_b': [{} for _ in range(n)],
            'ctx_n': [{} for _ in range(n)],
        }

    def aggregate_models(self, client_ids: List[int]) -> Dict[str, torch.Tensor]:
        # stream 1: ctx_b uses standard FedAvg
        global_ctx_b = average_weights(
            self.client_weights['components']['ctx_b'],
            client_ids, self.client_data_sizes, islist=False
        )

        # stream 2: ctx_n aggregation (EMA by default; falls back to FedAvg when ablated)
        fedavg_ctx_n = average_weights(
            self.client_weights['components']['ctx_n'],
            client_ids, self.client_data_sizes, islist=False
        )['ctx_n']
        ablation_fedavg = getattr(self.cfg.TRAINER.FEDNC, 'ABLATION_CTX_N_FEDAVG', False)
        if ablation_fedavg or self.global_ctx_n_prev is None:
            new_ctx_n = fedavg_ctx_n
        else:
            ema_alpha = getattr(self.cfg.TRAINER.FEDNC, 'EMA_ALPHA', -1.0)
            if ema_alpha >= 0:
                # EMA_ALPHA = momentum (keep weight); ema_n = update weight = 1 - alpha
                ema_n = float(1.0 - ema_alpha)
            else:
                # fewer shots -> scarcer gradient signal -> larger EMA step to speed up convergence
                shots = max(getattr(self.cfg.DATASET, 'NUM_SHOTS', 4), 1)
                ema_n = min(0.4, 0.4 / shots)   # shots=1->0.40, 2->0.20, 4->0.10, 8->0.05
            new_ctx_n = (1.0 - ema_n) * self.global_ctx_n_prev['ctx_n'].to(fedavg_ctx_n.device) \
                        + ema_n * fedavg_ctx_n
        self.global_ctx_n_prev = {'ctx_n': new_ctx_n.clone()}
        return {'ctx_b': global_ctx_b['ctx_b'], 'ctx_n': new_ctx_n}

    def _similarity_guided_aggregate(
        self,
        client_ctx_n_list: List[Dict],
        client_ids: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Cosine-similarity-weighted aggregation of ctx_n.
        Falls back to FedAvg on the first round when no previous global ctx_n exists.
        """
        if self.global_ctx_n_prev is None:
            return average_weights(
                client_ctx_n_list,
                client_ids,
                self.client_data_sizes,
                islist=False
            )

        prev = self.global_ctx_n_prev['ctx_n'].float()   # [N, n_ctx, dim]
        prev_flat = prev.reshape(-1)

        weights = []
        for cid in client_ids:
            local = client_ctx_n_list[cid].get('ctx_n')
            if local is None:
                weights.append(0.0)
                continue
            local_flat = local.float().reshape(-1)
            sim = F.cosine_similarity(
                prev_flat.unsqueeze(0),
                local_flat.unsqueeze(0)
            ).item()
            weights.append(sim)

        w = torch.tensor(weights)
        w = torch.softmax(w, dim=0)

        result_ctx_n = None
        for idx, cid in enumerate(client_ids):
            local = client_ctx_n_list[cid].get('ctx_n')
            if local is None:
                continue
            weighted = local.float() * w[idx].item()
            result_ctx_n = weighted if result_ctx_n is None \
                           else result_ctx_n + weighted

        return {'ctx_n': result_ctx_n.to(
            client_ctx_n_list[client_ids[0]]['ctx_n'].dtype
        )}

    def update_client_models(
        self,
        client_ids: List[int],
        global_component: Dict[str, torch.Tensor],
        all_clients: List[int] = None
    ) -> None:
        if all_clients is None:
            all_clients = client_ids

        for client_id in all_clients:
            client_weight = self.client_weights['local'][client_id]
            if 'ctx_b' in global_component:
                client_weight['prompt_learner.ctx_b'] = global_component['ctx_b'].clone()
            if 'ctx_n' in global_component:
                client_weight['prompt_learner.ctx_n'] = global_component['ctx_n'].clone()

    def _store_components(
        self,
        client_id: int,
        trained_state_dict: Dict[str, torch.Tensor]
    ) -> None:
        ctx_b_key = next((k for k in trained_state_dict if k.endswith('ctx_b')), None)
        if ctx_b_key:
            self.client_weights['components']['ctx_b'][client_id] = {
                'ctx_b': trained_state_dict[ctx_b_key].detach().clone()
            }

        ctx_n_key = next((k for k in trained_state_dict if k.endswith('ctx_n')), None)
        if ctx_n_key:
            self.client_weights['components']['ctx_n'][client_id] = {
                'ctx_n': trained_state_dict[ctx_n_key].detach().clone()
            }
