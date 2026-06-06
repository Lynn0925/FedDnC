# federated_core/trainers/fedopt_learner.py

from typing import Dict, List, Any
import torch
import copy
from ..base_federated_learner import BaseFederatedLearner
from ..fed_utils import average_weights

class FedOPTLearner(BaseFederatedLearner):
    """FedOPT/PromptFolio implementation with separate global/local prompts (ctx_0/ctx_1)."""

    def _get_trainer_specific_attributes(self) -> Dict[str, Any]:
        return {}

    def _initialize_components_storage(self) -> None:
        self.client_weights['components'] = {
            'ctx_0': [[] for _ in range(self.cfg.DATASET.USERS)],  # Global part
            'ctx_1': [[] for _ in range(self.cfg.DATASET.USERS)]   # Local part
        }

    def aggregate_models(self, client_ids: List[int]) -> torch.Tensor:
        """Aggregate only the global prompt part."""
        global_ctx_0 = average_weights(
            self.client_weights['components']['ctx_0'],
            client_ids,
            self.client_data_sizes,
            islist=True
        )
        return global_ctx_0

    def update_client_models(self, client_ids: List[int],
                           global_component: torch.Tensor,
                           all_clients: List[int] = None) -> None:
        """Concatenate global and local prompts."""
        if all_clients is None:
            all_clients = client_ids

        for client_id in all_clients:
            self.client_weights['local'][client_id]['prompt_learner.ctx'] = torch.cat(
                [global_component, self.client_weights['components']['ctx_1'][client_id]],
                dim=0
            )
            print(f"Client {client_id} FedOPT update completed")

    def _store_components(self, client_id: int, trained_state_dict: Dict[str, torch.Tensor]) -> None:
        """Separate ctx into global and local parts."""
        ctx = trained_state_dict['prompt_learner.ctx']
        self.client_weights['components']['ctx_0'][client_id] = copy.deepcopy(ctx[:1])
        self.client_weights['components']['ctx_1'][client_id] = copy.deepcopy(ctx[1:])
