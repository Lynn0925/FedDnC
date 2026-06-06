# federated_core/trainers/fedpgp_learner.py

from typing import Dict, List, Any
import torch
import copy
from ..base_federated_learner import BaseFederatedLearner
from ..fed_utils import average_weights

class FedPGPLearner(BaseFederatedLearner):
    """FedPGP implementation with separate aggregation of sigma, U, V components."""

    def _get_trainer_specific_attributes(self) -> Dict[str, Any]:
        return {}

    def _initialize_components_storage(self) -> None:
        self.client_weights['components'] = {
            'sigma': [[] for _ in range(self.cfg.DATASET.USERS)],
            'U': [[] for _ in range(self.cfg.DATASET.USERS)],
            'V': [[] for _ in range(self.cfg.DATASET.USERS)]
        }

    def aggregate_models(self, client_ids: List[int]) -> torch.Tensor:
        """Aggregate only the sigma parameter."""
        global_sigma = average_weights(
            self.client_weights['components']['sigma'],
            client_ids,
            self.client_data_sizes,
            islist=True
        )
        return global_sigma

    def update_client_models(self, client_ids: List[int],
                           global_component: torch.Tensor,
                           all_clients: List[int] = None) -> None:
        """Use global sigma, keep local U and V."""
        if all_clients is None:
            all_clients = client_ids

        for client_id in all_clients:
            self.client_weights['local'][client_id]['prompt_learner.sigma'] = global_component

            # Keep client-specific U and V (only for participating clients)
            if client_id in client_ids:
                self.client_weights['local'][client_id]['prompt_learner.U'] = \
                    self.client_weights['components']['U'][client_id]
                self.client_weights['local'][client_id]['prompt_learner.V'] = \
                    self.client_weights['components']['V'][client_id]

            print(f"Client {client_id} FedPGP update completed")

    def _store_components(self, client_id: int, trained_state_dict: Dict[str, torch.Tensor]) -> None:
        """Store sigma, U, V components."""
        sigma_key = next((k for k in trained_state_dict if k.endswith('prompt_learner.sigma') or k == 'prompt_learner.sigma'), None)
        u_key = next((k for k in trained_state_dict if k.endswith('prompt_learner.U') or k == 'prompt_learner.U'), None)
        v_key = next((k for k in trained_state_dict if k.endswith('prompt_learner.V') or k == 'prompt_learner.V'), None)

        if sigma_key is None or u_key is None or v_key is None:
            available = [k for k in trained_state_dict if 'prompt_learner' in k]
            raise KeyError(
                f"FedPGP: expected prompt_learner.sigma/U/V in state_dict, "
                f"found prompt_learner keys: {available}"
            )

        self.client_weights['components']['sigma'][client_id] = copy.deepcopy(trained_state_dict[sigma_key])
        self.client_weights['components']['U'][client_id] = copy.deepcopy(trained_state_dict[u_key])
        self.client_weights['components']['V'][client_id] = copy.deepcopy(trained_state_dict[v_key])
