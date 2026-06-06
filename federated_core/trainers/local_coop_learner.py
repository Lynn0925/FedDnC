# federated_core/trainers/local_coop_learner.py

from typing import Dict, List, Any, Optional
import torch
from ..base_federated_learner import BaseFederatedLearner


class LocalCoOpLearner(BaseFederatedLearner):
    """
    Truly non-federated per-client CoOp baseline.
    Each client trains its own prompt independently — no weight sharing across clients.
    Equivalent to running CoOp on each source domain in isolation.
    """

    def _get_trainer_specific_attributes(self) -> Dict[str, Any]:
        return {}

    def _initialize_components_storage(self) -> None:
        self.client_weights['components'] = {
            'prompt_learner': [{} for _ in range(self.cfg.DATASET.USERS)]
        }

    # ------------------------------------------------------------------ #
    # No aggregation: each client owns its own prompt the whole time       #
    # ------------------------------------------------------------------ #
    def aggregate_models(self, client_ids: List[int]):
        return None  # nothing to share

    def update_client_models(
        self,
        client_ids: List[int],
        global_component,
        all_clients: Optional[List[int]] = None
    ) -> None:
        pass  # intentionally empty — clients never receive foreign weights

    def _store_components(self, client_id: int, trained_state_dict: Dict[str, torch.Tensor]) -> None:
        trainable_params = self._get_trainable_params(state_dict=trained_state_dict)
        self.client_weights['local'][client_id] = trainable_params

    # ------------------------------------------------------------------ #
    # Override weight loading so each client always resumes its OWN state #
    # ------------------------------------------------------------------ #
    def _load_client_weights_for_training(self, client_id: int, round_idx: int) -> None:
        if round_idx == 0:
            # first round: everyone starts from the same random initialisation
            self._load_trainable_params(self.trainer.model, self.client_weights['global'])
        else:
            # subsequent rounds: resume from this client's last checkpoint
            self._load_trainable_params(self.trainer.model, self.client_weights['local'][client_id])
