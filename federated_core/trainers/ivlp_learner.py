# federated_core/trainers/ivlp_learner.py

from typing import Dict, List, Any
import torch
import copy
from ..base_federated_learner import BaseFederatedLearner
from ..fed_utils import average_weights, split_ivlp_params

class IVLPLearner(BaseFederatedLearner):
    """IVLP implementation with selective aggregation of vision/language parameters."""

    def _get_trainer_specific_attributes(self) -> Dict[str, Any]:
        return {}

    def _initialize_components_storage(self) -> None:
        self.client_weights['components'] = {
            'ivlp_vision': [{} for _ in range(self.cfg.DATASET.USERS)],
            'ivlp_language': [{} for _ in range(self.cfg.DATASET.USERS)]
        }

    def aggregate_models(self, client_ids: List[int]) -> Dict[str, torch.Tensor]:
        """Aggregate parameters based on mode: all/vision/language."""
        aggregate_mode = getattr(self.cfg.TRAINER.IVLP, 'AGGREGATE', "all")

        if aggregate_mode == "all":
            # Merge vision and language parameters
            all_params = {}
            for client_id in client_ids:
                client_params = {}
                client_params.update(self.client_weights['components']['ivlp_vision'][client_id])
                client_params.update(self.client_weights['components']['ivlp_language'][client_id])
                all_params[client_id] = client_params
            component_to_aggregate = all_params
        elif aggregate_mode == "vision":
            component_to_aggregate = self.client_weights['components']['ivlp_vision']
        elif aggregate_mode == "language":
            component_to_aggregate = self.client_weights['components']['ivlp_language']
        else:
            raise ValueError(f"Unsupported aggregate mode: {aggregate_mode}")

        global_params = average_weights(
            component_to_aggregate,
            client_ids,
            self.client_data_sizes,
            islist=False
        )
        return global_params

    def update_client_models(self, client_ids: List[int],
                           global_component: Dict[str, torch.Tensor],
                           all_clients: List[int] = None) -> None:
        """Update based on mode, preserving local parameters as needed."""
        if all_clients is None:
            all_clients = client_ids

        aggregate_mode = getattr(self.cfg.TRAINER.IVLP, 'AGGREGATE', "all")

        if len(all_clients) > 0:
            print(f"IVLP update mode: {aggregate_mode}")
            print(f"Global params count: {len(global_component)}")

        for client_id in all_clients:
            client_weight = self.client_weights['local'][client_id]

            # Update global aggregated parameters
            for key, value in global_component.items():
                client_weight[key] = value

            # Preserve local parameters based on mode
            if aggregate_mode == "vision":
                for key, value in self.client_weights['components']['ivlp_language'][client_id].items():
                    client_weight[key] = value
                print(f"Client {client_id} updated vision params, preserved local language params")

            elif aggregate_mode == "language":
                for key, value in self.client_weights['components']['ivlp_vision'][client_id].items():
                    client_weight[key] = value
                print(f"Client {client_id} updated language params, preserved local vision params")

            elif aggregate_mode == "all":
                print(f"Client {client_id} updated all params")

    def _store_components(self, client_id: int, trained_state_dict: Dict[str, torch.Tensor]) -> None:
        """Separate and store vision and language parameters."""
        vision_params, language_params = self._split_ivlp_params(trained_state_dict)
        self.client_weights['components']['ivlp_vision'][client_id] = vision_params
        self.client_weights['components']['ivlp_language'][client_id] = language_params

    def _split_ivlp_params(self, state_dict: Dict[str, torch.Tensor]) -> tuple:
        """Split IVLP model parameters into vision and language parts."""
        vision_params = {}
        language_params = {}

        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad and name in state_dict:
                if 'visual' in name or 'image_encoder' in name or ('VPT' in name and not 'text_encoder' in name):
                    vision_params[name] = copy.deepcopy(state_dict[name])
                elif 'prompt_learner.ctx' in name or ('VPT' in name and 'text_encoder' in name):
                    language_params[name] = copy.deepcopy(state_dict[name])

        return vision_params, language_params
