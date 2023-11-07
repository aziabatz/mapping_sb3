import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from PointerNetwork import PolicyNetwork
from gymnasium import spaces
from torch import nn

from typing import Optional, Union, Type, Dict, Any, Tuple
import torch as th

class PtrNetPolicy(BasePolicy):

    # FIXME Pass parameters as dictionary
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            params: Dict[str, Any],
            device,
            lr_schedule: Schedule,
            activation_fn: Type[nn.Module] == nn.Tahn, # Maybe unused
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True, # For actor/critic, but not needed
            normalize_images: bool = False, # Not needed
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            ):

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images
        )

        if device is None:
            device = "cpu"

        self.policy_network = PolicyNetwork(params, device)
        self._build(optimizer_kwargs=optimizer_kwargs,
                    optimizer_class=optimizer_class,
                    lr_schedule=None) # Ignore this for now

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:

        pass

    def _build(self, optimizer_kwargs, lr_schedule, optimizer_class: Type[th.optim.Optimizer]):
        # FIXME Do weights initialization here
        # self.policy_network.__initialize_weights()
        # Setup optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        self.optimizer = optimizer_class(self.parameters(), **self.optimizer_kwargs)

    # def forward(self, observation):
    #     action = self.policy_network.forward(observation)
    #     return action

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:

        observation = observation.to(self.device)
        self.set_training_mode(False)

        mapping, _ = self.policy_network(observation) # ignore log likehood?

        # TODO convert mapping to action
        action = mapping

        return action

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        return super()._dummy_schedule(progress_remaining)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        return super().scale_action(action)

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        return super().unscale_action(scaled_action)




