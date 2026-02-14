from typing import Optional

import torch
import torch.nn as nn

from skrl.skrl.models.torch import DeterministicMixin, Model


class ModuleWrapper(DeterministicMixin, Model):
    def __init__(
        self,
        module: nn.Module,
        device: Optional[str] = None,
        observation_space=None,
        action_space=None,
        clip_actions=False,
    ):
        if device is None:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # init base classes
        Model.__init__(self, observation_space, action_space, self.device)
        DeterministicMixin.__init__(self, clip_actions)

        self._unwrapped_module = module.to(self.device)

    def compute(self, inputs: dict, role):
        # pick states and actions if available
        return self._unwrapped_module.forward(**inputs), {}
