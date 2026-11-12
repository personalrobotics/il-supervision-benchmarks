"""Implicit behavioral cloning, derivative-free (Florence et al. 2021), ported from diffusion_policy's
IbcDfoLowdimPolicy (5ba07ac) with its default training and inference paths (andy_train False, kevin_inference False).

The energy is a scalar MLP over [observation, action]; its output is a logit (low energy = high logit). Training is
InfoNCE: the demonstrated action, jittered by 1e-4, against train_n_neg actions uniform over the normalized action
box [-1, 1]. Inference draws pred_n_samples uniform actions, then pred_n_iter - 1 times resamples them in proportion
to softmax(logits) and adds N(0, resample_std) noise, and finally draws one action from the last softmax.
"""
import torch
import torch.nn.functional as F
from lerobot.configs import FeatureType
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from torch import Tensor, nn

from .configuration_ibc import IBCConfig


class IBCPolicy(PreTrainedPolicy):
    config_class = IBCConfig
    name = "ibc"

    def __init__(self, config: IBCConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.obs_keys = [k for k, f in config.input_features.items() if f.type in (FeatureType.STATE, FeatureType.ENV)]
        d_obs = sum(config.input_features[k].shape[0] for k in self.obs_keys)
        self.d_act = config.action_feature.shape[0]
        layers, width = [], d_obs + self.d_act
        for _ in range(config.n_layers):
            layers += [nn.Linear(width, config.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout)]
            width = config.hidden_dim
        self.energy = nn.Sequential(*layers, nn.Linear(width, 1))
        # compiled for training only, and kept out of the module tree so the checkpoint's keys are unchanged
        self.__dict__["_compiled"] = torch.compile(self.energy) if config.compile else None

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        pass

    def observation(self, batch: dict[str, Tensor]) -> Tensor:
        return torch.cat([batch[k] for k in self.obs_keys], dim=-1)

    def logits(self, obs: Tensor, actions: Tensor) -> Tensor:
        """obs (B, Do), actions (B, N, Da) -> (B, N)."""
        b, n, _ = actions.shape
        x = torch.cat([obs.unsqueeze(1).expand(-1, n, -1), actions], dim=-1)
        net = self._compiled if self._compiled is not None and self.training else self.energy
        return net(x.reshape(b * n, -1)).reshape(b, n)

    def infonce(self, obs: Tensor, action: Tensor, negatives: Tensor) -> Tensor:
        """Mean cross-entropy of the demonstrated action (index 0) against the negatives, per batch row."""
        logits = self.logits(obs, torch.cat([action.unsqueeze(1), negatives], dim=1))
        return F.cross_entropy(logits, torch.zeros(len(obs), dtype=torch.long, device=obs.device), reduction="none")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        obs, action = self.observation(batch), batch[ACTION][:, 0]
        action = action + torch.randn_like(action) * self.config.positive_noise_std
        negatives = torch.rand(len(obs), self.config.train_n_neg, self.d_act, device=obs.device) * 2 - 1
        loss = self.infonce(obs, action, negatives).mean()
        return loss, {"infonce": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        obs = self.observation(batch)
        n = self.config.pred_n_samples
        samples = torch.rand(len(obs), n, self.d_act, device=obs.device) * 2 - 1
        rows = torch.arange(len(obs), device=obs.device).unsqueeze(-1)
        for i in range(self.config.pred_n_iter):
            prob = torch.softmax(self.logits(obs, samples), dim=-1)
            if i < self.config.pred_n_iter - 1:
                samples = samples[rows, torch.multinomial(prob, n, replacement=True)]
                samples = samples + torch.randn_like(samples) * self.config.resample_std
        return samples[rows, torch.multinomial(prob, 1, replacement=True)]  # (B, 1, Da)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        return self.predict_action_chunk(batch)[:, 0]
