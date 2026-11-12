"""Configuration of the IBC policy: the published PushT low-dimensional setting of IbcDfoLowdimPolicy.

Reference: real-stanford/diffusion_policy at 5ba07ac, config/train_ibc_dfo_lowdim_workspace.yaml and
policy/ibc_dfo_lowdim_policy.py: a 4 x 1024 ReLU MLP energy with dropout 0.1, InfoNCE against 1024 uniform negatives,
derivative-free inference with 1024 samples and 5 iterations, min/max normalization to [-1, 1], AdamW at 1e-4 with a
cosine schedule after 500 warmup steps. The reference's two observation steps are carried inside each observation
feature (tools/pusht_v2/derive.py), so n_obs_steps is 1 here.
"""
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("ibc")
@dataclass
class IBCConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    hidden_dim: int = 1024
    n_layers: int = 4
    dropout: float = 0.1
    train_n_neg: int = 1024
    positive_noise_std: float = 1e-4
    pred_n_samples: int = 1024
    pred_n_iter: int = 5
    resample_std: float = 3e-2
    # speed, not the method (pusht-v2 development evidence): lerobot-train reads dtype to choose Accelerate's mixed
    # precision (bfloat16: bf16 autocast over float32 weights); compile runs the energy network compiled in training
    dtype: str = "float32"
    compile: bool = False

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 100_000  # set to the run's step count: the cosine reaches 0 at its end

    def __post_init__(self):
        super().__post_init__()
        if self.n_obs_steps != 1:
            raise ValueError("ibc reads its history from each observation feature; n_obs_steps must be 1")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError(f"ibc dtype {self.dtype!r} is not float32 or bfloat16")

    def get_optimizer_preset(self) -> AdamWConfig:
        # the reference clips no gradients
        return AdamWConfig(lr=self.optimizer_lr, betas=self.optimizer_betas, eps=self.optimizer_eps,
                           weight_decay=self.optimizer_weight_decay, grad_clip_norm=1e12)

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps,
                                                    num_decay_steps=self.scheduler_decay_steps,
                                                    peak_lr=self.optimizer_lr, decay_lr=0.0)

    def validate_features(self) -> None:
        if self.image_features:
            raise ValueError("ibc takes low-dimensional observations only")
        if not self.robot_state_feature and not self.env_state_feature:
            raise ValueError("ibc needs observation.state or observation.environment_state")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return [0]

    @property
    def reward_delta_indices(self) -> None:
        return None
