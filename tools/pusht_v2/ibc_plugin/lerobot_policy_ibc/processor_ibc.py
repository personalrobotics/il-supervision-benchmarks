from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_ibc import IBCConfig


def make_ibc_pre_post_processors(
    config: IBCConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]:
    """LeRobot's default pipelines: normalization (min/max to [-1, 1]), batching and device placement."""
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
