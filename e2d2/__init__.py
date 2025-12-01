"""Toy MLX reimplementation of E2D2 (arXiv:2510.22852).

Encoder-decoder block diffusion: an encoder represents clean tokens once, a
lightweight decoder iteratively denoises a noised block, and the encoder is
invoked once per block rather than once per denoising step.

Public API::

    from e2d2 import E2D2, E2D2Config, SampleConfig, sample_blockwise
"""

from .diffusion import (
    DEFAULT_SCHEDULE,
    LinearNoiseSchedule,
    corruption_mask,
    discrete_t_grid,
    forward_corrupt,
    forward_corrupt_per_block,
    nelbo_loss,
    nelbo_loss_per_block,
    sample_train_times,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SCHEDULE",
    "LinearNoiseSchedule",
    "corruption_mask",
    "discrete_t_grid",
    "forward_corrupt",
    "forward_corrupt_per_block",
    "nelbo_loss",
    "nelbo_loss_per_block",
    "sample_train_times",
]
