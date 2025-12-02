"""Toy MLX reimplementation of E2D2 (arXiv:2510.22852).

Encoder-decoder block diffusion: an encoder represents clean tokens once, a
lightweight decoder iteratively denoises a noised block, and the encoder is
invoked once per block rather than once per denoising step.

Public API::

    from e2d2 import E2D2, E2D2Config, SampleConfig, sample_blockwise
"""

from .metrics import (
    count_parameters,
    exact_match,
    nelbo_estimate,
    per_position_accuracy,
    token_accuracy,
)
from .model import CallCounts, E2D2, E2D2Config
from .sampling import (
    SampleConfig,
    SampleResult,
    clean_counts,
    one_shot_predict,
    predict_tokens,
    sample_blockwise,
    unmask_step,
)
from .task import (
    BOS_ID,
    DIGIT_OFFSET,
    MASK_ID,
    OP_ID,
    TaskSpec,
    all_prompts,
    build_examples,
    held_out_prompts,
    make_batch,
)
from .layers import Attention, Block, MLP, ModelConfig, Stack
from .masks import (
    additive_mask,
    block_decode_mask,
    decoder_block_diagonal,
    decoder_only_mask,
    e2d2_decoder_mask,
    encoder_block_causal,
)
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
    "Attention",
    "BOS_ID",
    "Block",
    "CallCounts",
    "E2D2",
    "E2D2Config",
    "DEFAULT_SCHEDULE",
    "DIGIT_OFFSET",
    "MASK_ID",
    "MLP",
    "OP_ID",
    "SampleConfig",
    "SampleResult",
    "ModelConfig",
    "Stack",
    "TaskSpec",
    "additive_mask",
    "all_prompts",
    "build_examples",
    "clean_counts",
    "count_parameters",
    "exact_match",
    "held_out_prompts",
    "make_batch",
    "nelbo_estimate",
    "one_shot_predict",
    "per_position_accuracy",
    "predict_tokens",
    "sample_blockwise",
    "token_accuracy",
    "unmask_step",
    "block_decode_mask",
    "decoder_block_diagonal",
    "decoder_only_mask",
    "e2d2_decoder_mask",
    "encoder_block_causal",
    "LinearNoiseSchedule",
    "corruption_mask",
    "discrete_t_grid",
    "forward_corrupt",
    "forward_corrupt_per_block",
    "nelbo_loss",
    "nelbo_loss_per_block",
    "sample_train_times",
]
