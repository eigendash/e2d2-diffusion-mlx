"""Evaluation metrics for the toy task and for the efficiency claims.

Quality is measured two ways: exact match of the whole answer block (the strict
metric) and per-token accuracy over the answer block.  For a likelihood proxy the
masked-diffusion NELBO is estimated by averaging the weighted cross-entropy over a
grid of noise levels, which is the same objective the training loop minimises and
is an upper bound on ``-log p(x)`` under the variational bound the paper
reproduces.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.utils
import mlx.nn as nn

from .diffusion import DEFAULT_SCHEDULE, corruption_mask, forward_corrupt_per_block


def exact_match(predicted: mx.array, target: mx.array) -> float:
    """Fraction of rows where every token of the answer matches."""
    if predicted.shape != target.shape:
        raise ValueError("shapes must agree")
    same = mx.all(predicted == target, axis=1)
    return float(mx.mean(same.astype(mx.float32)).item())


def token_accuracy(predicted: mx.array, target: mx.array) -> float:
    return float(mx.mean((predicted == target).astype(mx.float32)).item())


def per_position_accuracy(predicted: mx.array, target: mx.array) -> list[float]:
    return [
        float(mx.mean((predicted[:, i] == target[:, i]).astype(mx.float32)).item())
        for i in range(target.shape[1])
    ]


def nelbo_estimate(
    model,
    tokens: mx.array,
    prompt_len: int,
    grid_steps: int = 32,
    n_samples: int = 4,
    seed: int = 0,
) -> float:
    """Importance-weighted answer-position cross-entropy averaged over a t grid.

    This is the training objective's own estimand, evaluated on a fixed grid
    instead of a random ``t``: for each level ``t = i / T`` the corruption is
    resampled ``n_samples`` times and the answer-half positions are scored with the
    masked-diffusion weight ``alpha'_t / (1 - alpha_t) = 1 / (1 - t)``.

    It is *not* a rigorous likelihood bound.  The paper's Eq. 4 is a NELBO whose
    expectation over ``t`` the sampler cancels in closed form; averaging over a
    finite grid here is a Monte-Carlo approximation of that integral that ignores
    the reconstruction and prior terms (both of which the paper shows vanish as
    ``T -> infinity``, Appendix A.3 Eqs. 5-6).  Call it a surrogate: valid for
    comparing two models trained the same way, not for quoting a perplexity.

    ``t`` is drawn **higher for the answer block only** -- the prompt block keeps
    ``t = 0`` -- because the question being asked is how well the model predicts
    the answer tokens given a clean prompt, which is the sampling-time setting.
    """
    cfg = model.config
    block_size = cfg.block_size
    answer_block_start = prompt_len // block_size
    n_blocks = tokens.shape[1] // block_size
    key = mx.random.key(seed)
    per_level = []
    for i in range(grid_steps):
        t_scalar = (i + 1) / grid_steps  # skip t = 0, whose weight is degenerate
        t = mx.zeros((tokens.shape[0], n_blocks), dtype=mx.float32)
        t = mx.concatenate(
            [
                t[:, :answer_block_start],
                mx.full(
                    (tokens.shape[0], n_blocks - answer_block_start),
                    t_scalar,
                    dtype=mx.float32,
                ),
            ],
            axis=1,
        )
        total, weight_sum = 0.0, 0.0
        for _ in range(n_samples):
            key, k1 = mx.random.split(key, 2)
            noisy = forward_corrupt_per_block(
                tokens, t, block_size, cfg.vocab_size - 1, k1
            )
            logits = model.denoise_logits_full(noisy)
            masked = corruption_mask(tokens, noisy).astype(mx.float32)
            masked_ans = mx.concatenate(
                [
                    mx.zeros((tokens.shape[0], prompt_len), dtype=mx.float32),
                    masked[:, prompt_len:],
                ],
                axis=1,
            )
            weight = DEFAULT_SCHEDULE.loss_weight(mx.array(t, dtype=mx.float32))
            weight = mx.repeat(weight, block_size, axis=1) * masked_ans
            logp = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1) + 1e-30)
            target_logp = mx.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]
            denominator = float(weight.sum().item())
            if denominator == 0.0:
                continue
            total += float((-target_logp * weight).sum().item()) / denominator
            weight_sum += 1.0
        if weight_sum:
            per_level.append(total / weight_sum)
    if not per_level:
        return float("nan")
    return sum(per_level) / len(per_level)


def count_parameters(model: nn.Module) -> int:
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
