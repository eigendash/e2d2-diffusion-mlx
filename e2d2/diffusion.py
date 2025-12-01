"""Training-time corruption process and loss weighting.

Absorbing-state (masked) discrete diffusion, as used by MDLM and taken up by the
block-diffusion parameterisation in the paper this repo reimplements.

The forward process applied to a clean token ``x`` at time ``t`` is

    q(z_t | x) = Cat(z_t ; alpha_t * x + (1 - alpha_t) * m)

with ``m`` the one-hot ``[MASK]`` state.  For the linear schedule used here
``alpha_t = 1 - t``, so every token is independently replaced by ``[MASK]`` with
probability ``t``.  ``t = 0`` is clean data, ``t = 1`` is the all-mask prior.

The simplified NELBO that the paper reproduces (its Appendix A.3, Eq. 4) is a
weighted sum of cross-entropies over masked positions:

    L = E_t E_q [ (alpha'_t / (1 - alpha_t)) * -log p_theta(x | z_t) ]

For ``alpha_t = 1 - t`` we have ``alpha'_t = -1`` and therefore the per-masked-token
weight ``1 / (1 - t)``.  In the discrete-time parameterisation with ``T`` steps
(``t = i / T``) that is ``T / (T - i)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class LinearNoiseSchedule:
    """``alpha_t = 1 - t``, the linear masking schedule.

    ``t`` is noise *level*: ``alpha(0) = 1`` (clean) and ``alpha(1) = 0`` (all
    mask), following the paper's convention that ``alpha_t`` decreases in ``t``.
    """

    def alpha(self, t):
        return 1.0 - t

    def mask_prob(self, t):
        """Probability that a token is corrupted to ``[MASK]`` at level ``t``."""
        return t

    def loss_weight(self, t):
        """``alpha'_t / (1 - alpha_t)``; ``0`` at ``t = 1`` where it is undefined."""
        t = mx.array(t, dtype=mx.float32)
        return mx.where(t >= 1.0, mx.zeros_like(t), 1.0 / mx.maximum(1.0 - t, 1e-6))


DEFAULT_SCHEDULE = LinearNoiseSchedule()


def discrete_t_grid(num_steps: int, schedule=DEFAULT_SCHEDULE):
    """The ``t`` grid ``{0, 1/T, ..., (T-1)/T}`` used for training-time weighting.

    ``t = 1`` is excluded: at that level every token is masked and the weight
    ``alpha'_t / (1 - alpha_t)`` is undefined.  The paper takes ``T -> infinity``
    and integrates over ``t in [0, 1)`` anyway.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be >= 2")
    return [i / num_steps for i in range(num_steps)]


def sample_train_times(num_steps: int, batch_size: int, key) -> mx.array:
    """Uniformly sample the discrete training grid, one ``t`` per batch element."""
    idx = mx.random.randint(0, num_steps, (batch_size,), key=key)
    return idx.astype(mx.float32) / float(num_steps)


def forward_corrupt(tokens: mx.array, t: mx.array, mask_id: int, key) -> mx.array:
    """Sample ``z_t ~ q(z_t | x)`` for a batch of token sequences.

    Args:
        tokens: ``(B, L)`` int32 clean tokens.
        t: ``(B,)`` or ``(B, 1)`` noise levels in ``[0, 1]``, one per sequence.
        mask_id: vocabulary index of ``[MASK]``.
        key: PRNG key.

    Returns:
        ``(B, L)`` int32 tokens with each position independently replaced by
        ``mask_id`` with probability ``t[b]``.
    """
    t = mx.array(t, dtype=mx.float32).reshape(-1)
    if t.shape[0] != tokens.shape[0]:
        raise ValueError("t must have one entry per sequence")
    u = mx.random.uniform(shape=tokens.shape, key=key, dtype=mx.float32)
    corrupt = u < t[:, None]
    return mx.where(corrupt, mx.full(tokens.shape, mask_id, dtype=tokens.dtype), tokens)


def forward_corrupt_per_block(
    tokens: mx.array, t: mx.array, block_size: int, mask_id: int, key
) -> mx.array:
    """Corruption with one noise level *per block* instead of per sequence.

    Block diffusion draws a separate ``t`` for every block (the paper's Appendix A
    says diffusion is performed within each block over ``T`` steps), which is what
    lets a single pass model the posterior of all blocks at once.

    ``t`` has shape ``(B, B_blocks)``.
    """
    t = mx.array(t, dtype=mx.float32)
    bsz, seq_len = tokens.shape
    if seq_len % block_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be a multiple of block_size ({block_size})")
    n_blocks = seq_len // block_size
    if t.shape != (bsz, n_blocks):
        raise ValueError(f"t must be {(bsz, n_blocks)}, got {t.shape}")
    t_expanded = mx.repeat(t, block_size, axis=1)
    u = mx.random.uniform(shape=tokens.shape, key=key, dtype=mx.float32)
    corrupt = u < t_expanded
    return mx.where(corrupt, mx.full(tokens.shape, mask_id, dtype=tokens.dtype), tokens)


def corruption_mask(tokens: mx.array, noisy: mx.array) -> mx.array:
    """Boolean ``(B, L)`` indicator of positions that were corrupted.

    For absorbing diffusion ``[MASK]`` is only ever produced by corruption, so
    ``noisy == mask_id`` is exactly the corruption indicator.
    """
    return noisy != tokens


def nelbo_loss(
    logits: mx.array,
    tokens: mx.array,
    noisy: mx.array,
    t: mx.array,
    schedule=DEFAULT_SCHEDULE,
) -> tuple[mx.array, mx.array, mx.array]:
    """Weighted masked-token cross-entropy, the toy stand-in for the paper's Eq. 4.

    The loss is evaluated only at corrupted positions (the paper's zero-masking-
    probability constraint: the model never has to predict ``[MASK]`` and clean
    positions carry no reconstruction term once ``T -> infinity``).

    Returns ``(mean_loss, sum_weight, n_masked)``; the two scalars let callers
    aggregate a correctly weighted average over batch elements with different
    numbers of masked positions.
    """
    masked = corruption_mask(tokens, noisy).astype(mx.float32)
    logp = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1) + 1e-30)
    target_logp = mx.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]
    weight = schedule.loss_weight(mx.array(t, dtype=mx.float32)).reshape(-1)
    if weight.shape[0] != tokens.shape[0]:
        raise ValueError("t must have one entry per sequence for nelbo_loss")
    per_token = -target_logp * weight[:, None] * masked
    n_masked = masked.sum()
    total_weight = (weight[:, None] * masked).sum()
    # Guard the all-clean case, which happens when t is sampled at 0.
    safe_weight = mx.where(total_weight > 0, total_weight, mx.array(1.0, mx.float32))
    return per_token.sum() / safe_weight, total_weight, n_masked


def nelbo_loss_per_block(
    logits: mx.array,
    tokens: mx.array,
    noisy: mx.array,
    t: mx.array,
    block_size: int,
    schedule=DEFAULT_SCHEDULE,
) -> mx.array:
    """Weighted masked-token cross-entropy with one noise level per block.

    For each block the loss is the importance-weighted mean of the per-token
    cross-entropies at the masked positions,

        L_b = (1 / m_b) * sum_{l in block b, masked} (alpha'_{t_b} / (1 - alpha_{t_b})) * -log p(x_l | z),

    with ``m_b`` the number of masked positions in the block, and the returned
    value is the mean over blocks.  Normalising per block rather than per sequence
    is what stops a block that happens to be fully masked from dominating the
    batch; it is the objective ``scripts/train_toy.py`` optimises.

    Blocks with ``t_b = 0`` have no masked positions and contribute nothing.
    """
    bsz, seq_len = tokens.shape
    if seq_len % block_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be a multiple of block_size ({block_size})")
    n_blocks = seq_len // block_size
    t_arr = t if isinstance(t, mx.array) else mx.array(t, dtype=mx.float32)
    if t_arr.shape != (bsz, n_blocks):
        raise ValueError(f"t must be {(bsz, n_blocks)}, got {t_arr.shape}")
    masked = corruption_mask(tokens, noisy).astype(mx.float32)
    logp = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1) + 1e-30)
    target_logp = mx.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]
    weight = schedule.loss_weight(t_arr.astype(mx.float32))  # (B, n_blocks)
    weight_per_token = mx.repeat(weight, block_size, axis=1)
    per_token = (-target_logp) * weight_per_token * masked
    per_block_sum = per_token.reshape(bsz, n_blocks, block_size).sum(axis=-1)
    per_block_masked = masked.reshape(bsz, n_blocks, block_size).sum(axis=-1)
    # Denominator is the number of masked positions times that block's weight, so
    # the block's loss is a weighted mean; using the count alone would leave a bare
    # factor of alpha'_t / (1 - alpha_t) in the objective.
    per_block_weight = per_block_masked * weight
    safe = mx.where(
        per_block_weight > 0, per_block_weight, mx.array(1.0, mx.float32)
    )
    return (per_block_sum / safe).mean()
