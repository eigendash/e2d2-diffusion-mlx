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

    The weighted sum is divided by the *sequence length* ``L``, a constant, which
    is the convention used by the masked-diffusion implementations: dividing by the
    number of masked positions instead would cancel the weight ``w(t)`` exactly and
    turn the NELBO into a plain masked-token mean.

    Returns ``(mean_loss, sum_weight, n_masked)``; the two scalars let callers
    aggregate over batch elements with different numbers of masked positions.
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
    # Constant denominator: the weight stays part of the objective.
    return per_token.sum() / float(tokens.shape[1]), total_weight, n_masked


def nelbo_loss_per_block(
    logits: mx.array,
    tokens: mx.array,
    noisy: mx.array,
    t: mx.array,
    block_size: int,
    schedule=DEFAULT_SCHEDULE,
    weighting: str = "nelbo",
) -> mx.array:
    """Masked-token cross-entropy with one noise level per block.

    Two aggregation choices, both used in this repo and both reported in the
    README:

    ``weighting="nelbo"`` (default)
        The paper's Eq. 4 / Eq. 5 diffusion loss.  Each masked token is weighted by
        ``alpha'_t / (1 - alpha_t)`` and the block's sum is divided by the *block
        size* -- a constant -- so the weight survives:

            L_b = (1 / S) * sum_{l in block b, masked} w(t_b) * -log p(x_l | z)

        For the linear schedule ``w(t) = 1 / (1 - t)``, which is large exactly when
        the block is close to fully masked.  That is the regime sampling starts
        from, so this objective is what makes denoising from an all-mask block
        learnable.

    ``weighting="uniform"``
        The unweighted mean cross-entropy over masked positions.  Note that
        dividing by the number of masked positions cancels ``w(t)`` exactly, so
        this is *not* a reparameterisation of the NELBO; it is the objective the
        authors' reference implementation uses during training (its
        ``_compute_loss`` in ``src/denoiser/diffusion.py`` branches on
        ``self.training`` and takes the unweighted sum over masked tokens).  It
        concentrates the loss on lightly noised blocks, where a block's own clean
        tokens are available, so at toy scale it under-trains the cross-attention
        that full-block denoising needs.

    Blocks with ``t_b = 0`` have no masked positions and contribute nothing.
    """
    bsz, seq_len = tokens.shape
    if seq_len % block_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be a multiple of block_size ({block_size})")
    n_blocks = seq_len // block_size
    t_arr = t if isinstance(t, mx.array) else mx.array(t, dtype=mx.float32)
    if t_arr.shape != (bsz, n_blocks):
        raise ValueError(f"t must be {(bsz, n_blocks)}, got {t_arr.shape}")
    if weighting not in ("nelbo", "uniform"):
        raise ValueError("weighting must be 'nelbo' or 'uniform'")

    masked = corruption_mask(tokens, noisy).astype(mx.float32)
    logp = mx.log(mx.softmax(logits.astype(mx.float32), axis=-1) + 1e-30)
    target_logp = mx.take_along_axis(logp, tokens[..., None], axis=-1)[..., 0]
    weight = schedule.loss_weight(t_arr.astype(mx.float32))  # (B, n_blocks)
    weight_per_token = mx.repeat(weight, block_size, axis=1)
    per_token = (-target_logp) * weight_per_token * masked
    per_block_sum = per_token.reshape(bsz, n_blocks, block_size).sum(axis=-1)

    if weighting == "uniform":
        # Dividing by the masked count cancels w(t) exactly, leaving the plain mean.
        per_block_count = masked.reshape(bsz, n_blocks, block_size).sum(axis=-1)
        safe = mx.where(per_block_count > 0, per_block_count, mx.array(1.0, mx.float32))
        return (per_block_sum / safe).mean()
    # The weighted sum is divided by the block size, a constant: the importance
    # weight is part of the objective, not a normalisation that cancels.
    return (per_block_sum / float(block_size)).mean()
