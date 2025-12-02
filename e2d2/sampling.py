"""Block-by-block sampling for E2D2 and for the decoder-only baseline.

This is the paper's Algorithm 1, made concrete where the paper leaves the
``Sample(...)`` step abstract.

The loop structure is fixed and is the object of the efficiency measurement:

    h <- encode(prompt)                     # 1 encoder invocation
    for b in 1..B:                          # B blocks
        z <- [MASK] * block_size
        repeat T times:
            logits <- decode(z, h)          # T decoder invocations
            z <- unmask(logits)
        h <- encode(all clean tokens so far)  # 1 encoder invocation

so per generated sequence the encoder is invoked ``B`` times and the decoder
``B * T`` times.  A decoder-only block-diffusion model instead runs its one whole
network ``B * T`` times.

Unmasking schedule
------------------
The paper does not specify how tokens are selected for unmasking, only that the
decoder is applied ``T`` times and ``Sample`` produces the next ``z``.  This repo
uses the standard confidence-based scheme for masked diffusion:

* at sampling step ``i = 1..T`` exactly ``ceil(i * block_size / T)`` positions of
  the block are clean, so the number of clean tokens follows the same linear
  ``1 - t`` schedule the training-time corruption uses (``t = 1 - i/T``);
* the positions that become clean are the currently-masked positions with the
  highest model probability for their predicted token;
* once a position is unmasked it is never re-masked (the paper's "carry-over
  unmasking" constraint, Appendix A.3).

With ``T = 1`` the whole block is unmasked from the all-mask initialisation in a
single step, i.e. one-shot prediction from ``z = m`` -- the documented limit that
``tests/test_sampling.py`` checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx


def clean_counts(block_size: int, num_steps: int) -> list[int]:
    """Number of clean positions in a block after each of ``num_steps`` steps.

    ``clean_counts(S, T)[i-1] = ceil(i * S / T)``, ending at exactly ``S``.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    return [-(-(i * block_size) // num_steps) for i in range(1, num_steps + 1)]


def unmask_step(
    noisy_block: mx.array, logits: mx.array, mask_id: int, n_clean: int
) -> mx.array:
    """Apply one confidence-based unmasking step to a single ``(S,)`` block.

    Positions already unmasked are carried over unchanged.  Among the still-masked
    positions the most confident predictions are filled in until ``n_clean``
    positions of the block are clean.  Confidence ties are broken by position
    order (stable argsort over the masked positions).

    The ``[MASK]`` logit is excluded before the argmax.  The paper enforces zero
    masking probability (Appendix A.3: ``p_theta(x = m | z_t) = 0``), so the model
    is never allowed to predict mask; without that exclusion a prediction of
    ``[MASK]`` would be indistinguishable from "not yet unmasked" and a block could
    never become fully clean.
    """
    seq_len = noisy_block.shape[0]
    predictions = predict_tokens(logits, mask_id, dtype=noisy_block.dtype)
    if n_clean >= seq_len:
        # Everything becomes clean at once: pure one-shot prediction.
        return predictions
    if n_clean <= 0:
        return noisy_block

    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    confidence = mx.max(probs, axis=-1)
    is_masked = noisy_block == mask_id
    n_masked = int(is_masked.sum().item())
    if n_masked == 0:
        return noisy_block
    to_unmask = min(n_clean - (seq_len - n_masked), n_masked)
    if to_unmask <= 0:
        return noisy_block

    # Rank only the masked positions, best confidence first, position order as the
    # tie-break.  Restricting the sort to the masked subset makes the already-clean
    # positions structurally irrelevant rather than something to filter afterwards.
    masked_idx = mx.array([i for i in range(seq_len) if bool(is_masked[i].item())])
    local_order = mx.argsort(-confidence[masked_idx], axis=-1)[:to_unmask]
    chosen = mx.sort(masked_idx[local_order])
    return _scatter(noisy_block, chosen, predictions[chosen])


def predict_tokens(logits: mx.array, mask_id: int, dtype=mx.int32) -> mx.array:
    """Argmax over the vocabulary with the ``[MASK]`` logit excluded."""
    logits = logits.astype(mx.float32)
    without_mask = mx.where(
        mx.arange(logits.shape[-1]) == mask_id,
        mx.full(logits.shape, -1e9, dtype=mx.float32),
        logits,
    )
    return mx.argmax(without_mask, axis=-1).astype(dtype)


def _scatter(block: mx.array, idx: mx.array, values: mx.array) -> mx.array:
    """Write ``values`` at ``idx`` in a flat 1-D ``block``.

    MLX has no differentiable boolean-mask scatter for this shape, and the block
    is tiny, so this goes through Python lists.  It is only ever used on the
    sampling path (never inside a gradient).
    """
    out = block.tolist()
    for i, v in zip([int(i) for i in idx.tolist()], values.tolist()):
        out[i] = v
    return mx.array(out, dtype=block.dtype)


@dataclass
class SampleResult:
    tokens: mx.array
    counts: dict = field(default_factory=dict)
    wall_seconds: float = 0.0
    step_history: list = field(default_factory=list)


@dataclass
class SampleConfig:
    num_steps: int = 8
    block_size: int = 4
    n_blocks: int = 1
    mask_id: int = 0
    greedy: bool = True

    @property
    def total_tokens(self) -> int:
        return self.block_size * self.n_blocks


def sample_blockwise(
    model,
    prompt_tokens: mx.array,
    config: SampleConfig,
    collect_steps: bool = False,
) -> SampleResult:
    """Generate ``config.n_blocks`` blocks conditioned on ``prompt_tokens``.

    ``prompt_tokens`` is ``(B, P)``.  Returns a :class:`SampleResult` whose
    ``counts`` dict is the model's :class:`~e2d2.model.CallCounts` deltas for this
    call, so a caller can assert on encoder/decoder invocations directly.
    """
    cfg = model.config
    if config.block_size != cfg.block_size:
        raise ValueError("SampleConfig.block_size must match the model's block size")
    if config.n_blocks > cfg.n_blocks:
        raise ValueError("asked for more blocks than the model was configured with")
    bsz = prompt_tokens.shape[0]
    if prompt_tokens.shape[1] % cfg.block_size != 0:
        raise ValueError("prompt length must be a multiple of the block size")

    model.reset_counts()
    schedule = clean_counts(cfg.block_size, config.num_steps)
    history: list = []

    start = time.perf_counter()
    clean = prompt_tokens
    h = model.encode_prefix(clean)
    generated = []

    for block_idx in range(config.n_blocks):
        noisy = mx.full((bsz, cfg.block_size), config.mask_id, dtype=prompt_tokens.dtype)
        for step_idx, n_clean in enumerate(schedule):
            logits = model.denoise_step(noisy, h)  # one decoder invocation
            new_rows = [
                unmask_step(noisy[b], logits[b], config.mask_id, n_clean)
                for b in range(bsz)
            ]
            noisy = mx.stack(new_rows, axis=0)
            if collect_steps:
                history.append(
                    {
                        "block": block_idx,
                        "step": step_idx,
                        "n_clean_target": n_clean,
                        "n_clean": int((noisy != config.mask_id).sum().item()),
                        "tokens": noisy.tolist(),
                    }
                )
        generated.append(noisy)
        # Hand the completed block back to the encoder so the next block's decoder
        # sees it as clean context.  One encoder invocation per block, and none
        # after the final block -- there is nothing left to condition.  This is
        # the paper's line 10 of Algorithm 1, which runs for b = 1..B but is only
        # *needed* for b < B; skipping the last one is what makes the count
        # exactly B rather than B + 1.
        if block_idx + 1 < config.n_blocks:
            clean = mx.concatenate([clean, noisy], axis=1)
            h = model.encode_prefix(clean)

    tokens = mx.concatenate(generated, axis=1)
    wall = time.perf_counter() - start
    return SampleResult(
        tokens=tokens, counts=model.counts.as_dict(), wall_seconds=wall, step_history=history
    )


def one_shot_predict(model, prompt_tokens: mx.array, mask_id: int) -> mx.array:
    """``T = 1`` limit, written out separately so tests can compare the two paths.

    Encode the prompt, feed an all-mask block and return the argmax prediction for
    every position without ever updating the block in between.  This must equal
    :func:`sample_blockwise` with ``num_steps=1``.
    """
    cfg = model.config
    h = model.encode_prefix(prompt_tokens)
    noisy = mx.full(
        (prompt_tokens.shape[0], cfg.block_size), mask_id, dtype=prompt_tokens.dtype
    )
    logits = model.denoise_step(noisy, h)
    return predict_tokens(logits, mask_id, dtype=prompt_tokens.dtype)
