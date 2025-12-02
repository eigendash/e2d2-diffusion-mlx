"""Boolean attention masks for block diffusion.

Two masks are needed, both straight from the paper's Figure 2.

``encoder_block_causal(L, S)`` -- the clean sequence is processed by the encoder
with a block-causal mask: a token in block ``b`` may attend to its own block and
to every earlier block, and to nothing later.  This is the paper's
``M_Enc`` and it is what makes one encoder pass able to produce the context
representation for every block at once.

``decoder_block_diagonal(L, S)`` -- the noised sequence is processed by the
decoder with a block-diagonal mask: a noised token attends only within its own
block (the paper's ``M_BD``).

``e2d2_decoder_mask(L, S)`` -- the full mask for the E2D2 decoder when its input
is the concatenation ``[h^{1:L} ; z_t^{1:L}]`` of encoder outputs and noised
embeddings.  Query/key positions ``< L`` are the encoder side (``h``) and
positions ``>= L`` are the noised side (``z``).  A noised query in block ``b``
attends to ``z`` within block ``b`` and to ``h`` from blocks ``< b`` only -- the
paper's "cross-attention to previous clean blocks".  Note in particular that a
noised query does *not* see ``h`` of its own block, which is what keeps the
denoising target from leaking through the encoder representation of the very
tokens being predicted; see the README note on this.

``decoder_only_mask(L, S)`` -- the mask for a decoder-only block diffusion model
(BD3LM), whose input is the concatenation of the clean token embeddings and the
noised token embeddings.  Clean queries are block-causal, noised queries are
block-diagonal plus offset-block-causal to *earlier* clean blocks.
"""

from __future__ import annotations

import mlx.core as mx


def _blocks(seq_len: int, block_size: int) -> mx.array:
    if block_size <= 0 or seq_len % block_size != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be a positive multiple of block_size ({block_size})"
        )
    return mx.arange(seq_len) // block_size


def encoder_block_causal(seq_len: int, block_size: int) -> mx.array:
    """``(L, L)`` bool: ``b_i >= b_j`` (own block and all earlier blocks)."""
    b = _blocks(seq_len, block_size)
    return b[:, None] >= b[None, :]


def decoder_block_diagonal(seq_len: int, block_size: int) -> mx.array:
    """``(L, L)`` bool: ``b_i == b_j`` (own block only)."""
    b = _blocks(seq_len, block_size)
    return b[:, None] == b[None, :]


def e2d2_decoder_mask(seq_len: int, block_size: int) -> mx.array:
    """``(2L, 2L)`` bool mask for ``[h^{1:L} ; z_t^{1:L}]``.

    The top-left ``(L, L)`` block is the paper's ``M_Enc``: the representations
    ``h`` themselves are produced by a block-causal encoder, so when the decoder
    treats them as keys it must do so with the same causal structure (otherwise a
    hidden-state position could be attended in a way its own computation did not
    allow).  All three remaining quadrants are fixed by Figure 2.
    """
    b = _blocks(seq_len, block_size)
    n = 2 * seq_len
    mask = mx.zeros((n, n), dtype=mx.bool_)

    same_blk = b[:, None] == b[None, :]
    causal_blk = b[:, None] >= b[None, :]

    # h-queries over h-keys: block-causal (matches the encoder's own mask).
    top_left = causal_blk
    # h-queries over z-keys: nothing (the encoder never reads the noised sequence).
    top_right = mx.zeros((seq_len, seq_len), dtype=mx.bool_)
    # z-queries over h-keys: strictly earlier clean blocks only.
    bottom_left = b[:, None] > b[None, :]
    # z-queries over z-keys: within the noised block.
    bottom_right = same_blk

    mask = mx.concatenate(
        [
            mx.concatenate([top_left, top_right], axis=1),
            mx.concatenate([bottom_left, bottom_right], axis=1),
        ],
        axis=0,
    )
    return mask


def decoder_only_mask(seq_len: int, block_size: int) -> mx.array:
    """``(2L, 2L)`` bool mask for ``[x^{1:L} ; z_t^{1:L}]`` (BD3LM)."""
    b = _blocks(seq_len, block_size)
    n = 2 * seq_len
    same_blk = b[:, None] == b[None, :]
    causal_blk = b[:, None] >= b[None, :]
    later = b[:, None] > b[None, :]

    top_left = causal_blk  # clean block-causal self-attention (M_BC)
    top_right = mx.zeros((seq_len, seq_len), dtype=mx.bool_)
    bottom_left = later  # noisy -> earlier clean blocks (M_OBC)
    bottom_right = same_blk  # noisy block-diagonal self-attention (M_BD)
    return mx.concatenate(
        [
            mx.concatenate([top_left, top_right], axis=1),
            mx.concatenate([bottom_left, bottom_right], axis=1),
        ],
        axis=0,
    )


def block_decode_mask(n_prefix_blocks: int, block_size: int) -> mx.array:
    """``(2P, 2P)`` mask used by the incremental sampler for one block.

    ``P = n_prefix_blocks * block_size`` is the number of clean tokens the encoder
    has seen so far (prompt plus already-decoded blocks).  The decoder input at
    this moment is ``[h^{1:P} ; z^b]`` with ``b = n_prefix_blocks`` (0-indexed),
    so this is just the relevant sub-block of :func:`e2d2_decoder_mask`:

    * ``h`` queries over ``h`` keys: block-causal (the representations were built
      under the encoder's own block-causal mask, so the same causal structure has
      to hold when they are read back);
    * ``h`` queries over ``z`` keys: nothing;
    * ``z`` queries over ``h`` keys: every clean block (all of them are earlier);
    * ``z`` queries over ``z`` keys: the whole noised block.
    """
    b = mx.arange(n_prefix_blocks)
    n_prefix = n_prefix_blocks * block_size
    causal_blk = b[:, None] >= b[None, :]
    top_left = mx.repeat(mx.repeat(causal_blk, block_size, axis=0), block_size, axis=1)
    top_right = mx.zeros((n_prefix, block_size), dtype=mx.bool_)
    bottom_left = mx.ones((block_size, n_prefix), dtype=mx.bool_)
    bottom_right = mx.ones((block_size, block_size), dtype=mx.bool_)
    return mx.concatenate(
        [
            mx.concatenate([top_left, top_right], axis=1),
            mx.concatenate([bottom_left, bottom_right], axis=1),
        ],
        axis=0,
    )


def additive_mask(mask: mx.array, dtype=mx.float32) -> mx.array:
    """Turn a boolean keep-mask into the additive bias MLX softmax wants."""
    return mx.where(mask, mx.array(0.0, dtype), mx.array(-1e9, dtype))
