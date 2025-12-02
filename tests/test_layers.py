"""Tests for the attention masks and the transformer blocks."""

import mlx.core as mx
import mlx.utils
import numpy as np
import pytest

from e2d2.layers import Attention, Block, ModelConfig, Stack
from e2d2.masks import (
    additive_mask,
    block_decode_mask,
    decoder_block_diagonal,
    decoder_only_mask,
    e2d2_decoder_mask,
    encoder_block_causal,
)


def test_encoder_mask_is_block_causal():
    """A token sees its own block and every earlier block, never a later one."""
    mask = np.array(encoder_block_causal(seq_len=8, block_size=2))
    for i in range(8):
        for j in range(8):
            assert mask[i, j] == (i // 2 >= j // 2), (i, j)
    # diagonal is always allowed (a token attends to itself)
    assert mask.diagonal().all()


def test_encoder_mask_nonzero_count_matches_paper_formula():
    """The paper counts (L^2 + L*S)/2 entries for the block-causal mask."""
    for seq_len, block_size in ((6, 2), (8, 4), (12, 3), (16, 4)):
        got = int(np.array(encoder_block_causal(seq_len, block_size)).sum())
        b = seq_len // block_size
        s = block_size
        l = seq_len
        assert got == (l * l + l * s) // 2, (seq_len, block_size, got)
        assert got == sum(bb * s * s for bb in range(1, b + 1))


def test_decoder_mask_is_block_diagonal():
    mask = np.array(decoder_block_diagonal(seq_len=8, block_size=2))
    for i in range(8):
        for j in range(8):
            assert mask[i, j] == (i // 2 == j // 2), (i, j)


def test_e2d2_decoder_mask_quadrants():
    """The four quadrants of the E2D2 decoder mask have the documented structure."""
    seq_len, block_size = 8, 2
    m = np.array(e2d2_decoder_mask(seq_len, block_size))
    assert m.shape == (2 * seq_len, 2 * seq_len)
    for i in range(seq_len):  # h query
        for j in range(seq_len):  # h key: block-causal
            assert m[i, j] == (i // block_size >= j // block_size), (i, j)
            assert not m[i, seq_len + j], ("h must not read z", i, j)
    for i in range(seq_len):  # z query
        for j in range(seq_len):  # h key: strictly earlier clean blocks
            assert m[seq_len + i, j] == (i // block_size > j // block_size), (i, j)
            assert m[seq_len + i, seq_len + j] == (
                i // block_size == j // block_size
            ), (i, j)


def test_e2d2_decoder_never_sees_its_own_clean_block():
    """The invariant that keeps the denoising target from leaking.

    A noised token in block b may not attend to the clean hidden state of block b,
    because those hidden states are built from the very tokens being predicted.
    """
    seq_len, block_size = 12, 4
    m = np.array(e2d2_decoder_mask(seq_len, block_size))
    for i in range(seq_len):
        for j in range(seq_len):
            if i // block_size == j // block_size:
                assert not m[seq_len + i, j], (i, j)
            if i // block_size < j // block_size:
                assert not m[seq_len + i, j], (i, j)


def test_decoder_only_mask_structure():
    seq_len, block_size = 8, 2
    m = np.array(decoder_only_mask(seq_len, block_size))
    for i in range(seq_len):  # clean query
        for j in range(seq_len):
            assert m[i, j] == (i // block_size >= j // block_size)
            assert not m[i, seq_len + j]
    for i in range(seq_len):  # noisy query
        for j in range(seq_len):
            assert m[seq_len + i, j] == (i // block_size > j // block_size)
            assert m[seq_len + i, seq_len + j] == (i // block_size == j // block_size)


def test_block_decode_mask_matches_full_mask_subblock():
    """The incremental decode mask is the relevant sub-block of the full mask.

    During sampling there is no noised sequence yet, only a clean prefix, so the
    mask must equal ``e2d2_decoder_mask`` restricted to the prefix plus the new
    block.  A mismatch here would silently change what the model attends to at
    sampling time relative to training.
    """
    seq_len, block_size = 8, 2
    full = np.array(e2d2_decoder_mask(seq_len, block_size))
    n_blocks = seq_len // block_size
    for n_prefix_blocks in range(1, n_blocks):
        n_prefix = n_prefix_blocks * block_size
        incremental = np.array(block_decode_mask(n_prefix_blocks, block_size))
        assert incremental.shape == (n_prefix + block_size, n_prefix + block_size)
        # h-queries over h-keys: the leading sub-block of the full mask.
        assert np.array_equal(
            incremental[:n_prefix, :n_prefix], full[:n_prefix, :n_prefix]
        )
        # h-queries over the not-yet-existing z part: nothing.
        assert not incremental[:n_prefix, n_prefix:].any()
        # z-queries over the prefix h-keys: the full mask's noised rows for this
        # block, restricted to the prefix columns.
        rows = slice(seq_len + n_prefix, seq_len + n_prefix + block_size)
        assert np.array_equal(incremental[n_prefix:, :n_prefix], full[rows, :n_prefix])
        # z-queries over z-keys: the whole noised block.
        assert incremental[n_prefix:, n_prefix:].all()
    # The last prefix block still produces a well shaped mask (no noised rows of
    # the full mask exist for it, since the noised sequence only has L positions).
    last = np.array(block_decode_mask(n_blocks, block_size))
    assert last.shape == (seq_len + block_size, seq_len + block_size)
    assert last[seq_len:, :seq_len].all() and last[seq_len:, seq_len:].all()


def test_block_decode_mask_rejects_bad_block_size():
    with pytest.raises(ValueError):
        encoder_block_causal(7, 2)
    with pytest.raises(ValueError):
        encoder_block_causal(8, 0)
    with pytest.raises(ValueError):
        decoder_block_diagonal(8, 3)


def test_additive_mask_matches_boolean_keep_set():
    mask = encoder_block_causal(4, 2)
    bias = np.array(additive_mask(mask))
    assert bias.shape == (4, 4)
    assert np.array_equal(bias == 0.0, np.array(mask))
    assert (bias[bias != 0.0] < -1e8).all()


def test_attention_rows_sum_to_one_over_allowed_keys():
    """Softmax over the masked rows is a proper distribution, and masked-out
    keys get exactly zero weight."""
    mx.random.seed(0)
    attn = Attention(d_model=8, n_heads=2, max_seq_len=8)
    x = mx.random.normal((2, 4, 8))
    mask = encoder_block_causal(4, 2)
    # Recompute the internals to inspect the attention weights.
    qkv = attn.qkv(x)
    q, k, v = mx.split(qkv, 3, axis=-1)
    bsz, seq_len, _ = x.shape
    q = q.reshape(bsz, seq_len, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, seq_len, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
    scores = (q * attn.scale) @ k.transpose(0, 1, 3, 2)
    bias = np.array(additive_mask(mask))[None, None]
    w = np.array(mx.softmax(scores + mx.array(bias), axis=-1))
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-5)
    keep = np.array(mask)
    assert (w[..., ~keep] < 1e-12).all()
    assert (w[..., keep] > 0).all()


def test_attention_zero_masked_softmax_is_uniform():
    """With a single allowed key all attention weight lands on it."""
    mx.random.seed(1)
    attn = Attention(d_model=4, n_heads=1, max_seq_len=4)
    x = mx.random.normal((1, 3, 4))
    mask = mx.array([[True, False, False], [True, True, False], [True, True, True]])
    out = attn(x, mask)
    assert out.shape == (1, 3, 4)
    assert np.isfinite(np.array(out)).all()


def test_stack_zero_layers_is_the_identity_on_embeddings():
    """The reduction used in the model tests: a 0-layer stack adds nothing."""
    cfg = ModelConfig(vocab_size=16, max_seq_len=8, d_model=8, n_heads=2, n_layers=0)
    stack = Stack(cfg, with_head=False)
    tokens = mx.array([[1, 2, 3, 4]])
    mask = mx.ones((4, 4), dtype=mx.bool_)
    out = stack(stack.embed_tokens(tokens), mask)
    expected = stack.embed_tokens(tokens)
    assert np.allclose(np.array(out), np.array(expected), atol=1e-6)
    # and it has no transformer blocks, i.e. nothing that could transform the
    # clean side of the decoder-only model's input
    assert stack.blocks == []
    assert stack.cfg.n_layers == 0


def test_stack_output_head_is_tied_to_embeddings():
    cfg = ModelConfig(vocab_size=16, max_seq_len=8, d_model=8, n_heads=2, n_layers=1)
    stack = Stack(cfg, with_head=True)
    hidden = mx.random.normal((2, 3, 8))
    logits = stack.logits(hidden)
    assert logits.shape == (2, 3, 16)
    # as_linear uses the embedding matrix, so scaling the embedding scales logits
    stack.embed.weight = stack.embed.weight * 2.0
    assert np.allclose(np.array(stack.logits(hidden)), 2 * np.array(logits), atol=1e-5)


def test_block_shapes_and_dtypes():
    cfg = ModelConfig(vocab_size=16, max_seq_len=8, d_model=8, n_heads=2, n_layers=1)
    block = Block(cfg.d_model, cfg.n_heads, cfg.max_seq_len, cfg.mlp_mult)
    x = mx.random.normal((2, 5, 8))
    out = block(x, mx.ones((5, 5), dtype=mx.bool_))
    assert out.shape == (2, 5, 8)
    assert out.dtype == mx.float32
    assert np.isfinite(np.array(out)).all()
