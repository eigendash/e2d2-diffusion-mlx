"""Tests for the E2D2 model: masking invariants, counting, and the decoder-only limit."""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import pytest

from e2d2.diffusion import forward_corrupt_per_block
from e2d2.layers import ModelConfig, Stack
from e2d2.model import CallCounts, E2D2, E2D2Config
from e2d2.task import MASK_ID, TaskSpec, held_out_prompts

SMALL = dict(d_model=32, n_heads=4, encoder_layers=2, decoder_layers=1)


def make_config(spec: TaskSpec | None = None, **overrides) -> E2D2Config:
    spec = spec or TaskSpec()
    kwargs = dict(
        vocab_size=spec.vocab_size,
        block_size=spec.block_size,
        n_blocks=spec.n_blocks,
        **SMALL,
    )
    kwargs.update(overrides)
    return E2D2Config(**kwargs)


def make_model(**overrides) -> E2D2:
    mx.random.seed(0)
    return E2D2(make_config(**overrides))


# -- structure --------------------------------------------------------------


def test_decoder_only_config_has_no_encoder_stack():
    baseline = make_model(encoder_layers=0, decoder_layers=3)
    assert baseline.config.is_decoder_only
    assert baseline.encoder.blocks == []
    assert len(baseline.decoder.blocks) == 3
    assert baseline.full_stack_layers() == 3


def test_encoder_decoder_config_has_both_stacks():
    model = make_model(encoder_layers=2, decoder_layers=1)
    assert len(model.encoder.blocks) == 2
    assert len(model.decoder.blocks) == 1
    assert model.config.encoder_layers == 2 and model.config.decoder_layers == 1
    assert model.full_stack_layers() == 3


def test_matched_parameter_budget_is_close():
    """The split and the baseline can be built at nearly the same parameter count.

    The experiment compares architectures "at matched parameter budget"; this
    records how close the two settings used there actually are.
    """
    split = make_model(encoder_layers=3, decoder_layers=1)
    baseline = make_model(encoder_layers=0, decoder_layers=4)
    a, b = split.num_parameters(), baseline.num_parameters()
    assert abs(a - b) / max(a, b) < 0.15, (a, b)
    # The split has to be the smaller model here for the comparison to be fair to
    # the baseline, otherwise the baseline is handicapped on capacity as well.
    assert a <= b * 1.02, (a, b)


def test_full_decoder_mask_is_the_figure_two_mask_for_both_models():
    """The baseline and the encoder-decoder model share the same decoder mask.

    That is exactly what makes the baseline the identity-encoder limit rather than
    a different masking scheme as well.
    """
    split = make_model()
    baseline = make_model(encoder_layers=0, decoder_layers=2)
    assert np.array_equal(
        np.array(split.full_decoder_mask()), np.array(baseline.full_decoder_mask())
    )


def test_identical_seeds_give_identical_initialisations_across_models():
    """Two models built identically from the same seed are the same function."""
    mx.random.seed(3)
    a = E2D2(make_config())
    mx.random.seed(3)
    b = E2D2(make_config())
    tokens = mx.array([[1, 5, 6, 2, 5, 6, 3, 4]])
    la = a.denoise_logits_full(tokens)
    lb = b.denoise_logits_full(tokens)
    assert np.allclose(np.array(la), np.array(lb), atol=1e-6)


# -- the structural reduction to decoder-only block diffusion ---------------


def test_identity_encoder_limit_matches_a_hand_built_decoder_only_model():
    """With encoder_layers=0 the model *is* a decoder-only block diffusion model.

    The limit tested here is: identity encoder, so the clean half of the decoder
    input is the token embedding; the rest of the graph is unchanged.  The test
    rebuilds that graph by hand out of the same parameter arrays and checks the
    logits agree to 1e-5.
    """
    model = make_model(encoder_layers=0, decoder_layers=3)
    spec = TaskSpec()
    _, full = held_out_prompts(spec, 4, seed=2)
    noisy = mx.where(
        mx.arange(spec.seq_len)[None, :] % 3 == 0,
        mx.full(full.shape, MASK_ID, dtype=full.dtype),
        full,
    )
    got = model.denoise_logits_full(noisy)

    # Hand-built equivalent: one stack, input [embed(clean) ; embed(noisy)].
    dec = model.decoder
    clean_emb = dec.embed_tokens(noisy)
    noisy_emb = dec.embed_tokens(noisy)
    x = mx.concatenate([clean_emb, noisy_emb], axis=1)
    hidden = dec(x, model.full_decoder_mask())
    expected = dec.logits(hidden[:, spec.seq_len :, :])
    assert np.allclose(np.array(got), np.array(expected), atol=1e-5)


def test_identity_encoder_is_an_embedding_lookup():
    """The clean representation in the limit is literally embed(x) plus position."""
    model = make_model(encoder_layers=0)
    tokens = mx.array([[1, 5, 6, 2]])
    got = model.clean_representation(tokens)
    dec = model.decoder
    assert np.allclose(np.array(got), np.array(dec.embed_tokens(tokens)), atol=1e-6)
    assert model.counts.encoder_calls == 0


def test_encoder_decoder_clean_representation_differs_from_embeddings():
    """A trained encoder is not the identity, so the limit is a real restriction."""
    model = make_model(encoder_layers=1)
    tokens = mx.array([[1, 5, 6, 2]])
    got = np.array(model.clean_representation(tokens))
    plain = np.array(model.decoder.embed_tokens(tokens))
    assert not np.allclose(got, plain, atol=1e-3)
    assert model.counts.encoder_calls == 1


# -- masking invariants -----------------------------------------------------


def test_noised_block_cannot_see_its_own_clean_block():
    """Changing the clean content of a block must not change that block's logits.

    The block is fully masked, so the only route by which its clean tokens could
    influence the prediction is through attention -- and the mask forbids it.
    """
    model = make_model()
    spec = TaskSpec()
    _, full = held_out_prompts(spec, 1, seed=4)
    n_blocks = spec.n_blocks
    answer_block = n_blocks - 1
    fully_masked_answer = mx.concatenate(
        [
            full[:, : answer_block * spec.block_size],
            mx.full(
                (1, spec.block_size), MASK_ID, dtype=full.dtype
            ),
        ],
        axis=1,
    )
    # corrupt the answer block's clean content in a way that must be invisible
    other = full + mx.where(
        mx.arange(spec.seq_len)[None, :] >= answer_block * spec.block_size,
        mx.ones(full.shape, dtype=full.dtype),
        mx.zeros(full.shape, dtype=full.dtype),
    )
    other = mx.where(other >= spec.vocab_size, mx.full(other.shape, 3, dtype=full.dtype), other)
    a = model.denoise_logits_full(fully_masked_answer)
    # Same noised input, different clean sequence: the encoder sees different clean
    # tokens in the answer block, but the decoder must not read them.
    b = model.denoise_logits_full(fully_masked_answer, clean_override=other)
    diff = float(mx.max(mx.abs(a - b)).item())
    assert diff < 1e-4, diff


def test_later_clean_blocks_do_not_affect_earlier_hidden_states():
    """Encoder block-causality: h for block j is unchanged when later clean blocks change."""
    model = make_model(encoder_layers=2)
    spec = TaskSpec()
    _, full = held_out_prompts(spec, 2, seed=5)
    bs = spec.block_size
    perturbed = full + mx.where(
        mx.arange(spec.seq_len)[None, :] >= bs,
        mx.ones(full.shape, dtype=full.dtype),
        mx.zeros(full.shape, dtype=full.dtype),
    )
    perturbed = mx.where(
        perturbed >= spec.vocab_size, mx.full(perturbed.shape, 4, dtype=full.dtype), perturbed
    )
    h_a = model.clean_representation(full)
    h_b = model.clean_representation(perturbed)
    first_block_a = np.array(h_a[:, :bs, :])
    first_block_b = np.array(h_b[:, :bs, :])
    assert np.allclose(first_block_a, first_block_b, atol=1e-5)
    assert not np.allclose(np.array(h_a), np.array(h_b), atol=1e-4)


def test_different_noise_levels_give_different_losses():
    """The loss actually depends on t, which catches a dropped weight."""
    mx.random.seed(11)
    model = make_model()
    spec = TaskSpec()
    _, full = held_out_prompts(spec, 8, seed=6)
    losses = []
    for step in (1, 3, 5):
        t = mx.full((8, spec.n_blocks), step / 8.0, dtype=mx.float32)
        noisy = forward_corrupt_per_block(full, t, spec.block_size, MASK_ID, mx.random.key(step))
        logits = model.denoise_logits_full(noisy)
        from e2d2.diffusion import DEFAULT_SCHEDULE, nelbo_loss_per_block

        losses.append(
            float(
                nelbo_loss_per_block(
                    logits, full, noisy, t, spec.block_size, DEFAULT_SCHEDULE
                )
            )
        )
    assert len(set(round(v, 4) for v in losses)) == 3, losses


# -- call counting ----------------------------------------------------------


def test_training_pass_counts_one_encoder_and_one_decoder_call():
    model = make_model()
    spec = TaskSpec()
    _, full = held_out_prompts(spec, 4, seed=7)
    model.reset_counts()
    model.training_loss(full, mx.random.key(0), num_steps=8)
    assert model.counts.encoder_calls == 1
    assert model.counts.decoder_calls == 1
    assert model.counts.encoder_layer_forwards == model.config.encoder_layers * spec.n_blocks
    assert model.counts.decoder_layer_forwards == model.config.decoder_layers * spec.n_blocks


def test_denoise_logits_full_counts_one_of_each():
    model = make_model()
    tokens = mx.array([[1, 5, 6, 2, 5, 6, 3, 4]])
    model.reset_counts()
    model.denoise_logits_full(tokens)
    assert model.counts.encoder_calls == 1
    assert model.counts.decoder_calls == 1


def test_callcounts_reset_and_serialise():
    counts = CallCounts()
    counts.encoder_calls = 3
    counts.decoder_calls = 5
    d = counts.as_dict()
    assert d["encoder_calls"] == 3 and d["decoder_calls"] == 5
    counts.reset()
    assert all(v == 0 for v in counts.as_dict().values())


# -- shapes, dtypes, gradients ---------------------------------------------


def test_training_loss_shapes_and_finiteness():
    model = make_model()
    spec = TaskSpec()
    _, full = held_out_prompts(spec, 6, seed=8)
    loss = model.training_loss(full, mx.random.key(0), num_steps=8)
    assert loss.shape == ()
    assert loss.dtype == mx.float32
    assert np.isfinite(np.array(loss)).all()
    assert float(loss) > 0.0


def test_logits_shape_and_dtype():
    model = make_model()
    spec = TaskSpec()
    _, full = held_out_prompts(spec, 3, seed=9)
    logits = model.denoise_logits_full(full)
    assert logits.shape == (3, spec.seq_len, spec.vocab_size)
    assert logits.dtype == mx.float32


def test_gradients_are_finite_and_reach_both_stacks():
    """Every parameter gets a finite gradient, and the encoder is not bypassed."""
    model = make_model()
    spec = TaskSpec()
    _, full = held_out_prompts(spec, 4, seed=10)
    key = mx.random.key(0)

    def loss_fn(m):
        return m.training_loss(full, key, num_steps=8)

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, grads)
    flat = mlx.utils.tree_flatten(grads)
    assert len(flat) > 0
    for name, g in flat:
        assert np.isfinite(np.array(g)).all(), name
    encoder_norms = {
        n: float(mx.sum(mx.abs(g)).item()) for n, g in flat if n.startswith("encoder.")
    }
    assert encoder_norms, "no encoder gradients at all"
    assert any(v > 0 for v in encoder_norms.values()), encoder_norms


def test_training_loss_rejects_wrong_sequence_length():
    model = make_model()
    with pytest.raises(ValueError):
        model.training_loss(mx.zeros((1, 4), dtype=mx.int32), mx.random.key(0))


def test_mask_cache_is_stable_across_calls():
    model = make_model()
    a = model.full_decoder_mask()
    b = model.full_decoder_mask()
    assert a is b
    assert model.encoder_mask(2) is model.encoder_mask(2)


def test_stack_config_shapes():
    cfg = make_config(encoder_layers=2, decoder_layers=3)
    assert cfg.seq_len == 8
    assert cfg.stack_config(2).n_layers == 2
    assert cfg.stack_config(3).n_layers == 3
    assert cfg.stack_config(0).n_layers == 0
    assert isinstance(cfg.stack_config(1), ModelConfig)
    assert isinstance(Stack(cfg.stack_config(1), with_head=True), Stack)
