"""Regression tests for the three train/sample mismatches found during development.

Each of these was a real bug that left every other test green while making the
model unable to sample at all.  They are collected here because they share one
shape: two code paths that must agree on a detail nobody was checking.

1. The mask token was implicit.  ``model.py`` corrupted with ``vocab_size - 1``
   while the data pipeline defined ``[MASK]`` as token 0, so training noised a real
   token and the sampler masked a different one.
2. The sampler dropped the noised block's absolute positional embedding, so the
   model saw the block at offset 0 while training had shown it at its true offset.
3. A fully masked block is a block of identical inputs, so without positions the
   block's query positions are interchangeable and the model cannot learn which
   answer token belongs where.  The loss parks at ``log(block_size)``.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from e2d2.sampling import one_shot_predict

from e2d2.diffusion import forward_corrupt_per_block
from e2d2.model import E2D2, E2D2Config
from e2d2.task import TaskSpec, held_out_prompts, split_prompts

SPEC = TaskSpec()


def make_model(seed: int = 0, **overrides) -> E2D2:
    kwargs = dict(
        vocab_size=SPEC.vocab_size,
        block_size=SPEC.block_size,
        n_blocks=SPEC.n_blocks,
        d_model=32,
        n_heads=4,
        encoder_layers=2,
        decoder_layers=1,
        mask_id=SPEC.mask_id,
    )
    kwargs.update(overrides)
    mx.random.seed(seed)
    return E2D2(E2D2Config(**kwargs))


# -- 1. the mask token is explicit -----------------------------------------


def test_config_defaults_mask_to_the_last_vocabulary_token():
    cfg = E2D2Config(vocab_size=20)
    assert cfg.mask_id == 19


def test_config_mask_id_is_honoured_not_recomputed():
    cfg = E2D2Config(vocab_size=20, mask_id=3)
    assert cfg.mask_id == 3


def test_config_rejects_an_out_of_range_mask_id():
    with pytest.raises(ValueError):
        E2D2Config(vocab_size=8, mask_id=8)
    with pytest.raises(ValueError):
        E2D2Config(vocab_size=8, mask_id=-1)


def test_training_corruption_uses_the_models_mask_id_not_vocab_minus_one():
    """The noised sequence must contain the model's mask id and nothing else new.

    A model configured with a non-default mask id must corrupt with exactly that
    id.  This is the regression: the old code hard-coded ``vocab_size - 1``, so a
    custom mask id silently corrupted a real token instead.
    """
    spec = SPEC
    model = make_model(mask_id=5)
    assert model.config.mask_id == 5
    _, clean = held_out_prompts(spec, 4, seed=0)
    t = mx.full((4, spec.n_blocks), 1.0, dtype=mx.float32)
    noisy = forward_corrupt_per_block(clean, t, spec.block_size, model.config.mask_id, mx.random.key(0))
    values = set(np.unique(np.array(noisy)).tolist())
    assert values == {5}, values
    assert 5 not in set(np.unique(np.array(clean)).tolist())


def test_task_mask_id_does_not_collide_with_any_real_token():
    _, full = held_out_prompts(SPEC, 512, seed=0)
    real = set(np.unique(np.array(full)).tolist())
    assert SPEC.mask_id not in real
    assert max(real) < SPEC.mask_id


def test_training_loss_accepts_the_task_mask_id_and_stays_finite():
    spec = SPEC
    model = make_model(mask_id=spec.mask_id)
    _, clean = held_out_prompts(spec, 8, seed=1)
    loss = model.training_loss(clean, mx.random.key(0), num_steps=8)
    assert np.isfinite(np.array(loss)).all()


# -- 2. both paths embed the noised block at the same absolute offsets ------


def test_training_and_sampling_paths_produce_identical_logits():
    """The two forward paths must agree given the same corruption.

    The training pass scores the whole noised sequence at once (positions 0..L-1);
    the sampler scores one block at a time.  For the answer block those must give
    the same logits, otherwise the model is evaluated at a different offset from
    the one it was trained at.
    """
    for enc, dec in ((2, 1), (0, 3), (1, 2)):
        model = make_model(encoder_layers=enc, decoder_layers=dec)
        prompt, full = held_out_prompts(SPEC, 6, seed=2)
        block = full[:, SPEC.prompt_len :]
        fully_masked = mx.full(block.shape, SPEC.mask_id, dtype=block.dtype)
        training_side = model.denoise_logits_full(
            mx.concatenate([prompt, fully_masked], axis=1)
        )
        training_side = training_side[:, (SPEC.seq_len - SPEC.block_size) :, :]
        sampling_side = model.denoise_step(fully_masked, model.encode_prefix(prompt))
        assert training_side.shape == sampling_side.shape
        diff = float(mx.max(mx.abs(training_side - sampling_side)).item())
        assert diff < 1e-5, (enc, dec, diff)


def test_denoise_step_positions_start_after_the_encoded_prefix():
    """The sampler must use absolute offsets, not offsets from zero.

    Checked behaviourally: embedding a block at ``[P, P+S)`` and at ``[0, S)``
    gives different embeddings, and the logits the sampler produces must match a
    hand-run decoder that used the shifted positions.
    """
    model = make_model()
    prompt, full = held_out_prompts(SPEC, 3, seed=3)
    block = mx.full((3, SPEC.block_size), SPEC.mask_id, dtype=mx.int32)
    h = model.encode_prefix(prompt)
    n_prefix = h.shape[1]

    got = model.denoise_step(block, h)
    shifted = model.decoder.embed_tokens(
        block, positions=mx.arange(n_prefix, n_prefix + SPEC.block_size)
    )
    x = mx.concatenate([h, shifted], axis=1)
    expected = model.decoder.logits(model.decoder(x, model.decode_mask(1)))[:, n_prefix:, :]
    assert np.allclose(np.array(got), np.array(expected), atol=1e-6)

    unshifted = model.decoder.embed_tokens(block, positions=mx.arange(SPEC.block_size))
    x0 = mx.concatenate([h, unshifted], axis=1)
    wrong = model.decoder.logits(model.decoder(x0, model.decode_mask(1)))[:, n_prefix:, :]
    assert not np.allclose(np.array(got), np.array(wrong), atol=1e-4)


# -- 3. a fully masked block needs position information --------------------


def test_fully_masked_block_rows_are_not_interchangeable():
    """With positions on, the four masked queries get four different outputs.

    Without positional information the block is permutation-symmetric and every
    query produces the same distribution, which caps the loss at ``log(S)`` and
    makes the reversal task unlearnable.
    """
    model = make_model()
    prompt, _ = held_out_prompts(SPEC, 2, seed=4)
    h = model.encode_prefix(prompt)
    block = mx.full((2, SPEC.block_size), SPEC.mask_id, dtype=mx.int32)
    logits = np.array(model.denoise_step(block, h))
    for i in range(SPEC.block_size):
        for j in range(i + 1, SPEC.block_size):
            assert not np.allclose(logits[:, i, :], logits[:, j, :], atol=1e-5), (i, j)


def test_disabling_positions_makes_masked_block_rows_identical():
    """Ablation that documents the failure mode: use_pos=False symmetrises the block.

    With no positions, ``use_pos=False`` on the decoder makes the four masked
    queries identical inputs under an all-to-all block mask, so their outputs are
    exactly equal.  This pins down *why* the decoder keeps positions.
    """
    mx.random.seed(0)
    model = make_model()
    model.decoder.use_pos = False
    prompt, _ = held_out_prompts(SPEC, 2, seed=4)
    h = model.encode_prefix(prompt)
    block = mx.full((2, SPEC.block_size), SPEC.mask_id, dtype=mx.int32)
    logits = np.array(model.denoise_step(block, h))
    for i in range(1, SPEC.block_size):
        assert np.allclose(logits[:, 0, :], logits[:, i, :], atol=1e-5), i


def test_model_can_fit_a_handful_of_examples():
    """An end-to-end check that the objective and sampler agree.

    Three examples, the answer block always fully masked, a fixed corruption key:
    the training path must be able to drive the loss down and the sampler must
    then reproduce the training-path predictions.  A mismatch anywhere in the two
    paths shows up here as a sampler that disagrees with the model's own logits.
    """
    spec = SPEC
    model = make_model(seed=0, encoder_layers=1, decoder_layers=1)
    prompt, full = held_out_prompts(spec, 3, seed=5)
    answer = full[:, spec.prompt_len :]
    prompt_block = prompt

    def loss_fn(m):
        block = mx.full(answer.shape, spec.mask_id, dtype=answer.dtype)
        noisy = mx.concatenate([prompt_block, block], axis=1)
        logits = m.denoise_logits_full(noisy)[:, spec.prompt_len :, :]
        logp = mx.log(mx.softmax(logits, axis=-1) + 1e-30)
        picked = mx.take_along_axis(logp, answer[..., None], axis=-1)[..., 0]
        return -picked.mean()

    optimizer = optim.AdamW(learning_rate=1e-3)
    grad_fn = nn.value_and_grad(model, loss_fn)
    for _ in range(1500):
        loss, grads = grad_fn(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    assert float(loss) < 0.05, float(loss)

    # the sampler must now reproduce the training path's predictions exactly
    training_pred = np.array(mx.argmax(model.denoise_logits_full(
        mx.concatenate([prompt_block, mx.full(answer.shape, spec.mask_id, dtype=answer.dtype)], axis=1)
    )[:, spec.prompt_len :, :], axis=-1))
    sampler_pred = np.array(one_shot_predict(model, prompt_block, spec.mask_id))
    assert np.array_equal(training_pred, sampler_pred)
    assert np.array_equal(training_pred, np.array(answer))
