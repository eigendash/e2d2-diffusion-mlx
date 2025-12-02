"""Tests for blockwise sampling, unmasking, and the call-count invariant.

The invariant that carries the paper's whole efficiency argument is that the
encoder is invoked once per generated block, independently of the number of
denoising steps.  These tests count the invocations on the model object.
"""

import mlx.core as mx
import numpy as np
import pytest

from e2d2.model import E2D2, E2D2Config
from e2d2.sampling import (
    SampleConfig,
    clean_counts,
    one_shot_predict,
    sample_blockwise,
    unmask_step,
)
from e2d2.task import MASK_ID, TaskSpec, held_out_prompts

SMALL = dict(d_model=32, n_heads=4, encoder_layers=2, decoder_layers=1)


def make_model(seed: int = 0, **overrides) -> E2D2:
    spec = TaskSpec()
    kwargs = dict(
        vocab_size=spec.vocab_size,
        block_size=spec.block_size,
        n_blocks=spec.n_blocks,
        **SMALL,
    )
    kwargs.update(overrides)
    mx.random.seed(seed)
    return E2D2(E2D2Config(**kwargs))


def make_prompt(n: int = 3):
    spec = TaskSpec()
    prompt, _ = held_out_prompts(spec, n, seed=1)
    return spec, prompt


# -- the efficiency invariant ----------------------------------------------


@pytest.mark.parametrize("num_steps", [1, 2, 4, 8, 16])
def test_encoder_is_invoked_once_regardless_of_denoising_steps(num_steps):
    model = make_model()
    spec, prompt = make_prompt()
    result = sample_blockwise(
        model,
        prompt,
        SampleConfig(
            num_steps=num_steps, block_size=spec.block_size, n_blocks=1, mask_id=MASK_ID
        ),
    )
    assert result.counts["encoder_calls"] == 1, result.counts
    assert result.counts["decoder_calls"] == num_steps, result.counts


@pytest.mark.parametrize("num_blocks", [1, 2, 3])
def test_encoder_calls_scale_with_blocks_not_steps(num_blocks):
    model = make_model(n_blocks=4 if num_blocks >= 2 else 2)
    spec = TaskSpec()
    prompt, _ = held_out_prompts(spec, 2, seed=3)
    num_steps = 6
    result = sample_blockwise(
        model,
        prompt,
        SampleConfig(
            num_steps=num_steps,
            block_size=spec.block_size,
            n_blocks=num_blocks,
            mask_id=MASK_ID,
        ),
    )
    assert result.counts["encoder_calls"] == num_blocks
    assert result.counts["decoder_calls"] == num_blocks * num_steps


def test_decoder_only_baseline_invokes_no_encoder_and_one_stack_per_step():
    baseline = make_model(encoder_layers=0, decoder_layers=3)
    spec, prompt = make_prompt()
    num_steps = 5
    result = sample_blockwise(
        baseline,
        prompt,
        SampleConfig(
            num_steps=num_steps, block_size=spec.block_size, n_blocks=1, mask_id=MASK_ID
        ),
    )
    assert result.counts["encoder_calls"] == 0
    assert result.counts["decoder_calls"] == num_steps
    # The baseline's one network is the whole model, so the per-step layer cost is
    # the full depth rather than the decoder's share of it.
    assert result.counts["decoder_layer_forwards"] == num_steps * 3


def test_layer_forward_accounting_splits_encoder_and_decoder():
    model = make_model(encoder_layers=3, decoder_layers=1)
    spec, prompt = make_prompt()
    num_steps = 4
    result = sample_blockwise(
        model,
        prompt,
        SampleConfig(
            num_steps=num_steps, block_size=spec.block_size, n_blocks=1, mask_id=MASK_ID
        ),
    )
    counts = result.counts
    # one encoder call over the prompt block (4 tokens, 3 layers)
    assert counts["encoder_layer_forwards"] == 3 * 1
    assert counts["decoder_layer_forwards"] == 1 * num_steps
    assert counts["encoder_layer_forwards"] < counts["decoder_layer_forwards"]


# -- the single-step limit -------------------------------------------------


def test_single_step_sampling_equals_one_shot_prediction():
    """T=1 unfixes every masked position from the all-mask initialisation."""
    model = make_model()
    spec, prompt = make_prompt()
    result = sample_blockwise(
        model,
        prompt,
        SampleConfig(num_steps=1, block_size=spec.block_size, n_blocks=1, mask_id=MASK_ID),
    )
    expected = one_shot_predict(model, prompt, MASK_ID)
    assert np.array_equal(np.array(result.tokens), np.array(expected))
    assert result.counts["decoder_calls"] == 1
    # no [MASK] can survive a single step
    assert not (np.array(result.tokens) == MASK_ID).any()


def test_single_step_of_a_two_block_model_ignores_earlier_generated_blocks():
    """T=1 predicts each block from the prompt plus earlier blocks, in one call each."""
    model = make_model()
    spec, prompt = make_prompt()
    result = sample_blockwise(
        model,
        prompt,
        SampleConfig(num_steps=1, block_size=spec.block_size, n_blocks=2, mask_id=MASK_ID),
    )
    assert result.counts["decoder_calls"] == 2
    assert result.counts["encoder_calls"] == 2
    assert not (np.array(result.tokens) == MASK_ID).any()


# -- unmasking schedule ----------------------------------------------------


def test_clean_counts_are_monotone_and_end_at_block_size():
    for block_size in (1, 4, 8):
        for num_steps in (1, 3, 5, 8):
            counts = clean_counts(block_size, num_steps)
            assert len(counts) == num_steps
            assert counts[-1] == block_size
            assert all(b >= a for a, b in zip(counts, counts[1:]))
            assert all(0 <= c <= block_size for c in counts)


def test_clean_counts_track_the_linear_schedule():
    """The number of clean positions after step i is ceil(i*S/T) = ceil((1-t)*S)."""
    for block_size, num_steps in ((4, 4), (4, 8), (8, 3)):
        counts = clean_counts(block_size, num_steps)
        for i, c in enumerate(counts, start=1):
            t = 1.0 - i / num_steps
            assert c == int(np.ceil((1.0 - t) * block_size))
            # and it is never fewer than the masked-diffusion expectation
            assert c >= int(np.floor((1.0 - t) * block_size))


def test_clean_counts_rejects_zero_steps():
    with pytest.raises(ValueError):
        clean_counts(4, 0)


def test_unmask_step_fills_the_most_confident_masked_positions():
    block = mx.array([MASK_ID, MASK_ID, 5, MASK_ID])
    logits = mx.zeros((4, 8))
    # give positions 0, 1, 3 very different confidences for token 6
    logits = logits.at[0, 6].add(10.0)
    logits = logits.at[1, 6].add(1.0)
    logits = logits.at[3, 6].add(5.0)
    out = np.array(unmask_step(block, logits, MASK_ID, n_clean=3))
    # positions 0 and 3 are the two most confident masked positions
    assert out[0] == 6 and out[3] == 6
    assert out[1] == MASK_ID
    assert out[2] == 5  # carried over, not remasked


def test_unmask_step_never_remasks():
    block = mx.array([3, 4, MASK_ID, MASK_ID])
    logits = mx.zeros((4, 8))
    out = np.array(unmask_step(block, logits, MASK_ID, n_clean=2))
    assert out[0] == 3 and out[1] == 4


def test_unmask_step_is_monotone_across_a_full_schedule():
    """Once a position is clean it keeps its value for the rest of the schedule."""
    mx.random.seed(0)
    block_size = 8
    logits = mx.random.normal((block_size, 12), key=mx.random.key(2))
    noisy = mx.full((block_size,), MASK_ID, dtype=mx.int32)
    previous = None
    for n_clean in clean_counts(block_size, 5):
        noisy = unmask_step(noisy, logits, MASK_ID, n_clean)
        current = np.array(noisy)
        assert (current != MASK_ID).sum() == n_clean
        if previous is not None:
            keep = previous != MASK_ID
            assert np.array_equal(current[keep], previous[keep])
        previous = current
    assert not (previous == MASK_ID).any()


def test_unmask_step_ties_break_on_position():
    block = mx.full((4,), MASK_ID, dtype=mx.int32)
    logits = mx.zeros((4, 8))  # every position equally confident
    out = np.array(unmask_step(block, logits, MASK_ID, n_clean=2))
    # the two lowest positions win the tie, and argmax's own tie-break picks the
    # lowest non-mask token, which is 1 because 0 is [MASK]
    assert list(out) == [1, 1, MASK_ID, MASK_ID]


def test_unmask_step_n_clean_zero_is_a_no_op():
    block = mx.array([MASK_ID, 3, MASK_ID, MASK_ID])
    out = np.array(unmask_step(block, mx.zeros((4, 8)), MASK_ID, n_clean=0))
    assert np.array_equal(out, np.array(block))


def test_unmask_step_full_block_replaces_everything():
    block = mx.array([MASK_ID, MASK_ID, MASK_ID, MASK_ID])
    logits = mx.zeros((4, 8))
    logits = logits.at[:, 7].add(3.0)
    out = np.array(unmask_step(block, logits, MASK_ID, n_clean=4))
    assert np.array_equal(out, np.full((4,), 7))


# -- sampler plumbing ------------------------------------------------------


def test_sampler_output_shape_and_no_masks_left():
    model = make_model()
    spec, prompt = make_prompt(4)
    result = sample_blockwise(
        model,
        prompt,
        SampleConfig(num_steps=4, block_size=spec.block_size, n_blocks=1, mask_id=MASK_ID),
    )
    assert result.tokens.shape == (4, 1 * spec.block_size)
    assert result.tokens.dtype == prompt.dtype
    assert not (np.array(result.tokens) == MASK_ID).any()
    # the completed prompt+answer sequence is a valid model input
    logits = model.denoise_logits_full(mx.concatenate([prompt, result.tokens], axis=1))
    assert np.isfinite(np.array(logits)).all()


def test_sampler_carries_generated_blocks_into_the_encoder_context():
    """The second block must be decoded with the first block as clean context.

    The encoder runs once before the loop and once between the two blocks, so it is
    invoked twice: on the 4-token prompt and then on the 8-token prefix.  There is
    no encoder call after the final block.
    """
    model = make_model()
    spec, prompt = make_prompt(2)
    result = sample_blockwise(
        model,
        prompt,
        SampleConfig(num_steps=3, block_size=spec.block_size, n_blocks=2, mask_id=MASK_ID),
    )
    # prompt has 4 tokens; after block 1, the prefix is 8 tokens
    assert result.counts["encoder_tokens"] == 4 + 8
    assert result.counts["decoder_tokens"] == 2 * 3 * spec.block_size


def test_sampler_rejects_mismatched_block_size_and_too_many_blocks():
    model = make_model()
    spec, prompt = make_prompt()
    with pytest.raises(ValueError):
        sample_blockwise(
            model, prompt, SampleConfig(num_steps=2, block_size=2, n_blocks=1, mask_id=MASK_ID)
        )
    with pytest.raises(ValueError):
        sample_blockwise(
            model,
            prompt,
            SampleConfig(num_steps=2, block_size=spec.block_size, n_blocks=99, mask_id=MASK_ID),
        )
    with pytest.raises(ValueError):
        sample_blockwise(
            model,
            mx.zeros((1, 3), dtype=mx.int32),
            SampleConfig(num_steps=2, block_size=spec.block_size, n_blocks=1, mask_id=MASK_ID),
        )


def test_sampler_resets_counts_between_calls():
    model = make_model()
    spec, prompt = make_prompt()
    cfg = SampleConfig(num_steps=4, block_size=spec.block_size, n_blocks=1, mask_id=MASK_ID)
    a = sample_blockwise(model, prompt, cfg)
    b = sample_blockwise(model, prompt, cfg)
    assert a.counts["decoder_calls"] == b.counts["decoder_calls"] == 4
    assert a.counts["encoder_calls"] == b.counts["encoder_calls"] == 1


def test_sampler_is_deterministic_for_a_fixed_model():
    model = make_model(seed=7)
    spec, prompt = make_prompt()
    cfg = SampleConfig(num_steps=5, block_size=spec.block_size, n_blocks=2, mask_id=MASK_ID)
    a = sample_blockwise(model, prompt, cfg)
    b = sample_blockwise(model, prompt, cfg)
    assert np.array_equal(np.array(a.tokens), np.array(b.tokens))


def test_step_history_records_the_schedule():
    model = make_model()
    spec, prompt = make_prompt(1)
    result = sample_blockwise(
        model,
        prompt,
        SampleConfig(num_steps=3, block_size=spec.block_size, n_blocks=1, mask_id=MASK_ID),
        collect_steps=True,
    )
    assert len(result.step_history) == 3
    assert [h["n_clean"] for h in result.step_history] == clean_counts(spec.block_size, 3)
    assert result.step_history[-1]["n_clean"] == spec.block_size


def test_sampler_wall_time_is_measured():
    model = make_model()
    spec, prompt = make_prompt()
    result = sample_blockwise(
        model,
        prompt,
        SampleConfig(num_steps=4, block_size=spec.block_size, n_blocks=1, mask_id=MASK_ID),
    )
    assert result.wall_seconds >= 0.0
