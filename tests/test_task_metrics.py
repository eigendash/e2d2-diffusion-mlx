"""Tests for the synthetic copy-and-compute task and the evaluation metrics."""

import mlx.core as mx
import numpy as np
import pytest

from e2d2.metrics import (
    exact_match,
    nelbo_estimate,
    per_position_accuracy,
    token_accuracy,
)
from e2d2.model import E2D2, E2D2Config
from e2d2.task import (
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

SPEC = TaskSpec()


def test_vocabulary_layout_is_fixed():
    assert MASK_ID == 0 and BOS_ID == 1 and OP_ID == 2
    assert DIGIT_OFFSET == 3
    assert SPEC.vocab_size == DIGIT_OFFSET + SPEC.modulus


def test_task_shapes():
    prompt, full = make_batch(SPEC, 5, mx.random.key(0))
    assert prompt.shape == (5, SPEC.prompt_len)
    assert full.shape == (5, SPEC.seq_len)
    assert prompt.dtype == mx.int32 and full.dtype == mx.int32
    assert np.array_equal(np.array(prompt), np.array(full)[:, : SPEC.prompt_len])


def test_prompt_and_answer_content():
    """The answer is [a, b, (a+b) % M, (a-b) % M] and the prompt carries a and b."""
    a = mx.array([0, 1, 5, 7], dtype=mx.int32)
    b = mx.array([0, 2, 6, 3], dtype=mx.int32)
    prompt, full = build_examples(SPEC, a, b)
    p, f = np.array(prompt), np.array(full)
    m = SPEC.modulus
    assert (p[:, 0] == BOS_ID).all()
    assert (p[:, 3] == OP_ID).all()
    assert np.array_equal(p[:, 1] - DIGIT_OFFSET, np.array(a))
    assert np.array_equal(p[:, 2] - DIGIT_OFFSET, np.array(b))
    ans = f[:, SPEC.prompt_len :]
    assert np.array_equal(ans[:, 0] - DIGIT_OFFSET, np.array(a))
    assert np.array_equal(ans[:, 1] - DIGIT_OFFSET, np.array(b))
    assert np.array_equal(ans[:, 2] - DIGIT_OFFSET, (np.array(a) + np.array(b)) % m)
    assert np.array_equal(ans[:, 3] - DIGIT_OFFSET, (np.array(a) - np.array(b)) % m)


def test_no_mask_tokens_in_clean_data():
    _, full = held_out_prompts(SPEC, 64, seed=0)
    assert not (np.array(full) == MASK_ID).any()


def test_all_prompts_covers_every_pair_exactly_once():
    prompt, full = all_prompts(SPEC)
    m = SPEC.modulus
    assert prompt.shape == (m * m, SPEC.prompt_len)
    pairs = {(int(x), int(y)) for x, y in zip(prompt[:, 1], prompt[:, 2])}
    assert len(pairs) == m * m
    # and the answers are consistent with those pairs
    for i in range(m * m):
        a = int(prompt[i, 1]) - DIGIT_OFFSET
        b = int(prompt[i, 2]) - DIGIT_OFFSET
        assert int(full[i, SPEC.prompt_len + 2]) - DIGIT_OFFSET == (a + b) % m
        assert int(full[i, SPEC.prompt_len + 3]) - DIGIT_OFFSET == (a - b) % m


def test_held_out_prompts_are_deterministic_and_not_the_whole_space():
    a, _ = held_out_prompts(SPEC, 8, seed=4)
    b, _ = held_out_prompts(SPEC, 8, seed=4)
    assert np.array_equal(np.array(a), np.array(b))
    c, _ = held_out_prompts(SPEC, 8, seed=5)
    assert not np.array_equal(np.array(a), np.array(c))
    # the task space is only M*M prompts, so a small held-out draw is a subset
    assert len({tuple(row) for row in np.array(a).tolist()}) <= SPEC.modulus**2


def test_different_specs_have_different_vocabularies():
    big = TaskSpec(modulus=16)
    assert big.vocab_size == 19
    assert big.seq_len == SPEC.seq_len
    assert big.answer_len == SPEC.answer_len


# -- metrics ---------------------------------------------------------------


def test_exact_match_and_token_accuracy_hand_computed():
    target = mx.array([[1, 2, 3], [4, 5, 6]])
    predicted = mx.array([[1, 2, 3], [4, 0, 6]])
    assert exact_match(predicted, target) == 0.5
    assert abs(token_accuracy(predicted, target) - 5 / 6) < 1e-6
    assert per_position_accuracy(predicted, target) == [1.0, 0.5, 1.0]


def test_exact_match_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        exact_match(mx.zeros((2, 3)), mx.zeros((2, 4)))


def test_exact_match_is_one_for_identical_and_zero_for_disjoint():
    x = mx.array([[1, 2], [3, 4]])
    assert exact_match(x, x) == 1.0
    assert exact_match(x, x + 10) == 0.0


def test_nelbo_estimate_is_finite_and_lower_for_a_sharper_model():
    """The surrogate must reward putting probability on the right tokens."""
    spec = SPEC
    _, full = held_out_prompts(spec, 16, seed=0)
    mx.random.seed(0)
    model = E2D2(
        E2D2Config(
            vocab_size=spec.vocab_size,
            block_size=spec.block_size,
            n_blocks=spec.n_blocks,
            d_model=32,
            n_heads=4,
            encoder_layers=1,
            decoder_layers=1,
        )
    )
    value = nelbo_estimate(
        model, full, prompt_len=spec.prompt_len, grid_steps=4, n_samples=2
    )
    assert np.isfinite(value)
    assert value > 0

    # A model with a deliberate bias towards the true next token for the *answer*
    # positions must score lower.  This is implemented by replacing the decoder
    # head with one that returns the target logits directly.
    class Oracle(E2D2):
        def denoise_logits_full(self, noisy_tokens, clean_override=None):
            del noisy_tokens, clean_override
            logits = mx.full((full.shape[0], spec.seq_len, spec.vocab_size), -6.0)
            return mx.put_along_axis(logits, full[..., None], mx.array(6.0), axis=-1)

    oracle = Oracle(model.config)
    oracle.update(model.parameters())
    better = nelbo_estimate(
        oracle, full, prompt_len=spec.prompt_len, grid_steps=4, n_samples=2
    )
    assert better < value, (better, value)
