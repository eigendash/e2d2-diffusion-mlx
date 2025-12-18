"""Tests for the synthetic reverse-copy task and the evaluation metrics."""

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
    TaskSpec,
    all_prompts,
    build_examples,
    held_out_prompts,
    make_batch,
    split_prompts,
)

SPEC = TaskSpec()


def test_vocabulary_layout_is_fixed():
    """[BOS], digits, then [MASK] last, following paper Appendix A.1."""
    assert BOS_ID == 0 and DIGIT_OFFSET == 1
    assert SPEC.vocab_size == DIGIT_OFFSET + SPEC.modulus + 1
    assert MASK_ID == SPEC.mask_id == SPEC.vocab_size - 1
    # the mask id must not collide with any real token
    assert MASK_ID == SPEC.modulus + 1


def test_task_shapes():
    prompt, full = make_batch(SPEC, 5, mx.random.key(0))
    assert prompt.shape == (5, SPEC.prompt_len)
    assert full.shape == (5, SPEC.seq_len)
    assert prompt.dtype == mx.int32 and full.dtype == mx.int32
    assert np.array_equal(np.array(prompt), np.array(full)[:, : SPEC.prompt_len])


def test_prompt_layout_and_reversed_answer():
    """The answer block is exactly the prompt block reversed."""
    content = mx.array([[0, 1, 2], [3, 4, 5], [7, 0, 7]], dtype=mx.int32)
    prompt, full = build_examples(SPEC, content)
    p, f = np.array(prompt), np.array(full)
    assert (p[:, 0] == BOS_ID).all()
    assert np.array_equal(p[:, 1:] - DIGIT_OFFSET, np.array(content))
    answer = f[:, SPEC.prompt_len :]
    assert np.array_equal(answer, p[:, ::-1])
    # spelled out: last digit first, BOS last
    assert np.array_equal(answer[:, 0] - DIGIT_OFFSET, np.array(content)[:, 2])
    assert np.array_equal(answer[:, 1] - DIGIT_OFFSET, np.array(content)[:, 1])
    assert np.array_equal(answer[:, 2] - DIGIT_OFFSET, np.array(content)[:, 0])
    assert (answer[:, 3] == BOS_ID).all()


def test_answer_is_not_a_positional_copy_of_the_prompt():
    """No fixed prompt position holds the answer for the same position.

    This is what makes the task a test of *ordering* rather than of copying: for
    every one of the four slots, the answer token differs from the token the
    prompt has in that slot for the overwhelming majority of examples, so no
    "copy k to k" or fixed-shift rule can solve it.
    """
    _, full = held_out_prompts(SPEC, 1024, seed=0)
    prompt = np.array(full[:, : SPEC.prompt_len])
    answer = np.array(full[:, SPEC.prompt_len :])
    for k in range(SPEC.prompt_len):
        differs = float((answer[:, k] != prompt[:, k]).mean())
        # slot 3 is the trailing BOS, which differs from d3 whenever d3 != 0
        assert differs > 1.0 - 1.0 / SPEC.modulus - 0.02, (k, differs)
    # and the answer is exactly the reversal, which is why the shifts above fail
    assert np.array_equal(answer, prompt[:, ::-1])


def test_no_mask_tokens_in_clean_data():
    _, full = held_out_prompts(SPEC, 64, seed=0)
    assert not (np.array(full) == MASK_ID).any()


def test_build_examples_rejects_wrong_content_width():
    with pytest.raises(ValueError):
        build_examples(SPEC, mx.zeros((2, 2), dtype=mx.int32))


def test_all_prompts_covers_every_triple_exactly_once():
    prompt, full = all_prompts(SPEC)
    m = SPEC.modulus
    assert prompt.shape == (m**3, SPEC.prompt_len)
    triples = {tuple(int(v) - DIGIT_OFFSET for v in row[1:]) for row in np.array(prompt)}
    assert len(triples) == m**3
    assert max(max(t) for t in triples) == m - 1


def test_split_prompts_are_exactly_disjoint():
    (tr_p, tr_f), (ho_p, ho_f) = split_prompts(SPEC, 400, 400, seed=1)
    train = {tuple(row) for row in np.array(tr_p).tolist()}
    held = {tuple(row) for row in np.array(ho_p).tolist()}
    assert len(train) == 400 and len(held) == 400
    assert train & held == set()
    for p, f in ((tr_p, tr_f), (ho_p, ho_f)):
        assert np.array_equal(np.array(f)[:, SPEC.prompt_len :], np.array(p)[:, ::-1])


def test_split_prompts_is_deterministic_and_validates_size():
    a, _ = split_prompts(SPEC, 10, 10, seed=3)
    b, _ = split_prompts(SPEC, 10, 10, seed=3)
    assert np.array_equal(np.array(a[0]), np.array(b[0]))
    c, _ = split_prompts(SPEC, 10, 10, seed=4)
    assert not np.array_equal(np.array(a[0]), np.array(c[0]))
    with pytest.raises(ValueError):
        split_prompts(SPEC, SPEC.prompt_space, 1, seed=0)


def test_held_out_prompts_are_deterministic():
    a, _ = held_out_prompts(SPEC, 8, seed=4)
    b, _ = held_out_prompts(SPEC, 8, seed=4)
    assert np.array_equal(np.array(a), np.array(b))
    c, _ = held_out_prompts(SPEC, 8, seed=5)
    assert not np.array_equal(np.array(a), np.array(c))


def test_different_specs_have_different_vocabularies():
    big = TaskSpec(modulus=32)
    assert big.vocab_size == 34
    assert big.mask_id == 33 and big.mask_id != SPEC.mask_id
    assert big.seq_len == SPEC.seq_len
    assert big.prompt_space == 32**3 > SPEC.prompt_space


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


def _random_model(spec: TaskSpec, seed: int) -> E2D2:
    mx.random.seed(seed)
    return E2D2(
        E2D2Config(
            vocab_size=spec.vocab_size,
            block_size=spec.block_size,
            n_blocks=spec.n_blocks,
            d_model=16,
            n_heads=2,
            encoder_layers=1,
            decoder_layers=1,
            mask_id=spec.mask_id,
        )
    )


def test_nelbo_estimate_is_finite_and_rewards_the_right_tokens():
    """The surrogate must be finite, positive, and lower for a sharper model."""
    spec = SPEC
    _, full = held_out_prompts(spec, 16, seed=0)
    model = _random_model(spec, 0)
    value = nelbo_estimate(
        model, full, prompt_len=spec.prompt_len, grid_steps=4, n_samples=2
    )
    assert np.isfinite(value) and value > 0

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
