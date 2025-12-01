"""Tests for the absorbing-state corruption process and the loss weighting."""

import math

import mlx.core as mx
import numpy as np
import pytest

from e2d2.diffusion import (
    DEFAULT_SCHEDULE,
    LinearNoiseSchedule,
    corruption_mask,
    discrete_t_grid,
    forward_corrupt,
    forward_corrupt_per_block,
    nelbo_loss,
    nelbo_loss_per_block,
    sample_train_times,
)

MASK = 7
VOCAB = 8


def test_clean_limit_and_prior_limit():
    """t=0 leaves the sequence untouched; t=1 masks every position."""
    tokens = mx.array([[0, 1, 2, 3], [4, 5, 6, 0]], dtype=mx.int32)
    clean = forward_corrupt(tokens, mx.array([0.0, 0.0]), MASK, mx.random.key(0))
    assert np.array_equal(np.array(clean), np.array(tokens))
    assert not np.array(corruption_mask(tokens, clean)).any()

    prior = forward_corrupt(tokens, mx.array([1.0, 1.0]), MASK, mx.random.key(0))
    assert np.array_equal(np.array(prior), np.full((2, 4), MASK))
    assert np.array(corruption_mask(tokens, prior)).all()


def test_empirical_mask_rate_matches_t():
    """The realised fraction of [MASK] tokens matches the requested t."""
    tokens = mx.zeros((64, 128), dtype=mx.int32)  # all token 0, never MASK
    for t in (0.25, 0.5, 0.9):
        noisy = forward_corrupt(tokens, mx.full((64,), t), MASK, mx.random.key(1))
        rate = float(np.array(noisy == MASK).mean())
        assert abs(rate - t) < 0.01, (t, rate)


def test_corruption_is_deterministic_per_key():
    tokens = mx.arange(32, dtype=mx.int32).reshape(2, 16)
    t = mx.array([0.3, 0.7])
    a = forward_corrupt(tokens, t, MASK, mx.random.key(5))
    b = forward_corrupt(tokens, t, MASK, mx.random.key(5))
    c = forward_corrupt(tokens, t, MASK, mx.random.key(6))
    assert np.array_equal(np.array(a), np.array(b))
    assert not np.array_equal(np.array(a), np.array(c))


def test_per_block_corruption_respects_block_levels():
    """A block with t=0 is untouched and a block with t=1 is entirely masked."""
    tokens = mx.arange(16, dtype=mx.int32).reshape(1, 16)
    t = mx.array([[0.0, 1.0, 0.0, 1.0]], dtype=mx.float32)
    noisy = forward_corrupt_per_block(tokens, t, block_size=4, mask_id=MASK, key=mx.random.key(0))
    got = np.array(noisy)
    assert np.array_equal(got[0, 0:4], np.array(tokens)[0, 0:4])
    assert np.array_equal(got[0, 8:12], np.array(tokens)[0, 8:12])
    assert (got[0, 4:8] == MASK).all()
    assert (got[0, 12:16] == MASK).all()


def test_per_block_corruption_normalises_int_and_float_levels():
    """Integer t=1 from a caller still means 'fully masked'."""
    tokens = mx.zeros((1, 8), dtype=mx.int32)
    noisy = forward_corrupt_per_block(tokens, mx.array([[1, 1]]), 4, MASK, mx.random.key(0))
    assert np.array_equal(np.array(noisy), np.full((1, 8), MASK))


def test_uniform_logits_loss_is_log_vocab():
    """With uniform logits every masked token costs exactly log(V)."""
    tokens = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    noisy = mx.array([[0, MASK, MASK, 3]], dtype=mx.int32)
    logits = mx.zeros((1, 4, VOCAB))
    loss, total_weight, n_masked = nelbo_loss(
        logits, tokens, noisy, mx.array([0.5]), DEFAULT_SCHEDULE
    )
    # two masked positions, weight 1/(1-0.5) = 2 each
    assert abs(float(total_weight) - 4.0) < 1e-5
    assert int(n_masked) == 2
    assert abs(float(loss) - math.log(VOCAB)) < 1e-5


def test_loss_only_counts_corrupted_positions():
    """An uncorrupted sequence has zero masked positions and a guarded loss."""
    tokens = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    logits = mx.zeros((1, 4, VOCAB))
    loss, total_weight, n_masked = nelbo_loss(
        logits, tokens, tokens, mx.array([0.0]), DEFAULT_SCHEDULE
    )
    assert int(n_masked) == 0
    assert float(total_weight) == 0.0
    assert float(loss) == 0.0


def test_loss_weight_matches_schedule_derivative():
    """weight(t) = alpha'(t) / (1 - alpha(t)) = 1/(1-t) for the linear schedule."""
    for t in (0.0, 0.25, 0.75, 0.99):
        got = float(DEFAULT_SCHEDULE.loss_weight(mx.array(t, dtype=mx.float32)))
        assert abs(got - 1.0 / (1.0 - t)) < 1e-3


def test_linear_schedule_is_monotone_decreasing():
    schedule = LinearNoiseSchedule()
    alphas = [float(schedule.alpha(t)) for t in np.linspace(0, 1, 11)]
    assert alphas[0] == 1.0 and alphas[-1] == 0.0
    assert all(b <= a for a, b in zip(alphas, alphas[1:]))


def test_t_grid_excludes_the_singular_endpoint():
    grid = discrete_t_grid(8)
    assert grid[0] == 0.0
    assert len(grid) == 8
    assert max(grid) < 1.0
    with pytest.raises(ValueError):
        discrete_t_grid(1)


def test_train_times_stay_on_the_grid():
    t = sample_train_times(16, 4096, mx.random.key(0))
    values = np.array(t)
    assert values.min() >= 0.0 and values.max() < 1.0
    # every value is a multiple of 1/16
    assert np.allclose(values * 16, np.round(values * 16))
    # and the sampler covers the grid roughly uniformly
    counts = np.bincount((values * 16).astype(int), minlength=16)
    assert counts.min() > 4096 / 16 * 0.7


def test_single_block_losses_agree():
    """With one block the per-block objective is exactly the whole-sequence one."""
    tokens = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    noisy = mx.array([[0, MASK, MASK, 3]], dtype=mx.int32)
    logits = mx.random.normal((1, 4, VOCAB), key=mx.random.key(3)) * 2.0
    whole, _, _ = nelbo_loss(logits, tokens, noisy, mx.array([0.5]), DEFAULT_SCHEDULE)
    per_block = nelbo_loss_per_block(
        logits, tokens, noisy, mx.array([[0.5]]), 4, DEFAULT_SCHEDULE
    )
    assert abs(float(whole) - float(per_block)) < 1e-4


def test_per_block_loss_is_a_weighted_mean_over_blocks():
    """Balance the masked positions per block and the two objectives coincide.

    ``nelbo_loss`` is one weighted mean over all masked tokens; the per-block form
    averages each block's weighted mean.  With equal masked counts and equal
    weights per block the two are the same number, so this pins the per-block
    normalisation (which is easy to get wrong by a factor of the block size).
    """
    tokens = mx.array([[0, 1, 2, 3, 4, 5, 6, 0]], dtype=mx.int32)
    noisy = mx.array([[0, MASK, 2, 3, 4, MASK, 6, 0]], dtype=mx.int32)
    logits = mx.random.normal((1, 8, VOCAB), key=mx.random.key(3)) * 2.0
    whole, _, _ = nelbo_loss(logits, tokens, noisy, mx.array([0.5]), DEFAULT_SCHEDULE)
    per_block = nelbo_loss_per_block(
        logits, tokens, noisy, mx.array([[0.5, 0.5]]), 4, DEFAULT_SCHEDULE
    )
    assert abs(float(whole) - float(per_block)) < 1e-4


def test_per_block_loss_ignores_zero_noise_blocks():
    """A block at t=0 contributes nothing regardless of its logits."""
    tokens = mx.array([[0, 1, 2, 3, 4, 5, 6, 0]], dtype=mx.int32)
    noisy = mx.concatenate([tokens[:, :4], mx.full((1, 4), MASK, dtype=mx.int32)], axis=1)
    good = mx.zeros((1, 8, VOCAB))
    bad = mx.concatenate(
        [mx.full((1, 4, VOCAB), 50.0), mx.zeros((1, 4, VOCAB))], axis=1
    )
    t = mx.array([[0.0, 0.5]], dtype=mx.float32)
    a = nelbo_loss_per_block(good, tokens, noisy, t, 4, DEFAULT_SCHEDULE)
    b = nelbo_loss_per_block(bad, tokens, noisy, t, 4, DEFAULT_SCHEDULE)
    assert abs(float(a) - float(b)) < 1e-6


def test_per_block_loss_ignores_uncorrupted_blocks():
    """A block with t=0 contributes nothing, even if its logits are terrible."""
    tokens = mx.array([[0, 1, 2, 3, 4, 5, 6, 0]], dtype=mx.int32)
    noisy = mx.concatenate([tokens[:, :4], mx.full((1, 4), MASK, dtype=mx.int32)], axis=1)
    good = mx.zeros((1, 8, VOCAB))
    bad = good.at[:, :4, :].add(mx.full((1, 4, VOCAB), 50.0))
    # t=0 for block 0, t=0.5 for block 1
    t = mx.array([[0.0, 0.5]])
    a = nelbo_loss_per_block(good, tokens, noisy, t, 4, DEFAULT_SCHEDULE)
    b = nelbo_loss_per_block(bad, tokens, noisy, t, 4, DEFAULT_SCHEDULE)
    assert abs(float(a) - float(b)) < 1e-6


def test_forward_corrupt_shape_and_error_contracts():
    tokens = mx.zeros((2, 8), dtype=mx.int32)
    with pytest.raises(ValueError):
        forward_corrupt(tokens, mx.array([0.5]), MASK, mx.random.key(0))
    with pytest.raises(ValueError):
        forward_corrupt_per_block(tokens, mx.array([[0.5, 0.5]]), 4, MASK, mx.random.key(0))
    # block_size must divide the sequence length
    with pytest.raises(ValueError):
        forward_corrupt_per_block(
            mx.zeros((2, 6), dtype=mx.int32), mx.ones((2, 2)), 4, MASK, mx.random.key(0)
        )
    out = forward_corrupt_per_block(
        tokens, mx.array([[0.5, 0.5], [0.5, 0.5]]), 4, MASK, mx.random.key(0)
    )
    assert out.dtype == mx.int32 and out.shape == (2, 8)
