"""A tiny synthetic sequence task: reverse the prompt block.

Each example is a prompt block followed by exactly one answer block:

    prompt = [BOS, d1, d2, d3]        d_i in [0, M)
    answer = [d3, d2, d1, BOS]        exactly the prompt block reversed

The answer is a deterministic function of the prompt, so the context has to be
carried into the answer block, and the required computation is an *ordering* of
the context tokens rather than a copy at a fixed offset: the model reads prompt
positions 1, 2, 3 and emits them in the opposite order.  That makes it a real
test of whether the decoder can read the encoder's representation of the prompt,
while staying learnable in a couple of minutes.

Why not arithmetic: an earlier version of this task asked for ``[a, b, a+b, a-b]``
from a two-digit prompt.  At toy scale every architecture simply memorised the
training prompts (training loss 0.0) while held-out token accuracy stalled at the
~0.5 you get from copying the operands without learning the arithmetic, so exact
match was 0.0 everywhere and the quality comparison carried no information.  A
64-prompt space against a 270k-parameter model is memorisation territory; here the
space is ``M ** n_content`` (4096 at the default ``M = 16``).

The vocabulary is ``[BOS], digits 0..M-1, [MASK]`` with the absorbing state last,
following the paper's Appendix A.1 (``argmax_i m_i = V``).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

BOS_ID = 0
DIGIT_OFFSET = 1


@dataclass(frozen=True)
class TaskSpec:
    """Shape of the task: ``modulus`` digit values and a fixed block layout.

    The prompt occupies ``prompt_blocks`` blocks and the answer is the same number
    of tokens, so the sequence is ``2 * prompt_blocks`` blocks:

        prompt = [BOS, d1, ..., d_{4P-1}]     (4P tokens, P blocks)
        answer = reverse(prompt)              (4P tokens, P blocks)

    ``prompt_blocks=1`` is the default and gives the single-answer-block setting the
    quality experiment uses.  Larger values exist so the *cost* of sampling can be
    measured with more than one generated block: the encoder is invoked once per
    generated block, so this is what turns "the encoder runs once" into a scaling
    claim rather than a single data point.
    """

    modulus: int = 16
    block_size: int = 4
    prompt_blocks: int = 1

    @property
    def vocab_size(self) -> int:
        return DIGIT_OFFSET + self.modulus + 1  # digits then [MASK]

    @property
    def mask_id(self) -> int:
        """The absorbing state is the last vocabulary entry (paper Appendix A.1)."""
        return self.vocab_size - 1

    @property
    def prompt_len(self) -> int:
        """Prompt tokens: one block per ``prompt_blocks``, BOS plus digits."""
        return self.block_size * self.prompt_blocks

    @property
    def n_answer_blocks(self) -> int:
        return self.prompt_blocks

    @property
    def n_blocks(self) -> int:
        return self.prompt_blocks + self.n_answer_blocks

    @property
    def seq_len(self) -> int:
        return self.block_size * self.n_blocks

    @property
    def answer_len(self) -> int:
        return self.block_size * self.n_answer_blocks

    @property
    def n_content(self) -> int:
        """Digits per prompt: the prompt length minus the leading BOS."""
        return self.prompt_len - 1

    @property
    def prompt_space(self) -> int:
        return self.modulus**self.n_content


#: Mask index of the default :class:`TaskSpec`.  ``TaskSpec.mask_id`` is the
#: authoritative value; this constant exists for callers that use the default spec.
MASK_ID = TaskSpec().mask_id


def digits_to_tokens(values: mx.array) -> mx.array:
    return values + DIGIT_OFFSET


def build_examples(spec: TaskSpec, content: mx.array) -> tuple[mx.array, mx.array]:
    """Build ``(prompt, full)`` from ``(B, n_content)`` digit values."""
    if content.shape[1] != spec.n_content:
        raise ValueError(f"expected {spec.n_content} content digits per example")
    bsz = content.shape[0]
    prompt = mx.concatenate(
        [mx.full((bsz, 1), BOS_ID, dtype=mx.int32), digits_to_tokens(content)],
        axis=1,
    )
    # The answer is the whole prompt block reversed: last digit first, BOS last.
    reversed_prompt = prompt[:, ::-1]
    return prompt, mx.concatenate([prompt, reversed_prompt], axis=1)


def make_batch(spec: TaskSpec, batch_size: int, key) -> tuple[mx.array, mx.array]:
    """Return ``(prompt_tokens, full_tokens)`` sampled uniformly."""
    content = mx.random.randint(0, spec.modulus, (batch_size, spec.n_content), key=key)
    return build_examples(spec, content)


def _decode_flat(spec: TaskSpec, flat: mx.array) -> mx.array:
    """Decode flat indices in ``[0, M ** n_content)`` into base-M digit columns."""
    digits = []
    for i in range(spec.n_content):
        power = spec.modulus ** (spec.n_content - 1 - i)
        digits.append(((flat // power) % spec.modulus).astype(mx.int32))
    return mx.stack(digits, axis=1)


def split_prompts(
    spec: TaskSpec, n_train: int, n_held_out: int, seed: int = 0
) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
    """Exactly disjoint train and held-out sets over the ``M ** n_content`` space.

    The two sets are drawn from a permutation of the entire prompt space, so they
    do not overlap at all rather than overlapping with small probability.
    """
    total = n_train + n_held_out
    space = spec.prompt_space
    if total > space:
        raise ValueError(f"only {space} distinct prompts exist, asked for {total}")
    key = mx.random.key(seed)
    perm = mx.random.permutation(mx.arange(space, dtype=mx.int32), key=key)[:total]
    content = _decode_flat(spec, perm)
    train = build_examples(spec, content[:n_train])
    held_out = build_examples(spec, content[n_train:])
    return train, held_out


def held_out_prompts(spec: TaskSpec, n: int, seed: int) -> tuple[mx.array, mx.array]:
    """A deterministic held-out set, sampled with its own key."""
    return make_batch(spec, n, mx.random.key(seed))


def all_prompts(spec: TaskSpec) -> tuple[mx.array, mx.array]:
    """Every distinct prompt, in a fixed order (``M ** n_content`` examples)."""
    flat = mx.arange(spec.prompt_space, dtype=mx.int32)
    return build_examples(spec, _decode_flat(spec, flat))
