"""A tiny synthetic sequence task: copy two operands and compute two functions.

Each example is a prompt block followed by exactly one answer block, with a
vocabulary small enough to train in a couple of minutes:

    prompt = [BOS, a, b, OP]          a, b in [0, M)
    answer = [a, b, (a + b) % M, (a - b) % M]

The model has to copy ``a`` and ``b`` out of the prompt and compute two modular
functions of them, which is the smallest task I could find that (a) needs the
context to be carried into the answer block rather than continued from it, and
(b) is not solvable by a purely local rule.  With ``M = 8`` there are 64 distinct
prompts, so a model can memorise the training stream; the experiment therefore
reports accuracy on a held-out set of prompts drawn with a different seed, and
what it measures is generalisation to unseen operand pairs, not novelty of
structure.

Vocabulary layout (fixed): ``0 = [MASK]``, ``1 = [BOS]``, ``2 = [OP]``, then
``M`` digit tokens ``3 .. M + 2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

MASK_ID = 0
BOS_ID = 1
OP_ID = 2
DIGIT_OFFSET = 3


@dataclass(frozen=True)
class TaskSpec:
    modulus: int = 8
    block_size: int = 4
    n_blocks: int = 2

    @property
    def vocab_size(self) -> int:
        return DIGIT_OFFSET + self.modulus

    @property
    def prompt_len(self) -> int:
        """Prompt is one block: BOS plus the three content tokens."""
        return self.block_size

    @property
    def seq_len(self) -> int:
        return self.block_size * self.n_blocks

    @property
    def n_answer_blocks(self) -> int:
        return self.n_blocks - 1

    @property
    def answer_len(self) -> int:
        return self.block_size * self.n_answer_blocks


def digits_to_tokens(values: mx.array) -> mx.array:
    return values + DIGIT_OFFSET


def make_batch(spec: TaskSpec, batch_size: int, key) -> tuple[mx.array, mx.array]:
    """Return ``(prompt_tokens, full_tokens)``, both ``(B, P)`` and ``(B, L)``."""
    key_a, key_b = mx.random.split(key, 2)
    a = mx.random.randint(0, spec.modulus, (batch_size,), key=key_a)
    b = mx.random.randint(0, spec.modulus, (batch_size,), key=key_b)
    return build_examples(spec, a, b)


def build_examples(spec: TaskSpec, a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
    m = spec.modulus
    bsz = a.shape[0]
    prompt = mx.stack(
        [
            mx.full((bsz,), BOS_ID, dtype=mx.int32),
            digits_to_tokens(a),
            digits_to_tokens(b),
            mx.full((bsz,), OP_ID, dtype=mx.int32),
        ],
        axis=1,
    )
    answer = mx.stack(
        [
            digits_to_tokens(a),
            digits_to_tokens(b),
            digits_to_tokens((a + b) % m),
            digits_to_tokens((a - b) % m),
        ],
        axis=1,
    )
    return prompt, mx.concatenate([prompt, answer], axis=1)


def held_out_prompts(spec: TaskSpec, n: int, seed: int) -> tuple[mx.array, mx.array]:
    """A deterministic held-out set, sampled with its own key."""
    key = mx.random.key(seed)
    prompt, full = make_batch(spec, n, key)
    return prompt, full


def all_prompts(spec: TaskSpec) -> tuple[mx.array, mx.array]:
    """Every distinct prompt, in a fixed order (``M * M`` examples)."""
    m = spec.modulus
    a = mx.array([i for i in range(m) for _ in range(m)], dtype=mx.int32)
    b = mx.array([j for _ in range(m) for j in range(m)], dtype=mx.int32)
    return build_examples(spec, a, b)
