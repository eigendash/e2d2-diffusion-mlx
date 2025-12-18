"""Transformer pieces shared by the E2D2 and decoder-only models.

Small and deliberately plain: pre-norm blocks with multi-head attention and a
gated MLP, learned positional embeddings, tied embedding/output weights.  There
is no KV-cache plumbing; the experiment measures decoder *invocations* and
per-invocation layer count rather than wall-clock latency, and the sampler builds
whatever it needs by calling these modules directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelConfig:
    """Shape of a single transformer stack (encoder or decoder)."""

    vocab_size: int
    max_seq_len: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_mult: int = 4
    n_blocks: int = 4
    block_size: int = 4

    @property
    def seq_len(self) -> int:
        return self.n_blocks * self.block_size


class Attention(nn.Module):
    """Multi-head attention over a fixed boolean keep-mask."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        """``x``: ``(B, L, D)``; ``mask``: ``(L, L)`` bool keep-mask."""
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        # (B, H, L, Dh)
        q = q.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        bias = mx.where(mask, mx.array(0.0, mx.float32), mx.array(-1e9, mx.float32))
        scores = scores + bias[None, None, :, :]
        weights = mx.softmax(scores, axis=-1)
        attended = (weights @ v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.out(attended)


class MLP(nn.Module):
    def __init__(self, d_model: int, mult: int):
        super().__init__()
        hidden = d_model * mult
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, mlp_mult: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, max_seq_len)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_mult)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


class Stack(nn.Module):
    """A stack of ``n_layers`` blocks with its own embeddings and head.

    ``n_layers = 0`` is the identity stack used by the structural-reduction test:
    ``__call__`` returns the token embeddings unchanged and the stack contributes
    only its embedding tables (about ``vocab_size * d_model`` parameters) to the
    trunk.

    The output head is the embedding matrix itself, via ``Embedding.as_linear``.
    That is exact weight tying without a second parameter array, so an optimiser
    cannot drift the two copies apart.
    """

    def __init__(self, cfg: ModelConfig, with_head: bool, use_pos: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_pos = use_pos
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = [
            Block(cfg.d_model, cfg.n_heads, cfg.max_seq_len, cfg.mlp_mult)
            for _ in range(cfg.n_layers)
        ]
        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.with_head = with_head

    def embed_tokens(self, tokens: mx.array, positions: mx.array | None = None):
        """Token embedding plus an absolute positional embedding.

        ``positions`` must be the tokens' absolute offsets in the generated
        sequence in *both* the training pass and the sampler, otherwise the model
        sees the same block at two different offsets.  ``use_pos=False`` exists for
        ablations only: a fully masked block is a block of identical inputs, so
        without positions its query positions are interchangeable and the model
        cannot learn which token goes where (the experiment found exactly that
        failure mode, stuck at ``log(block_size)``).
        """
        embedded = self.embed(tokens)
        if not self.use_pos:
            return embedded
        if positions is None:
            positions = mx.arange(tokens.shape[1])
        return embedded + self.pos(positions)[None, :, :]

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        """``x``: ``(B, L, D)``; ``mask``: ``(L, L)`` bool keep-mask."""
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        # (B, H, L, Dh)
        q = q.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        bias = mx.where(mask, mx.array(0.0, mx.float32), mx.array(-1e9, mx.float32))
        scores = scores + bias[None, None, :, :]
        weights = mx.softmax(scores, axis=-1)
        attended = (weights @ v).transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.out(attended)


class MLP(nn.Module):
    def __init__(self, d_model: int, mult: int):
        super().__init__()
        hidden = d_model * mult
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, mlp_mult: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, max_seq_len)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_mult)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


class Stack(nn.Module):
    """A stack of ``n_layers`` blocks with its own embeddings and head.

    ``n_layers = 0`` is the identity stack used by the structural-reduction test:
    ``__call__`` returns the token embeddings unchanged and the stack contributes
    only its embedding tables (about ``vocab_size * d_model`` parameters) to the
    trunk.

    The output head is the embedding matrix itself, via ``Embedding.as_linear``.
    That is exact weight tying without a second parameter array, so an optimiser
    cannot drift the two copies apart.
    """

    def __init__(self, cfg: ModelConfig, with_head: bool, use_pos: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_pos = use_pos
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = [
            Block(cfg.d_model, cfg.n_heads, cfg.max_seq_len, cfg.mlp_mult)
            for _ in range(cfg.n_layers)
        ]
        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.with_head = with_head

    def embed_tokens(self, tokens: mx.array, positions: mx.array | None = None):
        """Token embedding, plus positional embedding when this stack uses one.

        ``use_pos=False`` is set on the decoder.  The decoder only ever sees one
        noised block at a time, and that block sits at a different absolute
        position depending on how many blocks precede it, so an absolute
        positional embedding on the noised side cannot be made consistent between
        the training pass (all L noised tokens at once, positions 0..L-1) and the
        sampling pass (block b alone, positions b*S..(b+1)*S-1).  Leaving position
        out of the noised embeddings removes the inconsistency: the decoder knows
        where it is inside the block from the block-diagonal attention structure,
        and where the block is in the sequence from the encoder representation it
        attends to.  The encoder does use absolute positions, and those are
        consistent: it always sees a prefix starting from position 0.
        """
        embedded = self.embed(tokens)
        if not self.use_pos:
            return embedded
        if positions is None:
            positions = mx.arange(tokens.shape[1])
        return embedded + self.pos(positions)[None, :, :]

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        if not self.blocks:
            # A zero-layer stack is a pure identity on its input.  It must not
            # even normalise: the structural-reduction test relies on the clean
            # context of the decoder-only model being exactly the token
            # embeddings, with no encoder computation of any kind in between.
            return x
        for block in self.blocks:
            x = block(x, mask)
        return self.final_norm(x)

    def forward_tokens(self, tokens: mx.array, mask: mx.array) -> mx.array:
        return self(self.embed_tokens(tokens), mask)

    def logits(self, hidden: mx.array) -> mx.array:
        if not self.with_head:
            raise ValueError("this stack has no output head")
        return self.embed.as_linear(hidden)
