"""E2D2: an encoder-decoder block-diffusion language model, plus its decoder-only limit.

The architecture follows the paper's "last hidden state" design decision: the
final encoder output ``h`` is concatenated to the decoder's token embeddings along
the sequence dimension and every decoder layer runs one fused attention over
``[h ; z_t]``.

    Training (paper Algorithm 2)
        h^{1:L}        = Encoder(x^{1:L}, M_Enc)          # block-causal, clean tokens
        x_logit^{1:L}  = Decoder(z_t^{1:L}, h^{1:L}, M_Dec)

    Sampling (paper Algorithm 1)
        h <- Encoder(prompt)                              # encode the prompt once
        for b in 1..B:
            z^b <- m^b                                    # start from all-mask
            repeat T times: z^b <- Sample(Decoder(z^b, h))  # decoder only
            h <- Encoder(x^{1:b})                          # one encoder call per block

The decoder-only block-diffusion baseline is the *same* code path with
``encoder_layers = 0`` and an identity "encoder": the clean tokens then appear in
the decoder's context as their own embeddings, i.e. exactly the BD3LM-style input
``[x^{1:L} ; z_t^{1:L}]`` processed by one whole network.  That limit is
documented and tested; see README "Where this may differ from the paper".

Call accounting lives here rather than in the sampler, so the invariant the
paper's efficiency claim rests on -- *the encoder is invoked once per block, not
once per denoising step* -- is verified by counting on the model object.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.utils
import mlx.nn as nn

from . import masks
from .diffusion import (
    DEFAULT_SCHEDULE,
    forward_corrupt_per_block,
    nelbo_loss_per_block,
    sample_train_times,
)
from .layers import ModelConfig, Stack


@dataclass
class E2D2Config:
    """Configuration of an encoder-decoder (or decoder-only) block diffusion model."""

    vocab_size: int
    block_size: int = 4
    n_blocks: int = 2
    d_model: int = 64
    n_heads: int = 4
    encoder_layers: int = 3
    decoder_layers: int = 1
    mlp_mult: int = 4
    mask_id: int | None = None

    def __post_init__(self) -> None:
        if self.mask_id is None:
            # The paper puts the absorbing state last in the vocabulary
            # (Appendix A.1: argmax_i m_i = V).  It is carried explicitly rather
            # than re-derived at each use site, because an implicit
            # ``vocab_size - 1`` is exactly the kind of assumption that silently
            # disagrees with the data pipeline's [MASK] id.
            object.__setattr__(self, "mask_id", self.vocab_size - 1)
        if not 0 <= self.mask_id < self.vocab_size:
            raise ValueError("mask_id must be a valid vocabulary index")

    @property
    def seq_len(self) -> int:
        return self.n_blocks * self.block_size

    @property
    def is_decoder_only(self) -> bool:
        """The structural limit: an identity encoder, no trainable encoder stack."""
        return self.encoder_layers == 0

    def stack_config(self, n_layers: int) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            max_seq_len=2 * self.seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=n_layers,
            mlp_mult=self.mlp_mult,
            n_blocks=self.n_blocks,
            block_size=self.block_size,
        )


COUNT_FIELDS = (
    "encoder_calls",
    "decoder_calls",
    "encoder_layer_forwards",
    "decoder_layer_forwards",
    "encoder_tokens",
    "decoder_tokens",
)


@dataclass
class CallCounts:
    """How many module invocations happened, and how many layer-forwards that cost."""

    encoder_calls: int = 0
    decoder_calls: int = 0
    encoder_layer_forwards: int = 0
    decoder_layer_forwards: int = 0
    encoder_tokens: int = 0
    decoder_tokens: int = 0

    def reset(self) -> None:
        for name in COUNT_FIELDS:
            setattr(self, name, 0)

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in COUNT_FIELDS}


class E2D2(nn.Module):
    """Encoder-decoder block diffusion.

    With ``encoder_layers=0`` the encoder is the identity on token embeddings and
    the model is the decoder-only block-diffusion baseline.
    """

    def __init__(self, config: E2D2Config):
        super().__init__()
        self.config = config
        # Always built, so that ``model.encoder.blocks == []`` is the observable
        # statement of "this model has no encoder stack".  A zero-layer stack is a
        # pure identity and is never invoked by the decoder-only code path.
        self.encoder = Stack(
            config.stack_config(config.encoder_layers), with_head=False, use_pos=True
        )
        # The decoder keeps an absolute positional embedding.  It is not optional:
        # a fully masked block is a block of four *identical* inputs, and with an
        # all-to-all block-diagonal mask the four query positions would be
        # interchangeable, so the model could never learn which answer token belongs
        # at which offset.  Absolute positions are used in both paths and agree:
        # the training pass gives the noised sequence positions 0..L-1, and the
        # sampler gives the currently denoised block its absolute position range.
        self.decoder = Stack(
            config.stack_config(config.decoder_layers), with_head=True, use_pos=True
        )
        self.counts = CallCounts()
        self._mask_cache: dict[tuple, mx.array] = {}

    # -- masks ---------------------------------------------------------------

    def encoder_mask(self, n_blocks: int) -> mx.array:
        seq_len = n_blocks * self.config.block_size
        key = ("enc", seq_len)
        if key not in self._mask_cache:
            self._mask_cache[key] = masks.encoder_block_causal(
                seq_len, self.config.block_size
            )
        return self._mask_cache[key]

    def full_decoder_mask(self) -> mx.array:
        """``(2L, 2L)`` mask for the whole-sequence training pass.

        Both the encoder-decoder model and its decoder-only limit use the Figure 2
        decoder mask.  Clean queries are block-causal over the clean half and
        noised queries are block-diagonal plus cross-attention to *earlier* clean
        blocks.  That mask means a clean position never sees a later clean
        position, so when the encoder is the identity the whole network is exactly
        a decoder-only block-diffusion model: one stack, input
        ``[x^{1:L} ; z_t^{1:L}]``, block-causal context and block-diagonal denoising.
        The only difference between the two models is then whether the clean half
        is transformed by an encoder stack.

        Whether the clean context should be block-causal or bidirectional is not
        stated in the paper for the baseline, so this picks block-causal; see
        README "Where this may differ from the paper".
        """
        key = ("dec_full", self.config.seq_len)
        if key not in self._mask_cache:
            self._mask_cache[key] = masks.e2d2_decoder_mask(
                self.config.seq_len, self.config.block_size
            )
        return self._mask_cache[key]

    def decode_mask(self, n_prefix_blocks: int) -> mx.array:
        """``(2P, 2P)`` mask for one sampling step, ``P = 2 * prefix * block_size``."""
        key = ("dec_step", n_prefix_blocks)
        if key not in self._mask_cache:
            self._mask_cache[key] = masks.block_decode_mask(
                n_prefix_blocks, self.config.block_size
            )
        return self._mask_cache[key]

    # -- counting ------------------------------------------------------------

    def reset_counts(self) -> None:
        self.counts.reset()

    def _enc(self, clean_tokens: mx.array) -> mx.array:
        cfg = self.config
        if cfg.is_decoder_only:
            raise RuntimeError(
                "a decoder-only model has no encoder stack to invoke; its clean "
                "context is the token embedding (see clean_representation)"
            )
        n_blocks = clean_tokens.shape[1] // cfg.block_size
        hidden = self.encoder.embed_tokens(clean_tokens)
        hidden = self.encoder(hidden, self.encoder_mask(n_blocks))
        self.counts.encoder_calls += 1
        self.counts.encoder_layer_forwards += cfg.encoder_layers * n_blocks
        self.counts.encoder_tokens += clean_tokens.shape[-1]
        return hidden

    def _dec(self, x: mx.array, mask: mx.array, n_noisy: int, n_blocks: int) -> mx.array:
        hidden = self.decoder(x, mask)
        logits = self.decoder.logits(hidden)
        self.counts.decoder_calls += 1
        self.counts.decoder_layer_forwards += self.config.decoder_layers * n_blocks
        self.counts.decoder_tokens += n_noisy
        return logits

    # -- training ------------------------------------------------------------

    def clean_representation(self, clean_tokens: mx.array) -> mx.array:
        """The clean half fed to the decoder.

        For the encoder-decoder model this is ``Encoder(x)`` (one invocation); for
        the decoder-only limit it is the token embedding itself, so the single
        network sees ``[x^{1:L} ; z_t^{1:L}]``.
        """
        if self.config.is_decoder_only:
            return self.decoder.embed_tokens(clean_tokens)
        return self._enc(clean_tokens)

    def training_loss(
        self,
        clean_tokens: mx.array,
        key,
        num_steps: int = 32,
        context_blocks: int = 0,
        weighting: str = "nelbo",
    ):
        """Block-diffusion NELBO with an independent noise level per block.

        Follows Algorithm 2: one encoder pass over the entire clean sequence and
        one decoder pass over the entire noised sequence, with the Figure 2 masks
        making that single pass model the posterior of every block.

        ``weighting`` selects between the paper's weighted diffusion loss (default,
        Eq. 4) and the unweighted masked-token mean; see
        :func:`~e2d2.diffusion.nelbo_loss_per_block` for why they are not the same
        objective at toy scale.

        ``context_blocks`` keeps the first ``context_blocks`` blocks at ``t = 0``.
        That mirrors the sampling setup for a conditional task, where the prompt is
        given clean, and it avoids spending loss on the leading block, whose noised
        tokens have no earlier context to condition on (under the Figure 2 mask a
        noised token only sees *earlier* clean blocks, so block 0 is unconditioned).
        The default 0 is the paper's unconditional block-diffusion objective.
        """
        cfg = self.config
        bsz, seq_len = clean_tokens.shape
        if seq_len != cfg.seq_len:
            raise ValueError(f"expected sequences of length {cfg.seq_len}")
        n_blocks = seq_len // cfg.block_size
        if not 0 <= context_blocks <= n_blocks:
            raise ValueError("context_blocks must be within [0, n_blocks]")

        key_t, key_q = mx.random.split(key, 2)
        t = sample_train_times(num_steps, bsz * n_blocks, key_t).reshape(bsz, n_blocks)
        if context_blocks:
            t = mx.concatenate(
                [
                    mx.zeros((bsz, context_blocks), dtype=mx.float32),
                    t[:, context_blocks:],
                ],
                axis=1,
            )
        noisy = forward_corrupt_per_block(
            clean_tokens, t, cfg.block_size, cfg.mask_id, key_q
        )
        h = self.clean_representation(clean_tokens)
        z = self.decoder.embed_tokens(noisy)
        x = mx.concatenate([h, z], axis=1)
        logits = self._dec(x, self.full_decoder_mask(), n_noisy=seq_len, n_blocks=n_blocks)
        # Only the second half has a denoising target; the first half is the
        # clean-token representation itself.
        logits = logits[:, seq_len:, :]
        return nelbo_loss_per_block(
            logits,
            clean_tokens,
            noisy,
            t,
            cfg.block_size,
            DEFAULT_SCHEDULE,
            weighting=weighting,
        )

    def denoise_logits_full(
        self, noisy_tokens: mx.array, clean_override: mx.array | None = None
    ) -> mx.array:
        """Training-path logits for the answer half, with the noise supplied.

        The counterpart of :meth:`training_loss` where the caller controls the
        corruption, used by the NELBO estimate in ``metrics.py``.  Without
        ``clean_override`` the clean half is the noisy sequence itself, which is
        what the encoder is given during sampling.

        ``clean_override`` exists so tests can feed a *different* clean sequence
        while holding the noised sequence fixed; the mask must make the difference
        invisible to the logits of the block being denoised.
        """
        cfg = self.config
        seq_len = noisy_tokens.shape[1]
        if seq_len != cfg.seq_len:
            raise ValueError(f"expected sequences of length {cfg.seq_len}")
        clean = noisy_tokens if clean_override is None else clean_override
        if clean.shape != noisy_tokens.shape:
            raise ValueError("clean_override must have the same shape as noisy_tokens")
        h = self.clean_representation(clean)
        z = self.decoder.embed_tokens(noisy_tokens)
        x = mx.concatenate([h, z], axis=1)
        logits = self._dec(
            x,
            self.full_decoder_mask(),
            n_noisy=seq_len,
            n_blocks=seq_len // cfg.block_size,
        )
        return logits[:, seq_len:, :]

    # -- incremental (sampling) path ----------------------------------------

    def encode_prefix(self, clean_prefix: mx.array) -> mx.array:
        """Encode the accumulated clean tokens.

        One encoder invocation for the encoder-decoder model.  The decoder-only
        limit has no encoder, so this is only an embedding lookup and the counter
        stays at zero -- which is exactly the accounting difference the paper's
        efficiency argument is about.
        """
        return self.clean_representation(clean_prefix)

    def denoise_step(self, noisy_block: mx.array, h_prefix: mx.array) -> mx.array:
        """One decoder invocation over ``[h^{1:P} ; z^b]``.

        ``h_prefix`` has length ``P`` (clean tokens encoded so far) and
        ``noisy_block`` has length ``block_size``.  The position embeddings of the
        noised tokens continue after the clean prefix, which reproduces the
        position layout the model was trained with.
        """
        cfg = self.config
        n_prefix = h_prefix.shape[1]
        # The noised block gets its *absolute* offsets in the generated sequence,
        # which is exactly the position range those tokens have in the training
        # pass (where the whole noised sequence is embedded at 0..L-1).  Getting
        # this wrong is silent: shapes and masks stay valid while the block's
        # positional embedding is shifted by ``n_prefix``.
        z = self.decoder.embed_tokens(
            noisy_block, positions=mx.arange(n_prefix, n_prefix + cfg.block_size)
        )
        x = mx.concatenate([h_prefix, z], axis=1)
        n_prefix_blocks = n_prefix // cfg.block_size
        logits = self._dec(x, self.decode_mask(n_prefix_blocks), n_noisy=cfg.block_size, n_blocks=1)
        return logits[:, n_prefix:, :]

    # -- reporting -----------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(v.size for _, v in mlx.utils.tree_flatten(self.parameters()))

    def full_stack_layers(self) -> int:
        """Layer-forwards in one pass of the whole network (encoder + decoder)."""
        return self.config.encoder_layers + self.config.decoder_layers
