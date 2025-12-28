# e2d2-diffusion-mlx

A small independent MLX implementation of *Encoder-Decoder Diffusion Language Models for
Efficient Training and Inference* (arXiv:2510.22852), plus a toy block-diffusion language
model trained on a synthetic sequence task to try the mechanism out. It runs on Apple
silicon through MLX.

The paper's observation is that a discrete diffusion model does two different jobs:
it *represents clean tokens* and it *denoises corrupted ones*. Decoder-only diffusion
models have to do both inside one network, so every denoising step pays for the whole
model. E2D2 splits the work: an encoder consumes the clean tokens (the prompt and the
blocks decoded so far) and a lighter decoder is applied repeatedly to the noised block,
attending to the encoder's representation. The encoder is invoked once per generated
block instead of once per denoising step, so the expensive module is amortised over the
cheap iterations. The same split also makes block-diffusion training cheaper, because the
encoder sees the clean sequence and the decoder sees the noised sequence in one pass each,
rather than one network seeing both concatenated (the paper's Table 1: 2x fewer training
FLOPs than BD3LM at equal depth).

**This is not a reproduction.** It is about 330k parameters on a synthetic task that takes
seconds to learn, it measures nothing about the paper's Qwen3-based models or its
summarization / translation / GSM8K / OpenWebText results, and none of its numbers are
comparable to the paper's. What it does do is implement the mechanism end to end, check
the invariants that carry the paper's argument, count the encoder and decoder invocations
directly, and report what happens at toy scale. The structural half of the paper's argument
(the encoder runs once per block, not once per step) reproduces exactly; the quality and
throughput half does not, and the README says so where it happens.

## What is in here

**Training-time corruption** (`e2d2/diffusion.py`). Absorbing-state (masked) diffusion with
the linear schedule `alpha_t = 1 - t`, so a token is replaced by `[MASK]` with probability
`t`. `forward_corrupt_per_block` draws an independent `t` for every block, which is what
lets a single pass model every block's posterior at once. The loss is the paper's Eq. 4:
a cross-entropy at corrupted positions weighted by `alpha'_t / (1 - alpha_t) = 1 / (1 - t)`.
The module exposes two aggregations and the difference is not cosmetic:

- `weighting="nelbo"` (the paper's objective) divides the weighted sum by the block size, a
  constant, so the importance weight survives. This is the default.
- `weighting="uniform"` divides by the number of masked positions instead, which cancels
  `w(t)` exactly and leaves a plain masked-token mean. That is *not* a reparameterisation of
  the NELBO; it is what the authors' reference implementation uses during training (its
  `_compute_loss` takes the unweighted sum over masked tokens when `self.training`). The
  experiment reports which one it used, and both are tested.

**Masks** (`e2d2/masks.py`). The paper's Figure 2 as boolean keep-masks.
`encoder_block_causal` is the encoder's block-causal mask; a test checks its non-zero count
against the paper's reported `(L^2 + L*S)/2`. `e2d2_decoder_mask` is the four-quadrant mask
for the concatenated `[h ; z_t]` decoder input: hidden-state queries are block-causal,
noised queries are block-diagonal plus cross-attention to *earlier* clean blocks only.
The quadrant that matters is the one that is zero — a noised query never attends to the
hidden state of its own clean block, because that representation is built from the tokens
being predicted. `block_decode_mask` builds the smaller mask the incremental sampler needs
and a test checks it is exactly the relevant sub-block of the training mask.

**Model** (`e2d2/model.py`, `e2d2/layers.py`). An encoder stack and a decoder stack. The
architecture is the paper's "last hidden state" design decision: the decoder's input is
`[h^{1:L} ; z_t^{1:L}]` and every decoder layer runs one attention over that concatenation,
which stands in for the paper's fused self/cross-attention kernel. Training follows
Algorithm 2 (one encoder pass over the clean sequence, one decoder pass over the noised
sequence). The decoder keeps an absolute positional embedding even though the encoder
already has one, for a reason the experiment forced: a fully masked block is four
*identical* inputs, so with an all-to-all block-diagonal mask and no positions the four
query slots are interchangeable and the model cannot learn which answer token belongs
where. A test pins that failure mode by turning positions off and checking the four query
rows become identical. In a single-example overfit with that ablation the loss sat at
`log 4` — the model could only learn the *set* of answer tokens, not their order — which is
where I first located this bug.

**Sampling** (`e2d2/sampling.py`). Algorithm 1 as code: encode the prompt, then for each
block run `T` decoder invocations and hand the finished block back to the encoder. There is
no encoder call after the final block, since nothing is left to condition, so the count is
exactly one encoder invocation per generated block. The paper leaves its `Sample(...)` step
abstract; the rule here is the standard confidence-based one — after step `i` a block has
`ceil(i*S/T)` clean positions, following the same linear `1 - t` schedule as the training
corruption, positions are never re-masked ("carry-over unmasking", Appendix A.3), and the
`[MASK]` logit is excluded from the argmax because the paper's zero-masking-probability
constraint says the model may never predict mask.

**Task and metrics** (`e2d2/task.py`, `e2d2/metrics.py`). The task is: emit the prompt block
reversed. `prompt = [BOS, d1, d2, d3]`, `answer = [d3, d2, d1, BOS]`. The answer is a
deterministic function of the prompt, so the context has to reach the answer block, and it
is an *ordering* of the context tokens rather than a copy at a fixed offset — a test checks
that no fixed prompt position holds the answer for the same position. Prompt space is
`M ** 3`, and train and held-out sets are drawn from a permutation of the whole space, so
they are exactly disjoint. Metrics are exact match and token accuracy over the answer block,
plus an importance-weighted answer-position cross-entropy averaged over a `t` grid as a
likelihood surrogate.

## Checks

`pytest` runs 104 tests. The ones that matter most:

- **The two forward paths agree exactly.** Training scores the whole noised sequence at
  once (positions `0..L-1`); the sampler scores one block at a time. For the answer block
  those must produce identical logits. `denoise_logits_full` and `denoise_step` are compared
  for three encoder/decoder splits and differ by `< 1e-5` (`tests/test_path_consistency.py`).
  This is the invariant that catches the whole class of bug this repo actually hit.
- **The encoder is invoked once per generated block, whatever `T` is.** Counted on the model
  object rather than inferred from elapsed time, for `T` in `{1, 2, 4, 8, 16}` and for one,
  two and three generated blocks (`tests/test_sampling.py`).
- **`T=1` equals one-shot prediction.** Sampling with a single step equals a separately
  written `one_shot_predict` that never updates the block in between, and no `[MASK]`
  survives (`tests/test_sampling.py`). This is the "one step is one-shot prediction from the
  fully masked sequence" limit, checked rather than asserted.
- **The masking process and the reverse process are the same schedule.**
  `clean_counts(S, T)[i-1] == ceil(i*S/T)` is checked against `ceil((1-t)*S)` for the
  `t = 1 - i/T` the training corruption uses, and a step-by-step test checks the unmasking
  is monotone (once a position is clean it keeps its value) and ends fully clean.
- **The decoder-only limit is exact.** With `encoder_layers=0` the clean half of the decoder
  input is the token embedding, so the whole model is one stack over `[x^{1:L} ; z_t^{1:L}]`
  under the Figure 2 decoder mask. `tests/test_model.py` rebuilds that graph by hand from the
  same parameter arrays and checks the logits agree to `1e-5`, checks that a real encoder
  *does* change the clean representation (so the limit is a restriction, not an accident),
  and checks both models use the same decoder mask.
- **No leakage across blocks.** Changing the clean content of a fully noised block leaves
  that block's logits unchanged, and perturbing later clean blocks leaves earlier encoder
  hidden states unchanged (encoder block-causality). These guard the design choice that a
  noised block must not read the hidden state of its own clean block; the mask was right
  from the start, but nothing else in the suite would have caught it if it had not been.
- **The mask token is explicit.** `E2D2Config.mask_id` is a validated field used at every
  corruption site, with a test that a custom id actually appears in the noised sequence
  (`tests/test_path_consistency.py`). The vocabulary put `[MASK]` last because the paper's
  Appendix A.1 fixes the absorbing state at `argmax_i m_i = V`; that is now the only place
  the index is derived.
- **Hand-computable cases.** The corruption limits (`t=0` clean, `t=1` all mask), the
  realised mask rate, the loss on uniform logits (`log V`), the loss on an uncorrupted
  sequence (`0`), the per-block objective against the per-sequence one, the mask non-zero
  counts, and attention rows summing to one over exactly the allowed keys.
- **Gradients.** Every parameter receives a finite gradient through the full training loss,
  and the encoder stack is not bypassed.
- **The task.** The answer is exactly the reversed prompt block, the mask id collides with no
  real token, `all_prompts` enumerates every triple once, and the train/held-out split is
  disjoint.

## Experiment

`scripts/e2d2_experiment.py` trains three architectures at (approximately) matched parameter
budget on the reverse-the-prompt task and measures quality against inference cost. Script:
5000 AdamW steps at lr 1e-3, batch 64, three seeds, `M = 16` (a 4096-prompt space), 1024
training and 1024 held-out prompts, `context_blocks=1` so the prompt block is kept clean
(mirroring sampling) and the loss weighting is the reference implementation's unweighted
mean. Roughly 30 s per model per seed on this machine.

| configuration | layers (enc/dec) | parameters | held-out exact match | token accuracy | NELBO surrogate | train s/seed |
|---|---|---:|---:|---:|---:|---:|
| `e2d2-4/1` | 4 / 1 | 332800 | 0.9756 +- 0.0345 | 0.9939 +- 0.0086 | 0.000044 +- 0.000002 | 34.1 |
| `e2d2-1/3` | 1 / 3 | 267136 | 0.9622 +- 0.0534 | 0.9906 +- 0.0134 | 0.000035 +- 0.000001 | 29.1 |
| `decoder-only-4` | 0 / 4 | 267136 | 0.9987 +- 0.0018 | 0.9997 +- 0.0005 | 0.000033 +- 0.000001 | 30.5 |

Mean +- population std over the three seeds. The final training loss averaged 0.000085
(`e2d2-4/1`), 0.000069 (`e2d2-1/3`) and 0.000065 (`decoder-only-4`).

Quality above is measured at `T = 8` denoising steps. The likelihood surrogate (weighted
answer-position cross-entropy over a 16-point `t` grid, so lower is better) is essentially
zero for all three, and so is the training loss: the task is small enough that every
architecture fits it, and neither the surrogate nor the loss separates them.

Quality against denoising steps, mean exact match over the three seeds:

| configuration | T=1 | T=2 | T=4 | T=8 |
|---|---:|---:|---:|---:|
| `e2d2-4/1` | 1.0000 | 0.9990 | 0.9951 | 0.9756 |
| `e2d2-1/3` | 1.0000 | 1.0000 | 0.9967 | 0.9622 |
| `decoder-only-4` | 1.0000 | 1.0000 | 1.0000 | 0.9987 |

Inference cost per generated sequence, one generated block, 64 sequences, best of 10 runs
after a warm-up. The invocation counts are exact; the wall-clock numbers are from this
laptop through MLX and are not comparable to the paper's H100 throughput.

| configuration | T | encoder calls | decoder calls | layer-forwards | full-network calls for the baseline | wall clock |
|---|---:|---:|---:|---:|---:|---:|
| `e2d2-4/1` | 1 | 1 | 1 | 5 | 1 | 0.0122 ms |
| `e2d2-4/1` | 2 | 1 | 2 | 6 | 2 | 0.9375 ms |
| `e2d2-4/1` | 4 | 1 | 4 | 8 | 4 | 2.6055 ms |
| `e2d2-4/1` | 8 | 1 | 8 | 12 | 8 | 3.2284 ms |
| `e2d2-1/3` | 8 | 1 | 8 | 25 | 8 | 3.1356 ms |
| `decoder-only-4` | 1 | 0 | 1 | 4 | 1 | 0.0067 ms |
| `decoder-only-4` | 8 | 0 | 8 | 32 | 8 | 3.1120 ms |

The headline measurement, and the whole point of the architecture: **the encoder is invoked
once per generated sequence no matter how many denoising steps are used.** The scaling run
(one, two and three generated blocks, `T` in `{1, 2, 4, 8}`) gives encoder calls of exactly
1, 2 and 3 — i.e. one per generated block — while the decoder is called `blocks * T` times
and a decoder-only model must run its whole network `blocks * T` times. The ratio of
baseline full-network invocations to encoder invocations is therefore exactly `T`:

| generated blocks | T | encoder calls | decoder calls | baseline network calls | baseline / encoder |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 | 1 | 1.0 |
| 1 | 8 | 1 | 8 | 8 | 8.0 |
| 2 | 4 | 2 | 8 | 8 | 4.0 |
| 3 | 8 | 3 | 24 | 24 | 8.0 |

At `T = 8` with one generated block, `e2d2-4/1` performs 12 layer-forwards against the
baseline's 32 (ratio 0.3750) and calls the encoder 0.1250 times per baseline network call.

What the numbers mean:

- **The structural claim holds and is exact.** The encoder runs once per block regardless of
  the step count, and the decoder carries the iteration. This is counted, not inferred.
- **Quality is matched, not beaten, at this scale.** All three architectures reach ~0.96-1.00
  held-out exact match, so the task is solved and the quality column mostly says "this
  experiment cannot separate them". At `T=8` the baseline is marginally ahead (0.9987 against
  0.9756 and 0.9622) and the gap is at or within the seed spread (std 0.0345 and 0.0534).
- **The layer-forward saving does not become a wall-clock saving here.** `e2d2-4/1` needs
  1.0374x the baseline's wall clock at `T=8`: 3.2284 ms against 3.1120 ms, which is to say the
  same speed. Repeated runs of this same script on this machine moved that ratio around by
  tens of percent and to both sides of 1.0, so I read it as "no measurable difference" rather
  than as a small win or a small loss. The
  reason is the per-call overhead — `e2d2-4/1` makes 9 forward calls (1 encoder + 8 decoder)
  against the baseline's 8, and at this size each call costs roughly the same fixed Python and
  MLX overhead as the layer arithmetic inside it, so saving 20 of 32 layer-forwards buys
  nothing. The layer-forward saving is real and exact; I am not claiming a speedup.
- **More denoising steps slightly hurt the encoder-decoder variants.** `e2d2-1/3` falls from
  1.0000 exact match at `T=1,2` to 0.9622 at `T=8`, while the baseline stays at 0.9987. With no
  re-masking, every extra step is another chance to overwrite a position the model had
  already right, and the smaller decoder appears to be less reliable at the later,
  low-noise steps. The paper reports all its results at `T = L` steps and does not publish a
  quality-versus-`T` table, so there is nothing to compare this against; treat it as a
  property of this toy rather than a finding about E2D2.
- Only invocation counts, layer-forwards and wall clock are measured. There are no fused
  kernels, no KV cache, and no throughput comparison in tokens/sec.

## Where this may differ from the paper

- **The task is not the paper's.** The paper trains domain-specific models on CNN/DailyMail,
  WMT14 de-en, GSM8K and OpenWebText. Here it is a synthetic reversal task with an 18-token
  vocabulary and 8-token sequences. It is small enough to solve exactly, which is what makes
  the cost comparison clean, but it exercises almost nothing the paper's benchmarks do.
- **The paper's numbers are not reproduced anywhere here.** No Qwen3, no 7-8B model, no
  ROUGE/BLEU/pass@1, no `T = L` sampling protocol, no 3x-faster-than-MDLM or ~75%-throughput
  claims, no H100 tokens/sec, no training-FLOPs table.
- **Scale.** ~330k parameters and 8-token sequences against hundreds of millions of tokens
  of training data in the paper. Nothing here says anything about whether E2D2 helps at scale.
- **The decoder's own-clean-block quadrant.** The paper's Figure 2 says a noised token
  cross-attends to *previous* clean blocks, and the caption's "offset block-causal" label is
  a block-causal (not strictly-lower) relation. I implemented the strictly-earlier reading, so
  a noised block never sees the hidden state of its own clean block. This is the reading that
  makes training and sampling consistent (the sampler never has the current block's clean
  tokens) and I consider it the safer one, but the authors' reference implementation's
  `_decoder_block_mask` uses `block_q >= block_kv` for the cross-attention quadrant, which
  permits same-block access. If that is not a bug, my mask is stricter than theirs.
- **Fused attention is not implemented.** The paper's design decision is a single fused
  attention call over encoder outputs and decoder hidden states; here it is one attention over
  the concatenated `[h ; z_t]`, which is the same computation without the kernel-fusion
  motivation.
- **Only the "last hidden state" connection, not "shared KV cache".** The paper's second
  variant reuses encoder KVs per layer; it exists to fine-tune decoder-only checkpoints and
  needs matched layer pairings, so the toy keeps untied weights and the simpler connection.
- **No KV cache.** The sampler re-runs the encoder over the accumulated clean prefix on every
  block, which is O(prefix) per block rather than O(block). This does not change the encoder
  *invocation* count — which is the thing being measured — but it does make each encoder call
  more expensive than the paper's cached version, and it is a second reason (besides per-call
  overhead) not to expect a wall-clock win from the layer-forward saving.
- **The unmasking rule is mine.** The paper specifies `T` decoder applications and an abstract
  `Sample(...)`; the confidence-based schedule, the `ceil(i*S/T)` clean target and the
  treatment of `[MASK]` as unpredictable are all choices here.
- **The loss weighting used in the experiment is `"uniform"`, not the paper's Eq. 4.** I ran
  both; `"nelbo"` did not change the outcome at this scale, and `"uniform"` matches the
  reference implementation's training path, so that is what the reported numbers use. The
  default in the library is `"nelbo"`.
- **`context_blocks=1` keeps the prompt clean during training.** The paper's Algorithm 2
  noises the whole sequence. I keep the first block clean because the sampler is given a clean
  prompt, and because under the Figure 2 mask a first-block noised token has no earlier
  context to condition on at all, so its loss is unconditioned. Set `context_blocks=0` for the
  paper's unconditional objective.
- **Learned absolute positional embeddings, not RoPE.** The paper's models use Qwen3 with RoPE
  and QK-norm.
- **lr 1e-3 is load-bearing and was found by sweep.** At lr 3e-3 or 1e-2 the encoder-decoder
  models diverge (training loss above `log V`) while the decoder-only baseline still trains.
  I do not have an explanation beyond "the cross-attention path is less stable"; the paper
  does not report this because it fine-tunes from pretrained weights.
- **The task was changed mid-build.** The first version asked for `[a, b, (a+b) % M, (a-b) % M]`
  from a two-digit prompt. Every architecture memorised the training prompts (training loss
  0.0) while held-out exact match stayed at 0.0, because the space was 64 prompts against
  270k parameters. The reported task replaces it. `e2d2/task.py` documents this.

## Running it

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python -m pytest
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/e2d2_experiment.py
```

The experiment takes about ten minutes and rewrites `results/experiment.json` (machine
readable) and `results/run_log.txt` (the full log, which is what the tables above quote).
Useful flags:
`--seeds`, `--steps`, `--lr`, `--modulus`, `--n-train`, `--n-held-out`, `--sample-steps`,
`--weighting`, `--context-blocks`, `--cost-prompt-blocks`.
