#!/usr/bin/env python
"""Toy E2D2 vs. decoder-only block-diffusion experiment.

Trains three models at (approximately) the same parameter budget on the
reverse-the-prompt task in ``e2d2/task.py`` and reports quality against inference
cost:

    e2d2-4/1        encoder 4 layers, decoder 1 layer, identity encoder off
    e2d2-1/3        encoder 1 layer,  decoder 3 layers
    decoder-only-4  encoder_layers=0, decoder 4 layers (the BD3LM-style baseline)

Cost is reported two ways.  The structural counts (encoder invocations, decoder
invocations, layer-forwards) are exact and machine-independent: E2D2 calls the
encoder once per generated block, its decoder once per denoising step per block,
and the baseline runs its single network once per denoising step per block.  The
wall-clock numbers are measured on this laptop through MLX and are quoted as such
-- they include Python overhead per call and are not a speedup claim comparable to
the paper's throughput table.

Writes ``results/experiment.json`` (machine readable), ``results/run_log.txt``
(the full log) and ``results/experiment.txt`` (the summary table).

Run with::

    /Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/e2d2_experiment.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from e2d2.metrics import exact_match, nelbo_estimate, token_accuracy  # noqa: E402
from e2d2.model import E2D2, E2D2Config  # noqa: E402
from e2d2.sampling import SampleConfig, sample_blockwise  # noqa: E402
from e2d2.task import DIGIT_OFFSET, TaskSpec, split_prompts  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CONFIGURATIONS = {
    "e2d2-4/1": dict(encoder_layers=4, decoder_layers=1),
    "e2d2-1/3": dict(encoder_layers=1, decoder_layers=3),
    "decoder-only-4": dict(encoder_layers=0, decoder_layers=4),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modulus", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-held-out", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-t-grid", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--sample-seqs", type=int, default=64)
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--nelbo-grid", type=int, default=16)
    parser.add_argument("--nelbo-samples", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--weighting", choices=["uniform", "nelbo"], default="uniform")
    parser.add_argument("--context-blocks", type=int, default=1)
    parser.add_argument(
        "--cost-prompt-blocks",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="prompt sizes (in blocks) used for the encoder-call scaling measurement",
    )
    parser.add_argument("--out-prefix", type=str, default="experiment")
    return parser.parse_args(argv)


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.lines: list[str] = []

    def __call__(self, message: str = "") -> None:
        print(message, flush=True)
        self.lines.append(message)

    def dump(self) -> str:
        text = "\n".join(self.lines) + "\n"
        self.path.write_text(text)
        return text


def build_config(spec: TaskSpec, args, **arch) -> E2D2Config:
    return E2D2Config(
        vocab_size=spec.vocab_size,
        block_size=spec.block_size,
        n_blocks=spec.n_blocks,
        d_model=args.d_model,
        n_heads=args.n_heads,
        # Always the task's own mask id, never an implicit vocab_size - 1.
        mask_id=spec.mask_id,
        **arch,
    )


def sample_batch(
    key, train_prompt: mx.array, train_full: mx.array, batch_size: int
) -> tuple[mx.array, mx.array]:
    """Draw a random minibatch of whole examples from the training split."""
    idx = mx.random.randint(0, train_full.shape[0], (batch_size,), key=key)
    return train_prompt[idx], train_full[idx]


def train_one(config: E2D2Config, args, seed: int, log: Logger, label: str) -> dict:
    """Train one model and return the trained model plus training diagnostics."""
    mx.random.seed(seed)
    model = E2D2(config)
    n_params = model.num_parameters()
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(
        model,
        lambda m, t, k: m.training_loss(
            t,
            k,
            args.num_t_grid,
            context_blocks=args.context_blocks,
            weighting=args.weighting,
        ),
    )
    (train_prompt, train_full), (held_prompt, held_full) = split_prompts(
        TaskSpec(modulus=args.modulus, block_size=config.block_size, prompt_blocks=1),
        args.n_train,
        args.n_held_out,
        seed=seed,
    )

    key = mx.random.key(seed * 1000 + 1)
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        key, batch_key, loss_key = mx.random.split(key, 3)
        _, batch = sample_batch(batch_key, train_prompt, train_full, args.batch_size)
        loss, grads = loss_and_grad(model, batch, loss_key)
        optimizer.update(model, grads)
        # Evaluate the *current* parameters and optimiser state.  Holding a stale
        # reference (captured before the update) leaves every step's graph alive,
        # which grows without bound and eventually gets the process killed.
        mx.eval(model.parameters(), optimizer.state)
        value = float(loss.item())
        losses.append(value)
        if step % args.log_every == 0 or step == 1:
            log(
                f"    [{label}] step {step:5d}/{args.steps}  "
                f"loss {value:.4f}  mean(last 100) {statistics.fmean(losses[-100:]):.4f}"
            )
    train_seconds = time.perf_counter() - started

    return {
        "model": model,
        "config": config,
        "n_params": n_params,
        "losses": losses,
        "train_seconds": train_seconds,
        "train_prompt": train_prompt,
        "train_full": train_full,
        "held_prompt": held_prompt,
        "held_full": held_full,
    }


def evaluate_quality(model: E2D2, spec: TaskSpec, args, run: dict) -> dict:
    """Exact match, token accuracy and the likelihood surrogate on held-out data."""
    held_full = run["held_full"]
    prompt_len = spec.prompt_len
    answer = held_full[:, prompt_len:]

    predictions = {}
    for steps in args.sample_steps:
        result = sample_blockwise(
            model,
            run["held_prompt"],
            SampleConfig(
                num_steps=steps,
                block_size=spec.block_size,
                n_blocks=spec.n_answer_blocks,
                mask_id=spec.mask_id,
            ),
        )
        predicted = result.tokens
        predictions[str(steps)] = {
            "exact_match": exact_match(predicted, answer),
            "token_accuracy": token_accuracy(predicted, answer),
            "sample_calls": result.counts,
        }

    value = nelbo_estimate(
        model,
        held_full,
        prompt_len=prompt_len,
        grid_steps=args.nelbo_grid,
        n_samples=args.nelbo_samples,
        seed=0,
    )
    return {"predictions": predictions, "nelbo_surrogate": value}


def measure_calls(model: E2D2, spec: TaskSpec, steps: int, n_sequences: int, repeats: int) -> dict:
    """Structural call counts and wall-clock time for one sampling setting."""
    prompt = mx.zeros((n_sequences, spec.prompt_len), dtype=mx.int32)
    config = SampleConfig(
        num_steps=steps,
        block_size=spec.block_size,
        n_blocks=spec.n_answer_blocks,
        mask_id=spec.mask_id,
    )
    # one warm-up call so the timing is not dominated by first-call graph setup
    sample_blockwise(model, prompt, config)
    timings = []
    counts = None
    for _ in range(repeats):
        result = sample_blockwise(model, prompt, config)
        timings.append(result.wall_seconds)
        counts = result.counts
    return {
        "num_steps": steps,
        "n_sequences": n_sequences,
        "encoder_calls_per_sequence": counts["encoder_calls"],
        "decoder_calls_per_sequence": counts["decoder_calls"],
        "encoder_layer_forwards_per_sequence": counts["encoder_layer_forwards"],
        "decoder_layer_forwards_per_sequence": counts["decoder_layer_forwards"],
        "total_layer_forwards_per_sequence": counts["encoder_layer_forwards"]
        + counts["decoder_layer_forwards"],
        "wall_seconds_per_sequence_best": min(timings) / n_sequences,
        "wall_seconds_per_sequence_median": statistics.median(timings) / n_sequences,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = Logger(RESULTS_DIR / "run_log.txt")

    spec = TaskSpec(modulus=args.modulus, block_size=4, prompt_blocks=1)
    log("E2D2 toy experiment: encoder-decoder block diffusion vs decoder-only block diffusion")
    log("Paper: arXiv:2510.22852 (Encoder-Decoder Diffusion Language Models for Efficient")
    log("       Training and Inference).  This is an independent toy-scale implementation;")
    log("       nothing here is comparable to the paper's reported numbers.")
    log()
    log(f"host           : {platform.platform()} / {platform.machine()}")
    log(f"mlx            : {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    log(f"task           : [BOS, d1, d2, d3] then the same block reversed, M = {args.modulus}")
    log(f"prompt space   : {spec.prompt_space} distinct prompts")
    log(f"sequence       : {spec.n_blocks} blocks x {spec.block_size} tokens = {spec.seq_len}")
    log(f"vocabulary     : {spec.vocab_size} tokens ([BOS]=0, digits {DIGIT_OFFSET}..{spec.vocab_size - 2}, [MASK]={spec.mask_id} last)")
    log(
        f"train / held-out prompts: {args.n_train} / {args.n_held_out} "
        f"(disjoint, total space {spec.prompt_space})"
    )
    log(f"training       : {args.steps} AdamW steps, lr {args.lr}, batch {args.batch_size}, t grid {args.num_t_grid}")
    log(f"loss           : weighting={args.weighting}, context_blocks={args.context_blocks} (prompt block kept clean)")
    log(f"seeds          : {args.seeds}")
    log()

    runs: dict[str, list[dict]] = {}
    trained_models: dict[str, E2D2] = {}
    for label, arch in CONFIGURATIONS.items():
        config = build_config(spec, args, **arch)
        log(f"== {label} (encoder_layers={arch['encoder_layers']}, decoder_layers={arch['decoder_layers']}) ==")
        per_seed = []
        for seed in args.seeds:
            log(f"  seed {seed}: training")
            run = train_one(config, args, seed, log, label)
            quality = evaluate_quality(run["model"], spec, args, run)
            entry = {
                "seed": seed,
                "n_params": run["n_params"],
                "train_final_loss": statistics.fmean(run["losses"][-100:]),
                "train_seconds": run["train_seconds"],
                **quality,
            }
            per_seed.append(entry)
            trained_models.setdefault(label, run["model"])
            em = quality["predictions"][str(args.sample_steps[-1])]["exact_match"]
            log(
                f"    seed {seed}: params {run['n_params']}, "
                f"final train loss {entry['train_final_loss']:.6f}, "
                f"held-out exact match (T={args.sample_steps[-1]}) {em:.4f}, "
                f"NELBO surrogate {quality['nelbo_surrogate']:.6f}"
            )
            for steps in args.sample_steps:
                p = quality["predictions"][str(steps)]
                log(
                    f"      T={steps:>3}: exact match {p['exact_match']:.4f}  "
                    f"token accuracy {p['token_accuracy']:.4f}  "
                    f"encoder calls/seq {p['sample_calls']['encoder_calls']}  "
                    f"decoder calls/seq {p['sample_calls']['decoder_calls']}"
                )
        runs[label] = per_seed
        log()

    # ---- cost: structural counts for every step count, and wall clock ---------
    log("== inference cost ==")
    cost: dict[str, dict] = {}
    for label, arch in CONFIGURATIONS.items():
        model = trained_models[label]
        per_steps = {}
        for steps in args.sample_steps:
            per_steps[str(steps)] = measure_calls(
                model, spec, steps, args.sample_seqs, args.timing_repeats
            )
        cost[label] = {
            "n_params": model.num_parameters(),
            "full_stack_layers": model.full_stack_layers(),
            "per_steps": per_steps,
        }
        log(f"  {label}: params {model.num_parameters()}, full stack depth {model.full_stack_layers()}")
        for steps in args.sample_steps:
            row = per_steps[str(steps)]
            log(
                f"    T={steps:3d}: encoder calls/seq {row['encoder_calls_per_sequence']:3d}  "
                f"decoder calls/seq {row['decoder_calls_per_sequence']:4d}  "
                f"layer-forwards/seq {row['total_layer_forwards_per_sequence']:5d}  "
                f"best wall {row['wall_seconds_per_sequence_best'] * 1e3:.4f} ms/seq"
            )
    log()

    scaling = measure_scaling(args, log)

    summary = summarise(args, spec, runs, cost)
    summary["encoder_call_scaling"] = scaling
    _write_summary(log, summary)

    payload = {
        "paper": {
            "title": "Encoder-Decoder Diffusion Language Models for Efficient Training and Inference",
            "arxiv": "2510.22852",
        },
        "note": (
            "Independent toy-scale reimplementation. Numbers are not comparable to the "
            "paper's summarization/translation/GSM8K results or its H100 throughput."
        ),
        "args": vars(args),
        "task": {
            "modulus": args.modulus,
            "block_size": spec.block_size,
            "prompt_blocks": spec.prompt_blocks,
            "n_blocks": spec.n_blocks,
            "n_answer_blocks": spec.n_answer_blocks,
            "seq_len": spec.seq_len,
            "vocab_size": spec.vocab_size,
            "mask_id": spec.mask_id,
            "prompt_len": spec.prompt_len,
            "answer_len": spec.answer_len,
            "n_content": spec.n_content,
            "prompt_space": spec.prompt_space,
            "n_train": args.n_train,
            "n_held_out": args.n_held_out,
        },
        "runs": runs,
        "cost": cost,
        "summary": summary,
    }
    (RESULTS_DIR / f"{args.out_prefix}.json").write_text(json.dumps(payload, indent=2))
    log(f"wrote {RESULTS_DIR / (args.out_prefix + '.json')}")
    log(f"wrote {RESULTS_DIR / 'run_log.txt'}")
    # Flush last, so the written log contains the lines that name the files it wrote.
    log.dump()
    return 0


def measure_scaling(args, log: Logger) -> list[dict]:
    """Encoder/decoder invocation counts as the number of generated blocks grows.

    The counts are exact and independent of the weights: the encoder is invoked
    once per generated block and the decoder once per denoising step per block,
    while a decoder-only model runs its single network once per step per block.
    Freshly initialised models are used here because the *shape* of the computation
    - and therefore both the counts and the wall-clock cost - does not depend on
    the parameter values for a dense transformer with no early exit.
    """
    log("== encoder/decoder invocation scaling ==")
    log(
        "  fresh models; counts are structural (weight-independent), wall clock is "
        "this machine"
    )
    log(
        f"  {'prompt blocks':>13} {'T':>3} {'blocks gen':>10} {'enc calls':>10} "
        f"{'dec calls':>10} {'baseline calls':>15} {'base/enc':>9}"
    )
    rows = []
    for prompt_blocks in args.cost_prompt_blocks:
        spec = TaskSpec(
            modulus=args.modulus,
            block_size=4,
            prompt_blocks=prompt_blocks,
        )
        mx.random.seed(args.seeds[0])
        config = E2D2Config(
            vocab_size=spec.vocab_size,
            block_size=spec.block_size,
            n_blocks=spec.n_blocks,
            d_model=args.d_model,
            n_heads=args.n_heads,
            encoder_layers=3,
            decoder_layers=1,
            mask_id=spec.mask_id,
        )
        model = E2D2(config)
        prompt = mx.zeros((args.sample_seqs, spec.prompt_len), dtype=mx.int32)
        for steps in args.sample_steps:
            sample_config = SampleConfig(
                num_steps=steps,
                block_size=spec.block_size,
                n_blocks=spec.n_answer_blocks,
                mask_id=spec.mask_id,
            )
            sample_blockwise(model, prompt, sample_config)  # warm up
            timings = []
            counts = None
            for _ in range(args.timing_repeats):
                result = sample_blockwise(model, prompt, sample_config)
                timings.append(result.wall_seconds)
                counts = result.counts
            baseline_calls = counts["decoder_calls"]  # one whole network per step
            row = {
                "prompt_blocks": prompt_blocks,
                "n_blocks": spec.n_blocks,
                "generated_blocks": spec.n_answer_blocks,
                "num_steps": steps,
                "encoder_calls_per_sequence": counts["encoder_calls"],
                "decoder_calls_per_sequence": counts["decoder_calls"],
                "encoder_layer_forwards_per_sequence": counts["encoder_layer_forwards"],
                "decoder_layer_forwards_per_sequence": counts["decoder_layer_forwards"],
                "decoder_only_network_calls_per_sequence": baseline_calls,
                "wall_seconds_per_sequence_median": statistics.median(timings)
                / args.sample_seqs,
                "n_params": model.num_parameters(),
            }
            rows.append(row)
            log(
                f"  {prompt_blocks:>13} {steps:>3} {spec.n_answer_blocks:>10} "
                f"{counts['encoder_calls']:>10} {counts['decoder_calls']:>10} "
                f"{baseline_calls:>15} {baseline_calls / counts['encoder_calls']:>9.1f}"
            )
    log()
    return rows


def summarise(args, spec: TaskSpec, runs, cost) -> dict:
    last_steps = str(args.sample_steps[-1])
    table = {}
    for label, per_seed in runs.items():
        on_held = [e["predictions"][last_steps] for e in per_seed]
        table[label] = {
            "n_params_mean": statistics.fmean(e["n_params"] for e in per_seed),
            "train_final_loss_mean": statistics.fmean(e["train_final_loss"] for e in per_seed),
            "train_seconds_mean": statistics.fmean(e["train_seconds"] for e in per_seed),
            "train_loss_mean": statistics.fmean(e["train_final_loss"] for e in per_seed),
            "exact_match_mean": statistics.fmean(p["exact_match"] for p in on_held),
            "exact_match_std": statistics.pstdev(p["exact_match"] for p in on_held),
            "token_accuracy_mean": statistics.fmean(p["token_accuracy"] for p in on_held),
            "token_accuracy_std": statistics.pstdev(p["token_accuracy"] for p in on_held),
            "nelbo_surrogate_mean": statistics.fmean(e["nelbo_surrogate"] for e in per_seed),
            "nelbo_surrogate_std": statistics.pstdev(e["nelbo_surrogate"] for e in per_seed),
            "sample_steps": last_steps,
            "per_steps": {
                str(steps): {
                    "exact_match_mean": statistics.fmean(
                        e["predictions"][str(steps)]["exact_match"] for e in per_seed
                    ),
                    "exact_match_std": statistics.pstdev(
                        e["predictions"][str(steps)]["exact_match"] for e in per_seed
                    ),
                    "token_accuracy_mean": statistics.fmean(
                        e["predictions"][str(steps)]["token_accuracy"] for e in per_seed
                    ),
                    "token_accuracy_std": statistics.pstdev(
                        e["predictions"][str(steps)]["token_accuracy"] for e in per_seed
                    ),
                }
                for steps in args.sample_steps
            },
        }
    ratios = {}
    split = table["e2d2-4/1"]
    baseline = table["decoder-only-4"]
    ratios["exact_match_split_over_baseline"] = (
        split["exact_match_mean"] / baseline["exact_match_mean"] if baseline["exact_match_mean"] else None
    )
    baseline_cost = cost["decoder-only-4"]["per_steps"][last_steps]
    split_cost = cost["e2d2-4/1"]["per_steps"][last_steps]
    ratios["layer_forward_ratio_split_over_baseline"] = (
        split_cost["total_layer_forwards_per_sequence"]
        / baseline_cost["total_layer_forwards_per_sequence"]
    )
    ratios["wall_ratio_split_over_baseline"] = (
        split_cost["wall_seconds_per_sequence_best"]
        / baseline_cost["wall_seconds_per_sequence_best"]
    )
    ratios["encoder_call_ratio_split_over_baseline"] = (
        split_cost["encoder_calls_per_sequence"] / baseline_cost["decoder_calls_per_sequence"]
    )
    return {"quality": table, "cost_ratio": ratios}


def _write_summary(log: Logger, summary: dict) -> None:
    log("== summary ==")
    log(f"{'configuration':<16} {'params':>8} {'EM':>18} {'tok-acc':>18} {'NELBO-surr':>20}")
    for label, row in summary["quality"].items():
        log(
            f"{label:<16} {int(row['n_params_mean']):>8} "
            f"{row['exact_match_mean']:.4f}+-{row['exact_match_std']:.4f}".ljust(36)
            + f"{row['token_accuracy_mean']:.4f}+-{row['token_accuracy_std']:.4f}".rjust(18)
            + f"{row['nelbo_surrogate_mean']:.6f}+-{row['nelbo_surrogate_std']:.6f}".rjust(21)
        )
        log(
            f"    mean train loss {row['train_loss_mean']:.6f}  "
            f"mean train seconds/seed {row['train_seconds_mean']:.1f}"
        )
        for steps, cell in row["per_steps"].items():
            log(
                f"    T={int(steps):>3}: exact match {cell['exact_match_mean']:.4f}"
                f"+-{cell['exact_match_std']:.4f}  token accuracy "
                f"{cell['token_accuracy_mean']:.4f}+-{cell['token_accuracy_std']:.4f}"
            )
    log()
    log("cost ratios at the largest step count, split (e2d2-4/1) over decoder-only baseline:")
    for key, value in summary["cost_ratio"].items():
        shown = "n/a" if value is None else f"{value:.4f}"
        log(f"  {key:<44} {shown}")


if __name__ == "__main__":
    raise SystemExit(main())
