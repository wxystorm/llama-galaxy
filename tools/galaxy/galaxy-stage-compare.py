#!/usr/bin/env python3
"""Compare Galaxy logical intermediate states after reconstructing rank sharding.

The Meta backend writes two selected stages in GLG1/F32 format:
  stages/ffn_inp-N.f32   (residual + attention, before FFN normalization)
  stages/ffn_norm-N.f32  (normalized per-token FFN input)
Existing layer snapshots remain in layers/l_out-N.f32.

Mathematically equivalent residual contributions can be reconstructed offline:
  attention contribution at layer N = ffn_inp[N] - l_out[N-1] (N >= 1)
  FFN contribution at layer N = l_out[N] - ffn_inp[N]
No graph ops or collectives are added to construct these differences.
The residual differences may incur small float32 subtraction rounding.
"""
import argparse
from array import array
import json
import math
from pathlib import Path
import struct
import sys


def read_snapshot(path):
    raw = path.read_bytes()
    if len(raw) < 8 or raw[:4] != b"GLG1":
        raise ValueError(f"invalid GLG1 header: {path}")
    count = struct.unpack_from("<i", raw, 4)[0]
    if count <= 0 or len(raw) != 8 + 4 * count:
        raise ValueError(f"invalid GLG1 payload length: {path}")
    data = array("f")
    data.frombytes(raw[8:])
    if sys.byteorder != "little":
        data.byteswap()
    if len(data) != count or not all(math.isfinite(v) for v in data):
        raise ValueError(f"non-finite or malformed snapshot: {path}")
    return data


def metrics(a, b, hidden, weights):
    if len(a) != len(b) or len(a) % hidden:
        raise ValueError("snapshot shapes differ")
    tokens = len(a) // hidden
    if tokens < 2:
        raise ValueError("expected prefill with at least two tokens")
    split = min(tokens - 1, max(1, int(tokens * weights[0] / sum(weights) + 0.5)))
    squared = [0.0, 0.0]
    norm_a = norm_b = dot = sqsum = 0.0
    maximum = -1.0
    worst = -1
    for idx, (x, y) in enumerate(zip(a, b)):
        err = float(x) - float(y)
        ae = abs(err)
        squared[0 if idx // hidden < split else 1] += err * err
        sqsum += err * err
        dot += float(x) * float(y)
        norm_a += float(x) * float(x)
        norm_b += float(y) * float(y)
        if ae > maximum:
            maximum, worst = ae, idx
    return {
        "tokens": tokens,
        "hidden": hidden,
        "split": [split, tokens - split],
        "rmse": math.sqrt(sqsum / len(a)),
        "pc_rmse": math.sqrt(squared[0] / (split * hidden)),
        "phone_rmse": math.sqrt(squared[1] / ((tokens - split) * hidden)),
        "max_abs": maximum,
        "max_token": worst // hidden,
        "max_channel": worst % hidden,
        "max_owner": "PC" if worst // hidden < split else "Phone",
        "reference_value": float(a[worst]),
        "candidate_value": float(b[worst]),
        "cosine": dot / math.sqrt(norm_a * norm_b) if norm_a and norm_b else None,
    }


def read_state(directory, layer, stage):
    folder = "layers" if stage == "l_out" else "stages"
    return read_snapshot(directory / folder / f"{stage}-{layer}.f32")


def subtraction(a, b):
    if len(a) != len(b):
        raise ValueError("incompatible stage shapes")
    return [float(x) - float(y) for x, y in zip(a, b)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="cap=0 or TP run directory")
    parser.add_argument("--candidate", type=Path, required=True, help="cap=1 or Galaxy2 run directory")
    parser.add_argument("--layers", default="0,1", help="comma-separated layer indices")
    parser.add_argument("--hidden", type=int, default=3072, help="hidden width (Llama 3.2 3B is 3072)")
    parser.add_argument("--ffn-ops", action="store_true", help="compare Up/Gate/activated FFN operator snapshots")
    parser.add_argument("--ffn-width", type=int, default=8192, help="FFN feature width (Llama 3.2 3B is 8192)")
    parser.add_argument("--seq-weights", default="1,1", help="rank 0/1 token weights")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    weights = tuple(int(x) for x in args.seq_weights.split(","))
    if len(weights) != 2 or min(weights) <= 0 or args.hidden <= 0 or args.ffn_width <= 0:
        parser.error("need positive --hidden, --ffn-width and --seq-weights a,b")
    layers = sorted(set(int(v) for v in args.layers.split(",")))
    if not layers or layers[0] < 0:
        parser.error("need nonnegative --layers")
    states = {}
    rows = []
    for layer in layers:
        stage_set = ["ffn_inp", "ffn_norm", "l_out"]
        for stage in stage_set:
            a, b = (read_state(run, layer, stage) for run in (args.reference, args.candidate))
            states[(layer, stage)] = (a, b)
            rows.append({"layer": layer, "stage": stage, **metrics(a, b, args.hidden, weights)})
        if args.ffn_ops:
            for op in ("ffn_up", "ffn_gate", "ffn_swiglu"):
                op_path = f"ffn_ops/{op}-{layer}.f32"
                a, b = (read_snapshot(run / op_path) for run in (args.reference, args.candidate))
                rows.append({"layer": layer, "stage": op,
                             **metrics(a, b, args.ffn_width, weights)})
        in_a, in_b = states[(layer, "ffn_inp")]
        out_a, out_b = states[(layer, "l_out")]
        rows.append({"layer": layer, "stage": "FFN_contribution",
                     **metrics(subtraction(out_a, in_a), subtraction(out_b, in_b),
                               args.hidden, weights)})
        if layer > 0 and (layer - 1, "l_out") in states:
            prev_a, prev_b = states[(layer - 1, "l_out")]
            rows.append({"layer": layer, "stage": "Attention_contribution",
                         **metrics(subtraction(in_a, prev_a), subtraction(in_b, prev_b),
                                   args.hidden, weights)})
    print(f"{'Layer':>5} {'Stage':<24} {'RMSE':>12} {'PC_RMSE':>12} {'Phone_RMSE':>12} {'MaxAbs':>12} {'Token':>6} {'Channel':>8} {'Owner':>6}")
    for r in rows:
        print(f"{r['layer']:>5} {r['stage']:<24} {r['rmse']:>12.8f} "
              f"{r['pc_rmse']:>12.8f} {r['phone_rmse']:>12.8f} "
              f"{r['max_abs']:>12.6f} {r['max_token']:>6} "
              f"{r['max_channel']:>8} {r['max_owner']:>6}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"reference": str(args.reference),
            "candidate": str(args.candidate), "layers": layers,
            "ffn_ops": args.ffn_ops, "ffn_width": args.ffn_width,
            "results": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[GALAXY_STAGE_COMPARE] output={args.output}")
    print("[GALAXY_STAGE_COMPARE] PASS (diagnostics only, not a numerical-parity verdict)")


if __name__ == "__main__":
    main()
