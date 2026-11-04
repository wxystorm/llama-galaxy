#!/usr/bin/env python3
"""Compare GGUF/Llama Galaxy GLG1 producer snapshots across two complete runs.

Usage:
  python tools/galaxy/galaxy-activation-diff.py \
    --tp-run <baseline tp-... directory> \
    --sp-run <REPLACE-1 tp-... directory> --tokens 98 --layer 0 \
    --json <optional results.json>

The tool is intentionally dependency-free (stdlib only). It compares exactly
the data produced by existing QKV, stage and layer probes. In particular:
- Q/K/V: tests whether the input to native SP Attention changed.
- fattn_raw/kqv_out: tests whether the SP replacement equals the *independent*
  TP Flash output (not just whether it was correctly injected).
- ffn_inp: observes the post-W_o, post-residual boundary.
- ffn_norm/l_out: locates divergence inside or after the selected layer.
- first_logits: checks the downstream 28-layer propagation, not as a substitute
  for the existing strict top1-aware galaxy-logits-compare.exe.
"""

from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import struct
import sys


def read_glg1(path: Path) -> array:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open("rb") as f:
        magic = f.read(8)
        if len(magic) != 8 or magic[:4] != b"GLG1":
            raise ValueError(f"Invalid GLG1 header: {path}")
        (count,) = struct.unpack("<i", magic[4:])
        if not 0 < count < 100_000_000:
            raise ValueError(f"Invalid GLG1 element count {count}: {path}")
        values = f.read()
    if len(values) != count * 4:
        raise ValueError(f"Unexpected GLG1 payload length: {path}")
    arr = array("f")
    arr.frombytes(values)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr


def metrics(a: array, b: array, tokens: int | None) -> dict:
    if len(a) != len(b):
        raise ValueError(f"Tensor element counts differ: {len(a)} vs {len(b)}")
    if not len(a):
        raise ValueError("Empty activation tensor")
    if tokens is not None and len(a) % tokens:
        raise ValueError(f"Element count {len(a)} not divisible by {tokens} tokens")
    squared = abs_sum = biggest = dot = norm_a = norm_b = 0.0
    biggest_idx = -1
    exact = True
    last_sq = last_max = 0.0
    last_n = 0
    width = len(a) // tokens if tokens is not None else len(a)
    worst_token = -1
    for i, (x, y) in enumerate(zip(a, b)):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Nonfinite element at {i}")
        if x != y:
            exact = False
        d = float(x) - float(y)
        ad = abs(d)
        abs_sum += ad
        squared += d * d
        dot += float(x) * y
        norm_a += float(x) * x
        norm_b += float(y) * y
        if ad > biggest:
            biggest = ad
            biggest_idx = i
            worst_token = i // width if tokens is not None else -1
        if tokens is not None and i >= len(a) - width:
            last_n += 1
            last_sq += d * d
            last_max = max(last_max, ad)
    n = len(a)
    return {
        "elements": n,
        "tokens": tokens,
        "features_per_token": width if tokens else None,
        "byte_exact": exact,
        "rmse": math.sqrt(squared / n),
        "max_abs": biggest,
        "max_abs_index": biggest_idx,
        "max_abs_token": worst_token,
        "mean_abs": abs_sum / n,
        "cosine": dot / math.sqrt(norm_a * norm_b) if norm_a > 0 and norm_b > 0 else None,
        "last_token_rmse": math.sqrt(last_sq / last_n) if last_n else None,
        "last_token_max_abs": last_max if last_n else None,
    }


def snapshot_paths(run: Path, layer: int, tokens: int) -> list[tuple[str, Path, int | None]]:
    stages = ["Qcur", "Kcur", "Vcur", "fattn_raw", "kqv_out"]
    result = [(name, Path("real-qkv") / f"tokens-{tokens}" /
               f"{name}-{layer}.f32", tokens) for name in stages]
    result += [(name, Path("stages") / f"{name}-{layer}.f32", tokens)
               for name in ("ffn_inp", "ffn_norm")]
    result.append(("l_out", Path("layers") / f"l_out-{layer}.f32", tokens))
    result.append(("first_logits", Path("first_logits.f32"), None))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tp-run", required=True, type=Path)
    parser.add_argument("--sp-run", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.tokens < 8 or args.layer < 0:
        parser.error("tokens must be >= 8 and layer must be nonnegative")
    rows = {}
    for name, rel, token_count in snapshot_paths(args.tp_run, args.layer, args.tokens):
        reference = read_glg1(args.tp_run / rel)
        candidate = read_glg1(args.sp_run / rel)
        row = metrics(reference, candidate, token_count)
        rows[name] = row
        print("[GALAXY_REPLACE_DIFF] "
              f"tensor={name}-{args.layer if name != 'first_logits' else 'all'} "
              f"count={row['elements']} exact={int(row['byte_exact'])} "
              f"rmse={row['rmse']:.9g} max_abs={row['max_abs']:.9g} "
              f"worst_token={row['max_abs_token']} "
              f"last_rmse={row['last_token_rmse'] if row['last_token_rmse'] is not None else 'NA'} "
              "status=MEASURED")
    upstream_exact = all(rows[k]["byte_exact"] for k in ("Qcur", "Kcur", "Vcur"))
    print(f"[GALAXY_REPLACE_DIFF_SUM] inputs_byte_exact={int(upstream_exact)} "
          f"attention_rmse={rows['kqv_out']['rmse']:.9g} "
          f"attention_last_rmse={rows['kqv_out']['last_token_rmse']:.9g} "
          f"post_wo_rmse={rows['ffn_inp']['rmse']:.9g} "
          f"last_layer_rmse={rows['l_out']['rmse']:.9g} "
          f"logits_rmse={rows['first_logits']['rmse']:.9g} "
          "status=MEASURED")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "reference": str(args.tp_run), "candidate": str(args.sp_run),
            "layer": args.layer, "tokens": args.tokens,
            "input_byte_exact": upstream_exact, "stages": rows
        }, indent=2), encoding="utf-8")
    if not upstream_exact:
        print("[GALAXY_REPLACE_DIFF_DIAG] stage=PRE_ATTENTION "
              "finding=QKV_INPUTS_CHANGED investigate=producer_time_capture")
    elif rows["kqv_out"]["rmse"] > 0.001:
        print("[GALAXY_REPLACE_DIFF_DIAG] stage=ATTENTION "
              "finding=SP_OUTPUT_DIFFERS_FROM_INDEPENDENT_TP_REFERENCE "
              "investigate=native_flash_qkv_cache_mask")
    elif rows["ffn_inp"]["rmse"] > 0.001:
        print("[GALAXY_REPLACE_DIFF_DIAG] stage=POST_WO "
              "finding=DOWNSTREAM_DIVERGES_AFTER_MATCHING_ATTENTION "
              "investigate=output_buffer_lifetime_wo_reduce")
    elif rows["l_out"]["rmse"] > 0.001:
        print("[GALAXY_REPLACE_DIFF_DIAG] stage=FFN "
              "finding=DIFFERENCE_AMPLIFIES_WITHIN_SELECTED_LAYER")
    else:
        print("[GALAXY_REPLACE_DIFF_DIAG] stage=LATER_LAYERS "
              "finding=INITIAL_LAYER_CLOSE_CHECK_DOWNSTREAM_AMPLIFICATION")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, OSError, ValueError) as ex:
        print(f"[GALAXY_REPLACE_DIFF] status=FAIL reason={ex}", file=sys.stderr)
        sys.exit(1)
