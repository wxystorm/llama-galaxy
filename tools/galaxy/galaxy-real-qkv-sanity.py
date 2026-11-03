#!/usr/bin/env python3
"""Inspect causal first-token consistency of a real GGUF QKV probe.

For a fresh, single-sequence causal prefill, the first query can attend
only to the first key. Therefore the pre-Wo Attention output for token 0
must equal the corresponding Vcur[0] GQA head, up to any V-cache
quantization / kernel rounding. This powerful invariant does not depend
on Q/K, RoPE, score scaling, or softmax implementation.

This program reads existing GLG1 snapshots offline. It does not run a
model, access any devices, or assert numerical parity based on an
arbitrary tolerance.
"""
import argparse
from array import array
import math
from pathlib import Path
import struct
import sys


def load(path, elements):
    raw = path.read_bytes()
    if len(raw) < 8 or raw[:4] != b"GLG1":
        raise ValueError(f"bad GLG1 header: {path}")
    count = struct.unpack_from("<i", raw, 4)[0]
    if count != elements or len(raw) != 8 + 4 * elements:
        raise ValueError(f"shape or length mismatch in {path}: {count} vs {elements}")
    values = array("f")
    values.frombytes(raw[8:])
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"nonfinite data in {path}")
    return values


def metrics(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("incompatible arrays")
    squares = dot = na = nb = 0.0
    max_abs = 0.0
    for x, y in zip(a, b):
        diff = float(x) - float(y)
        squares += diff * diff
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
        max_abs = max(max_abs, abs(diff))
    return (math.sqrt(squares / len(a)), max_abs,
            dot / math.sqrt(na * nb) if na > 0 and nb > 0 else float("nan"))


def as_row(data, index, width):
    return data[index * width:(index + 1) * width]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-dir", type=Path, required=True)
    p.add_argument("--compare-probe-dir", type=Path,
                   help="optional earlier GLG1 probe for detecting snapshot lifetime changes")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--tokens", type=int, default=98)
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    args = p.parse_args()
    if args.tokens < 2 or args.heads < 1 or args.kv_heads < 1 or args.head_dim < 1:
        p.error("positive dimensions and at least 2 tokens required")
    if args.heads % args.kv_heads:
        p.error("--heads must be divisible by --kv-heads")
    if args.layer < 0:
        p.error("layer must be >= 0")

    qwidth = args.heads * args.head_dim
    kvwidth = args.kv_heads * args.head_dim
    snapshots = {}
    for name, width in [("Qcur", qwidth), ("Kcur", kvwidth),
                        ("Vcur", kvwidth), ("kqv_out", qwidth)]:
        snapshots[name] = load(args.probe_dir / f"{name}-{args.layer}.f32",
                               args.tokens * width)

    if args.compare_probe_dir:
        # Compare two independently captured runs *without recomputation*.
        # In these investigations only diagnostic graph-output pinning
        # changed. Identical Q/K and changed V/Attention is strong evidence
        # that an end-of-graph snapshot should not be assumed stable.
        for name, width in (("Qcur", qwidth), ("Kcur", kvwidth),
                            ("Vcur", kvwidth), ("kqv_out", qwidth)):
            earlier = load(args.compare_probe_dir / f"{name}-{args.layer}.f32",
                           args.tokens * width)
            current = snapshots[name]
            rmse, max_abs, cosine = metrics(earlier, current)
            exact = sum(x == y for x, y in zip(earlier, current))
            print(f"[GALAXY_QKV_CROSS_RUN] tensor={name} scope=all "
                  f"rmse={rmse:.9g} max_abs={max_abs:.9g} "
                  f"cosine={cosine:.9g} exact={exact}/{len(current)}")
            a, b = earlier[:width], current[:width]
            tr, tm, tc = metrics(a, b)
            te = sum(x == y for x, y in zip(a, b))
            print(f"[GALAXY_QKV_CROSS_RUN] tensor={name} scope=token0 "
                  f"rmse={tr:.9g} max_abs={tm:.9g} "
                  f"cosine={tc:.9g} exact={te}/{width}")
            if width % 2 == 0:
                for rank in (0, 1):
                    ra, rb = [], []
                    half = width // 2
                    for token in range(args.tokens):
                        first = token * width + rank * half
                        ra.extend(earlier[first:first + half])
                        rb.extend(current[first:first + half])
                    rr, rm, rc = metrics(ra, rb)
                    re = sum(x == y for x, y in zip(ra, rb))
                    print(f"[GALAXY_QKV_CROSS_RUN] tensor={name} scope=rank{rank} "
                          f"rmse={rr:.9g} max_abs={rm:.9g} "
                          f"cosine={rc:.9g} exact={re}/{len(ra)}")

    # An exceptionally specific symptom: the best KV-head sequence and
    # best RMSE repeat in h=0..11 and h=12..23. Test whether exported
    # Attention/Query/KV activations themselves duplicate their rank
    # halves. Equal best matches alone are only suggestive; an exact
    # activation equality is decisive evidence of duplicated data.
    for name, head_count in (("Qcur", args.heads), ("Kcur", args.kv_heads),
                             ("Vcur", args.kv_heads), ("kqv_out", args.heads)):
        data = snapshots[name]
        if head_count % 2:
            print(f"[GALAXY_QKV_DUPLICATE] tensor={name} skip=odd_head_count")
            continue
        stride = head_count * args.head_dim
        half = stride // 2
        a = []
        b = []
        for token in range(args.tokens):
            begin = token * stride
            a.extend(data[begin:begin + half])
            b.extend(data[begin + half:begin + stride])
        rmse, maxabs, cosine = metrics(a, b)
        exact = sum(x == y for x, y in zip(a, b))
        print(f"[GALAXY_QKV_DUPLICATE] tensor={name} halves={head_count//2}+{head_count//2} "
              f"rmse={rmse:.9g} max_abs={maxabs:.9g} cosine={cosine:.9g} "
              f"exact={exact}/{len(a)}")
        # Emit token0 metrics separately, to correlate with the
        # first-token causal invariant, rather than guessing from averages.
        t_rmse, t_max, t_cos = metrics(a[:half], b[:half])
        print(f"[GALAXY_QKV_DUPLICATE] tensor={name} token=0 "
              f"rmse={t_rmse:.9g} max_abs={t_max:.9g} cosine={t_cos:.9g}")

    # For an exact rank-half duplication, every aligned head-pair and
    # every token must match. The per-head check helps spot just one
    # duplicated logical Q head even when the entire half differs.
    for name in ("Qcur", "kqv_out"):
        width = qwidth
        data = snapshots[name]
        half_heads = args.heads // 2
        if args.heads % 2:
            continue
        for h in range(half_heads):
            va = []
            vb = []
            for token in range(args.tokens):
                base = token * width
                i = base + h * args.head_dim
                j = base + (h + half_heads) * args.head_dim
                va.extend(data[i:i+args.head_dim])
                vb.extend(data[j:j+args.head_dim])
            h_rmse, h_max, h_cos = metrics(va, vb)
            if h == 0 or h_rmse < 1.0e-7:
                print(f"[GALAXY_QKV_HEAD_PAIR] tensor={name} "
                      f"head_a={h} head_b={h+half_heads} "
                      f"rmse={h_rmse:.9g} max_abs={h_max:.9g} "
                      f"cosine={h_cos:.9g} exact={sum(x == y for x, y in zip(va, vb))}/{len(va)}")

    # The raw FLASH_ATTN_EXT output is four-dimensional before its
    # kqv_out 2D reshape. The Meta exporter reconstructs both into the
    # same logical [heads*head_dim, tokens] F32 order. This source-level
    # identity test distinguishes faulty reshape/export from upstream
    # Flash Attention output duplication. The raw snapshot is optional,
    # so historical probes remain readable without rerunning the model.
    raw_path = args.probe_dir / f"fattn_raw-{args.layer}.f32"
    if raw_path.is_file():
        raw_attn = load(raw_path, args.tokens * qwidth)
        final_attn = snapshots["kqv_out"]
        rmse_all, maxabs_all, cosine_all = metrics(raw_attn, final_attn)
        matches = sum(a == b for a, b in zip(raw_attn, final_attn))
        print(f"[GALAXY_QKV_RAW_RESHAPE] all rmse={rmse_all:.9g} "
              f"max_abs={maxabs_all:.9g} cosine={cosine_all:.9g} "
              f"exact={matches}/{len(raw_attn)}")
        per_head = args.head_dim
        half_width = qwidth // 2
        for rank in (0, 1):
            a = []
            b = []
            for token in range(args.tokens):
                start = token*qwidth + rank*half_width
                a.extend(raw_attn[start:start+half_width])
                b.extend(final_attn[start:start+half_width])
            r, m, cos = metrics(a, b)
            print(f"[GALAXY_QKV_RAW_RESHAPE] rank={rank} rmse={r:.9g} "
                  f"max_abs={m:.9g} cosine={cos:.9g} "
                  f"exact={sum(x == y for x, y in zip(a, b))}/{len(a)}")
        a, b = [], []
        for token in range(args.tokens):
            start = token*qwidth
            a.extend(raw_attn[start:start+half_width])
            b.extend(raw_attn[start+half_width:start+qwidth])
        r, m, cos = metrics(a, b)
        print(f"[GALAXY_QKV_RAW_DUPLICATE] halves=12+12 rmse={r:.9g} "
              f"max_abs={m:.9g} cosine={cos:.9g} "
              f"exact={sum(x == y for x, y in zip(a, b))}/{len(a)}")
        if matches == len(raw_attn):
            print("[GALAXY_QKV_RAW_VERDICT] reshape_preserves_raw_data=YES")
        else:
            print("[GALAXY_QKV_RAW_VERDICT] reshape_preserves_raw_data=NO "
                  "inspect_meta_view_or_export=YES")
    else:
        print("[GALAXY_QKV_RAW_RESHAPE] unavailable=1 "
              "reason=previous_probe_without_raw_flash_snapshot")

    original = as_row(snapshots["kqv_out"], 0, qwidth)
    v0 = as_row(snapshots["Vcur"], 0, kvwidth)
    ratio = args.heads // args.kv_heads
    grouped = [v0[(h // ratio) * args.head_dim + d]
               for h in range(args.heads) for d in range(args.head_dim)]
    interleaved = [v0[(h % args.kv_heads) * args.head_dim + d]
                   for h in range(args.heads) for d in range(args.head_dim)]
    r_group = metrics(original, grouped)
    r_inter = metrics(original, interleaved)
    print("[GALAXY_QKV_SANITY] conditional=fresh_single_sequence_causal_prefill")
    print("[GALAXY_QKV_SANITY] invariant=original_token0_equals_Vcur_token0")
    print(f"[GALAXY_QKV_SANITY] grouped_GQA rmse={r_group[0]:.9g} "
          f"max_abs={r_group[1]:.9g} cosine={r_group[2]:.9g}")
    print(f"[GALAXY_QKV_SANITY] interleaved_GQA rmse={r_inter[0]:.9g} "
          f"max_abs={r_inter[1]:.9g} cosine={r_inter[2]:.9g}")
    print(f"{'QHead':>6} {'ExpectedKV':>10} {'BestKV':>7} "
          f"{'BestRMSE':>12} {'ExpectedRMSE':>14}")
    best_sequence = []
    for head in range(args.heads):
        output_head = original[head*args.head_dim:(head+1)*args.head_dim]
        candidates = []
        for k_head in range(args.kv_heads):
            values = v0[k_head*args.head_dim:(k_head+1)*args.head_dim]
            candidates.append(metrics(output_head, values)[0])
        winner = min(range(args.kv_heads), key=lambda h: candidates[h])
        best_sequence.append(winner)
        expected_head = head // ratio
        print(f"{head:>6d} {expected_head:>10d} {winner:>7d} "
              f"{candidates[winner]:>12.7g} {candidates[expected_head]:>14.7g}")
    print("[GALAXY_QKV_SANITY] best_kv_head_sequence=" +
          ",".join(map(str, best_sequence)))
    print("[GALAXY_QKV_SANITY] expected_sequence=" +
          ",".join(str(h // ratio) for h in range(args.heads)))

    # Check whether original token 0 resembles some *other* token's V.
    # This can expose a non-identity physical KV slot mapping or an
    # incorrect exporter token stride, without guessing the cause.
    target_heads = sorted({0, args.heads // 2, args.heads - 1})
    for h in target_heads:
        output_head = original[h*args.head_dim:(h+1)*args.head_dim]
        kv_head = h // ratio
        candidates = []
        for token in range(args.tokens):
            vi = token * kvwidth + kv_head * args.head_dim
            candidate = snapshots["Vcur"][vi:vi + args.head_dim]
            candidates.append(metrics(output_head, candidate)[0])
        best_token = min(range(args.tokens), key=lambda t: candidates[t])
        print(f"[GALAXY_QKV_SANITY] head={h} expected_kv={kv_head} "
              f"best_v_token={best_token} best_rmse={candidates[best_token]:.9g} "
              f"v0_rmse={candidates[0]:.9g}")

    for name, width in [("Qcur", qwidth), ("Kcur", kvwidth),
                        ("Vcur", kvwidth), ("kqv_out", qwidth)]:
        values = snapshots[name]
        rms = math.sqrt(sum(float(x)*x for x in values)/len(values))
        print(f"[GALAXY_QKV_SANITY] tensor={name} elements={len(values)} "
              f"rms={rms:.9g} min={min(values):.9g} max={max(values):.9g}")
    print("[GALAXY_QKV_SANITY] FINISHED; invariants are diagnostic, not a PASS claim")


if __name__ == "__main__":
    main()
