#!/usr/bin/env python3
"""Analyze Galaxy TP Fig. 3 profile without treating RPC wait as pure Wi-Fi.

Inputs are diagnostic "-PureTP -Profile" CSVs. Non-profile Prefill latency
must be obtained separately from the runner's stderr.log.
"""
from __future__ import annotations

import argparse
import bisect
import csv
from collections import Counter
from pathlib import Path


def rows(path: Path, required: bool = True) -> list[dict[str, str]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"missing profile CSV: {path}")
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0)


def numeric_sum(data: list[dict[str, str]], key: str) -> float:
    return sum(number(row, key) for row in data)


def main() -> int:
    ap = argparse.ArgumentParser(description="Galaxy pure-TP Fig. 3 AllReduce / RPC diagnostic breakdown")
    ap.add_argument("--profile-dir", required=True, type=Path)
    ap.add_argument("--last-graphs", default=5, type=int,
                    help="Choose final N graph invocations; 1041 tokens / ubatch 256 -> 5 (verify against log)")
    ap.add_argument("--graph-ids", default="", help="Override selection with comma-separated explicit graph IDs")
    ap.add_argument("--out", type=Path, default=None, help="Output one-row diagnostic CSV")
    args = ap.parse_args()

    folder = args.profile_dir.resolve()
    summaries = rows(folder / "summary.csv")
    allreduce = rows(folder / "allreduce_breakdown.csv")
    legacy_comm = rows(folder / "communication_timeline.csv")
    compute = rows(folder / "compute_timeline.csv")
    rpc = rows(folder / "rpc_transfer_timeline.csv", required=False)
    if not summaries:
        raise ValueError("summary.csv is empty")
    available = sorted({int(row["graph"]) for row in summaries})
    if args.graph_ids:
        wanted = sorted({int(s.strip()) for s in args.graph_ids.split(",") if s.strip()})
        unknown = sorted(set(wanted) - set(available))
        if unknown:
            raise ValueError(f"unrecognized graph ID(s): {unknown}; available={available}")
    else:
        if args.last_graphs < 1 or args.last_graphs > len(available):
            raise ValueError(f"--last-graphs must be 1..{len(available)}")
        wanted = available[-args.last_graphs:]

    selected_summaries = [r for r in summaries if int(r["graph"]) in wanted]
    selected_ar = sorted((r for r in allreduce if int(r["graph"]) in wanted),
                         key=lambda r: int(r["start_steady_ns"]))
    selected_comm = [r for r in legacy_comm
                     if int(r["graph"]) in wanted and r["kind"] == "ALLREDUCE"]
    selected_compute = [r for r in compute if int(r["graph"]) in wanted]

    if not selected_ar:
        raise ValueError("No AllReduce breakdown rows in selected graphs. Rebuild PC from Fig. 3 branch.")
    if len(selected_ar) != len(selected_comm):
        raise ValueError(f"AllReduce detail/legacy mismatch: {len(selected_ar)} != {len(selected_comm)}")
    expected_calls = int(numeric_sum(selected_summaries, "all_reduce"))
    if len(selected_ar) != expected_calls:
        raise ValueError(f"AllReduce detail/summary mismatch: {len(selected_ar)} != {expected_calls}")

    starts = [int(r["start_steady_ns"]) for r in selected_ar]
    ends = [int(r["end_steady_ns"]) for r in selected_ar]
    matched_rpc: list[dict[str, str]] = []
    unmatched_rpc = 0
    for event in rpc:
        begin = int(event["start_steady_ns"])
        end = int(event["end_steady_ns"])
        idx = bisect.bisect_right(starts, begin) - 1
        if idx >= 0 and begin >= starts[idx] and end <= ends[idx]:
            matched_rpc.append(event)
        else:
            unmatched_rpc += 1

    graph_ms = numeric_sum(selected_summaries, "graph_wall_ms")
    ar_ms = numeric_sum(selected_ar, "wall_ms")
    native_ms = numeric_sum(selected_ar, "native_call_ms")
    copy_ms = numeric_sum(selected_ar, "copy_call_ms")
    add_ms = numeric_sum(selected_ar, "add_dispatch_ms")
    fence_ms = numeric_sum(selected_ar, "completion_wait_ms")
    other_ms = numeric_sum(selected_ar, "other_ms")
    other_graph_ms = max(0.0, graph_ms - ar_ms)
    total_rpc_ms = numeric_sum(matched_rpc, "wall_ms")
    rpc_send_ms = numeric_sum(matched_rpc, "send_ms")
    rpc_recv_ms = numeric_sum(matched_rpc, "recv_wait_ms")
    rpc_commands = Counter(row["command"] for row in matched_rpc)
    compute_by_rank: dict[int, float] = {}
    for item in selected_compute:
        rank = int(item["rank"])
        compute_by_rank[rank] = compute_by_rank.get(rank, 0.0) + number(item, "observed_ms")

    result = {
        "graphs": ",".join(map(str, wanted)),
        "profile_forced_sync": 1,
        "profile_graph_wall_ms": round(graph_ms, 3),
        "allreduce_count": len(selected_ar),
        "allreduce_wall_ms": round(ar_ms, 3),
        "allreduce_avg_ms": round(ar_ms / len(selected_ar), 3),
        "allreduce_graph_share_pct": round(100 * ar_ms / graph_ms, 3) if graph_ms > 0 else 0,
        "native_collective_call_ms": round(native_ms, 3),
        "copy_call_ms": round(copy_ms, 3),
        "copy_calls": int(numeric_sum(selected_ar, "copy_calls")),
        "copy_requested_mib": round(numeric_sum(selected_ar, "copy_bytes") / (1024 * 1024), 3),
        "add_dispatch_ms": round(add_ms, 3),
        "add_dispatches": int(numeric_sum(selected_ar, "add_dispatches")),
        "completion_wait_ms": round(fence_ms, 3),
        "allreduce_other_ms": round(other_ms, 3),
        "non_allreduce_graph_ms": round(other_graph_ms, 3),
        "rpc_matched_calls": len(matched_rpc),
        "rpc_unmatched_calls": unmatched_rpc,
        "rpc_matched_wall_ms_nested": round(total_rpc_ms, 3),
        "rpc_send_ms_nested": round(rpc_send_ms, 3),
        "rpc_recv_wait_ms_nested": round(rpc_recv_ms, 3),
        "rpc_request_mib_nested": round(numeric_sum(matched_rpc, "input_bytes") / (1024 * 1024), 3),
        "rpc_response_mib_nested": round(numeric_sum(matched_rpc, "output_bytes") / (1024 * 1024), 3),
        "pc_rank0_observed_ms_overlapping": round(compute_by_rank.get(0, 0.0), 3),
        "phone_rpc_rank1_observed_ms_overlapping": round(compute_by_rank.get(1, 0.0), 3),
    }
    for cmd in ("GET_TENSOR", "SET_TENSOR", "SET_TENSOR_HASH", "COPY_TENSOR",
                "GRAPH_COMPUTE", "GRAPH_RECOMPUTE"):
        result[f"rpc_{cmd.lower()}_count"] = rpc_commands[cmd]

    output = args.out or (folder / "fig3_tp_diagnostic.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    print(f"[FIG3_GALAXY] graphs={result['graphs']} allreduce={len(selected_ar)}")
    print(f"  graph_wall={graph_ms:.2f} ms  allreduce={ar_ms:.2f} ms  share={result['allreduce_graph_share_pct']:.2f}%")
    print(f"  copy_call={copy_ms:.2f} ms  ADD_dispatch={add_ms:.2f} ms  "
          f"completion_wait={fence_ms:.2f} ms  native={native_ms:.2f} ms  "
          f"other_AR={other_ms:.2f} ms")
    print(f"  RPC calls within AllReduce={len(matched_rpc)} "
          f"send={rpc_send_ms:.2f} ms  recv/wait={rpc_recv_ms:.2f} ms")
    print(f"  RPC counts={dict(rpc_commands)}")
    print(f"  rank0 observed={compute_by_rank.get(0, 0.0):.2f} ms  "
          f"rank1 RPC observed={compute_by_rank.get(1, 0.0):.2f} ms")
    print("  NOTE: RPC intervals are NESTED inside copy/dispatch/AllReduce; NEVER add them again.")
    print("  NOTE: recv/wait includes remote processing; neither RPC wait nor copy-call is pure Wi-Fi.")
    print("  NOTE: profile forced fences. Use the separate non-profile run for normal Prefill latency.")
    if not rpc:
        print("  WARNING: RPC detailed CSV absent. Rebuild PC ggml-rpc and check profiling env.")
    print(f"[FIG3_GALAXY] csv={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
