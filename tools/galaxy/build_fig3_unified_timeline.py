#!/usr/bin/env python3
"""Build one aligned Galaxy TP Prefill timeline with a single Phone Compute class.

The Phone backend (CPU/GPU/HTP) is deliberately NOT inferred. All Phone
native graph compute intervals have category 'Phone Compute'.

This requires the new, opt-in Phone monotonic graph intervals and the PC
clock_sync.csv from the SAME run. A legacy phone events.csv with only
timestamp_unix_ms is intentionally not accepted for an aligned timeline.

This is a measured diagnostic occupancy projection, NOT proof of a unique
critical path or of actual hardware idle time. PC GPU/CPU/PCIe requires
separate PC device activity traces (optional --pc-events).
"""

from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path

ACTIVITIES = (
    "Phone Compute",
    "GPU Compute",
    "CPU Compute",
    "PCIe Copy",
    "Wi-Fi Sync",
    "Waiting/Other",
)
# A deterministic EXCLUSIVE projection, not proof of causality. In
# particular, Phone Compute overrides the containing AllReduce/RPC wait.
PRIORITY = ACTIVITIES[:-1]
COLORS = {
    "Phone Compute": "#3978AB",
    "GPU Compute": "#7E5AA7",
    "CPU Compute": "#B18B45",
    "PCIe Copy": "#46A38C",
    "Wi-Fi Sync": "#DC8858",
    "Waiting/Other": "#A6AFBD",
}
PC_EVENT_CATEGORIES = {"GPU Compute", "CPU Compute", "PCIe Copy"}


def load(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input absent: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        result = list(csv.DictReader(f))
    if not result:
        raise ValueError(f"Empty CSV: {path}")
    return result


def write_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def interval(start: int, end: int, category: str, source: str, label: str) -> dict:
    if end <= start:
        raise ValueError(f"Non-positive interval: {label} ({start}..{end})")
    return {
        "start_ns": start,
        "end_ns": end,
        "category": category,
        "source": source,
        "label": label,
    }


def choose_clock_sync(rows: list[dict[str, str]], endpoint: str) -> tuple[int, float, str]:
    candidates = [r for r in rows if not endpoint or r["endpoint"] == endpoint]
    if not candidates:
        raise ValueError(f"No clock calibration for endpoint {endpoint!r}")
    # Minimum uncertainty bounds offset error under symmetric-delay
    # assumption; do not claim perfect one-way network clock correction.
    best = min(candidates, key=lambda r: (float(r["uncertainty_ms"]), float(r["rtt_ms"])))
    return int(best["offset_phone_minus_pc_ns"]), float(best["uncertainty_ms"]), best["endpoint"]


def graph_selection(folder: Path, last_graphs: int, ids: str):
    summary = load(folder / "summary.csv")
    graph_intervals = load(folder / "graph_timeline.csv")
    by_graph = {int(r["graph"]): r for r in graph_intervals}
    available = sorted({int(r["graph"]) for r in summary})
    if ids:
        chosen = sorted({int(x.strip()) for x in ids.split(",") if x.strip()})
        if not set(chosen) <= set(available):
            raise ValueError(f"Unknown graph ID; available={available}")
    else:
        if last_graphs < 1 or last_graphs > len(available):
            raise ValueError(f"--last-graphs must be 1..{len(available)}")
        chosen = available[-last_graphs:]
    if any(g not in by_graph for g in chosen):
        raise ValueError("graph_timeline.csv does not contain each selected graph")
    spans = []
    for g in chosen:
        r = by_graph[g]
        start, end = int(r["start_steady_ns"]), int(r["end_steady_ns"])
        expected_ms = float(r["wall_ms"])
        if abs((end - start) / 1e6 - expected_ms) > 0.5:
            raise ValueError(f"Graph {g} has inconsistent monotonic boundaries")
        spans.append((g, start, end))
    selected_summary = [r for r in summary if int(r["graph"]) in chosen]
    expected_calls = sum(int(r["all_reduce"]) for r in selected_summary)
    return chosen, sorted(spans), expected_calls


def project(spans: list[tuple[int, int, int]], activity: list[dict]):
    """Segment graph coverage into a single, exhaustive exclusive timeline."""
    events: list[tuple[int, str, int]] = []
    for _, start, end in spans:
        events += [(start, "graph", 1), (end, "graph", -1)]
    for item in activity:
        events += [(item["start_ns"], item["category"], 1),
                   (item["end_ns"], item["category"], -1)]
    events.sort(key=lambda e: e[0])
    active = Counter()
    out = []
    previous = None
    i = 0
    overlap_ms = 0.
    while i < len(events):
        t = events[i][0]
        if previous is not None and t > previous and active["graph"] > 0:
            category = next((cat for cat in PRIORITY if active[cat] > 0),
                            "Waiting/Other")
            enabled = [cat for cat in PRIORITY if active[cat] > 0]
            if len(enabled) > 1:
                overlap_ms += (t - previous) / 1e6
            if out and out[-1]["category"] == category and out[-1]["end_ns"] == previous:
                out[-1]["end_ns"] = t
            else:
                out.append({"start_ns": previous, "end_ns": t, "category": category})
        while i < len(events) and events[i][0] == t:
            _, key, delta = events[i]
            active[key] += delta
            if active[key] < 0:
                raise ValueError(f"Invalid sweep count for {key}")
            i += 1
        previous = t
    return out, overlap_ms


def render_svg(path: Path, spans: list[tuple[int, int, int]], segments: list[dict],
               total_ms: float, info: str) -> None:
    t0 = min(s for _, s, _ in spans)
    t1 = max(e for _, _, e in spans)
    width, left, right, top = 1120, 105, 35, 80
    chart = width - left - right
    def x(ns):
        return left + (ns - t0) / max(1, t1 - t0) * chart
    total_s = (t1 - t0) / 1e9
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 225">',
        '<rect width="1120" height="225" fill="white"/>',
        '<text x="105" y="27" font-size="18" font-weight="bold" fill="#1C2735">'
        'Galaxy 14B: unified prefill timeline (diagnostic)</text>',
        f'<text x="105" y="47" font-size="12" fill="#53606F">{html.escape(info)}</text>',
        '<text x="18" y="99" font-size="13" fill="#263748">Activity</text>',
        f'<rect x="{left}" y="{top}" width="{chart}" height="32" fill="#EEF0F3"/>',
    ]
    for item in segments:
        a, b = x(item["start_ns"]), x(item["end_ns"])
        lines.append(f'<rect x="{a:.3f}" y="{top}" width="{max(0.001,b-a):.3f}" '
                     f'height="32" fill="{COLORS[item["category"]]}"/>')
    for tick in range(6):
        s = total_s * tick / 5
        xx = left + chart * tick / 5
        lines.append(f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="116" y2="123" '
                     'stroke="#647386" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.2f}" y="140" fill="#415166" '
                     f'text-anchor="middle" font-size="11">{s:.1f} s</text>')
    for i, (name, color) in enumerate(COLORS.items()):
        col, row = i % 3, i // 3
        xx, yy = left + col * 312, 167 + row * 23
        lines.append(f'<rect x="{xx}" y="{yy - 10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{xx+20}" y="{yy}" font-size="12" fill="#263748">'
                     f'{html.escape(name)}</text>')
    lines.append(f'<text x="{left}" y="219" font-size="10" fill="#657282">'
                 f'Profile-covered graph time: {total_ms / 1000:.2f} s; '
                 'overlapping activities are projected by priority, not added.</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile-dir", type=Path, required=True,
                    help="PC profile folder containing summary, graph_timeline, AllReduce, RPC and clock_sync CSVs")
    ap.add_argument("--phone-timeline", type=Path, required=True,
                    help="Phone fig3-galaxy-phone.timeline.csv copied from Termux")
    ap.add_argument("--run-id", default="", help="Optional unique profile run ID in phone timeline")
    ap.add_argument("--endpoint", default="", help="Select PC clock calibration for this Phone endpoint")
    ap.add_argument("--last-graphs", type=int, default=5)
    ap.add_argument("--graph-ids", default="")
    ap.add_argument("--pc-events", type=Path, default=None,
                    help="Optional PC events CSV: category,start_steady_ns,end_steady_ns")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-clock-uncertainty-ms", type=float, default=20.0)
    ap.add_argument("--normal-prefill-ms", type=float, default=None,
                    help="Separate unprofiled Prefill timing for the label; never changes Profile proportions")
    args = ap.parse_args()

    profile_dir = args.profile_dir.resolve()
    out = (args.out_dir or profile_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    chosen, graph_spans, expected_ar = graph_selection(profile_dir, args.last_graphs, args.graph_ids)
    clock_offset_ns, uncertainty_ms, endpoint = choose_clock_sync(
        load(profile_dir / "clock_sync.csv"), args.endpoint)
    if uncertainty_ms > args.max_clock_uncertainty_ms:
        raise ValueError(
            f"Clock calibration uncertainty {uncertainty_ms:.2f} ms exceeds "
            f"{args.max_clock_uncertainty_ms:.2f} ms. Check PC/Phone Wi-Fi.")
    ar_rows = [r for r in load(profile_dir / "allreduce_breakdown.csv")
               if int(r["graph"]) in chosen]
    if len(ar_rows) != expected_ar:
        raise ValueError(f"AllReduce count mismatch: {len(ar_rows)} != {expected_ar}")

    activity: list[dict] = []
    for r in ar_rows:
        activity.append(interval(int(r["start_steady_ns"]), int(r["end_steady_ns"]),
                                 "Wi-Fi Sync", "PC_ALLREDUCE", r["tensor"]))
    rpc_rows = load(profile_dir / "rpc_transfer_timeline.csv")
    for r in rpc_rows:
        if r["command"] not in ("GET_TENSOR", "SET_TENSOR", "SET_TENSOR_HASH", "COPY_TENSOR"):
            continue
        if int(r.get("ok", "1")) != 1:
            continue
        start, end = int(r["start_steady_ns"]), int(r["end_steady_ns"])
        activity.append(interval(start, end, "Wi-Fi Sync", "PC_RPC", r["command"]))

    all_phone = load(args.phone_timeline)
    phone_rows = [r for r in all_phone if not args.run_id or r["run_id"] == args.run_id]
    if not phone_rows:
        raise ValueError("No Phone graph intervals matching the selected run-id")
    for r in phone_rows:
        start = int(r["compute_start_phone_ns"]) - clock_offset_ns
        end = int(r["compute_end_phone_ns"]) - clock_offset_ns
        if end <= start:
            continue
        activity.append(interval(start, end, "Phone Compute", "PHONE_NATIVE",
                                 f'{r["mode"]}:{r["graph_uid"]}'))

    pc_events_present = args.pc_events is not None
    if args.pc_events:
        for r in load(args.pc_events):
            cat = r["category"]
            if cat not in PC_EVENT_CATEGORIES:
                raise ValueError(f"Invalid PC event category {cat!r}; use {sorted(PC_EVENT_CATEGORIES)}")
            activity.append(interval(int(r["start_steady_ns"]), int(r["end_steady_ns"]),
                                     cat, "PC_DEVICE", cat))

    # Keep raw events only when they intersect selected graph windows; no
    # phone preload / startup graph work is included in Fig. 3 Prefill.
    def touches_graph(a):
        return any(a["start_ns"] < end and a["end_ns"] > start
                   for _, start, end in graph_spans)

    activity = [a for a in activity if touches_graph(a)]
    segments, overlapping_ms = project(graph_spans, activity)
    if not segments:
        raise ValueError("No covered Prefill graph intervals")
    origin = min(s for _, s, _ in graph_spans)
    total_ms = sum((end - start) / 1e6 for _, start, end in graph_spans)
    if abs(sum((s["end_ns"]-s["start_ns"])/1e6 for s in segments) - total_ms) > .02:
        raise ValueError("Projected segments do not sum to measured PC graph time")
    by_category = defaultdict(float)
    for s in segments:
        by_category[s["category"]] += (s["end_ns"] - s["start_ns"]) / 1e6

    source_rows = [{
        "start_ms": round((a["start_ns"] - origin) / 1e6, 4),
        "end_ms": round((a["end_ns"] - origin) / 1e6, 4),
        "duration_ms": round((a["end_ns"] - a["start_ns"]) / 1e6, 4),
        "source": a["source"], "category": a["category"], "label": a["label"]
    } for a in sorted(activity, key=lambda x: (x["start_ns"], x["end_ns"]))]
    write_csv(out / "fig3_source_events.csv", source_rows,
              ["start_ms", "end_ms", "duration_ms", "source", "category", "label"])

    line_rows = [{
        "start_ms": round((r["start_ns"] - origin) / 1e6, 4),
        "end_ms": round((r["end_ns"] - origin) / 1e6, 4),
        "duration_ms": round((r["end_ns"] - r["start_ns"]) / 1e6, 4),
        "category": r["category"]
    } for r in segments]
    write_csv(out / "fig3_unified_timeline.csv", line_rows,
              ["start_ms", "end_ms", "duration_ms", "category"])

    formal = "" if args.normal_prefill_ms is None else round(args.normal_prefill_ms, 3)
    values = {
        "baseline": "Galaxy",
        "graphs": ",".join(str(x) for x in chosen),
        "profile_total_graph_ms": round(total_ms, 3),
        "normal_prefill_ms": formal,
        "phone_backend": "UNIFIED (CPU/GPU/HTP NOT DISTINGUISHED)",
        "pc_device_events_included": int(pc_events_present),
        "clock_endpoint": endpoint,
        "clock_offset_phone_minus_pc_ns": clock_offset_ns,
        "clock_uncertainty_ms": round(uncertainty_ms, 4),
        "overlapping_activity_wall_ms": round(overlapping_ms, 3),
        "allreduce_count": expected_ar,
        "method": "exclusive_priority_projection; NOT causal critical path; Waiting/Other includes unattributed PC work",
    }
    for name in ACTIVITIES:
        key = name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        values[key + "_ms"] = round(by_category[name], 3)
        values[key + "_pct"] = round(100*by_category[name] / total_ms, 4)
    write_csv(out / "fig3_unified_breakdown.csv", [values], list(values))

    render_svg(out / "fig3_unified_timeline.svg", graph_spans, segments, total_ms,
               f'Graphs {values["graphs"]}; PC/Phone clock uncertainty '
               f'{uncertainty_ms:.2f} ms; Phone backend unified')
    print(f"[FIG3_TIMELINE] graphs={values['graphs']} AllReduce={expected_ar} "
          f"profile_graph_wall={total_ms:.2f} ms")
    print(f"[FIG3_TIMELINE] endpoint={endpoint} offset={clock_offset_ns} ns "
          f"uncertainty={uncertainty_ms:.3f} ms")
    for name in ACTIVITIES:
        print(f"  {name:>16s}  {by_category[name]:10.2f} ms  "
              f"{100*by_category[name]/total_ms:6.2f}%")
    print(f"[FIG3_TIMELINE] projected source overlaps={overlapping_ms:.2f} ms "
          "(not added to percentages)")
    if not pc_events_present:
        print("WARNING: PC GPU/CPU/PCIe event telemetry absent. Waiting/Other is "
              "a residual; NOT hardware idle. Do not publish as a verified six-category bar.")
    print("NOTE: -Profile is invasive; use normal Prefill time separately above the bar.")
    for name in ("fig3_unified_timeline.csv", "fig3_unified_breakdown.csv",
                 "fig3_source_events.csv", "fig3_unified_timeline.svg"):
        print(f"[FIG3_TIMELINE] output={out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
