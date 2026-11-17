#!/usr/bin/env python3
"""Galaxy TP Fig.3: validated PC CUDA/CPU/PCIe + unified Phone event timeline.

Required SAME-run files:
 PC: summary.csv, graph_timeline.csv, clock_sync.csv,
     allreduce_breakdown.csv, rpc_transfer_timeline.csv,
     pc_cuda_events.csv, pc_cpu_events.csv (may contain just the header)
 Phone: fig3-galaxy-phone-v3.timeline.csv with the selected run_id.

The output has TWO presentations:
 (1) raw concurrent device/synchronization intervals and device lanes;
 (2) one fully covered, exclusive 100%-stacked PROJECTION of the time axis.
Projection priority is an explicitly chosen accounting policy, not a proof of
the causal critical path. Device stream intervals are not per-kernel active
utilization; Wi-Fi Sync is RPC/AllReduce synchronization, not pure wire time.
Phone CPU/GPU/HTP are deliberately unified as Phone Compute.
"""
from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = (
    "Phone Compute",
    "PCIe Copy",
    "GPU Compute",
    "CPU Compute",
    "Wi-Fi Sync",
    "Waiting/Other",
)
PRIORITY = CATEGORIES[:-1]
COLORS = {
    "Phone Compute": "#317BB3",
    "PCIe Copy": "#31A98D",
    "GPU Compute": "#8F62B6",
    "CPU Compute": "#C3A34D",
    "Wi-Fi Sync": "#E17E55",
    "Waiting/Other": "#A9B1BD",
}
PC_CATEGORIES = {"GPU Compute", "CPU Compute", "PCIe Copy"}


def read_rows(path: Path, allow_header_only: bool = False):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required telemetry: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header: {path}")
        data = list(reader)
    if not data and not allow_header_only:
        raise ValueError(f"Missing data rows: {path}")
    return data


def write_rows(path: Path, rows: list[dict], cols: list[str]):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def itv(start: int, end: int, category: str, source: str, label: str):
    if start >= end:
        raise ValueError(f"Invalid {source} interval {start}..{end}")
    return dict(start=start, end=end, category=category, source=source, label=label)


def select_graphs(profile: Path, ids: str, last_graphs: int):
    summ = read_rows(profile / "summary.csv")
    raw = read_rows(profile / "graph_timeline.csv")
    by_id = {int(r["graph"]): r for r in raw}
    available = sorted({int(r["graph"]) for r in summ})
    if ids:
        selected = sorted(set(int(i.strip()) for i in ids.split(",") if i.strip()))
    else:
        if last_graphs < 1 or last_graphs > len(available):
            raise ValueError(f"last_graphs must be 1..{len(available)}")
        selected = available[-last_graphs:]
    if not selected or any(i not in available or i not in by_id for i in selected):
        raise ValueError(f"Graph IDs absent in summary or graph timeline. Available: {available}")
    spans = []
    for graph in selected:
        r = by_id[graph]
        a, b = int(r["start_steady_ns"]), int(r["end_steady_ns"])
        if not a < b or abs((b - a) / 1e6 - float(r["wall_ms"])) > 0.5:
            raise ValueError(f"Graph {graph} has invalid timing bounds")
        spans.append((graph, a, b))
    spans.sort(key=lambda z: z[1])
    if any(spans[i-1][2] > spans[i][1] for i in range(1, len(spans))):
        raise ValueError("Selected PC Graph spans overlap; cannot use graph-gap semantics")
    expected_ar = sum(int(r["all_reduce"]) for r in summ
                      if int(r["graph"]) in selected)
    return selected, spans, expected_ar


def calibration(profile: Path, endpoint: str, limit_ms: float):
    candidates = read_rows(profile / "clock_sync.csv")
    if endpoint:
        candidates = [r for r in candidates if r["endpoint"] == endpoint]
    if not candidates:
        raise ValueError("No Phone clock calibration for selected endpoint")
    row = min(candidates, key=lambda r: (float(r["uncertainty_ms"]), float(r["rtt_ms"])))
    error = float(row["uncertainty_ms"])
    if error > limit_ms:
        raise ValueError(f"Phone clock uncertainty {error:.3f}ms exceeds {limit_ms}ms")
    return int(row["offset_phone_minus_pc_ns"]), error, row["endpoint"]


def clip(item: dict, a: int, b: int):
    start = max(item["start"], a)
    end = min(item["end"], b)
    if start >= end:
        return None
    return dict(item, start=start, end=end)


def exclusive_projection(events: list[dict], t0: int, t1: int):
    ticks = [(t0, "", 0), (t1, "", 0)]
    for r in events:
        ticks.append((r["start"], r["category"], 1))
        ticks.append((r["end"], r["category"], -1))
    ticks.sort(key=lambda z: z[0])
    active = Counter()
    segments = []
    overlap_ns = 0
    i = 0
    old = t0
    while i < len(ticks):
        now = ticks[i][0]
        if now > old and old >= t0 and now <= t1:
            present = [name for name in PRIORITY if active[name] > 0]
            category = present[0] if present else "Waiting/Other"
            if len(present) > 1:
                overlap_ns += now - old
            if segments and segments[-1]["category"] == category and segments[-1]["end"] == old:
                segments[-1]["end"] = now
            else:
                segments.append({"start": old, "end": now, "category": category})
        while i < len(ticks) and ticks[i][0] == now:
            _, name, delta = ticks[i]
            if name:
                active[name] += delta
                if active[name] < 0:
                    raise ValueError(f"Negative active interval count: {name}")
            i += 1
        old = now
    expected = t1 - t0
    actual = sum(r["end"] - r["start"] for r in segments)
    if actual != expected or not segments or segments[0]["start"] != t0 or segments[-1]["end"] != t1:
        raise ValueError(f"Timeline coverage mismatch: {actual} != {expected}")
    return segments, overlap_ns


def svg_preview(path: Path, activity: list[dict], timeline: list[dict],
                t0: int, t1: int, graph_spans, values):
    width, left, right = 1200, 170, 25
    usable = width-left-right
    row_y = {
        "Exclusive projection": 85, "GPU Compute": 133,
        "Phone Compute": 172, "CPU Compute": 211,
        "PCIe Copy": 250, "Wi-Fi Sync": 289
    }
    def x(ns):
        return left + (ns-t0) / (t1-t0) * usable
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 410">',
        f'<rect width="{width}" height="410" fill="white"/>',
        '<text x="170" y="28" font-size="19" fill="#1D2938" font-weight="bold">'
        'Galaxy 14B: PC/Phone aligned raw lanes and exclusive projection</text>',
        '<text x="170" y="49" font-size="12" fill="#627083">'
        + html.escape("Total %.3f s, clock bound +/- %.3f ms. Overlap is counted once in the projection." %
                      (values["profile_envelope_ms"]/1000, values["clock_uncertainty_ms"])) + '</text>',
    ]
    for name, y in row_y.items():
        lines.append(f'<text x="{left-12}" y="{y+15}" text-anchor="end" fill="#344256" font-size="12">'
                     + html.escape(name) + '</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{usable}" height="23" fill="#EFF2F6"/>')
    for graph, a, b in graph_spans:
        gx = x(a)
        lines.append(f'<line x1="{gx:.3f}" x2="{gx:.3f}" y1="79" y2="315" stroke="#8CA0B2" opacity=".35" stroke-dasharray="2 2"/>')
        lines.append(f'<text x="{gx+3:.3f}" y="74" font-size="10" fill="#728092">G{graph}</text>')
    for seg in timeline:
        w = x(seg["end"])-x(seg["start"])
        lines.append(f'<rect x="{x(seg["start"]):.3f}" y="85" width="{max(.08,w):.3f}" height="23" fill="{COLORS[seg["category"]]}"/>')
    for item in activity:
        name = item["category"]
        if name not in row_y:
            continue
        w = x(item["end"])-x(item["start"])
        lines.append(f'<rect x="{x(item["start"]):.3f}" y="{row_y[name]}" width="{max(.06,w):.3f}" height="23" fill="{COLORS[name]}"/>')
    tspan = (t1-t0)/1e9
    for j in range(6):
        xx = left + usable*j/5
        lines.append(f'<text x="{xx:.2f}" y="336" font-size="11" text-anchor="middle" fill="#596777">{tspan*j/5:.1f}s</text>')
    # Visible, readable summary complements the granular native timeline.
    xx = left
    for name in CATEGORIES:
        pct = values[name.lower().replace(" ","_").replace("/","_").replace("-","_")+"_pct"]
        w = usable*pct/100.0
        if w > 0:
            lines.append(f'<rect x="{xx:.3f}" y="356" width="{w:.3f}" height="21" fill="{COLORS[name]}"/>')
            if w > 65:
                lines.append(f'<text x="{xx+w/2:.3f}" y="370" text-anchor="middle" fill="white" font-size="10">{pct:.1f}%</text>')
        xx += w
    lines.append('<text x="170" y="398" font-size="10" fill="#66788D">'
                 'Device lanes may overlap. Wi-Fi Sync is collective/RPC path, not isolated RF transmission. Waiting/Other is a residual.</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--phone-timeline", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--graph-ids", default="")
    parser.add_argument("--last-graphs", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-clock-uncertainty-ms", type=float, default=20)
    parser.add_argument("--normal-prefill-ms", type=float, default=None)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Diagnostic fallback only; cannot certify Fig.3 bar")
    args = parser.parse_args()
    profile = args.profile_dir.resolve()
    out = (args.out_dir or profile).resolve()
    out.mkdir(parents=True, exist_ok=True)
    chosen, graph_spans, expected_ar = select_graphs(profile, args.graph_ids, args.last_graphs)
    t0, t1 = graph_spans[0][1], graph_spans[-1][2]
    envelope_ms = (t1-t0)/1e6
    graph_ms = sum((b-a)/1e6 for _, a, b in graph_spans)
    gap_ms = envelope_ms - graph_ms
    if gap_ms < -.01:
        raise ValueError("Selected graph coverage exceeds Prefill envelope")

    offset, clock_error_ms, endpoint = calibration(profile, args.endpoint, args.max_clock_uncertainty_ms)
    allreduce = [r for r in read_rows(profile/"allreduce_breakdown.csv") if int(r["graph"]) in chosen]
    if len(allreduce) != expected_ar:
        raise ValueError(f"AllReduce mismatch: {len(allreduce)} != {expected_ar}")
    activity = []
    for r in allreduce:
        activity.append(itv(int(r["start_steady_ns"]), int(r["end_steady_ns"]),
                            "Wi-Fi Sync", "ALLREDUCE", r["tensor"]))
    rpc = read_rows(profile/"rpc_transfer_timeline.csv")
    for r in rpc:
        if r["command"] not in ("GET_TENSOR","SET_TENSOR","SET_TENSOR_HASH","COPY_TENSOR"):
            continue
        if int(r.get("ok","1")) != 1:
            continue
        activity.append(itv(int(r["start_steady_ns"]), int(r["end_steady_ns"]),
                            "Wi-Fi Sync", "PC_RPC", r["command"]))

    phone = read_rows(args.phone_timeline)
    phone = [r for r in phone if r["run_id"] == args.run_id]
    if not phone:
        raise ValueError("Phone timeline does not contain this --run-id; check the RPC Server profile env")
    for r in phone:
        a, b = int(r["compute_start_phone_ns"])-offset, int(r["compute_end_phone_ns"])-offset
        if b > a:
            activity.append(itv(a,b,"Phone Compute","PHONE_NATIVE",r["mode"]))

    # Both backend sidecars are required: never silently interpret the
    # absence of CUDA events as "GPU Compute = 0%".
    missing = []
    gpu_events = 0
    for filename, allow_empty in (("pc_cuda_events.csv", False), ("pc_cpu_events.csv", True)):
        path = profile/filename
        if not path.is_file():
            missing.append(filename)
            continue
        rows = read_rows(path, allow_header_only=allow_empty)
        for r in rows:
            cat = r["category"]
            if cat not in PC_CATEGORIES:
                raise ValueError(f"Bad device event category: {cat}")
            a, b = int(r["start_steady_ns"]), int(r["end_steady_ns"])
            if b <= a:
                continue
            if cat == "GPU Compute" and a < t1 and b > t0:
                gpu_events += 1
            activity.append(itv(a,b,cat,filename,r.get("source",cat)))
    if gpu_events == 0:
        missing.append("GPU Compute intervals within selected Prefill span")
    if missing and not args.allow_incomplete:
        raise ValueError("INCOMPLETE FIG3 PROFILE: "+", ".join(missing)+
                         ". Rebuild the modified PC CUDA/CPU backends BEFORE the next run.")
    activity = [r2 for r in activity if (r2:=clip(r,t0,t1)) is not None]
    if not any(r["category"]=="Phone Compute" for r in activity):
        raise ValueError("No Phone Compute overlaps the selected PC Prefill window; verify Run ID / clock calibration")

    segments, overlap_ns = exclusive_projection(activity,t0,t1)
    totals_ns = defaultdict(int)
    raw_ns = defaultdict(int)
    for seg in segments:
        totals_ns[seg["category"]] += seg["end"]-seg["start"]
    for r in activity:
        raw_ns[r["category"]] += r["end"]-r["start"]
    if sum(totals_ns.values()) != t1-t0:
        raise ValueError("Six-way percentages would not sum to 100%")

    values = {
        "baseline": "Galaxy",
        "graphs": ",".join(map(str,chosen)),
        "allreduce_count": expected_ar,
        "profile_graph_sum_ms": round(graph_ms,3),
        "graph_gap_ms": round(gap_ms,3),
        "profile_envelope_ms": round(envelope_ms,3),
        "normal_prefill_ms": round(args.normal_prefill_ms,3) if args.normal_prefill_ms is not None else "",
        "phone_backend": "Phone Compute (unified CPU/GPU/HTP)",
        "clock_endpoint": endpoint,
        "clock_offset_phone_minus_pc_ns": offset,
        "clock_uncertainty_ms": round(clock_error_ms,4),
        "pc_cuda_activity_count": sum(r["source"]=="pc_cuda_events.csv" for r in activity),
        "pc_cpu_activity_count": sum(r["source"]=="pc_cpu_events.csv" for r in activity),
        "phone_activity_count": sum(r["source"]=="PHONE_NATIVE" for r in activity),
        "overlap_ms": round(overlap_ns/1e6,3),
        "full_coverage": 1,
        "complete_inputs": int(not missing),
        "accounting_method": "exclusive_priority_projection_not_causal_critical_path",
        "wi_fi_definition": "ALLREDUCE_AND_RPC_SYNC_NOT_WIRE_TIME",
        "gpu_definition": "CUDA_STREAM_INTERVAL_NOT_KERNEL_UTILIZATION",
        "waiting_definition": "RESIDUAL_INCLUDING_UNINSTRUMENTED_HOST_WORK",
    }
    for cat in CATEGORIES:
        key = cat.lower().replace(" ","_").replace("/","_").replace("-","_")
        values[key+"_ms"] = round(totals_ns[cat]/1e6,3)
        values[key+"_pct"] = round(100*totals_ns[cat]/(t1-t0),4)
        values[key+"_raw_aggregate_ms"] = round(raw_ns[cat]/1e6,3)
    write_rows(out/"fig3_unified_breakdown.csv",[values],list(values))

    source_rows = []
    for r in sorted(activity,key=lambda z:(z["start"],z["end"])):
        source_rows.append({
            "start_ms":round((r["start"]-t0)/1e6,4),
            "end_ms":round((r["end"]-t0)/1e6,4),
            "duration_ms":round((r["end"]-r["start"])/1e6,4),
            "category":r["category"],"source":r["source"],"label":r["label"]})
    write_rows(out/"fig3_source_events.csv",source_rows,
               ["start_ms","end_ms","duration_ms","category","source","label"])
    line_rows = [{"start_ms":round((s["start"]-t0)/1e6,4),
                  "end_ms":round((s["end"]-t0)/1e6,4),
                  "duration_ms":round((s["end"]-s["start"])/1e6,4),
                  "category":s["category"]} for s in segments]
    write_rows(out/"fig3_unified_timeline.csv",line_rows,
               ["start_ms","end_ms","duration_ms","category"])
    graphs = [{"graph":g,"start_ms":round((a-t0)/1e6,4),
               "end_ms":round((b-t0)/1e6,4),
               "wall_ms":round((b-a)/1e6,4)} for g,a,b in graph_spans]
    write_rows(out/"fig3_graph_coverage.csv",graphs,["graph","start_ms","end_ms","wall_ms"])
    svg_preview(out/"fig3_unified_timeline.svg",activity,segments,t0,t1,graph_spans,values)

    print("[FIG3_COMPLETE] Graphs=%s AR=%d envelope=%.3fms graph_sum=%.3fms gaps=%.3fms" %
          (values["graphs"], expected_ar,envelope_ms,graph_ms,gap_ms))
    print("[FIG3_COMPLETE] PC/Phone clock bound=%.3f ms, Phone event count=%d; complete=%s" %
          (clock_error_ms,values["phone_activity_count"],bool(not missing)))
    for cat in CATEGORIES:
        key=cat.lower().replace(" ","_").replace("/","_").replace("-","_")
        print("  %-18s %10.2f ms (%6.2f%%) | raw-device-aggregate %.2f ms" %
              (cat,values[key+"_ms"],values[key+"_pct"],values[key+"_raw_aggregate_ms"]))
    print("[FIG3_COMPLETE] simultaneous-activity-overlap=%.3f ms" % (overlap_ns/1e6))
    print("[FIG3_COMPLETE] results=%s" % out)
    if missing:
        print("WARNING: INCOMPLETE device sources:",", ".join(missing))
    if args.normal_prefill_ms is None:
        print("WARNING: No unprofiled Prefill total supplied; profile wall is not the normal baseline latency.")
    print("CAUTION: exclusive accounting is an explicit priority projection, not hardware utilization or pure Wi-Fi wire time.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
