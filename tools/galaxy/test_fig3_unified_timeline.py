#!/usr/bin/env python3
"""Local synthetic self-test of the FULL Galaxy PC/Phone timeline analyzer.

Does not access model, GPU, Phone or network. This is intended as an early
preflight before expensive 14B inference and catches schema, syntax and
envelope-coverage regressions.
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


def write(path: Path, cols: list[str], rows: list[tuple]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(cols)
        writer.writerows(rows)


def test():
    analyzer = Path(__file__).with_name("build_fig3_unified_timeline.py")
    with tempfile.TemporaryDirectory(prefix="fig3_galaxy_test_") as tmp:
        root = Path(tmp)
        profile = root / "profile"
        profile.mkdir()
        phone = root / "phone.timeline.csv"
        base, phone_offset = 1_000_000_000_000, 500_000_000
        def ms(v):
            return base + v * 1_000_000
        graphs = list(range(3,8))
        spans = [(g, ms(i*25), ms(i*25+20)) for i,g in enumerate(graphs)]
        write(profile/"summary.csv",
              ["graph","all_reduce"], [(g, 1 if g==3 else 0) for g in graphs])
        write(profile/"graph_timeline.csv",
              ["graph","start_steady_ns","end_steady_ns","wall_ms"],
              [(g,a,b,20) for g,a,b in spans])
        write(profile/"clock_sync.csv",
              ["endpoint","offset_phone_minus_pc_ns","uncertainty_ms","rtt_ms"],
              [("phone:50052",phone_offset,0.01,0.03)])
        write(profile/"allreduce_breakdown.csv",
              ["graph","start_steady_ns","end_steady_ns","tensor"],
              [(3,ms(5),ms(15),"x")])
        write(profile/"rpc_transfer_timeline.csv",
              ["command","start_steady_ns","end_steady_ns","ok"],
              [("GET_TENSOR",ms(6),ms(16),1)])
        pc_header = ["category","start_steady_ns","end_steady_ns",
                     "source","device","bytes","device_ms"]
        write(profile/"pc_cuda_events.csv",pc_header,[
            ("GPU Compute",ms(0),ms(60),"CUDA_BACKEND_GRAPH",0,0,60),
            ("PCIe Copy",ms(42),ms(57),"H2D_ASYNC",0,1048576,15),
        ])
        write(profile/"pc_cpu_events.csv",pc_header,[
            ("CPU Compute",ms(60),ms(90),"CPU_BACKEND_GRAPH",-1,0,30)
        ])
        write(phone,
              ["run_id","compute_start_phone_ns","compute_end_phone_ns","mode"],
              [("synthetic-run",ms(10)+phone_offset,ms(30)+phone_offset,"GRAPH_COMPUTE")])
        cmd = [
            sys.executable,str(analyzer),
            "--profile-dir",str(profile),
            "--phone-timeline",str(phone),
            "--run-id","synthetic-run",
            "--normal-prefill-ms","115",
            "--last-graphs","5",
        ]
        result = subprocess.run(cmd,text=True,capture_output=True)
        if result.returncode:
            raise AssertionError(f"Valid full 6-way schema failed:\n{result.stdout}\n{result.stderr}")
        with (profile/"fig3_unified_breakdown.csv").open(newline="",encoding="utf-8-sig") as f:
            vals = next(csv.DictReader(f))
        if vals["complete_inputs"] != "1" or vals["full_coverage"] != "1":
            raise AssertionError("Full-input and full-coverage flags are not set")
        if abs(float(vals["graph_gap_ms"])-20) > 0.01:
            raise AssertionError(f"Inter-Graph gaps lost: {vals['graph_gap_ms']}")
        if abs(float(vals["profile_envelope_ms"])-120) > 0.01:
            raise AssertionError("Envelope and Graph sums are confused")
        names = ["phone_compute","gpu_compute","cpu_compute","pcie_copy","wi_fi_sync","waiting_other"]
        total_pct = sum(float(vals[name+"_pct"]) for name in names)
        if abs(total_pct-100.0) > 0.001:
            raise AssertionError(f"Six categories do not sum to 100: {total_pct}")
        with (profile/"fig3_unified_timeline.csv").open(newline="",encoding="utf-8-sig") as f:
            pieces = list(csv.DictReader(f))
        if abs(float(pieces[0]["start_ms"])) > 1e-5 or \
           abs(float(pieces[-1]["end_ms"])-120) > 1e-5:
            raise AssertionError("Timeline start/end does not cover the full 120 ms")
        for before, after in zip(pieces, pieces[1:]):
            if abs(float(before["end_ms"])-float(after["start_ms"])) > .0002:
                raise AssertionError("Uncovered hole or overlapping exclusive segments")
        (profile/"pc_cuda_events.csv").unlink()
        bad = subprocess.run(cmd,text=True,capture_output=True)
        if bad.returncode == 0 or "INCOMPLETE FIG3 PROFILE" not in bad.stderr+bad.stdout:
            raise AssertionError("Missing CUDA telemetry was not rejected")
        print("[FIG3_TEST] PASS: complete PC/Phone schemas, 100% coverage,"
              " 20ms graph gaps, overlap projection and strict missing-CUDA rejection")


if __name__=="__main__":
    test()
