#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("Warning: tabulate not available. Install with: pip install tabulate", file=sys.stderr)


def find_latest_yali_csv(results_root: Path) -> Optional[Path]:
    candidates = list(results_root.rglob("yali_sweep.csv"))
    if not candidates:
        return None
    # Pick latest by modification time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_yali_csv(csv_path: Path) -> Dict[int, Dict[str, Optional[float]]]:
    """Return mapping: size_bytes -> dict with best row metrics."""
    best: Dict[int, Dict[str, Optional[float]]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                size_bytes = int(row.get("actual_bytes") or row.get("requested_bytes") or 0)
                lanes = int(row.get("lanes") or 0)
                busbw = row.get("bus_bw_gbps")
                if busbw is None or busbw == "":
                    continue
                bw = float(busbw)
                lat = row.get("time_us")
                latency = float(lat) if lat not in (None, "") else None
            except Exception:
                continue
            cur = best.get(size_bytes)
            if cur is None or bw > (cur.get("bw") or -1.0):
                best[size_bytes] = {
                    "bw": bw,
                    "lanes": lanes,
                    "lat": latency,
                    "nvlink_cap": float(row["nvlink_cap_gbps"]) if row.get("nvlink_cap_gbps") else None,
                    "nvlink_lane_cap": float(row["nvlink_lane_cap_gbps"]) if row.get("nvlink_lane_cap_gbps") else None,
                    "nvlink_util": float(row["nvlink_util_percent"]) if row.get("nvlink_util_percent") else None,
                }
    return best


def parse_nccl_log(log_path: Path) -> Dict[int, Tuple[float, float]]:
    """Return mapping: size_bytes -> (best_busbw_gbps, best_latency_us).

    We take the maximum busbw across out-of-place/in-place, and the minimum
    latency across out-of-place/in-place for each size.

    Columns per row after split:
      size(B) count type redop root time_o algbw_o busbw_o #wrong time_i algbw_i busbw_i #wrong
    Indices: size=0, time_o=5, busbw_o=7, time_i=9, busbw_i=11
    """
    out_idx_time = 5
    out_idx_bus = 7
    in_idx_time = 9
    in_idx_bus = 11
    size_to_vals: Dict[int, Tuple[float, float]] = {}
    num_re = re.compile(r"^\s*(\d+)\s+")
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not num_re.match(line):
                continue
            parts = line.split()
            try:
                size_bytes = int(parts[0])
                out_busbw = float(parts[out_idx_bus]) if len(parts) > out_idx_bus else 0.0
                in_busbw = float(parts[in_idx_bus]) if len(parts) > in_idx_bus else 0.0
                out_time = float(parts[out_idx_time]) if len(parts) > out_idx_time else 0.0
                in_time = float(parts[in_idx_time]) if len(parts) > in_idx_time else 0.0
                best_bw = max(out_busbw, in_busbw)
                best_lat = min(x for x in (out_time, in_time) if x > 0.0) if any(x > 0.0 for x in (out_time, in_time)) else 0.0
                if best_bw > 0.0:
                    size_to_vals[size_bytes] = (best_bw, best_lat)
            except Exception:
                continue
    return size_to_vals


def verify_math(nccl_bw: float, nccl_lat: float, my_bw: float, my_lat: Optional[float], size_bytes: int) -> Tuple[float, float, float, float]:
    """Verify and compute ratios with explicit checks.
    
    Returns: (bw_ratio, bw_impr, lat_ratio, lat_impr)
    """
    # Bandwidth: higher is better
    # Ratio = Yali / NCCL (how many times better Yali is)
    if nccl_bw <= 0:
        bw_ratio = 0.0
    else:
        bw_ratio = my_bw / nccl_bw
    bw_impr = (bw_ratio - 1.0) * 100.0
    
    # Latency: lower is better
    # Ratio = NCCL / Yali (speedup: how many times faster Yali is)
    # If NCCL takes 100µs and Yali takes 50µs, ratio = 100/50 = 2.0x (Yali is 2x faster)
    if my_lat is None or my_lat <= 0:
        lat_ratio = 0.0
    else:
        if nccl_lat <= 0:
            lat_ratio = 0.0
        else:
            lat_ratio = nccl_lat / my_lat
    lat_impr = (lat_ratio - 1.0) * 100.0
    
    # Sanity check: bandwidth and latency should be inversely related
    # If bandwidth is X times better, latency should be roughly X times better (for same size)
    if my_lat and my_lat > 0 and nccl_lat > 0:
        expected_bw_ratio = (nccl_lat / my_lat)  # Expected BW ratio from latency
        actual_bw_ratio = bw_ratio
        # Allow 10% tolerance due to measurement noise
        if abs(expected_bw_ratio - actual_bw_ratio) / max(expected_bw_ratio, actual_bw_ratio) > 0.15:
            print(f"Warning: Size {size_bytes} B - BW ratio {actual_bw_ratio:.2f}x doesn't match latency ratio {expected_bw_ratio:.2f}x", file=sys.stderr)
    
    return bw_ratio, bw_impr, lat_ratio, lat_impr


def write_consolidated(out_path: Path, nccl: Dict[int, Tuple[float, float]], yali: Dict[int, Tuple[float, int, Optional[float]]], print_table: bool = False, md_out: Optional[Path] = None):
    common_sizes = sorted(set(nccl.keys()) & set(yali.keys()))
    rows: List[List] = []
    
    for size in common_sizes:
        nccl_bw, nccl_lat = nccl[size]
        my_info = yali[size]
        my_bw = my_info["bw"]
        lanes = int(my_info["lanes"])
        my_lat = my_info["lat"]
        my_cap = my_info.get("nvlink_cap")
        my_lane_cap = my_info.get("nvlink_lane_cap")
        my_util = my_info.get("nvlink_util")
        
        # Verify and compute ratios
        bw_ratio, bw_impr, lat_ratio, lat_impr = verify_math(nccl_bw, nccl_lat, my_bw, my_lat, size)
        
        rows.append([
            size,
            f"{size/1024/1024:.2f}",
            f"{nccl_lat:.2f}",
            f"{my_lat:.2f}" if my_lat is not None else "",
            f"{lat_ratio:.4f}",
            f"{lat_impr:.2f}",
            f"{nccl_bw:.4f}",
            f"{my_bw:.4f}",
            f"{bw_ratio:.4f}",
            f"{bw_impr:.2f}",
            lanes,
            f"{my_cap:.2f}" if my_cap else "",
            f"{my_lane_cap:.2f}" if my_lane_cap else "",
            f"{my_util:.2f}" if my_util is not None else "",
        ])
    
    # Write CSV
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "size_bytes",
            "size_mib",
            # Latency metrics
            "nccl_latency_us",
            "yali_latency_us",
            "latency_ratio_yali_vs_nccl",  # defined as NCCL / Yali (speedup)
            "latency_improvement_percent",
            # Bandwidth metrics
            "nccl_busbw_gbps",
            "yali_busbw_gbps",
            "bw_ratio_yali_over_nccl",
            "bw_improvement_percent",
            # Configuration
            "yali_lanes",
            "yali_nvlink_cap_gbps",
            "yali_nvlink_lane_cap_gbps",
            "yali_nvlink_util_percent",
        ])
        writer.writerows(rows)
    
    # Print formatted table if requested
    if print_table and HAS_TABULATE:
        table_data = []
        for row in rows:
            size_mib = row[1]
            nccl_lat = row[2]
            my_lat = row[3]
            lat_ratio = float(row[4])
            lat_impr = float(row[5])
            nccl_bw = row[6]
            my_bw = row[7]
            bw_ratio = float(row[8])
            bw_impr = float(row[9])
            lanes = row[10]
            cap = row[11] if len(row) > 11 else ""
            lane_cap = row[12] if len(row) > 12 else ""
            util = row[13] if len(row) > 13 else ""
            
            # Format ratios with × symbol and percentage
            lat_str = f"{lat_ratio:.2f}× ({lat_impr:+.1f}%)" if lat_ratio > 0 else "N/A"
            bw_str = f"{bw_ratio:.2f}× ({bw_impr:+.1f}%)" if bw_ratio > 0 else "N/A"
            
            table_data.append([
                size_mib,
                f"{nccl_lat}",
                f"{my_lat}" if my_lat else "N/A",
                lat_str,
                f"{nccl_bw}",
                f"{my_bw}",
                bw_str,
                lanes,
                cap if cap else "N/A",
                lane_cap if lane_cap else "N/A",
                util if util else "N/A",
            ])
        
        headers = [
            "Size (MiB)",
            "NCCL Lat (µs)",
            "Yali Lat (µs)",
            "Latency",
            "NCCL BW (GB/s)",
            "Yali BW (GB/s)",
            "Bandwidth",
            "Lanes",
            "NVLink Cap (GB/s)",
            "Lane Cap (GB/s)",
            "NVLink Util (%)",
        ]
        
        table_str = tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f")
        
        # Print to stdout if requested
        if print_table:
            print("\n" + "="*100)
            print("Yali vs NCCL Performance Comparison")
            print("="*100)
            print(table_str)
            print("\nNotes:")
            print("  - Latency ratio = NCCL / Yali (speedup: higher is better)")
            print("  - Bandwidth ratio = Yali / NCCL (higher is better)")
            print("  - Improvement % = (ratio - 1) × 100")
            print("="*100 + "\n")
        
        # Write to markdown file if requested
        if md_out:
            md_out.parent.mkdir(parents=True, exist_ok=True)
            with md_out.open("w", encoding="utf-8") as f:
                f.write("# Yali vs NCCL Performance Comparison\n\n")
                f.write(table_str + "\n\n")
                f.write("## Notes\n\n")
                f.write("- **Latency ratio** = NCCL / Yali (speedup: higher is better)\n")
                f.write("- **Bandwidth ratio** = Yali / NCCL (higher is better)\n")
                f.write("- **Improvement %** = (ratio - 1) × 100\n\n")
                f.write("### Why Latency and Bandwidth Ratios Are The Same\n\n")
                f.write("For the **same data size**, latency and bandwidth improvements are mathematically identical:\n\n")
                f.write("```\n")
                f.write("Bandwidth = Size / Time\n\n")
                f.write("Bandwidth Ratio = Yali_BW / NCCL_BW\n")
                f.write("                = (Size / Yali_Time) / (Size / NCCL_Time)\n")
                f.write("                = NCCL_Time / Yali_Time\n")
                f.write("                = Latency Ratio\n")
                f.write("```\n\n")
                f.write("**Key insight**: When transferring the same amount of data, if you reduce latency by factor X, bandwidth increases by factor X. This is because bandwidth is inversely proportional to time for a fixed size.\n\n")
                f.write("Small differences (e.g., 94.7% vs 94.8%) are due to rounding and measurement precision.\n\n")
                f.write("---\n\n")
                f.write(f"*Generated from: {out_path.name}*\n")
            print(f"Wrote formatted table to {md_out}")
    elif print_table:
        print("Warning: tabulate not available. Install with: pip install tabulate", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Consolidate Yali vs NCCL perf into CSV")
    ap.add_argument("--results-root", required=True, help="Directory containing yali_sweep.csv")
    ap.add_argument("--nccl-log", required=True, help="Path to NCCL baseline log (stdout captured)")
    ap.add_argument("--out", required=True, help="Output consolidated CSV path")
    ap.add_argument("--yali-csv", help="Explicit path to yali_sweep.csv (optional)")
    ap.add_argument("--print-table", action="store_true", help="Print formatted table to stdout")
    ap.add_argument("--md-out", help="Write formatted table to markdown file (optional)")
    args = ap.parse_args(argv)

    root = Path(args.results_root)
    nccl_log = Path(args.nccl_log)
    out_path = Path(args.out)
    yali_csv = Path(args.yali_csv) if args.yali_csv else find_latest_yali_csv(root)

    if not yali_csv or not yali_csv.exists():
        print(f"error: yali_sweep.csv not found under {root}", file=sys.stderr)
        return 2
    if not nccl_log.exists():
        print(f"error: NCCL log not found: {nccl_log}", file=sys.stderr)
        return 2

    yali = parse_yali_csv(yali_csv)
    nccl = parse_nccl_log(nccl_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md_out = Path(args.md_out) if args.md_out else None
    write_consolidated(out_path, nccl, yali, print_table=args.print_table, md_out=md_out)
    print(f"Wrote consolidated CSV to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
