# Benchmark Artifacts - 2026-01-15

Platform: 2x NVIDIA A100-SXM4-80GB (NVLink NV2)

## Available Reports

| Sweep Mode | Dtypes | Sizes | Report |
|:-----------|:-------|:------|:-------|
| Standard   | FP32, FP16, BF16 | 8 sizes (256B - 2GB) | [summary.md](151357-standard/summary.md) |
| Extensive  | FP32, FP16, BF16 | 15 sizes (256B - 2GB) | [summary.md](152933-extensive/summary.md) |
| Quick + Profiler | FP32 | 5 sizes + nsys profiling | [summary.md](164632-quick-profiler/summary.md) |

## Key Results

```
Peak Performance @ 2GB (FP32):
  YALI:  44.0 GB/s (94% SoL)
  NCCL:  37.3 GB/s (79% SoL)
  Speedup: 1.18x (+18%)
```

## Profiler Insights

The profiler sweep captures kernel-level timing via NVIDIA Nsight Systems:

- **Effective Bandwidth**: Fair comparison using wall clock time
- **Per-Kernel Duration**: Shown only for comparable kernel counts

See [164632-quick-profiler/profiler/](164632-quick-profiler/profiler/) for:
- `effective_bandwidth.png` - GB/s comparison
- `kernel_duration_comparison.png` - Per-kernel timing
- `profiler_summary.json` - Raw data

## Reproduce

```bash
# Standard sweep
python scripts/sweep.py --standard

# Extensive sweep
python scripts/sweep.py --extensive

# Quick sweep with profiler
python scripts/sweep.py --quick --profiler
```
