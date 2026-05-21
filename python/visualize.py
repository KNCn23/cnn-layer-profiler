#!/usr/bin/env python3
"""CNN Layer Profiler — visualize profile_results.csv"""

import sys
import csv
import argparse

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "layer":  row["layer"].strip(),
                "time_ms": float(row["time_ms"]),
                "memory_kb": float(row["memory_bytes"]) / 1024,
                "flops":  int(row["flops"]),
            })
    return rows

def bar_chart_ascii(rows, key, label, fmt=".3f"):
    values = [r[key] for r in rows]
    max_v  = max(values) if values else 1
    WIDTH  = 40
    print(f"\n  {label}")
    print("  " + "─" * (WIDTH + 35))
    for r in rows:
        bar_len = int(r[key] / max_v * WIDTH) if max_v else 0
        bar = "█" * bar_len
        val_str = f"{r[key]:{fmt}}"
        print(f"  {r['layer']:<26} {bar:<{WIDTH}} {val_str}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Visualize CNN profiler CSV")
    parser.add_argument("csv", nargs="?", default="profile_results.csv")
    args = parser.parse_args()

    try:
        rows = load_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: '{args.csv}' not found — run ./profiler first.")
        sys.exit(1)

    print(f"\nLoaded {len(rows)} layers from {args.csv}\n")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        labels = [r["layer"] for r in rows]
        x = np.arange(len(labels))

        for ax, key, title, color in [
            (axes[0], "time_ms",   "Inference Time (ms)", "#4C72B0"),
            (axes[1], "memory_kb", "Output Memory (KB)",  "#55A868"),
            (axes[2], "flops",     "FLOPs",               "#C44E52"),
        ]:
            vals = [r[key] for r in rows]
            bars = ax.bar(x, vals, color=color, edgecolor="white", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel(title)
            ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("CNN Layer Profiler — Layer-by-Layer Breakdown",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        out = args.csv.replace(".csv", "_plot.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {out}")
        plt.show()

    except ImportError:
        print("matplotlib/numpy not found — falling back to ASCII charts.")
        bar_chart_ascii(rows, "time_ms",   "Inference Time (ms)")
        bar_chart_ascii(rows, "memory_kb", "Output Memory (KB)", fmt=".1f")
        bar_chart_ascii(rows, "flops",     "FLOPs", fmt="d")

if __name__ == "__main__":
    main()
