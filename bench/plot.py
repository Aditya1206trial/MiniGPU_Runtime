"""Plot paraforge_bench CSV: speedup vs threads per size, and speedup vs grain.

Usage: python bench/plot.py [csv] [png]   (defaults: docs/cpu_bench.csv docs/cpu_speedup.png)

Expects rows in the order paraforge_bench prints them: each cpu_parallel row follows the
cpu_single row of the same run, so speedup = that single median / that parallel median.
A config (size, threads, grain) run several times is plotted as the median of its speedups.
"""

import csv
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "docs/cpu_bench.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "docs/cpu_speedup.png"

speedups = defaultdict(list)  # (size, threads, grain) -> speedup per run
single = None
with open(src, newline="") as f:
    for row in csv.DictReader(f):
        ms = float(row["time_ms"])
        if row["impl"] == "cpu_single":
            single = (int(row["size"]), ms)
        else:
            size = int(row["size"])
            assert single and single[0] == size, f"cpu_parallel row without matching cpu_single: {row}"
            speedups[size, int(row["threads"]), int(row["grain"])].append(single[1] / ms)
            single = None
assert speedups, f"no cpu_parallel rows in {src}"
runs = [(*k, statistics.median(v)) for k, v in speedups.items()]

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Thread sweep uses grain 16 only; the grain sweep is the right panel.
sizes = sorted({r[0] for r in runs if r[2] == 16})
max_t = max(r[1] for r in runs)
ax1.plot([1, max_t], [1, max_t], ls=":", color="#9a9893", lw=1, label="ideal")
for color, n in zip(COLORS, sizes):
    pts = sorted((t, s) for size, t, g, s in runs if size == n and g == 16)
    ax1.plot(*zip(*pts), marker="o", ms=5, lw=2, color=color, label=f"{n}x{n}")
ax1.set(xlabel="pool threads", ylabel="speedup vs cpu_single", title="Speedup vs threads (grain 16)")
ax1.set_xticks(sorted({r[1] for r in runs}))

g_size, g_threads = 512, max_t
pts = sorted((g, s) for size, t, g, s in runs if size == g_size and t == g_threads)
ax2.plot(*zip(*pts), marker="o", ms=5, lw=2, color=COLORS[0])
ax2.set_xscale("log", base=2)
ax2.set_xticks([p[0] for p in pts], [str(p[0]) for p in pts])
ax2.set(xlabel="grain (rows per task)", ylabel="speedup vs cpu_single",
        title=f"Speedup vs grain ({g_size}x{g_size}, {g_threads} threads)")

for ax in (ax1, ax2):
    ax.grid(color="#e4e3df", lw=0.8)
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
ax1.legend(frameon=False)
fig.tight_layout()
fig.savefig(dst, dpi=120)
print(f"wrote {dst}: {len(runs)} configs, {sum(map(len, speedups.values()))} runs")
