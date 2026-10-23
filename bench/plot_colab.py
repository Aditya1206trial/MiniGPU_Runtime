"""Bar chart of the four 512x512 methods from one Colab machine (CPU and GPU CSVs).

Usage: python bench/plot_colab.py [cpu_csv] [cuda_csv] [png]
(defaults: docs/colab_t4_cpu_bench.csv docs/colab_t4_cuda_bench.csv docs/colab_t4_512.png)

CPU bars are paraforge_bench time_ms (median wall time of the multiply); GPU bars are
paraforge_cuda_bench end_to_end_ms (median wall time of the whole call: alloc, H2D, kernel,
D2H, free). Kernel-only GPU time is not plotted.
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt

cpu_src = sys.argv[1] if len(sys.argv) > 1 else "docs/colab_t4_cpu_bench.csv"
gpu_src = sys.argv[2] if len(sys.argv) > 2 else "docs/colab_t4_cuda_bench.csv"
dst = sys.argv[3] if len(sys.argv) > 3 else "docs/colab_t4_512.png"
SIZE = 512

with open(cpu_src, newline="") as f:
    cpu = {r["impl"]: r for r in csv.DictReader(f) if int(r["size"]) == SIZE}
with open(gpu_src, newline="") as f:
    gpu = {r["impl"]: r for r in csv.DictReader(f) if (int(r["m"]), int(r["k"]), int(r["n"])) == (SIZE,) * 3}
assert set(cpu) == {"cpu_single", "cpu_parallel"}, f"need one cpu_single and one cpu_parallel {SIZE} row in {cpu_src}"
assert set(gpu) == {"cuda_naive", "cuda_tiled"}, f"need one cuda_naive and one cuda_tiled {SIZE}^3 row in {gpu_src}"

bars = [  # (label, ms, is_gpu)
    ("cpu_single\n(1 thread)", float(cpu["cpu_single"]["time_ms"]), False),
    (f"cpu_parallel\n({cpu['cpu_parallel']['threads']} threads, grain {cpu['cpu_parallel']['grain']})",
     float(cpu["cpu_parallel"]["time_ms"]), False),
    ("cuda_naive\n(end-to-end)", float(gpu["cuda_naive"]["end_to_end_ms"]), True),
    ("cuda_tiled\n(end-to-end)", float(gpu["cuda_tiled"]["end_to_end_ms"]), True),
]
COLORS = {False: "#2a78d6", True: "#eb6834"}  # same order as bench/plot.py

fig, ax = plt.subplots(figsize=(8, 4.2))
y = range(len(bars))[::-1]
ax.barh(y, [b[1] for b in bars], color=[COLORS[b[2]] for b in bars], height=0.6)
for yi, (_, ms, _) in zip(y, bars):
    ax.text(ms * 1.08, yi, f"{ms:g} ms", va="center", color="#3d3c38")
ax.set_yticks(list(y), [b[0] for b in bars])
ax.set_xscale("log")  # ~100x spread
ax.set_xlim(1, 1000)
ax.set(xlabel="median of 3 repeats, ms (log scale; lower is better)",
       title=f"{SIZE}x{SIZE} matmul on one Colab VM (Tesla T4), one CPU and one GPU CLI run")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[False]),
                   plt.Rectangle((0, 0), 1, 1, color=COLORS[True])],
          labels=["CPU wall time (naive i,j,k kernel)", "GPU end-to-end wall time (alloc + copies + kernel + free)"],
          frameon=False, loc="lower right", fontsize=8)
ax.grid(axis="x", color="#e4e3df", lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(dst, dpi=120)
print(f"wrote {dst}: " + ", ".join(f"{b[0].split(chr(10))[0]}={b[1]:g} ms" for b in bars))
