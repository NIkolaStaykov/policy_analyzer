#!/usr/bin/env python3
"""Violin plots of per-finger contact-force distributions, one panel per policy.

Reads the persisted datasets
    fingertip_forces_downwards_rotate.csv   (fingers: tip1..tip5)
    fingertip_forces_pinch_sinusoid.csv     (fingers: thumb, index)
and renders a 2x2 grid (baseline / proprio.target / proprio.delta / force.magnitude).
Each violin is the KDE of that finger's contact force pooled over all rollout_seeds
and steps; markers show median (o), mean (diamond) and the p5-p95 range; the % label
is the fraction of steps in contact (|f| > 0.05 N).

Usage:
    python plot_fingertip_force_violins.py <task> <in.csv> <out.png>
        task = downwards | pinch
"""
import sys, csv, collections, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]   # dataviz categorical, fixed order
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e7e7e3"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "svg.fonttype": "none"})

BUNDLE_ORDER = ["baseline", "proprio.target", "proprio.delta", "force.magnitude"]
TASKCFG = {
    "downwards": dict(cols=["f_tip1","f_tip2","f_tip3","f_tip4","f_tip5"],
                      labels=["tip 1","tip 2","tip 3","tip 4","tip 5"], ylim=6.6,
                      title="Per-fingertip contact-force distribution by policy — DownwardsRotateZ"),
    "pinch":     dict(cols=["f_thumb","f_index"], labels=["thumb","index"], ylim=None,
                      title="Per-finger contact-force distribution by policy — Pinch (sinusoid)"),
}

task, in_csv, out_png = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = TASKCFG[task]

# load: bundle -> list of per-finger arrays
rows = collections.defaultdict(lambda: collections.defaultdict(list))
seeds = collections.defaultdict(set)
with open(in_csv) as f:
    for r in csv.DictReader(f):
        b = r["sensor_bundle"]; seeds[b].add(r["seed"])
        for c in cfg["cols"]:
            rows[b][c].append(float(r[c]))
data = {b: {c: np.array(v) for c, v in d.items()} for b, d in rows.items()}
F = len(cfg["cols"])
ymax = cfg["ylim"] or max(np.percentile(np.concatenate(list(d.values())), 99.5)
                          for d in data.values()) * 1.12

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(cfg["title"] + "   (best seed per bundle, 8 rollouts)", fontsize=12.5,
             fontweight="bold", color=INK, y=0.99)
for ax, b in zip(axes.ravel(), BUNDLE_ORDER):
    series = [data[b][c] for c in cfg["cols"]]
    parts = ax.violinplot(series, positions=range(F), showextrema=False, widths=0.82)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(CAT[i]); body.set_edgecolor(CAT[i]); body.set_alpha(0.45); body.set_linewidth(0)
    for i, x in enumerate(series):
        med, p95, mean = np.median(x), np.percentile(x, 95), x.mean()
        ax.vlines(i, np.percentile(x, 5), p95, color=CAT[i], lw=2, zorder=3)
        ax.plot(i, med, "o", ms=7, color=CAT[i], mec=SURF, mew=1.5, zorder=4)
        ax.plot(i, mean, "D", ms=5, color="white", mec=CAT[i], mew=1.6, zorder=4)
        ax.text(i, min(p95 + 0.15, ymax * 0.98), f"{100*np.mean(x>0.05):.0f}%",
                ha="center", va="bottom", fontsize=8, color=INK2)
    total = np.sum([s.mean() for s in series])
    ax.set_title(f"{b}   (seed {sorted(seeds[b])[0]})   — mean total {total:.1f} N",
                 color=INK, fontsize=11, loc="left", pad=7)
    ax.set_xticks(range(F)); ax.set_xticklabels(cfg["labels"])
    ax.set_ylabel("contact force |f|  (N)"); ax.set_ylim(-0.3, ymax)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes.ravel()[0].text(0.02, 0.97, "median o   mean <>   | p5-p95   %=in contact",
                     transform=axes.ravel()[0].transAxes, fontsize=8, color=MUTED, va="top")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=SURF)
print("wrote", out_png)
