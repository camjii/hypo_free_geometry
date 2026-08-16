"""Robustness analysis: does each concept geometry survive 10 paraphrases?"""

# ruff: noqa: E402  (sys.path setup must precede sibling imports)

import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore", RuntimeWarning)  # spurious OpenBLAS matmul FPE flags

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from concept_families import FAMILIES

VAR = REPO / "outputs/activations/gemma-7b/variants"
FIG = REPO / "outputs/figures"
FIG.mkdir(parents=True, exist_ok=True)

EXPECTED = {f["name"]: f["expected"] for f in FAMILIES}
EXPECTED.update(
    {
        "days": "weekday circle (S1)",
        "months": "calendar circle (S1)",
        "chess_squares": "8x8 grid (2D)",
        "calendar_dates": "year cycle / torus (S1)",
    }
)


def is_cyclic(exp):
    e = exp.lower()
    return "s1" in e or "circle" in e or "circumplex" in e


def pca2(X):
    p = PCA(n_components=2, svd_solver="full").fit(X)
    return p.transform(X), p.explained_variance_ratio_


def score(Y, expected):
    """1.0 = points sit in exactly the expected order in the 2D projection."""
    n = len(Y)
    if is_cyclic(expected):
        ang = np.arctan2(Y[:, 1] - Y[:, 1].mean(), Y[:, 0] - Y[:, 0].mean())
        order = np.argsort(ang)
        steps = np.diff(np.concatenate([order, order[:1]])) % n
        return float(np.mean((steps == 1) | (steps == n - 1)))
    idx = np.arange(n)
    return float(max(abs(spearmanr(idx, Y[:, 0])[0]), abs(spearmanr(idx, Y[:, 1])[0])))


rows = []
for set_name, expected in EXPECTED.items():
    files = sorted(VAR.glob(f"{set_name}__*_layer_6.npz"))
    if not files:
        continue
    scores, evrs = [], []
    for f in files:
        d = np.load(f, allow_pickle=False)
        Y, evr = pca2(d["activations"].astype(np.float64))
        scores.append(score(Y, expected))
        evrs.append(float(evr[:2].sum()))
    rows.append(
        {
            "family": set_name,
            "expected": expected,
            "n_variants": len(files),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean_var_2d": float(np.mean(evrs)),
            "scores": scores,
        }
    )

rows.sort(key=lambda r: -r["mean"])
(FIG.parent / "variant_robustness.json").write_text(json.dumps(rows, indent=2) + "\n")

print(
    f"{'family':<17}{'expected':<26}{'n':>3}  {'mean':>5} {'sd':>5} {'min':>5} {'max':>5}"
)
for r in rows:
    flag = (
        "  <-- robust"
        if r["mean"] >= 0.8
        else ("  ~ partial" if r["mean"] >= 0.6 else "")
    )
    print(
        f"{r['family']:<17}{r['expected']:<26}{r['n_variants']:>3}  "
        f"{r['mean']:5.2f} {r['std']:5.2f} {r['min']:5.2f} {r['max']:5.2f}{flag}"
    )

# ---- figure 1: robustness strip plot ----
fig, ax = plt.subplots(figsize=(11, 7))
for i, r in enumerate(rows):
    ax.scatter(
        r["scores"], [i] * len(r["scores"]), s=45, alpha=0.55, color="#4C78A8", zorder=3
    )
    ax.scatter([r["mean"]], [i], s=150, marker="|", color="#D62728", zorder=4)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{r['family']}  ({r['expected']})" for r in rows], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(-0.02, 1.05)
ax.axvline(0.8, ls="--", lw=1, color="#888")
ax.set_xlabel("structure-recovery score in 2D PCA (1.0 = exact expected order)")
ax.set_title(
    "Does the geometry survive rewording?\n"
    "each dot = one of 10 paraphrases · red bar = mean",
    fontweight="bold",
)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "variant_robustness.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- figure 2: months across all 10 wordings ----
files = sorted(VAR.glob("months__*_layer_6.npz"))
if files:
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    for ax, f in zip(axes.flat, files):
        d = np.load(f, allow_pickle=False)
        labels = [str(x) for x in d["labels"]]
        Y, evr = pca2(d["activations"].astype(np.float64))
        colors = cm.twilight(np.linspace(0, 1, len(Y), endpoint=False))
        idx = list(range(len(Y))) + [0]
        ax.plot(Y[idx, 0], Y[idx, 1], "-", color="#C4C9D2", lw=1.1, zorder=1)
        ax.scatter(
            Y[:, 0], Y[:, 1], c=colors, s=55, zorder=3, edgecolor="white", lw=0.8
        )
        for (x, y), lab in zip(Y, labels):
            ax.annotate(
                lab[:3], (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7
            )
        name = f.name.split("__")[1].replace("_layer_6.npz", "")
        ax.set_title(
            f"{name} · {score(Y, 'circle'):.2f}\n{evr[:2].sum() * 100:.0f}% in 2D",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(
        "Months: the same circle under 10 different sentences "
        "(Gemma-7b layer 6, concept token)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG / "months_across_wordings.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

print(
    "\nwrote variant_robustness.png, months_across_wordings.png, "
    "variant_robustness.json"
)
