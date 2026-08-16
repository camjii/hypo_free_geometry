"""Cross-model comparison: does each concept geometry replicate across models?"""

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
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from concept_families import FAMILIES

ACT = REPO / "outputs/activations"
FIG = REPO / "outputs/figures"
MODELS = [
    ("gemma-7b", "Gemma-7B"),
    ("gemma-2-2b", "Gemma-2-2B"),
    ("qwen3-8b", "Qwen3-8B"),
]

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


def score(X, expected):
    Y = PCA(n_components=2, svd_solver="full").fit_transform(X)
    n = len(Y)
    if is_cyclic(expected):
        ang = np.arctan2(Y[:, 1] - Y[:, 1].mean(), Y[:, 0] - Y[:, 0].mean())
        order = np.argsort(ang)
        steps = np.diff(np.concatenate([order, order[:1]])) % n
        return float(np.mean((steps == 1) | (steps == n - 1)))
    idx = np.arange(n)
    return float(max(abs(spearmanr(idx, Y[:, 0])[0]), abs(spearmanr(idx, Y[:, 1])[0])))


table = {}
for slug, label in MODELS:
    var = ACT / slug / "variants"
    if not var.exists():
        print(f"skip {label}: no variants directory")
        continue
    per_family = {}
    for fam, expected in EXPECTED.items():
        files = sorted(var.glob(f"{fam}__*_layer_6.npz"))
        if not files:
            continue
        s = [
            score(
                np.load(f, allow_pickle=False)["activations"].astype(np.float64),
                expected,
            )
            for f in files
        ]
        per_family[fam] = {
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
            "n": len(s),
        }
    table[label] = per_family

common = sorted(
    set.intersection(*(set(v) for v in table.values())),
    key=lambda f: -table[MODELS[0][1]][f]["mean"],
)

print(f"{'family':<17}{'expected':<26}" + "".join(f"{lab:>16}" for _, lab in MODELS))
rows = []
for fam in common:
    line = f"{fam:<17}{EXPECTED[fam]:<26}"
    vals = []
    for _, lab in MODELS:
        m, s = table[lab][fam]["mean"], table[lab][fam]["std"]
        vals.append(m)
        line += f"{m:>9.2f}±{s:.2f}"
    rows.append((fam, vals))
    both = all(v >= 0.8 for v in vals)
    line += "   <-- replicates" if both else ""
    print(line)

(REPO / "outputs/model_comparison.json").write_text(json.dumps(table, indent=2) + "\n")

# ---- grouped bar chart ----
fig, ax = plt.subplots(figsize=(12, 7.5))
y = np.arange(len(rows))
h = 0.27
colors = ["#4C78A8", "#F58518", "#54A24B"]
for k, (_, lab) in enumerate(MODELS):
    means = [table[lab][f]["mean"] for f, _ in rows]
    errs = [table[lab][f]["std"] for f, _ in rows]
    ax.barh(
        y + (k - 1) * h,
        means,
        height=h,
        xerr=errs,
        label=lab,
        color=colors[k],
        error_kw={"lw": 1, "ecolor": "#555"},
    )
ax.set_yticks(y)
ax.set_yticklabels([f"{f}  ({EXPECTED[f]})" for f, _ in rows], fontsize=9)
ax.invert_yaxis()
ax.axvline(0.8, ls="--", lw=1, color="#666")
ax.set_xlim(0, 1.05)
ax.set_xlabel("structure-recovery score (mean over 10 paraphrases, ± sd)")
ax.set_title(
    "Concept geometry across models — layer 6, concept token", fontweight="bold"
)
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "model_comparison.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("\nwrote model_comparison.png + model_comparison.json")
