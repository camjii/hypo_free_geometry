"""Render a PCA gallery of every extracted concept family + labeled per-family PNGs."""

# ruff: noqa: E402  (sys.path setup must precede sibling imports)

import warnings
from pathlib import Path

warnings.simplefilter("ignore", RuntimeWarning)  # spurious OpenBLAS matmul FPE flags

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from concept_families import FAMILIES, EXISTING

ACT = REPO / "outputs/activations/gemma-7b"
FIG = REPO / "outputs/figures"
CONC = FIG / "concepts"
CONC.mkdir(parents=True, exist_ok=True)

# Order in which panels appear: existing validated sets first, then new families.
PANELS = [
    {
        "file": e["name"],
        "name": e["name"].replace("_layer_6", "").replace("-", " "),
        "expected": e["expected"],
    }
    for e in EXISTING
] + [
    {"file": f"{f['name']}_layer_6", "name": f["name"], "expected": f["expected"]}
    for f in FAMILIES
]


def wants_path(expected):
    e = expected.lower()
    return any(k in e for k in ("s1", "circle", "line", "poset"))


def is_cyclic(expected):
    e = expected.lower()
    return "s1" in e or "circle" in e or "circumplex" in e


def load(file):
    d = np.load(ACT / f"{file}.npz", allow_pickle=False)
    return d["activations"].astype(np.float64), [str(x) for x in d["labels"]]


def pca2(X):
    p = PCA(n_components=2, svd_solver="full").fit(X)
    return p.transform(X), p.explained_variance_ratio_


def draw(ax, Y, labels, expected, annotate):
    n = len(Y)
    colors = cm.viridis(np.linspace(0, 1, n))
    if wants_path(expected):
        idx = list(range(n)) + ([0] if is_cyclic(expected) else [])
        ax.plot(Y[idx, 0], Y[idx, 1], "-", color="#C4C9D2", lw=1.1, zorder=1)
    ax.scatter(
        Y[:, 0],
        Y[:, 1],
        c=colors,
        s=70 if annotate else 40,
        zorder=3,
        edgecolor="white",
        lw=0.6,
    )
    if annotate:
        for (x, y), lab in zip(Y, labels):
            ax.annotate(
                lab, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8
            )
    ax.set_aspect("equal", adjustable="datalim")


# ---- individual labeled PNGs ----
for p in PANELS:
    X, labels = load(p["file"])
    Y, evr = pca2(X)
    fig, ax = plt.subplots(figsize=(6, 6))
    draw(ax, Y, labels, p["expected"], annotate=True)
    ax.set_title(f"{p['name']}  —  expected: {p['expected']}", fontweight="bold")
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    fig.tight_layout()
    fig.savefig(CONC / f"{p['name']}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

# ---- one gallery grid ----
cols = 5
rows = -(-len(PANELS) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
for ax, p in zip(axes.flat, PANELS):
    X, labels = load(p["file"])
    Y, evr = pca2(X)
    draw(ax, Y, labels, p["expected"], annotate=False)
    ax.set_title(
        f"{p['name']}\n{p['expected']} · {evr[:2].sum() * 100:.0f}% in 2D", fontsize=9
    )
    ax.set_xticks([])
    ax.set_yticks([])
for ax in axes.flat[len(PANELS) :]:
    ax.axis("off")
fig.suptitle(
    "Gemma-7b layer-6 activations: concept geometry vs. expected ground truth",
    fontsize=15,
    fontweight="bold",
    y=1.005,
)
fig.tight_layout()
fig.savefig(FIG / "concept_gallery.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote gallery + %d individual PNGs" % len(PANELS))
