"""Validate Karkada-style month geometry against the fixed-spectrum null.

The primary analysis is the contextualized Gemma-2-2B residual stream at a
pre-specified layer. All-layer results are exploratory and are not used to
select the primary layer. Raw metrics are saved before plots are rendered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hypo-free-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA

from null_cloud import (
    Manifold,
    ManifoldComparator,
    empirical_pvalue,
    fit_low_rank_gaussian,
    sample_low_rank_gaussian,
)


MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
DEFAULT_CONTEXTUALIZED = Path(
    "ground_truth_verification/months_contextualized_reps_by_layer.npy"
)
DEFAULT_BARE = Path("ground_truth_verification/months_reps_by_layer.npy")
DEFAULT_OUTPUT = Path("outputs/month_circle_validation")
PRIMARY_METRICS = (
    "fourier_alignment",
    "circulant_r2",
    "neighbor_gap",
    "h1_max_persistence",
)
NULL_CLOUD_METRICS = (
    "H0_wasserstein",
    "H0_bottleneck",
    "H1_wasserstein",
    "H1_bottleneck",
)


class TopologyPipeline:
    """Small measurement adapter that exercises null_cloud's real topology path."""

    def reduce_pca(self, cloud: np.ndarray, var_threshold: float = 0.95) -> np.ndarray:
        pca = PCA(n_components=min(cloud.shape))
        full = pca.fit_transform(cloud)
        count = int(
            np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_threshold) + 1
        )
        return full[:, :count]

    def create_persistence_diagram(self, projected: np.ndarray) -> dict[str, Any]:
        return ripser(projected, maxdim=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _month_index(label: str) -> int:
    lowered = str(label).lower()
    for index, month in enumerate(MONTHS):
        if month.lower() in lowered:
            return index
    raise ValueError(f"could not identify a month in label {label!r}")


def load_month_layers(path: str | Path) -> dict[str, np.ndarray]:
    """Load an activation dictionary and enforce canonical calendar ordering."""
    source = Path(path)
    raw = np.load(source, allow_pickle=True).item()
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must contain a dictionary of layers")

    layers: dict[str, np.ndarray] = {}
    for layer, values in raw.items():
        if isinstance(values, Mapping):
            ordered: dict[int, np.ndarray] = {}
            for label, vector in values.items():
                index = _month_index(str(label))
                if index in ordered:
                    raise ValueError(
                        f"{layer} contains month {MONTHS[index]} more than once"
                    )
                ordered[index] = np.asarray(vector, dtype=float)
            if set(ordered) != set(range(len(MONTHS))):
                raise ValueError(
                    f"{layer} does not contain exactly the twelve calendar months"
                )
            cloud = np.stack([ordered[index] for index in range(len(MONTHS))])
        else:
            cloud = np.asarray(values, dtype=float)
        if cloud.ndim != 2 or cloud.shape[0] != len(MONTHS):
            raise ValueError(
                f"{layer} must have shape [12, features]; got {cloud.shape}"
            )
        if not np.isfinite(cloud).all():
            raise ValueError(f"{layer} contains non-finite activations")
        layers[str(layer)] = cloud
    return dict(sorted(layers.items(), key=lambda item: int(item[0].split("_")[-1])))


def _center(cloud: np.ndarray) -> np.ndarray:
    values = np.asarray(cloud, dtype=float)
    return values - values.mean(axis=0, keepdims=True)


def fourier_alignment(cloud: np.ndarray) -> float:
    """Fraction of centered activation energy explained by the first cyclic harmonic."""
    centered = _center(cloud)
    theta = 2.0 * np.pi * np.arange(len(centered)) / len(centered)
    design = np.column_stack((np.cos(theta), np.sin(theta)))
    fitted = design @ np.linalg.lstsq(design, centered, rcond=None)[0]
    denominator = float(np.sum(centered**2))
    return float(np.sum(fitted**2) / denominator) if denominator > 0.0 else float("nan")


def centered_gram(cloud: np.ndarray) -> np.ndarray:
    centered = _center(cloud)
    gram = centered @ centered.T
    scale = float(np.linalg.norm(gram))
    return gram / scale if scale > 0.0 else np.full_like(gram, np.nan)


def circulant_r2(cloud: np.ndarray) -> float:
    """Variance in off-diagonal Gram entries explained by cyclic month separation."""
    gram = centered_gram(cloud)
    values: list[float] = []
    groups: list[int] = []
    for i in range(len(MONTHS)):
        for j in range(i + 1, len(MONTHS)):
            separation = min((j - i) % len(MONTHS), (i - j) % len(MONTHS))
            values.append(float(gram[i, j]))
            groups.append(separation)
    array = np.asarray(values)
    fitted = np.asarray(
        [np.mean(array[np.asarray(groups) == group]) for group in groups]
    )
    total = float(np.sum((array - array.mean()) ** 2))
    return (
        float(1.0 - np.sum((array - fitted) ** 2) / total)
        if total > 0.0
        else float("nan")
    )


def neighbor_gap(cloud: np.ndarray) -> float:
    """Mean non-neighbor distance minus cyclic-neighbor distance, scale normalized."""
    centered = _center(cloud)
    distances = np.linalg.norm(centered[:, None, :] - centered[None, :, :], axis=2)
    diameter = float(distances.max())
    if diameter <= 0.0:
        return float("nan")
    neighbor, other = [], []
    for i in range(len(MONTHS)):
        for j in range(i + 1, len(MONTHS)):
            target = (
                neighbor
                if (j - i) in {1, len(MONTHS) - 1} or (i == 0 and j == 11)
                else other
            )
            target.append(float(distances[i, j] / diameter))
    return float(np.mean(other) - np.mean(neighbor))


def h1_max_persistence(cloud: np.ndarray, variance_threshold: float = 0.95) -> float:
    """Longest finite H1 bar after per-cloud PCA and diameter normalization."""
    values = np.asarray(cloud, dtype=float)
    pca = PCA(n_components=min(values.shape))
    full = pca.fit_transform(values)
    count = int(
        np.searchsorted(np.cumsum(pca.explained_variance_ratio_), variance_threshold)
        + 1
    )
    projected = full[:, :count]
    diameter = float(
        np.linalg.norm(projected[:, None, :] - projected[None, :, :], axis=2).max()
    )
    if not np.isfinite(diameter) or diameter <= 0.0:
        return float("nan")
    diagrams = ripser(projected / diameter, maxdim=1)["dgms"]
    if len(diagrams) < 2 or not len(diagrams[1]):
        return 0.0
    bars = np.asarray(diagrams[1], dtype=float)
    persistence = bars[:, 1] - bars[:, 0]
    persistence = persistence[np.isfinite(persistence)]
    return float(persistence.max()) if persistence.size else 0.0


def cloud_metrics(cloud: np.ndarray) -> dict[str, float]:
    return {
        "fourier_alignment": fourier_alignment(cloud),
        "circulant_r2": circulant_r2(cloud),
        "neighbor_gap": neighbor_gap(cloud),
        "h1_max_persistence": h1_max_persistence(cloud),
    }


def validate_layer(
    cloud: np.ndarray, *, n_nulls: int, base_seed: int
) -> dict[str, Any]:
    """Evaluate one observed cloud against exact-spectrum nulls and label permutations."""
    if n_nulls < 19:
        raise ValueError("n_nulls must be at least 19")
    observed = cloud_metrics(cloud)
    fit = fit_low_rank_gaussian(cloud)
    null_values = {name: [] for name in PRIMARY_METRICS}
    permutation_values = {name: [] for name in PRIMARY_METRICS[:-1]}
    rng = np.random.default_rng(base_seed + 1_000_000)
    for index in range(n_nulls):
        null_metrics = cloud_metrics(
            sample_low_rank_gaussian(fit, seed=base_seed + index)
        )
        for name in PRIMARY_METRICS:
            null_values[name].append(null_metrics[name])
        permuted = cloud[rng.permutation(len(cloud))]
        permuted_metrics = cloud_metrics(permuted)
        for name in permutation_values:
            permutation_values[name].append(permuted_metrics[name])

    inference = {
        name: empirical_pvalue(observed[name], null_values[name])
        for name in PRIMARY_METRICS
    }
    permutation_inference = {
        name: empirical_pvalue(observed[name], permutation_values[name])
        for name in permutation_values
    }
    return {
        "observed": observed,
        "null_values": null_values,
        "inference": inference,
        "permutation_values": permutation_values,
        "permutation_inference": permutation_inference,
        "null_rank": int(fit["rank"]),
    }


def run_null_cloud_test(
    cloud: np.ndarray, *, n_nulls: int, base_seed: int
) -> dict[str, Any]:
    """Run the Karkada cloud through null_cloud's unchanged public inference API."""
    manifold = Manifold(TopologyPipeline(), cloud, metrics=("topology",))
    return ManifoldComparator().compare_against_nulls(
        manifold,
        n_nulls=n_nulls,
        base_seed=base_seed,
        metrics=("topology",),
    )


def validate_karkada_null(
    cloud: np.ndarray, *, n_nulls: int, base_seed: int
) -> dict[str, Any]:
    """Test the paper's translation-symmetry prediction against the null.

    The paper-derived statistic is circulant R² of the centered Gram matrix:
    the fraction of pairwise-similarity variance explained by cyclic calendar
    separation. This uses the null generator unchanged and fits it only once.
    """
    if n_nulls < 19:
        raise ValueError("n_nulls must be at least 19")
    fit = fit_low_rank_gaussian(cloud)
    observed = circulant_r2(cloud)
    null_values = [
        circulant_r2(sample_low_rank_gaussian(fit, seed=base_seed + index))
        for index in range(n_nulls)
    ]
    return {
        "statistic": "circulant_r2",
        "inference": empirical_pvalue(observed, null_values),
        "null_rank": int(fit["rank"]),
    }


def _pca2(cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2)
    return pca.fit_transform(cloud), pca.explained_variance_ratio_


def _plot_cycle(ax, coordinates: np.ndarray, title: str) -> None:
    colors = plt.colormaps["hsv"](np.arange(len(MONTHS)) / len(MONTHS))
    closed = np.vstack((coordinates, coordinates[0]))
    ax.plot(closed[:, 0], closed[:, 1], color="#8b8b8b", linewidth=1.2, zorder=1)
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=colors,
        s=42,
        edgecolor="black",
        linewidth=0.3,
    )
    for point, month in zip(coordinates, MONTHS):
        ax.annotate(
            month[:3], point, xytext=(4, 4), textcoords="offset points", fontsize=7
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="datalim")


def render_primary_figure(
    contextualized: np.ndarray,
    bare: np.ndarray,
    null_cloud_result: Mapping[str, Any],
    karkada_result: Mapping[str, Any],
    *,
    base_seed: int,
    output: Path,
) -> None:
    fit = fit_low_rank_gaussian(contextualized)
    null = sample_low_rank_gaussian(fit, seed=base_seed)
    observed_xy, observed_evr = _pca2(contextualized)
    bare_xy, bare_evr = _pca2(bare)
    null_xy, null_evr = _pca2(null)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    _plot_cycle(
        axes[0, 0],
        observed_xy,
        f"Contextualized months ({observed_evr.sum():.1%} in PC1–2)",
    )
    _plot_cycle(
        axes[0, 1], bare_xy, f"Bare month tokens ({bare_evr.sum():.1%} in PC1–2)"
    )
    _plot_cycle(
        axes[0, 2], null_xy, f"Fixed-spectrum null ({null_evr.sum():.1%} in PC1–2)"
    )

    grams = (centered_gram(contextualized), centered_gram(null))
    limit = max(float(np.nanmax(np.abs(gram))) for gram in grams)
    for ax, gram, title in zip(
        axes[1, :2], grams, ("Observed centered Gram", "Null centered Gram")
    ):
        ax.imshow(gram, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_xticks(range(12), [month[:1] for month in MONTHS], fontsize=7)
        ax.set_yticks(range(12), [month[:1] for month in MONTHS], fontsize=7)
        ax.set_title(title, fontsize=10)
    ax = axes[1, 2]
    primary = karkada_result["inference"]
    ax.hist(primary["null_values"], bins=30, color="#b8c4d8", edgecolor="white")
    ax.axvline(primary["observed"], color="#d62728", linewidth=2.5, label="observed")
    ax.set_xlabel("Circulant $R^2$")
    ax.set_ylabel("Null draws")
    ax.set_title("Primary Karkada symmetry test", fontsize=10)
    topology = null_cloud_result["metrics"]
    ax.text(
        0.03,
        0.95,
        f"primary p={primary['pvalue']:.3f}\n"
        f"H1 Wasserstein p={topology['H1_wasserstein']['pvalue']:.2f}\n"
        f"H1 bottleneck p={topology['H1_bottleneck']['pvalue']:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Karkada translation symmetry tested against the null cloud", fontsize=14
    )
    fig.subplots_adjust(wspace=0.38, hspace=0.34, bottom=0.12)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_layer_figure(
    rows: list[dict[str, Any]], output: Path, primary_layer: str
) -> None:
    layers = [row["layer_index"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for ax, name in zip(axes.flat, PRIMARY_METRICS):
        for condition, color in (("contextualized", "#1f77b4"), ("bare", "#ff7f0e")):
            subset = [row for row in rows if row["condition"] == condition]
            ax.plot(
                layers[: len(subset)],
                [row[name] for row in subset],
                marker="o",
                markersize=3,
                color=color,
                label=condition,
            )
        ax.axvline(
            int(primary_layer.split("_")[-1]),
            color="black",
            linestyle="--",
            linewidth=1,
            label="primary layer",
        )
        ax.set_title(name.replace("_", " "))
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Layer")
    axes[1, 1].set_xlabel("Layer")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Exploratory month geometry across Gemma-2-2B layers")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    contextualized_path, bare_path = Path(args.contextualized), Path(args.bare)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    contextualized_layers = load_month_layers(contextualized_path)
    bare_layers = load_month_layers(bare_path)
    if contextualized_layers.keys() != bare_layers.keys():
        raise ValueError("contextualized and bare files must contain identical layers")
    if args.primary_layer not in contextualized_layers:
        raise ValueError(f"primary layer {args.primary_layer!r} is absent")

    karkada_result = validate_karkada_null(
        contextualized_layers[args.primary_layer],
        n_nulls=args.n_nulls,
        base_seed=args.seed,
    )
    null_cloud_result = run_null_cloud_test(
        contextualized_layers[args.primary_layer],
        n_nulls=args.topology_n_nulls,
        base_seed=args.seed,
    )
    paper_diagnostics = validate_layer(
        contextualized_layers[args.primary_layer],
        n_nulls=args.topology_n_nulls,
        base_seed=args.seed,
    )
    layer_rows: list[dict[str, Any]] = []
    for condition, layers in (
        ("contextualized", contextualized_layers),
        ("bare", bare_layers),
    ):
        for layer, cloud in layers.items():
            layer_rows.append(
                {
                    "condition": condition,
                    "layer": layer,
                    "layer_index": int(layer.split("_")[-1]),
                    **cloud_metrics(cloud),
                }
            )

    metrics = {
        "experiment_id": args.run_id,
        "primary_layer": args.primary_layer,
        "n_nulls": args.n_nulls,
        "topology_n_nulls": args.topology_n_nulls,
        "base_seed": args.seed,
        "primary_karkada_null_test": karkada_result,
        "secondary_null_cloud_topology": null_cloud_result,
        "paper_diagnostics": paper_diagnostics,
        "exploratory_layers": layer_rows,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True) + "\n"
    )
    with (output / "layer_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=layer_rows[0].keys())
        writer.writeheader()
        writer.writerows(layer_rows)

    render_primary_figure(
        contextualized_layers[args.primary_layer],
        bare_layers[args.primary_layer],
        null_cloud_result,
        karkada_result,
        base_seed=args.seed,
        output=output / "primary_validation.png",
    )
    render_layer_figure(
        layer_rows, output / "layerwise_geometry.png", args.primary_layer
    )

    manifest = {
        "experiment": {
            "id": args.run_id,
            "question": "Does the low-rank null destroy the translation-symmetric month organization predicted by Karkada et al.?",
            "hypothesis": "At pre-specified layer_6, observed circulant Gram R² exceeds the exact-spectrum null distribution.",
            "source_revision": _git_revision(),
            "command": " ".join(sys.argv),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "os": platform.platform(),
        },
        "inputs": {
            "contextualized": {
                "path": str(contextualized_path),
                "sha256": _sha256(contextualized_path),
            },
            "bare": {"path": str(bare_path), "sha256": _sha256(bare_path)},
            "model": "gemma-2-2b",
            "activation": "resid_post, final token, float32 source arrays, 12 x 2304 per layer",
            "prompts": {
                "contextualized": "The month of the year is {month}",
                "bare": "{month}",
            },
            "paper": "https://arxiv.org/abs/2602.15029",
        },
        "method": {
            "analysis_status": (
                "paper-derived reanalysis after inspecting this activation snapshot; "
                "requires confirmation on a fresh held-out extraction"
            ),
            "primary_layer": args.primary_layer,
            "primary_metric": "circulant_r2",
            "secondary_topology_metrics": list(NULL_CLOUD_METRICS),
            "secondary_paper_diagnostics": [
                name for name in PRIMARY_METRICS if name != "circulant_r2"
            ],
            "n_nulls": args.n_nulls,
            "topology_n_nulls": args.topology_n_nulls,
            "seed": args.seed,
            "null": "exact-spectrum low-rank surrogate from null_cloud.py",
            "label_baseline": "independent month-label permutations",
            "all_layer_analysis": "exploratory; not used to select primary layer",
        },
        "outputs": {
            "directory": str(output),
            "raw_metrics": ["metrics.json", "layer_metrics.csv"],
            "figures": ["primary_validation.png", "layerwise_geometry.png"],
            "elapsed_seconds": time.time() - started,
        },
        "claims": {
            "observations": [
                {
                    "metric": "circulant_r2",
                    "observed": karkada_result["inference"]["observed"],
                    "null_pvalue": karkada_result["inference"]["pvalue"],
                }
            ],
            "hypothesis_result": (
                "supported"
                if karkada_result["inference"]["pvalue"] <= 0.05
                else "not supported"
            ),
            "supported_claim": "The exact-spectrum null removes Karkada-style translation symmetry at the pre-specified layer; generic H1 remains a secondary underpowered diagnostic.",
            "non_claims": [
                "A rejection does not prove semantic meaning or uniquely identify a circle.",
                "The twelve-month sample is small and the all-layer curves are exploratory.",
                "This statistic was not formally preregistered before the activation snapshot was inspected.",
                "The local activation files do not record an immutable upstream model revision or tokenizer revision.",
            ],
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contextualized", default=str(DEFAULT_CONTEXTUALIZED))
    parser.add_argument("--bare", default=str(DEFAULT_BARE))
    parser.add_argument("--primary-layer", default="layer_6")
    parser.add_argument("--n-nulls", type=int, default=999)
    parser.add_argument("--topology-n-nulls", type=int, default=99)
    parser.add_argument("--seed", type=int, default=260215)
    parser.add_argument("--run-id", default="karkada-month-circle-layer6-v1")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = run(args)
    print(
        json.dumps(
            {
                "output": args.output,
                "primary_layer": args.primary_layer,
                "primary_statistic": "circulant_r2",
                "primary_pvalue": metrics["primary_karkada_null_test"]["inference"][
                    "pvalue"
                ],
                "secondary_topology_pvalues": {
                    name: metrics["secondary_null_cloud_topology"]["metrics"][name][
                        "pvalue"
                    ]
                    for name in NULL_CLOUD_METRICS
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
