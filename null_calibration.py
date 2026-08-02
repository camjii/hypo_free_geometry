#!/usr/bin/env python3
"""Synthetic null-calibration benchmark for the geometry pipeline.

Reuses ``null_cloud.Manifold`` / ``ManifoldComparator`` measurement and robust
null comparison. Synthetic generators live here as helpers to keep the change
compact; semantic ``ground_truths`` references are not dense point-cloud
controls for this experiment.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Match pipeline_draft OpenMP isolation before native imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import numpy as np
import pandas as pd
import skdim
import yaml
from ripser import ripser
from sklearn.decomposition import PCA
from sklearn.neighbors import radius_neighbors_graph

from null_cloud import (
    CURVATURE_UNAVAILABLE,
    Manifold,
    _finite,
    _graph_diagnostics,
    curvature_distribution_difference,
    diagram_distance_pair,
    NULL_KIND,
    empirical_pvalue,
    fit_low_rank_gaussian,
    null_diagnostics,
    resolve_null_kind,
    sample_low_rank_gaussian,
    unavailable,
)


DEFAULT_CONFIG = PROJECT_ROOT / "null_calibration.yaml"
MIN_SEEDS_FOR_RATE_CLAIMS = 10
MIN_CURVATURE_FOR_REPORT = 10
PRODUCTION_PCA_DEFAULT = "variance_95"
DEPLOYABLE_PCA_MODES = ("none", "variance_95", "parallel_analysis")

PCA_MODES = ("none", "variance_95", "parallel_analysis", "oracle")

RUN_COLUMNS = [
    "dataset",
    "seed",
    "sample_size",
    "ambient_dim",
    "noise_level",
    "pca_mode",
    "pca_components",
    "pca_explained_variance",
    "expected_intrinsic_dimension",
    "intrinsic_dimension",
    "intrinsic_dimension_error",
    "h0_max_persistence",
    "h0_total_persistence",
    "h1_max_persistence",
    "h1_total_persistence",
    # One null model, so one set of comparison columns.
    "h0_bottleneck",
    "h0_wasserstein",
    "h0_pvalue",
    "h1_bottleneck",
    "h1_wasserstein",
    "h1_pvalue",
    "id_difference",
    "id_pvalue",
    "null_rank",
    "null_spectrum_error",
    "curvature_enabled",
    "curvature_wasserstein",
    "curvature_wasserstein_pvalue",
    "curvature_mean_difference",
    "curvature_negative_fraction_difference",
    "graph_n_nodes",
    "graph_n_edges",
    "graph_n_components",
    "graph_mean_degree",
    "selected_epsilon",
    "runtime_seconds",
    "failure",
    "failure_detail",
]


# ---------------------------------------------------------------------------
# Synthetic datasets (helpers; metadata encodes expected structure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    generator: Callable[[int, np.random.Generator], np.ndarray]
    expected_intrinsic_dimension: float | None
    # Linear embedding rank needed for oracle PCA to preserve the target signal
    # (e.g. a circle has intrinsic dim 1 but needs 2 PCA axes).
    oracle_pca_components: int
    expected_homology: Mapping[str, str]
    main_signal: str
    notes: str = ""


def generate_isotropic_gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(n, 3))


def generate_spiked_gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    # Rank-2 linear covariance control in 3D ambient latent space.
    factors = rng.normal(size=(n, 2))
    loadings = np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=float)
    return factors @ loadings.T + 0.05 * rng.normal(size=(n, 3))


def generate_line(n: int, rng: np.random.Generator) -> np.ndarray:
    t = rng.uniform(-1.0, 1.0, size=n)
    return np.column_stack((t, np.zeros(n), np.zeros(n)))


def generate_two_clusters(n: int, rng: np.random.Generator) -> np.ndarray:
    n1 = n // 2
    n2 = n - n1
    a = rng.normal(loc=(-1.5, 0.0, 0.0), scale=0.12, size=(n1, 3))
    b = rng.normal(loc=(1.5, 0.0, 0.0), scale=0.12, size=(n2, 3))
    return np.vstack((a, b))


def generate_circle(n: int, rng: np.random.Generator) -> np.ndarray:
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.column_stack((np.cos(angles), np.sin(angles), np.zeros(n)))


def generate_swiss_roll(n: int, rng: np.random.Generator) -> np.ndarray:
    # Classic Swiss-roll parameterization (nonlinear intrinsic dim 2).
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.uniform(size=n))
    height = 21.0 * rng.uniform(size=n)
    x = t * np.cos(t)
    y = height
    z = t * np.sin(t)
    points = np.column_stack((x, y, z))
    return points / max(float(np.std(points)), 1e-9)


def generate_y_tree(n: int, rng: np.random.Generator) -> np.ndarray:
    # Three rays from the origin: branching / tree structure, no H1 loop.
    n_arm = max(n // 3, 1)
    remainder = n - 3 * n_arm
    arms = []
    directions = np.array(
        [[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2.0, 0.0], [-0.5, -np.sqrt(3) / 2.0, 0.0]],
        dtype=float,
    )
    for index, direction in enumerate(directions):
        count = n_arm + (1 if index < remainder else 0)
        radii = rng.uniform(0.05, 1.0, size=count)
        arms.append(radii[:, None] * direction[None, :])
    return np.vstack(arms)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "isotropic_gaussian": DatasetSpec(
        name="isotropic_gaussian",
        generator=generate_isotropic_gaussian,
        expected_intrinsic_dimension=3.0,
        oracle_pca_components=3,
        expected_homology={"H0": "single_blob", "H1": "none"},
        main_signal="pure_noise_negative_control",
    ),
    "spiked_gaussian": DatasetSpec(
        name="spiked_gaussian",
        generator=generate_spiked_gaussian,
        expected_intrinsic_dimension=2.0,
        oracle_pca_components=2,
        expected_homology={"H0": "single_blob", "H1": "none"},
        main_signal="linear_covariance_anisotropy",
    ),
    "line": DatasetSpec(
        name="line",
        generator=generate_line,
        expected_intrinsic_dimension=1.0,
        oracle_pca_components=1,
        expected_homology={"H0": "path", "H1": "none"},
        main_signal="intrinsic_dimension_1",
    ),
    "two_clusters": DatasetSpec(
        name="two_clusters",
        generator=generate_two_clusters,
        expected_intrinsic_dimension=3.0,
        oracle_pca_components=2,
        expected_homology={"H0": "two_components", "H1": "none"},
        main_signal="h0_separation",
    ),
    "circle": DatasetSpec(
        name="circle",
        generator=generate_circle,
        expected_intrinsic_dimension=1.0,
        oracle_pca_components=2,
        expected_homology={"H0": "single_component", "H1": "one_loop"},
        main_signal="h1_loop",
    ),
    "swiss_roll": DatasetSpec(
        name="swiss_roll",
        generator=generate_swiss_roll,
        expected_intrinsic_dimension=2.0,
        oracle_pca_components=3,
        expected_homology={"H0": "single_component", "H1": "none_dominant"},
        main_signal="nonlinear_intrinsic_dimension_2",
    ),
    "y_tree": DatasetSpec(
        name="y_tree",
        generator=generate_y_tree,
        expected_intrinsic_dimension=1.0,
        oracle_pca_components=2,
        expected_homology={"H0": "branching", "H1": "none"},
        main_signal="branching_without_h1",
        notes="Tree-like geometry; lack of H1 is not lack of structure.",
    ),
}


def dataset_metadata(name: str) -> dict[str, Any]:
    spec = DATASET_SPECS[name]
    return {
        "name": spec.name,
        "expected_intrinsic_dimension": spec.expected_intrinsic_dimension,
        "oracle_pca_components": spec.oracle_pca_components,
        "expected_homology": dict(spec.expected_homology),
        "main_signal": spec.main_signal,
        "notes": spec.notes,
    }


# ---------------------------------------------------------------------------
# Embedding / PCA helpers
# ---------------------------------------------------------------------------


def random_orthonormal_matrix(ambient_dim: int, latent_dim: int, seed: int) -> np.ndarray:
    if ambient_dim < latent_dim:
        raise ValueError("ambient_dim must be >= latent dimension")
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(ambient_dim, latent_dim))
    q, _ = np.linalg.qr(gaussian)
    return np.asarray(q[:, :latent_dim], dtype=float)


def embed_with_noise(
    latent: np.ndarray,
    ambient_dim: int,
    noise_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal isometric embedding into high-D, then isotropic Gaussian noise."""
    basis = random_orthonormal_matrix(ambient_dim, latent.shape[1], seed=seed)
    embedded = latent @ basis.T
    rng = np.random.default_rng(seed + 17)
    if noise_level > 0.0:
        embedded = embedded + noise_level * rng.normal(size=embedded.shape)
    return np.asarray(embedded, dtype=float), basis


def parallel_analysis_components(
    cloud: np.ndarray,
    n_draws: int = 30,
    seed: int = 0,
) -> int:
    """Horn's parallel analysis: keep PCs with variance above noise mean."""
    x = np.asarray(cloud, dtype=float)
    n_components = min(x.shape)
    if n_components < 1:
        return 1
    observed = PCA(n_components=n_components).fit(x).explained_variance_
    rng = np.random.default_rng(seed)
    null_variances = np.zeros((n_draws, n_components), dtype=float)
    for draw in range(n_draws):
        noise = rng.normal(size=x.shape)
        null_variances[draw] = PCA(n_components=n_components).fit(noise).explained_variance_
    threshold = null_variances.mean(axis=0)
    kept = int(np.sum(observed > threshold))
    return max(kept, 1)


def select_pca_projection(
    cloud: np.ndarray,
    mode: str,
    *,
    variance_threshold: float = 0.95,
    oracle_dim: int | None = None,
    parallel_draws: int = 30,
    seed: int = 0,
) -> tuple[np.ndarray, int, float]:
    x = np.asarray(cloud, dtype=float)
    if mode == "none":
        return x.copy(), int(x.shape[1]), 1.0

    max_components = min(x.shape)
    pca = PCA(n_components=max_components)
    full = pca.fit_transform(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    if mode == "variance_95":
        count = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    elif mode == "parallel_analysis":
        count = parallel_analysis_components(x, n_draws=parallel_draws, seed=seed)
    elif mode == "oracle":
        if oracle_dim is None:
            raise ValueError("oracle PCA requires oracle_dim")
        count = int(max(1, min(int(oracle_dim), max_components)))
    else:
        raise ValueError(f"unknown pca mode: {mode}")

    count = min(max(count, 1), max_components)
    explained = float(cumulative[count - 1])
    return np.asarray(full[:, :count], dtype=float), count, explained


# ---------------------------------------------------------------------------
# Lightweight pipeline backend (no transformer load)
# ---------------------------------------------------------------------------


class CalibrationPipeline:
    """Measurement backend compatible with ``null_cloud.Manifold``."""

    def __init__(
        self,
        *,
        pca_mode: str = "variance_95",
        oracle_dim: int | None = None,
        variance_threshold: float = 0.95,
        parallel_draws: int = 30,
        seed: int = 0,
        enable_curvature: bool = True,
    ) -> None:
        if pca_mode not in PCA_MODES:
            raise ValueError(f"pca_mode must be one of {PCA_MODES}")
        self.pca_mode = pca_mode
        self.oracle_dim = oracle_dim
        self.variance_threshold = float(variance_threshold)
        self.parallel_draws = int(parallel_draws)
        self.seed = int(seed)
        self.enable_curvature = bool(enable_curvature)
        self.last_pca_components = 0
        self.last_pca_explained_variance = float("nan")
        self.curvature_skip_reason: str | None = None

    def reduce_pca(self, opt_pos_activations, var_threshold: float = 0.95):
        threshold = float(var_threshold) if var_threshold is not None else self.variance_threshold
        projected, count, explained = select_pca_projection(
            np.asarray(opt_pos_activations, dtype=float),
            self.pca_mode,
            variance_threshold=threshold,
            oracle_dim=self.oracle_dim,
            parallel_draws=self.parallel_draws,
            seed=self.seed,
        )
        self.last_pca_components = count
        self.last_pca_explained_variance = explained
        return projected

    def get_intrinsic_dim(self, contrastive_diff) -> float:
        x = np.asarray(contrastive_diff, dtype=float)
        return float(skdim.id.TwoNN().fit(x).dimension_)

    def create_persistence_diagram(self, projected):
        return ripser(
            np.asarray(projected, dtype=float),
            maxdim=1,
            distance_matrix=False,
            do_cocycles=False,
            n_perm=None,
        )

    def create_epsilon_graph(self, projected, eps):
        adjacency = radius_neighbors_graph(
            np.asarray(projected, dtype=float),
            radius=float(eps),
            mode="distance",
            metric="euclidean",
            include_self=False,
        )
        return nx.Graph(adjacency)

    def compute_ollivier_ricci(self, graph):
        if not self.enable_curvature:
            self.curvature_skip_reason = "skipped_by_config"
            return {
                "graph": graph,
                "mean_curvature": float("nan"),
                "raw_values": [],
                "skipped": True,
            }
        from GraphRicciCurvature.OllivierRicci import OllivierRicci

        if graph.number_of_edges() == 0:
            self.curvature_skip_reason = "empty_graph"
            return {
                "graph": graph,
                "mean_curvature": float("nan"),
                "raw_values": [],
                "skipped": True,
            }
        calculator = OllivierRicci(graph, alpha=0.5, proc=1, verbose="ERROR")
        curved = calculator.compute_ricci_curvature()
        raw_values = [edge[-1] for edge in curved.edges(data="ricciCurvature")]
        finite = [float(v) for v in raw_values if v is not None and np.isfinite(v)]
        mean_curv = float(np.mean(finite)) if finite else float("nan")
        self.curvature_skip_reason = None
        return {"graph": curved, "mean_curvature": mean_curv, "raw_values": finite}


class CalibrationManifold(Manifold):
    """Manifold measurement with optional curvature and graph diagnostics.

    Curvature can be switched off per instance because Ollivier-Ricci dominates
    benchmark runtime. A switched-off or failed curvature measurement sets
    ``curvature_skipped`` and leaves topology and intrinsic dimension intact.
    """

    def __init__(self, *args: Any, enable_curvature: bool = True, **kwargs: Any) -> None:
        self.enable_curvature = bool(enable_curvature)
        self.curvature_skipped = False
        self.graph_n_nodes = 0
        self.graph_n_edges = 0
        self.graph_n_components = 0
        self.graph_mean_degree = float("nan")
        super().__init__(*args, **kwargs)

    def _measure_curvature(self, projected: np.ndarray) -> None:
        """Always build the graph for diagnostics; run Ricci only when enabled."""
        try:
            graph = self.pipeline.create_epsilon_graph(projected, self.eps)
        except Exception as exc:
            self.curvature_error = f"{type(exc).__name__}: {exc}"
            return
        self.graph_diagnostics = _graph_diagnostics(graph)
        if not self.enable_curvature:
            self.curvature_error = "skipped_by_config"
            return
        try:
            values = np.asarray(
                self.pipeline.compute_ollivier_ricci(graph).get("raw_values", []),
                dtype=float,
            )
        except Exception as exc:
            self.curvature_error = f"{type(exc).__name__}: {exc}"
            return
        if values.size == 0 or not np.isfinite(values).all():
            self.curvature_error = "empty_or_nonfinite_curvature"
        else:
            self.curvature_values = values

    def _measure(self) -> None:
        super()._measure()
        diagnostics = self.graph_diagnostics or {}
        self.graph_n_nodes = int(diagnostics.get("node_count", 0))
        self.graph_n_edges = int(diagnostics.get("edge_count", 0))
        self.graph_n_components = int(diagnostics.get("connected_components", 0))
        self.graph_mean_degree = float(diagnostics.get("mean_degree", float("nan")))
        self.curvature_skipped = self.curvature_values.size == 0

    def null(
        self,
        kind: str | None = None,
        seed: int | None = None,
        fit: Mapping[str, Any] | None = None,
        enable_curvature: bool | None = None,
    ) -> "CalibrationManifold":
        """Draw and measure one null cloud from the canonical low-rank generator.

        ``fit`` reuses a :func:`null_cloud.fit_low_rank_gaussian` result so an
        ensemble fits once. ``enable_curvature`` selects curvature for this draw
        only and never mutates the observed manifold.
        """
        resolve_null_kind(kind)
        if fit is None:
            fit = fit_low_rank_gaussian(self.cloud)
        null_seed = int(self.rng.integers(1 << 30)) if seed is None else int(seed)
        return CalibrationManifold(
            self.pipeline,
            sample_low_rank_gaussian(fit, seed=null_seed),
            label=f"null:{NULL_KIND}",
            seed=null_seed,
            eps_density=self.eps_density,
            var_threshold=self.var_threshold,
            enable_curvature=(
                self.enable_curvature if enable_curvature is None else bool(enable_curvature)
            ),
        )


# ---------------------------------------------------------------------------
# Persistence summaries and robust null comparison
# ---------------------------------------------------------------------------


def persistence_summaries(dgms: list[np.ndarray]) -> dict[str, float]:
    out = {
        "h0_max_persistence": 0.0,
        "h0_total_persistence": 0.0,
        "h1_max_persistence": 0.0,
        "h1_total_persistence": 0.0,
    }
    for dim, key_prefix in ((0, "h0"), (1, "h1")):
        if dim >= len(dgms):
            continue
        finite = _finite(dgms[dim])
        if finite.size == 0:
            continue
        persistences = finite[:, 1] - finite[:, 0]
        persistences = persistences[np.isfinite(persistences) & (persistences >= 0)]
        if persistences.size == 0:
            continue
        out[f"{key_prefix}_max_persistence"] = float(np.max(persistences))
        out[f"{key_prefix}_total_persistence"] = float(np.sum(persistences))
    return out


def _safe_diagram_distance(d1: np.ndarray, d2: np.ndarray) -> dict[str, float]:
    """Bottleneck / Wasserstein between two diagrams; NaN when persim refuses."""
    return diagram_distance_pair(d1, d2)


def _safe_curvature_difference(m1, m2) -> dict[str, float]:
    """Curvature comparison, or all-NaN when either side has no usable edges."""
    if getattr(m1, "curvature_skipped", False) or getattr(m2, "curvature_skipped", False):
        return dict(CURVATURE_UNAVAILABLE)
    return curvature_distribution_difference(m1.curvature_values, m2.curvature_values)


def flatten_calibration_distances(
    m1: CalibrationManifold,
    m2: CalibrationManifold,
    *,
    include_curvature: bool,
) -> dict[str, float]:
    """Named scalar distances. H0/H1/ID/curvature stay separate."""
    distances: dict[str, float] = {
        "id_difference": float(abs(m1.intrinsic_dim - m2.intrinsic_dim)),
    }
    for dim in (0, 1):
        if dim < len(m1.dgms) and dim < len(m2.dgms):
            values = _safe_diagram_distance(m1.dgms[dim], m2.dgms[dim])
        else:
            values = {"wasserstein": float("nan"), "bottleneck": float("nan")}
        distances[f"H{dim}_wasserstein"] = values["wasserstein"]
        distances[f"H{dim}_bottleneck"] = values["bottleneck"]

    if include_curvature:
        curvature = _safe_curvature_difference(m1, m2)
        # Signed quantities only -- do not multiply by |negative-fraction|.
        distances["curvature_wasserstein"] = float(curvature["distribution_distance"])
        distances["curvature_mean_difference"] = float(curvature["mean_difference"])
        distances["curvature_negative_fraction_difference"] = float(
            curvature["negative_fraction_difference"]
        )
    return distances


def _median_to_others(row: np.ndarray) -> float:
    """Median of the finite entries of one row; NaN when none are finite."""
    values = np.asarray(row, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def scores_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    """Rank the observed object's median distance against the null objects'.

    ``matrix`` is the symmetric (1 + n_nulls) square of pairwise distances with
    the observed object at index 0 and NaN on the diagonal. Inference is
    delegated to :func:`null_cloud.empirical_pvalue`, so a non-finite observed
    score or too few finite null scores yields ``inference_available=False``
    rather than a spuriously small p-value.
    """
    null_matrix = matrix[1:, 1:]
    return empirical_pvalue(
        _median_to_others(matrix[0, 1:]),
        [
            _median_to_others(np.delete(null_matrix[index], index))
            for index in range(null_matrix.shape[0])
        ],
    )


def robust_null_comparison(
    manifold: CalibrationManifold,
    *,
    n_nulls: int,
    base_seed: int,
    include_curvature: bool,
    curvature_nulls: int = 5,
) -> dict[str, Any]:
    """Repeated-null empirical inference against the low-rank Gaussian null.

    The null is fitted once and reused for every draw. Topology and ID nulls are
    measured without Ricci; curvature uses a smaller ensemble because
    Ollivier-Ricci dominates runtime. Draws that fail to measure are recorded and
    excluded rather than aborting. ``manifold`` is never mutated.
    """
    if n_nulls < 3:
        raise ValueError("n_nulls must be at least 3")

    fit = fit_low_rank_gaussian(manifold.cloud)
    nulls, failures = _draw_nulls(manifold, n_nulls, base_seed, fit, False)
    metric_names = list(
        flatten_calibration_distances(manifold, manifold, include_curvature=False)
    )

    if len(nulls) < 3:
        reason = (
            f"only {len(nulls)} of {n_nulls} null draws could be measured; "
            "at least 3 are required"
        )
        results = {name: unavailable(reason) for name in metric_names}
        diagnostics = None
    else:
        matrices = _pairwise_matrices(
            [manifold, *nulls], metric_names, include_curvature=False
        )
        results = {name: scores_from_matrix(matrix) for name, matrix in matrices.items()}
        diagnostics = null_diagnostics(fit, nulls[0].cloud)

    if include_curvature and not manifold.curvature_skipped:
        results.update(
            _curvature_comparison(
                manifold,
                n_nulls=max(3, min(int(curvature_nulls), n_nulls)),
                base_seed=base_seed + 10_000,
                fit=fit,
            )
        )

    return {
        "null_kind": NULL_KIND,
        "n_requested": n_nulls,
        "n_drawn": len(nulls),
        "failures": failures,
        "base_seed": base_seed,
        "null_diagnostics": diagnostics,
        "metrics": results,
    }


def _draw_nulls(
    manifold: CalibrationManifold,
    count: int,
    base_seed: int,
    fit: Mapping[str, Any],
    enable_curvature: bool,
) -> tuple[list[CalibrationManifold], list[dict[str, Any]]]:
    """Draw ``count`` nulls at deterministic seeds, recording failures."""
    nulls: list[CalibrationManifold] = []
    failures: list[dict[str, Any]] = []
    for index in range(count):
        seed = base_seed + index
        try:
            nulls.append(
                manifold.null(seed=seed, fit=fit, enable_curvature=enable_curvature)
            )
        except Exception as exc:
            failures.append(
                {"index": index, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            )
    return nulls, failures


def _pairwise_matrices(
    objects: Sequence[CalibrationManifold],
    names: Sequence[str],
    *,
    include_curvature: bool,
) -> dict[str, np.ndarray]:
    """Symmetric pairwise distance matrices, one per metric, NaN on the diagonal."""
    n = len(objects)
    matrices = {name: np.full((n, n), np.nan, dtype=float) for name in names}
    for i in range(n):
        for j in range(i + 1, n):
            distances = flatten_calibration_distances(
                objects[i], objects[j], include_curvature=include_curvature
            )
            for name, value in distances.items():
                if name in matrices:
                    matrices[name][i, j] = value
                    matrices[name][j, i] = value
    return matrices


def _curvature_comparison(
    manifold: CalibrationManifold,
    *,
    n_nulls: int,
    base_seed: int,
    fit: Mapping[str, Any],
) -> dict[str, Any]:
    """Curvature inference on its own smaller ensemble.

    The signed mean and negative-fraction differences are descriptive summaries
    of the observed cloud against its nulls, not tests, so they carry no p-value.
    """
    nulls, _ = _draw_nulls(manifold, n_nulls, base_seed, fit, True)
    if len(nulls) < 3:
        return {
            "curvature_wasserstein": unavailable(
                f"only {len(nulls)} curvature null draws could be measured"
            )
        }

    objects = [manifold, *nulls]
    matrix = _pairwise_matrices(
        objects, ["curvature_wasserstein"], include_curvature=True
    )["curvature_wasserstein"]
    signed = [
        _safe_curvature_difference(manifold, null) for null in nulls
    ]
    return {
        "curvature_wasserstein": scores_from_matrix(matrix),
        "signed_mean_curvature_difference": {
            "observed_score": _median_to_others(
                np.array([value["mean_difference"] for value in signed])
            )
        },
        "signed_negative_fraction_difference": {
            "observed_score": _median_to_others(
                np.array([value["negative_fraction_difference"] for value in signed])
            )
        },
    }


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    return config


def curvature_enabled_for_run(config: Mapping[str, Any], dataset: str, seed: int, noise: float, pca_mode: str) -> bool:
    curv = config.get("curvature", {}) or {}
    enabled_datasets = set(curv.get("enabled_datasets", []))
    enabled_seeds = set(int(s) for s in curv.get("enabled_seeds", config.get("seeds", [])))
    enabled_noise = set(float(x) for x in curv.get("enabled_noise_levels", config.get("noise_levels", [])))
    enabled_modes = set(curv.get("enabled_pca_modes", config.get("pca_modes", [])))
    return (
        dataset in enabled_datasets
        and int(seed) in enabled_seeds
        and float(noise) in enabled_noise
        and pca_mode in enabled_modes
    )


def empty_run_row() -> dict[str, Any]:
    return {column: np.nan for column in RUN_COLUMNS}


def _metric_or_nan(metrics: Mapping[str, Any], name: str, field: str) -> float:
    block = metrics.get(name)
    if not isinstance(block, Mapping):
        return float("nan")
    value = block.get(field, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def run_single(
    *,
    dataset: str,
    seed: int,
    sample_size: int,
    ambient_dim: int,
    noise_level: float,
    pca_mode: str,
    n_nulls: int,
    null_base_seed: int,
    eps_density: float,
    variance_threshold: float,
    parallel_draws: int,
    enable_curvature: bool,
    curvature_nulls: int = 5,
) -> dict[str, Any]:
    row = empty_run_row()
    row.update(
        {
            "dataset": dataset,
            "seed": int(seed),
            "sample_size": int(sample_size),
            "ambient_dim": int(ambient_dim),
            "noise_level": float(noise_level),
            "pca_mode": pca_mode,
            "curvature_enabled": bool(enable_curvature),
            "failure": "",
            "failure_detail": "",
        }
    )
    started = time.perf_counter()
    spec = DATASET_SPECS[dataset]
    row["expected_intrinsic_dimension"] = (
        float("nan")
        if spec.expected_intrinsic_dimension is None
        else float(spec.expected_intrinsic_dimension)
    )

    try:
        rng = np.random.default_rng(seed)
        latent = spec.generator(sample_size, rng)
        cloud, _ = embed_with_noise(latent, ambient_dim, noise_level, seed=seed + 1000)

        oracle_dim = None
        if pca_mode == "oracle":
            oracle_dim = int(max(1, spec.oracle_pca_components))

        pipeline = CalibrationPipeline(
            pca_mode=pca_mode,
            oracle_dim=oracle_dim,
            variance_threshold=variance_threshold,
            parallel_draws=parallel_draws,
            seed=seed,
            enable_curvature=enable_curvature,
        )
        manifold = CalibrationManifold(
            pipeline,
            cloud,
            label=dataset,
            seed=seed,
            eps_density=eps_density,
            var_threshold=variance_threshold,
            enable_curvature=enable_curvature,
        )

        row["pca_components"] = int(pipeline.last_pca_components or manifold.m)
        row["pca_explained_variance"] = float(pipeline.last_pca_explained_variance)
        row["intrinsic_dimension"] = float(manifold.intrinsic_dim)
        if spec.expected_intrinsic_dimension is not None:
            row["intrinsic_dimension_error"] = float(
                abs(manifold.intrinsic_dim - spec.expected_intrinsic_dimension)
            )
        summaries = persistence_summaries(list(manifold.dgms))
        row.update(summaries)
        row["graph_n_nodes"] = int(manifold.graph_n_nodes)
        row["graph_n_edges"] = int(manifold.graph_n_edges)
        row["graph_n_components"] = int(manifold.graph_n_components)
        row["graph_mean_degree"] = float(manifold.graph_mean_degree)
        row["selected_epsilon"] = float(manifold.eps)

        include_curvature = enable_curvature and not manifold.curvature_skipped
        comparison = robust_null_comparison(
            manifold,
            n_nulls=n_nulls,
            base_seed=null_base_seed,
            include_curvature=include_curvature,
            curvature_nulls=curvature_nulls,
        )
        metrics = comparison["metrics"]
        row["h0_bottleneck"] = _metric_or_nan(metrics, "H0_bottleneck", "observed")
        row["h0_wasserstein"] = _metric_or_nan(metrics, "H0_wasserstein", "observed")
        row["h0_pvalue"] = _metric_or_nan(metrics, "H0_bottleneck", "pvalue")
        row["h1_bottleneck"] = _metric_or_nan(metrics, "H1_bottleneck", "observed")
        row["h1_wasserstein"] = _metric_or_nan(metrics, "H1_wasserstein", "observed")
        row["h1_pvalue"] = _metric_or_nan(metrics, "H1_bottleneck", "pvalue")
        row["id_difference"] = _metric_or_nan(metrics, "id_difference", "observed")
        row["id_pvalue"] = _metric_or_nan(metrics, "id_difference", "pvalue")
        row["curvature_wasserstein"] = _metric_or_nan(
            metrics, "curvature_wasserstein", "observed"
        )
        row["curvature_wasserstein_pvalue"] = _metric_or_nan(
            metrics, "curvature_wasserstein", "pvalue"
        )
        row["curvature_mean_difference"] = _metric_or_nan(
            metrics, "signed_mean_curvature_difference", "observed"
        )
        row["curvature_negative_fraction_difference"] = _metric_or_nan(
            metrics, "signed_negative_fraction_difference", "observed"
        )

        diagnostics = comparison["null_diagnostics"] or {}
        row["null_rank"] = float(diagnostics.get("rank", float("nan")))
        row["null_spectrum_error"] = float(
            diagnostics.get("relative_spectrum_error", float("nan"))
        )

        if not include_curvature and enable_curvature and manifold.curvature_error:
            row["failure"] = "curvature_skipped"
            row["failure_detail"] = str(manifold.curvature_error)

    except Exception as exc:
        row["failure"] = type(exc).__name__
        row["failure_detail"] = f"{exc}\n{traceback.format_exc(limit=4)}"
    finally:
        row["runtime_seconds"] = float(time.perf_counter() - started)
    return row


def run_benchmark(config: Mapping[str, Any]) -> pd.DataFrame:
    datasets = list(config.get("datasets", list(DATASET_SPECS)))
    seeds = [int(s) for s in config.get("seeds", [0])]
    noise_levels = [float(x) for x in config.get("noise_levels", [0.0])]
    pca_modes = list(config.get("pca_modes", ["variance_95"]))
    rows: list[dict[str, Any]] = []

    for dataset in datasets:
        if dataset not in DATASET_SPECS:
            raise KeyError(f"unknown dataset: {dataset}")
        for seed in seeds:
            for noise in noise_levels:
                for pca_mode in pca_modes:
                    enable_curvature = curvature_enabled_for_run(
                        config, dataset, seed, noise, pca_mode
                    )
                    print(
                        f"[null_calibration] {dataset} seed={seed} noise={noise} "
                        f"pca={pca_mode} curvature={enable_curvature}"
                    )
                    rows.append(
                        run_single(
                            dataset=dataset,
                            seed=seed,
                            sample_size=int(config.get("sample_size", 36)),
                            ambient_dim=int(config.get("ambient_dim", 12)),
                            noise_level=noise,
                            pca_mode=pca_mode,
                            n_nulls=int(config.get("n_nulls", 4)),
                            null_base_seed=int(config.get("null_base_seed", 100)),
                            eps_density=float(config.get("eps_density", 0.10)),
                            variance_threshold=float(config.get("variance_threshold", 0.95)),
                            parallel_draws=int(config.get("parallel_analysis_draws", 30)),
                            enable_curvature=enable_curvature,
                            curvature_nulls=int(
                                (config.get("curvature") or {}).get("n_nulls", 5)
                            ),
                        )
                    )
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


# ---------------------------------------------------------------------------
# Pipeline benchmark scoring + single HTML report
# ---------------------------------------------------------------------------


REJECTION_COLUMNS = (
    ("reject_h0", "h0_pvalue"),
    ("reject_h1", "h1_pvalue"),
    ("reject_id", "id_pvalue"),
)


def annotate_rejections(runs: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add one boolean rejection column per p-value column.

    A missing p-value means inference was unavailable, which is neither a
    rejection nor a confirmed non-rejection. It is marked ``False`` here so it
    can never be counted as a detection, and the matching ``*_available`` column
    records that the run contributed no evidence either way -- rate helpers must
    use that mask instead of treating unavailable as a passed test.
    """
    frame = runs.copy()
    for rejection_column, pvalue_column in REJECTION_COLUMNS:
        values = pd.to_numeric(frame[pvalue_column], errors="coerce")
        available = np.isfinite(values.to_numpy(dtype=float))
        frame[f"{pvalue_column}_available"] = available
        frame[rejection_column] = available & (values <= alpha).fillna(False).to_numpy()
    return frame


def _subset(
    runs: pd.DataFrame,
    *,
    dataset: str | None = None,
    noise: float | None = 0.0,
    pca_mode: str | None = None,
) -> pd.DataFrame:
    frame = runs
    if dataset is not None:
        frame = frame[frame["dataset"] == dataset]
    if noise is not None:
        frame = frame[np.isclose(frame["noise_level"].astype(float), float(noise))]
    if pca_mode is not None:
        frame = frame[frame["pca_mode"] == pca_mode]
    return frame


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    return float(np.nanmean(frame[column].to_numpy(dtype=float)))


def _rate(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    return float(np.nanmean(frame[column].astype(float).to_numpy()))


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _strength_label(fraction: float | None, *, scored: bool) -> str:
    if not scored or fraction is None or not np.isfinite(fraction):
        return "N/A"
    if fraction >= 0.85:
        return "STRONG"
    if fraction >= 0.55:
        return "MODERATE"
    return "WEAK"


def compute_pipeline_benchmark(
    runs: pd.DataFrame,
    *,
    alpha: float = 0.05,
    production_pca_mode: str = PRODUCTION_PCA_DEFAULT,
    min_seeds_for_rates: int = MIN_SEEDS_FOR_RATE_CLAIMS,
    min_curvature: int = MIN_CURVATURE_FOR_REPORT,
) -> dict[str, Any]:
    """Compute the pipeline benchmark score from in-memory run results."""
    frame = annotate_rejections(runs, alpha=alpha)
    n_seeds = int(frame["seed"].nunique()) if not frame.empty else 0
    noise_levels = sorted({float(x) for x in frame["noise_level"].tolist()}) if not frame.empty else [0.0]
    max_noise = float(max(noise_levels)) if noise_levels else 0.0
    production_mode = str(production_pca_mode)
    if production_mode == "oracle":
        raise ValueError("production_pca_mode cannot be 'oracle'")

    # Prefer no-PCA for clean topology; fall back if absent.
    topology_mode = "none" if "none" in set(frame["pca_mode"]) else production_mode

    def metric(dataset: str, column: str, noise: float, mode: str) -> float:
        return _mean(_subset(frame, dataset=dataset, noise=noise, pca_mode=mode), column)

    # ----- 1. Topology recovery (30) -----
    circle_h1 = metric("circle", "h1_max_persistence", 0.0, topology_mode)
    cluster_h0 = metric("two_clusters", "h0_max_persistence", 0.0, topology_mode)
    line_h1 = metric("line", "h1_max_persistence", 0.0, topology_mode)
    tree_h1 = metric("y_tree", "h1_max_persistence", 0.0, topology_mode)

    circle_unit = _clip01(_safe_ratio(circle_h1, 0.15))
    cluster_unit = _clip01(_safe_ratio(cluster_h0, 0.25))
    line_unit = _clip01(1.0 - _safe_ratio(line_h1, 0.08))
    tree_unit = _clip01(1.0 - _safe_ratio(tree_h1, 0.08))

    topo_parts = {
        "circle_h1": {"metric": circle_h1, "unit": circle_unit, "points": 10.0 * circle_unit, "max": 10.0},
        "cluster_h0": {"metric": cluster_h0, "unit": cluster_unit, "points": 10.0 * cluster_unit, "max": 10.0},
        "line_no_h1": {"metric": line_h1, "unit": line_unit, "points": 5.0 * line_unit, "max": 5.0},
        "ytree_no_h1": {"metric": tree_h1, "unit": tree_unit, "points": 5.0 * tree_unit, "max": 5.0},
    }
    topology_earned = float(sum(part["points"] for part in topo_parts.values()))
    topology_available = 30.0

    # ----- 2. PCA preservation (20) -----
    circle_base = metric("circle", "h1_max_persistence", 0.0, "none")
    cluster_base = metric("two_clusters", "h0_max_persistence", 0.0, "none")
    pca_modes_detail = {}
    for mode in PCA_MODES:
        c_after = metric("circle", "h1_max_persistence", 0.0, mode)
        k_after = metric("two_clusters", "h0_max_persistence", 0.0, mode)
        comps = metric("circle", "pca_components", 0.0, mode)
        c_ret = _clip01(_safe_ratio(c_after, circle_base))
        k_ret = _clip01(_safe_ratio(k_after, cluster_base))
        mode_score = float(np.mean([c_ret, k_ret]))
        pca_modes_detail[mode] = {
            "circle_h1": c_after,
            "cluster_h0": k_after,
            "circle_components": comps,
            "circle_retention": c_ret,
            "cluster_retention": k_ret,
            "mode_score": mode_score,
            "diagnostic_only": mode == "oracle",
            "destroys_circle_h1": bool(np.isfinite(c_after) and c_after < 0.05 and circle_base > 0.15),
        }

    prod = pca_modes_detail.get(production_mode, {"mode_score": 0.0})
    pca_earned = 20.0 * float(prod.get("mode_score", 0.0))
    pca_available = 20.0

    # ----- 3. Noise robustness (20) -----
    circle_clean = metric("circle", "h1_max_persistence", 0.0, production_mode)
    circle_noisy = metric("circle", "h1_max_persistence", max_noise, production_mode)
    cluster_clean = metric("two_clusters", "h0_max_persistence", 0.0, production_mode)
    cluster_noisy = metric("two_clusters", "h0_max_persistence", max_noise, production_mode)
    circle_noise_ret = _clip01(_safe_ratio(circle_noisy, circle_clean))
    cluster_noise_ret = _clip01(_safe_ratio(cluster_noisy, cluster_clean))
    noise_unit = float(np.mean([circle_noise_ret, cluster_noise_ret]))
    noise_earned = 20.0 * noise_unit
    noise_available = 20.0

    # ----- 4. Null calibration (20 or N/A) -----
    null_scored = n_seeds >= min_seeds_for_rates
    null_detail: dict[str, Any] = {
        "scored": null_scored,
        "n_seeds": n_seeds,
        "reason": None if null_scored else "insufficient seeds for reliable rejection-rate estimates",
    }
    if null_scored:
        # Linear-Gaussian datasets must not reject their own matched null; the
        # nonlinear ones should. With a single null there is no iso-vs-cov
        # asymmetry to score, so false-positive control is measured on every
        # dataset the null is supposed to explain.
        gaussian_like = pd.concat(
            [
                _subset(frame, dataset=name, noise=0.0, pca_mode=production_mode)
                for name in ("isotropic_gaussian", "spiked_gaussian")
            ]
        )
        circle = _subset(frame, dataset="circle", noise=0.0, pca_mode=production_mode)
        clusters = _subset(frame, dataset="two_clusters", noise=0.0, pca_mode=production_mode)

        fpr_h0 = _rate(gaussian_like, "reject_h0")
        fpr_h1 = _rate(gaussian_like, "reject_h1")
        fpr_id = _rate(gaussian_like, "reject_id")
        power_h1 = _rate(circle, "reject_h1")
        power_h0 = _rate(clusters, "reject_h0")

        tests = {
            "gaussian_fpr_h0": {"value": fpr_h0, "unit": _clip01(1.0 - fpr_h0 / 0.10), "kind": "fpr"},
            "gaussian_fpr_h1": {"value": fpr_h1, "unit": _clip01(1.0 - fpr_h1 / 0.10), "kind": "fpr"},
            "gaussian_fpr_id": {"value": fpr_id, "unit": _clip01(1.0 - fpr_id / 0.10), "kind": "fpr"},
            "circle_power_h1": {"value": power_h1, "unit": _clip01(power_h1 / 0.80), "kind": "power"},
            "cluster_power_h0": {"value": power_h0, "unit": _clip01(power_h0 / 0.80), "kind": "power"},
        }
        null_unit = float(np.mean([item["unit"] for item in tests.values()]))
        null_earned = 20.0 * null_unit
        null_available = 20.0
        null_detail.update({"tests": tests, "unit": null_unit})
    else:
        null_earned = 0.0
        null_available = 0.0
        null_detail["tests"] = {}

    # ----- 5. Intrinsic dimension (10) -----
    id_shapes = ("line", "circle", "swiss_roll", "y_tree")
    clean_accs = []
    noisy_accs = []
    id_rows = []
    for name in id_shapes:
        expected = DATASET_SPECS[name].expected_intrinsic_dimension
        if expected is None:
            continue
        clean_err = metric(name, "intrinsic_dimension_error", 0.0, production_mode)
        noisy_err = metric(name, "intrinsic_dimension_error", max_noise, production_mode)
        clean_acc = _clip01(1.0 - _safe_ratio(clean_err, max(float(expected), 1.0)))
        noisy_acc = _clip01(1.0 - _safe_ratio(noisy_err, max(float(expected), 1.0)))
        clean_accs.append(clean_acc)
        noisy_accs.append(noisy_acc)
        id_rows.append(
            {
                "dataset": name,
                "expected": float(expected),
                "clean_error": clean_err,
                "noisy_error": noisy_err,
                "clean_accuracy": clean_acc,
                "noisy_accuracy": noisy_acc,
            }
        )
    clean_dim_acc = float(np.mean(clean_accs)) if clean_accs else 0.0
    noisy_dim_acc = float(np.mean(noisy_accs)) if noisy_accs else 0.0
    id_unit = 0.6 * clean_dim_acc + 0.4 * noisy_dim_acc
    id_earned = 10.0 * id_unit
    id_available = 10.0

    # ----- Curvature (diagnostic only) -----
    curv = frame[frame["curvature_enabled"].fillna(False).astype(bool)]
    n_finite = int(curv["curvature_wasserstein"].notna().sum()) if not curv.empty else 0
    curvature = {
        "scored": False,
        "available": 0.0,
        "earned": 0.0,
        "n_finite": n_finite,
        "status": "N/A",
        "reason": None,
    }
    if n_finite < min_curvature:
        curvature["reason"] = "insufficient repeated measurements"
    else:
        curvature.update(
            {
                "status": "EXPLORATORY",
                "mean_curvature_difference": float(np.nanmean(curv["curvature_mean_difference"])),
                "signed_negative_fraction_difference": float(
                    np.nanmean(curv["curvature_negative_fraction_difference"])
                ),
                "curvature_wasserstein": float(np.nanmean(curv["curvature_wasserstein"])),
                "reason": "reported separately; excluded from overall score",
            }
        )

    components = {
        "topology_recovery": {
            "label": "Topology recovery",
            "earned": topology_earned,
            "available": topology_available,
            "fraction": topology_earned / topology_available,
            "status": _strength_label(topology_earned / topology_available, scored=True),
            "details": {"mode": topology_mode, "parts": topo_parts},
        },
        "pca_preservation": {
            "label": "PCA preservation",
            "earned": pca_earned,
            "available": pca_available,
            "fraction": pca_earned / pca_available,
            "status": _strength_label(pca_earned / pca_available, scored=True),
            "details": {
                "production_mode": production_mode,
                "baseline_circle_h1": circle_base,
                "baseline_cluster_h0": cluster_base,
                "modes": pca_modes_detail,
            },
        },
        "noise_robustness": {
            "label": "Noise robustness",
            "earned": noise_earned,
            "available": noise_available,
            "fraction": noise_earned / noise_available,
            "status": _strength_label(noise_earned / noise_available, scored=True),
            "details": {
                "mode": production_mode,
                "max_noise": max_noise,
                "circle_clean": circle_clean,
                "circle_noisy": circle_noisy,
                "circle_retention": circle_noise_ret,
                "cluster_clean": cluster_clean,
                "cluster_noisy": cluster_noisy,
                "cluster_retention": cluster_noise_ret,
            },
        },
        "null_calibration": {
            "label": "Null calibration",
            "earned": null_earned,
            "available": null_available,
            "fraction": (null_earned / null_available) if null_available else None,
            "status": _strength_label(
                (null_earned / null_available) if null_available else None,
                scored=null_scored,
            ),
            "details": null_detail,
        },
        "intrinsic_dimension": {
            "label": "Intrinsic dimension",
            "earned": id_earned,
            "available": id_available,
            "fraction": id_earned / id_available,
            "status": _strength_label(id_earned / id_available, scored=True),
            "details": {
                "mode": production_mode,
                "clean_accuracy": clean_dim_acc,
                "noisy_accuracy": noisy_dim_acc,
                "shapes": id_rows,
            },
        },
        "curvature": {
            "label": "Curvature",
            "earned": 0.0,
            "available": 0.0,
            "fraction": None,
            "status": curvature["status"],
            "details": curvature,
        },
    }

    earned_points = float(
        sum(comp["earned"] for key, comp in components.items() if key != "curvature" and comp["available"] > 0)
    )
    available_points = float(
        sum(comp["available"] for key, comp in components.items() if key != "curvature")
    )
    overall = 100.0 * earned_points / available_points if available_points > 0 else 0.0
    evidence_coverage = available_points  # out of 100 nominal

    # Confidence
    curv_ok = n_finite >= min_curvature
    if n_seeds < min_seeds_for_rates or evidence_coverage < 80.0:
        confidence = "LOW"
    elif (
        n_seeds >= 30
        and evidence_coverage >= 90.0
        and null_scored
        and curv_ok
    ):
        confidence = "HIGH"
    elif n_seeds >= min_seeds_for_rates and evidence_coverage >= 80.0:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    scored = [
        (key, comp)
        for key, comp in components.items()
        if key != "curvature" and comp["available"] > 0
    ]
    strongest = max(scored, key=lambda item: item[1]["fraction"] if item[1]["fraction"] is not None else -1)
    weakest = min(scored, key=lambda item: item[1]["fraction"] if item[1]["fraction"] is not None else 2)

    # Findings (≤5)
    findings = [
        f"Clean topology recovery scored {topology_earned:.1f}/30.",
    ]
    destroyers = [
        mode
        for mode, detail in pca_modes_detail.items()
        if detail.get("destroys_circle_h1") and mode != "oracle"
    ]
    if destroyers:
        findings.append(
            f"{destroyers[0].replace('_', '-')} PCA erased the circle’s H1 loop "
            f"(components≈{pca_modes_detail[destroyers[0]]['circle_components']:.1f})."
        )
    else:
        findings.append(
            f"Production PCA mode `{production_mode}` retained "
            f"{100.0 * float(prod.get('mode_score', 0.0)):.0f}% of clean topology."
        )
    if noisy_dim_acc < 0.5:
        findings.append("TwoNN intrinsic dimension became unstable under noise.")
    else:
        findings.append(
            f"Intrinsic-dimension noise accuracy averaged {100.0 * noisy_dim_acc:.0f}%."
        )
    if not null_scored:
        findings.append(
            f"Null calibration was not scored because only {n_seeds} seed(s) were available."
        )
    else:
        findings.append(
            f"Null calibration scored {null_earned:.1f}/20 with {n_seeds} seeds."
        )
    if not null_scored:
        findings.append("Run at least 10 seeds before interpreting rejection rates.")
    elif n_finite < min_curvature:
        findings.append("Collect ≥10 finite curvature runs before using Ricci diagnostically.")
    else:
        findings.append("Increase seeds toward 30+ for HIGH confidence evidence.")
    findings = findings[:5]

    # One-sentence verdict
    if topology_earned >= 27 and (not null_scored or null_earned >= 12):
        if id_earned < 5 or (destroyers and production_mode in destroyers):
            verdict = (
                "Strong clean-topology recovery, but intrinsic-dimension calibration "
                "and/or PCA mode choice remain limiting."
            )
        else:
            verdict = (
                "Strong clean-topology recovery with solid PCA retention on the "
                "production preprocessor."
            )
    elif topology_earned >= 20:
        verdict = (
            "Moderate topology recovery; PCA preservation, noise robustness, or "
            "calibration still constrain the pipeline benchmark score."
        )
    else:
        verdict = (
            "Weak recovery of known synthetic topology; pipeline preprocessing or "
            "measurement settings need revision before production use."
        )
    if confidence == "LOW":
        verdict = verdict.rstrip(".") + "; statistical evidence remains limited."

    return {
        "overall_score": overall,
        "earned_points": earned_points,
        "available_points": available_points,
        "evidence_coverage": evidence_coverage,
        "confidence": confidence,
        "verdict": verdict,
        "components": components,
        "findings": findings,
        "strongest": strongest[0],
        "weakest": weakest[0],
        "n_seeds": n_seeds,
        "n_runs": int(len(frame)),
        "max_noise": max_noise,
        "production_pca_mode": production_mode,
        "topology_mode": topology_mode,
        "alpha": float(alpha),
    }


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html_report(
    result: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    runtime_seconds: float,
) -> str:
    """Self-contained HTML dashboard (no external assets)."""
    components = result["components"]
    order = [
        "topology_recovery",
        "pca_preservation",
        "noise_robustness",
        "null_calibration",
        "intrinsic_dimension",
        "curvature",
    ]

    def bar_row(key: str) -> str:
        comp = components[key]
        available = float(comp["available"])
        earned = float(comp["earned"])
        status = comp["status"]
        if available <= 0:
            fill = 0
            label = "N/A"
            width_note = "n/a"
        else:
            fill = 100.0 * earned / available
            label = f"{earned:.1f} / {available:.0f}"
            width_note = f"{fill:.0f}%"
        return f"""
        <div class="bar-block">
          <div class="bar-head">
            <span class="bar-label">{_html_escape(comp['label'])}</span>
            <span class="badge">{_html_escape(status)}</span>
            <span class="bar-score">{label}</span>
          </div>
          <div class="bar-track" role="img" aria-label="{_html_escape(comp['label'])} {width_note}">
            <div class="bar-fill" style="width:{fill if available > 0 else 0:.1f}%"></div>
          </div>
        </div>
        """

    # Detailed sections
    topo = components["topology_recovery"]["details"]
    pca = components["pca_preservation"]["details"]
    noise = components["noise_robustness"]["details"]
    null = components["null_calibration"]["details"]
    idet = components["intrinsic_dimension"]["details"]
    curv = components["curvature"]["details"]

    topo_rows = "".join(
        f"<tr><td>{_html_escape(name)}</td><td>{part['metric']:.3f}</td>"
        f"<td>{part['points']:.1f}</td><td>{part['max']:.0f}</td></tr>"
        for name, part in topo["parts"].items()
    )
    pca_rows = "".join(
        f"<tr><td>{_html_escape(mode)}"
        f"{' (production)' if mode == result['production_pca_mode'] else ''}"
        f"{' (diagnostic)' if detail['diagnostic_only'] else ''}</td>"
        f"<td>{detail['circle_h1']:.3f}</td><td>{detail['cluster_h0']:.3f}</td>"
        f"<td>{detail['circle_retention']:.2f}</td><td>{detail['cluster_retention']:.2f}</td>"
        f"<td>{detail['mode_score']:.2f}</td>"
        f"<td>{'destroys H1' if detail['destroys_circle_h1'] else 'ok'}</td></tr>"
        for mode, detail in pca["modes"].items()
    )

    if null["scored"]:
        null_rows = "".join(
            f"<tr><td>{_html_escape(name)}</td><td>{test['value']:.3f}</td>"
            f"<td>{test['unit']:.2f}</td><td>{_html_escape(test['kind'])}</td></tr>"
            for name, test in null["tests"].items()
        )
        null_block = f"""
        <p>Scored with {null['n_seeds']} seeds against the single low-rank Gaussian
        null. Axis-specific rates are kept separate. False-positive rates are
        measured on the linear-Gaussian datasets the null is meant to explain
        (isotropic and spiked); power is measured on the nonlinear ones.</p>
        <table><thead><tr><th>Test</th><th>Value</th><th>Unit score</th><th>Kind</th></tr></thead>
        <tbody>{null_rows}</tbody></table>
        """
    else:
        null_block = (
            "<p><strong>Null calibration: N/A</strong><br>"
            f"Reason: {_html_escape(null.get('reason') or 'insufficient seeds')}</p>"
        )

    id_rows = "".join(
        f"<tr><td>{_html_escape(row['dataset'])}</td><td>{row['expected']:.1f}</td>"
        f"<td>{row['clean_error']:.3f}</td><td>{row['noisy_error']:.3f}</td>"
        f"<td>{row['clean_accuracy']:.2f}</td><td>{row['noisy_accuracy']:.2f}</td></tr>"
        for row in idet["shapes"]
    )

    if curv.get("n_finite", 0) < MIN_CURVATURE_FOR_REPORT:
        curv_block = (
            "<p><strong>Curvature: N/A</strong><br>"
            f"Reason: {_html_escape(curv.get('reason') or 'insufficient repeated measurements')}</p>"
        )
    else:
        curv_block = f"""
        <p><strong>Curvature: EXPLORATORY</strong> (excluded from overall score)</p>
        <ul>
          <li>Mean curvature difference: {curv.get('mean_curvature_difference', float('nan')):.3f}</li>
          <li>Signed negative-fraction difference: {curv.get('signed_negative_fraction_difference', float('nan')):.3f}</li>
          <li>Curvature Wasserstein: {curv.get('curvature_wasserstein', float('nan')):.3f}</li>
          <li>Finite runs: {curv.get('n_finite', 0)}</li>
        </ul>
        """

    summary_rows = "".join(
        f"<tr><td>{_html_escape(components[key]['label'])}</td>"
        f"<td>{_fmt(components[key]['earned'])}</td>"
        f"<td>{'N/A' if components[key]['available'] <= 0 else _fmt(components[key]['available'])}</td>"
        f"<td><span class='badge'>{_html_escape(components[key]['status'])}</span></td>"
        f"<td>{_html_escape(_component_interpretation(key, components[key], result))}</td></tr>"
        for key in order
    )

    findings = "".join(f"<li>{_html_escape(item)}</li>" for item in result["findings"])
    conf_badge = "LOW CONFIDENCE" if result["confidence"] == "LOW" else f"{result['confidence']} CONFIDENCE"

    # Small inline SVG for component fractions
    svg_bars = []
    y = 20
    for key in order:
        comp = components[key]
        frac = comp["fraction"]
        width = 0 if frac is None else max(0.0, min(1.0, float(frac))) * 220
        label = comp["label"]
        score_txt = "N/A" if comp["available"] <= 0 else f"{comp['earned']:.1f}/{comp['available']:.0f}"
        svg_bars.append(
            f'<text x="0" y="{y}" class="svg-label">{_html_escape(label)}</text>'
            f'<rect x="130" y="{y-12}" width="220" height="14" fill="#e8e6e1"/>'
            f'<rect x="130" y="{y-12}" width="{width:.1f}" height="14" fill="#2f5d50"/>'
            f'<text x="360" y="{y}" class="svg-label">{score_txt}</text>'
        )
        y += 28
    svg = (
        f'<svg viewBox="0 0 430 {y}" width="100%" height="{y}" '
        f'role="img" aria-label="Component scores">'
        f"<style>.svg-label{{font:12px Georgia,serif;fill:#222}}</style>"
        + "".join(svg_bars)
        + "</svg>"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Pipeline benchmark score</title>
<style>
:root {{
  --ink:#1d1c1a; --muted:#5c584f; --line:#d9d4c8; --bg:#f7f4ee; --card:#fffdf8;
  --accent:#2f5d50; --warn:#8a5a2b; --weak:#7a3e3e;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; font-family:Georgia,"Times New Roman",serif; color:var(--ink);
  background:linear-gradient(180deg,#efe8da 0%, var(--bg) 220px, var(--bg) 100%);
  line-height:1.45;
}}
.wrap {{ max-width:980px; margin:0 auto; padding:28px 18px 48px; }}
h1,h2,h3 {{ font-weight:600; letter-spacing:0.01em; }}
h1 {{ font-size:1.8rem; margin:0 0 8px; }}
h2 {{ font-size:1.25rem; margin:28px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
h3 {{ font-size:1.05rem; margin:18px 0 8px; }}
.card {{
  background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:18px 18px 14px; box-shadow:0 1px 0 rgba(0,0,0,0.03);
}}
.headline {{
  display:grid; grid-template-columns:1.3fr 1fr; gap:16px; align-items:stretch;
}}
@media (max-width:800px) {{ .headline {{ grid-template-columns:1fr; }} }}
.score-xl {{ font-size:2.8rem; line-height:1; margin:8px 0; }}
.meta {{ color:var(--muted); font-size:0.95rem; }}
.verdict {{ margin-top:10px; font-size:1.05rem; }}
.badge {{
  display:inline-block; border:1px solid var(--line); border-radius:999px;
  padding:2px 8px; font-size:0.72rem; letter-spacing:0.04em; text-transform:uppercase;
  background:#f3efe6; color:var(--muted); white-space:nowrap;
}}
.badge.low {{ border-color:#c9a07a; color:var(--warn); }}
.bar-block {{ margin:10px 0 14px; }}
.bar-head {{ display:flex; gap:8px; align-items:center; justify-content:space-between; margin-bottom:4px; }}
.bar-label {{ font-weight:600; }}
.bar-score {{ font-variant-numeric:tabular-nums; color:var(--muted); font-size:0.92rem; }}
.bar-track {{ height:12px; background:#e8e6e1; border-radius:999px; overflow:hidden; }}
.bar-fill {{ height:100%; background:var(--accent); }}
table {{ width:100%; border-collapse:collapse; font-size:0.92rem; margin:8px 0 14px; }}
th, td {{ border-bottom:1px solid var(--line); padding:7px 6px; text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.03em; }}
ul {{ padding-left:1.15rem; }}
li {{ margin:4px 0; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
@media (max-width:800px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
.note {{ color:var(--muted); font-size:0.9rem; }}
code {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:0.88em; }}
</style>
</head>
<body>
<div class="wrap">
  <section class="card headline">
    <div>
      <div class="meta">Synthetic geometry pipeline benchmark</div>
      <h1>Pipeline benchmark score</h1>
      <div class="score-xl">{result['overall_score']:.1f} <span class="meta">/ 100</span></div>
      <div class="meta">
        Evidence coverage: {result['evidence_coverage']:.1f}%
        &nbsp;·&nbsp; Earned {result['earned_points']:.1f} of {result['available_points']:.1f} available points
      </div>
      <div style="margin-top:10px">
        <span class="badge {'low' if result['confidence']=='LOW' else ''}">{_html_escape(conf_badge)}</span>
      </div>
      <p class="verdict">{_html_escape(result['verdict'])}</p>
    </div>
    <div>
      <h3>Component snapshot</h3>
      {svg}
      <p class="note">Unmeasured components are N/A and excluded from the denominator.</p>
    </div>
  </section>

  <h2>2. Component scores</h2>
  <section class="card">
    {''.join(bar_row(key) for key in order)}
  </section>

  <h2>3. Main findings</h2>
  <section class="card"><ul>{findings}</ul></section>

  <h2>4. Detailed scoring</h2>
  <section class="card">
    <table>
      <thead><tr><th>Component</th><th>Earned</th><th>Available</th><th>Status</th><th>Interpretation</th></tr></thead>
      <tbody>{summary_rows}</tbody>
    </table>

    <h3>Topology recovery — 30 points</h3>
    <p class="note">Clean controls at noise=0 using mode <code>{_html_escape(topo['mode'])}</code>.
    Formulas: circle=clip(H1/0.15), clusters=clip(H0/0.25), line/tree=clip(1−H1/0.08).</p>
    <table><thead><tr><th>Control</th><th>Metric</th><th>Earned</th><th>Max</th></tr></thead>
    <tbody>{topo_rows}</tbody></table>

    <h3>PCA preservation — 20 points</h3>
    <p class="note">Baseline is no-PCA. Overall points use production mode
    <code>{_html_escape(result['production_pca_mode'])}</code> only. Oracle is diagnostic and does not raise the score.</p>
    <table><thead><tr><th>Mode</th><th>Circle H1</th><th>Cluster H0</th><th>Circle ret.</th><th>Cluster ret.</th><th>Mode score</th><th>Flag</th></tr></thead>
    <tbody>{pca_rows}</tbody></table>

    <h3>Noise robustness — 20 points</h3>
    <p class="note">Retention at max noise={noise['max_noise']} using production PCA. Non-loop controls are not rewarded for noise-created H1.</p>
    <table><thead><tr><th>Control</th><th>Clean</th><th>Max noise</th><th>Retained</th></tr></thead>
    <tbody>
      <tr><td>Circle H1</td><td>{noise['circle_clean']:.3f}</td><td>{noise['circle_noisy']:.3f}</td><td>{100*noise['circle_retention']:.1f}%</td></tr>
      <tr><td>Cluster H0</td><td>{noise['cluster_clean']:.3f}</td><td>{noise['cluster_noisy']:.3f}</td><td>{100*noise['cluster_retention']:.1f}%</td></tr>
    </tbody></table>
    <p>Earned: {components['noise_robustness']['earned']:.1f} / 20</p>

    <h3>Null calibration — 20 points</h3>
    {null_block}

    <h3>Intrinsic dimension — 10 points</h3>
    <p class="note">id_score = 0.6·clean_accuracy + 0.4·noisy_accuracy on line/circle/Swiss-roll/Y-tree.
    Null rejection belongs under null calibration, not here.</p>
    <table><thead><tr><th>Shape</th><th>Expected</th><th>Clean |err|</th><th>Noisy |err|</th><th>Clean acc</th><th>Noisy acc</th></tr></thead>
    <tbody>{id_rows}</tbody></table>
    <p>Earned: {components['intrinsic_dimension']['earned']:.1f} / 10
    (clean={idet['clean_accuracy']:.2f}, noisy={idet['noisy_accuracy']:.2f})</p>

    <h3>Curvature — diagnostic only</h3>
    {curv_block}
  </section>

  <h2>5. Configuration</h2>
  <section class="card">
    <div class="grid2">
      <ul>
        <li>Datasets: {_html_escape(', '.join(map(str, config.get('datasets', []))))}</li>
        <li>Sample size: {config.get('sample_size')}</li>
        <li>Ambient dimension: {config.get('ambient_dim')}</li>
        <li>Noise levels: {_html_escape(str(config.get('noise_levels')))}</li>
        <li>Seeds: {_html_escape(str(config.get('seeds')))} (n={result['n_seeds']})</li>
      </ul>
      <ul>
        <li>PCA modes: {_html_escape(str(config.get('pca_modes')))}</li>
        <li>Production PCA: <code>{_html_escape(result['production_pca_mode'])}</code></li>
        <li>Null draws: {config.get('n_nulls')}</li>
        <li>Alpha: {result['alpha']}</li>
        <li>Runtime: {runtime_seconds:.1f}s</li>
        <li>Total runs: {result['n_runs']}</li>
      </ul>
    </div>
  </section>

  <h2>6. Limitations and recommendations</h2>
  <section class="card">
    <ul>
      <li>{"Only " + str(result['n_seeds']) + " seeds: rejection-rate null calibration is N/A until ≥10 seeds." if result['n_seeds'] < MIN_SEEDS_FOR_RATE_CLAIMS else "Rejection-rate estimates use discrete Monte Carlo p-values with minimum 1/(n_nulls+1)."}</li>
      <li>Discrete Monte Carlo p-values are coarse; interpret rates cautiously near alpha.</li>
      <li>Some PCA selectors (notably parallel analysis) can collapse loop-bearing clouds and erase H1.</li>
      <li>TwoNN intrinsic-dimension estimates are noise-sensitive in ambient embeddings.</li>
      <li>{"Curvature has fewer than 10 finite measurements and stays exploratory/unavailable." if curv.get('n_finite', 0) < MIN_CURVATURE_FOR_REPORT else "Curvature is exploratory and excluded from the overall score."}</li>
      <li>This score evaluates predefined synthetic controls; it does not prove correctness on real LLM activations.</li>
    </ul>
  </section>
</div>
</body>
</html>
"""
    return html


def _component_interpretation(key: str, comp: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    if key == "null_calibration" and comp["available"] <= 0:
        return "Excluded from denominator; fewer than 10 seeds."
    if key == "curvature":
        return comp["details"].get("reason") or "Diagnostic only."
    if key == "pca_preservation":
        mode = result["production_pca_mode"]
        detail = comp["details"]["modes"].get(mode, {})
        if detail.get("destroys_circle_h1"):
            return f"Production mode {mode} destroys circle H1."
        return f"Production mode {mode} retention {100*float(detail.get('mode_score', 0)):.0f}%."
    if key == "topology_recovery":
        return f"Earned {comp['earned']:.1f} of 30 on clean controls."
    if key == "noise_robustness":
        return f"Mean topology retention at max noise: {100*float(comp['fraction'] or 0):.0f}%."
    if key == "intrinsic_dimension":
        d = comp["details"]
        return f"Clean acc {d['clean_accuracy']:.2f}; noisy acc {d['noisy_accuracy']:.2f}."
    return comp["status"]


def save_report(
    runs: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    runtime_seconds: float,
) -> Path:
    """Write the sole artifact: report.html. Remove any legacy outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    alpha = float(config.get("significance_alpha", 0.05))
    production_mode = str(config.get("production_pca_mode", PRODUCTION_PCA_DEFAULT))
    result = compute_pipeline_benchmark(
        runs,
        alpha=alpha,
        production_pca_mode=production_mode,
    )
    html = render_html_report(result, config, runtime_seconds=runtime_seconds)
    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")

    # Cleanup: only report.html should remain.
    for path in output_dir.rglob("*"):
        if path.is_file() and path.resolve() != report_path.resolve():
            path.unlink()
    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic null-calibration benchmark")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to YAML config (default: null_calibration.yaml)",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output_dir = Path(config.get("output_dir", "outputs/null_calibration"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    started = time.perf_counter()
    runs = run_benchmark(config)
    runtime_seconds = float(time.perf_counter() - started)
    report_path = save_report(runs, config, output_dir, runtime_seconds=runtime_seconds)

    result = compute_pipeline_benchmark(
        runs,
        alpha=float(config.get("significance_alpha", 0.05)),
        production_pca_mode=str(config.get("production_pca_mode", PRODUCTION_PCA_DEFAULT)),
    )
    print(f"Wrote: {report_path}")
    print(
        f"Pipeline benchmark score: {result['overall_score']:.1f} / 100 | "
        f"coverage {result['evidence_coverage']:.1f}% | confidence {result['confidence']}"
    )
    failures = runs[runs["failure"].fillna("").astype(str).str.len() > 0]
    if not failures.empty:
        print(f"Runs with structured failure info: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

