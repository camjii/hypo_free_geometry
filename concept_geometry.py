"""Hypothesis-free geometry fingerprints and clustering for concept clouds.

One fingerprint summarizes one entire concept dataset.  A row produced from
this module does not represent an individual prompt, token, or activation.
The measurements describe the point cloud through persistent homology and
Ollivier--Ricci curvature without comparing it with a named reference shape.

Ollivier--Ricci curvature depends on graph construction.  This module reuses
the project's matched-density epsilon rule and records the selected graph
parameters in metadata.  Persistence counts and total persistence can still
depend on sample count, so sample count is metadata rather than a clustering
feature.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

# GraphRicciCurvature's native shortest-path backend can conflict with other
# OpenMP runtimes on macOS. Keep the existing pipeline's single-thread setting
# before importing NumPy/SciPy-backed libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

# The pipeline saves figures by default and must remain safe in headless shells.
# This is intentionally unconditional and set before pyplot is imported.
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Constants and feature-column definitions
# ---------------------------------------------------------------------------


H0_FEATURE_COLUMNS = (
    "h0_count_finite",
    "h0_max_persistence",
    "h0_mean_persistence",
    "h0_total_persistence",
)

H1_FEATURE_COLUMNS = (
    "h1_count_finite",
    "h1_max_persistence",
    "h1_mean_persistence",
    "h1_total_persistence",
    "h1_second_max_persistence",
    "h1_dominance_ratio",
)

RICCI_FEATURE_COLUMNS = (
    "ricci_edge_count",
    "ricci_mean",
    "ricci_std",
    "ricci_min",
    "ricci_q25",
    "ricci_median",
    "ricci_q75",
    "ricci_max",
    "ricci_fraction_negative",
    "ricci_fraction_zero",
    "ricci_fraction_positive",
)

PH_FEATURE_COLUMNS = H0_FEATURE_COLUMNS + H1_FEATURE_COLUMNS
GEOMETRY_FEATURE_COLUMNS = PH_FEATURE_COLUMNS + RICCI_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 2. Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryFeatureResult:
    """A numerical fingerprint and provenance for one complete concept cloud."""

    concept_name: str
    features: dict[str, float]
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# 3. Input validation and preprocessing
# ---------------------------------------------------------------------------


def validate_point_cloud(
    point_cloud: Any,
    *,
    minimum_points: int = 3,
) -> np.ndarray:
    """Return a validated floating-point ``[n_samples, n_features]`` array."""

    try:
        cloud = np.asarray(point_cloud, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("point_cloud must be a two-dimensional numerical array") from exc

    if cloud.ndim != 2:
        raise ValueError(
            "point_cloud must be a two-dimensional array shaped "
            "[n_samples, n_features]"
        )
    if cloud.shape[0] < minimum_points:
        raise ValueError(
            f"point_cloud must contain at least {minimum_points} points; "
            f"received {cloud.shape[0]}"
        )
    if cloud.shape[1] < 1:
        raise ValueError("point_cloud must contain at least one feature dimension")
    if not np.isfinite(cloud).all():
        raise ValueError("point_cloud must contain only finite values")

    return cloud


def reduce_point_cloud_pca(
    point_cloud: Any,
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, PCA, dict[str, float | int]]:
    """Fit PCA per cloud and retain the requested cumulative variance.

    This is the model-independent equivalent of ``Pipeline.reduce_pca``.  PCA
    is preprocessing here; the separate two-component PCA in clustering is
    only a visualization.
    """

    cloud = validate_point_cloud(point_cloud, minimum_points=2)
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in the interval (0, 1]")

    pca = PCA(n_components=min(cloud.shape))
    full_projection = pca.fit_transform(cloud)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    if not np.isfinite(cumulative_variance).all():
        raise ValueError("point_cloud has insufficient variance for PCA preprocessing")

    component_count = int(
        np.searchsorted(cumulative_variance, variance_threshold, side="left") + 1
    )
    component_count = min(component_count, full_projection.shape[1])
    projection = full_projection[:, :component_count]
    metadata: dict[str, float | int] = {
        "pca_variance_threshold": float(variance_threshold),
        "pca_selected_components": component_count,
        "pca_explained_variance": float(cumulative_variance[component_count - 1]),
    }
    return projection, pca, metadata


def normalize_point_cloud_diameter(
    point_cloud: Any,
) -> tuple[np.ndarray, float]:
    """Scale a point cloud to unit diameter, matching ``Manifold._measure``."""

    cloud = validate_point_cloud(point_cloud, minimum_points=2)
    distances = pdist(cloud, metric="euclidean")
    diameter = float(np.max(distances))
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("point_cloud must contain at least two distinct points")
    return cloud / diameter, diameter


# ---------------------------------------------------------------------------
# 4. Persistent-homology summaries
# ---------------------------------------------------------------------------


def compute_persistence_diagrams(
    point_cloud: Any,
    *,
    max_dimension: int = 1,
) -> list[np.ndarray]:
    """Compute Vietoris--Rips persistence diagrams using the existing backend."""

    if max_dimension < 0:
        raise ValueError("max_dimension must be non-negative")
    cloud = validate_point_cloud(point_cloud, minimum_points=2)
    try:
        from ripser import ripser
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep
        raise ImportError(
            "persistent homology requires the 'ripser' package"
        ) from exc

    result = ripser(
        cloud,
        maxdim=max_dimension,
        distance_matrix=False,
        do_cocycles=False,
        n_perm=None,
    )
    return [np.asarray(diagram, dtype=float) for diagram in result["dgms"]]


def get_finite_persistences(persistence_diagram: Any) -> np.ndarray:
    """Return ``death - birth`` for finite bars, excluding infinite deaths."""

    diagram = np.asarray(persistence_diagram, dtype=float)
    if diagram.size == 0:
        return np.empty(0, dtype=float)
    if diagram.ndim != 2 or diagram.shape[1] != 2:
        raise ValueError("persistence_diagram must be shaped [n_pairs, 2]")

    finite_pairs = diagram[np.isfinite(diagram).all(axis=1)]
    persistences = finite_pairs[:, 1] - finite_pairs[:, 0]
    if np.any(persistences < 0.0):
        raise ValueError("persistence deaths must be greater than or equal to births")
    return persistences


def summarize_persistence_diagram(persistence_diagram: Any) -> dict[str, float]:
    """Summarize finite bars without imposing a significance threshold."""

    persistences = get_finite_persistences(persistence_diagram)
    if persistences.size == 0:
        return {
            "count_finite": 0.0,
            "max_persistence": 0.0,
            "mean_persistence": 0.0,
            "total_persistence": 0.0,
            "second_max_persistence": 0.0,
            "dominance_ratio": 0.0,
        }

    ordered = np.sort(persistences)[::-1]
    total = float(np.sum(ordered))
    return {
        "count_finite": float(ordered.size),
        "max_persistence": float(ordered[0]),
        "mean_persistence": float(np.mean(ordered)),
        "total_persistence": total,
        "second_max_persistence": float(ordered[1]) if ordered.size > 1 else 0.0,
        "dominance_ratio": float(ordered[0] / total) if total > 0.0 else 0.0,
    }


def extract_persistence_features(
    persistence_diagrams: Sequence[Any],
) -> dict[str, float]:
    """Extract the deterministic H0/H1 feature set from persistence diagrams."""

    if len(persistence_diagrams) < 1:
        raise ValueError("persistence_diagrams must contain an H0 diagram")

    empty_diagram = np.empty((0, 2), dtype=float)
    h0 = summarize_persistence_diagram(persistence_diagrams[0])
    h1 = summarize_persistence_diagram(
        persistence_diagrams[1] if len(persistence_diagrams) > 1 else empty_diagram
    )
    features = {
        "h0_count_finite": h0["count_finite"],
        "h0_max_persistence": h0["max_persistence"],
        "h0_mean_persistence": h0["mean_persistence"],
        "h0_total_persistence": h0["total_persistence"],
        "h1_count_finite": h1["count_finite"],
        "h1_max_persistence": h1["max_persistence"],
        "h1_mean_persistence": h1["mean_persistence"],
        "h1_total_persistence": h1["total_persistence"],
        "h1_second_max_persistence": h1["second_max_persistence"],
        "h1_dominance_ratio": h1["dominance_ratio"],
    }
    return {column: float(features[column]) for column in PH_FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# 5. Ollivier--Ricci graph construction and summaries
# ---------------------------------------------------------------------------


def select_matched_density_epsilon(
    point_cloud: Any,
    *,
    target_density: float = 0.10,
) -> float:
    """Select epsilon by the existing pairwise-distance quantile rule."""

    cloud = validate_point_cloud(point_cloud, minimum_points=2)
    if not 0.0 < target_density <= 1.0:
        raise ValueError("target_density must be in the interval (0, 1]")

    distances = np.sort(pdist(cloud, metric="euclidean"))
    distance_index = min(int(target_density * len(distances)), len(distances) - 1)
    epsilon = float(distances[distance_index])
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            "matched-density graph selection produced a non-positive epsilon; "
            "check for duplicate or degenerate points"
        )
    return epsilon


def build_epsilon_graph(point_cloud: Any, epsilon: float) -> nx.Graph:
    """Build the project's weighted Euclidean epsilon-neighborhood graph."""

    cloud = validate_point_cloud(point_cloud, minimum_points=2)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a positive finite number")

    adjacency = radius_neighbors_graph(
        cloud,
        radius=float(epsilon),
        mode="distance",
        metric="euclidean",
        include_self=False,
    )
    graph = nx.from_scipy_sparse_array(adjacency, create_using=nx.Graph)
    if graph.number_of_edges() == 0:
        raise ValueError(
            "epsilon-neighborhood graph has no usable edges; curvature cannot be computed"
        )
    return graph


def compute_ollivier_ricci_curvatures(
    graph: nx.Graph,
    *,
    alpha: float = 0.5,
    processes: int = 1,
    verbose: str = "ERROR",
) -> tuple[nx.Graph, np.ndarray]:
    """Compute and collect Ollivier--Ricci curvature for every graph edge."""

    if graph.number_of_edges() == 0:
        raise ValueError("graph has no usable edges; curvature cannot be computed")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in the interval [0, 1]")
    if processes < 1:
        raise ValueError("processes must be at least 1")
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep
        raise ImportError(
            "Ollivier--Ricci curvature requires the 'GraphRicciCurvature' package"
        ) from exc

    calculator = OllivierRicci(
        graph,
        alpha=float(alpha),
        weight="weight",
        proc=int(processes),
        verbose=verbose,
    )
    curved_graph = calculator.compute_ricci_curvature()
    values = np.asarray(
        [
            edge_data.get("ricciCurvature", np.nan)
            for _, _, edge_data in curved_graph.edges(data=True)
        ],
        dtype=float,
    )
    if values.size != graph.number_of_edges() or not np.isfinite(values).all():
        raise RuntimeError(
            "Ollivier--Ricci computation did not produce one finite value per edge"
        )
    return curved_graph, values


def summarize_curvature_values(
    curvature_values: Any,
    *,
    zero_tolerance: float = 1e-8,
) -> dict[str, float]:
    """Summarize a non-empty array of finite edge curvatures."""

    values = np.asarray(curvature_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("curvature_values must be a one-dimensional array")
    if values.size == 0:
        raise ValueError("curvature_values must contain at least one edge value")
    if not np.isfinite(values).all():
        raise ValueError("curvature_values must contain only finite values")
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be a non-negative finite number")

    negative = values < -zero_tolerance
    zero = np.abs(values) <= zero_tolerance
    positive = values > zero_tolerance
    features = {
        "ricci_edge_count": float(values.size),
        "ricci_mean": float(np.mean(values)),
        "ricci_std": float(np.std(values)),
        "ricci_min": float(np.min(values)),
        "ricci_q25": float(np.quantile(values, 0.25)),
        "ricci_median": float(np.median(values)),
        "ricci_q75": float(np.quantile(values, 0.75)),
        "ricci_max": float(np.max(values)),
        "ricci_fraction_negative": float(np.mean(negative)),
        "ricci_fraction_zero": float(np.mean(zero)),
        "ricci_fraction_positive": float(np.mean(positive)),
    }
    return {column: float(features[column]) for column in RICCI_FEATURE_COLUMNS}


# ---------------------------------------------------------------------------
# 6. Geometry feature extraction
# ---------------------------------------------------------------------------


def extract_geometry_features(
    point_cloud: Any,
    concept_name: str,
    *,
    pca_variance_threshold: float = 0.95,
    graph_density: float = 0.10,
    curvature_alpha: float = 0.5,
    curvature_processes: int = 1,
    curvature_verbose: str = "ERROR",
    curvature_zero_tolerance: float = 1e-8,
) -> GeometryFeatureResult:
    """Return one PH-plus-Ricci fingerprint for one complete concept dataset."""

    if not isinstance(concept_name, str) or not concept_name.strip():
        raise ValueError("concept_name must be a non-empty string")
    cloud = validate_point_cloud(point_cloud)

    projected, _, pca_metadata = reduce_point_cloud_pca(
        cloud, variance_threshold=pca_variance_threshold
    )
    normalized, diameter = normalize_point_cloud_diameter(projected)

    diagrams = compute_persistence_diagrams(normalized, max_dimension=1)
    persistence_features = extract_persistence_features(diagrams)

    epsilon = select_matched_density_epsilon(
        normalized, target_density=graph_density
    )
    graph = build_epsilon_graph(normalized, epsilon)
    curved_graph, curvature_values = compute_ollivier_ricci_curvatures(
        graph,
        alpha=curvature_alpha,
        processes=curvature_processes,
        verbose=curvature_verbose,
    )
    curvature_features = summarize_curvature_values(
        curvature_values, zero_tolerance=curvature_zero_tolerance
    )

    combined = {**persistence_features, **curvature_features}
    features = {column: float(combined[column]) for column in GEOMETRY_FEATURE_COLUMNS}
    if not np.isfinite(np.fromiter(features.values(), dtype=float)).all():
        raise RuntimeError("geometry feature extraction produced a non-finite value")

    metadata: dict[str, Any] = {
        "number_of_samples": int(cloud.shape[0]),
        "activation_dimension": int(cloud.shape[1]),
        "preprocessing": {
            **pca_metadata,
            "diameter_before_normalization": diameter,
            "diameter_normalized": True,
        },
        "graph_node_count": int(curved_graph.number_of_nodes()),
        "graph_edge_count": int(curved_graph.number_of_edges()),
        "graph_density": float(nx.density(curved_graph)),
        "selected_epsilon": epsilon,
        "graph_method": "matched-density weighted epsilon-neighborhood graph",
        "graph_density_target": float(graph_density),
        "persistent_homology_settings": {
            "backend": "ripser",
            "complex": "Vietoris-Rips",
            "max_dimension": 1,
            "distance_matrix": False,
            "infinite_deaths_excluded_from_summaries": True,
        },
        "curvature_settings": {
            "backend": "GraphRicciCurvature.OllivierRicci",
            "alpha": float(curvature_alpha),
            "processes": int(curvature_processes),
            "verbose": curvature_verbose,
            "edge_weight_attribute": "weight",
            "zero_tolerance": float(curvature_zero_tolerance),
        },
    }
    return GeometryFeatureResult(
        concept_name=concept_name,
        features=features,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 7. Feature-table construction and saving
# ---------------------------------------------------------------------------


def build_concept_feature_table(
    concept_clouds: Mapping[str, Any],
    **feature_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Build one deterministic feature row per complete concept point cloud."""

    if not isinstance(concept_clouds, Mapping) or not concept_clouds:
        raise ValueError("concept_clouds must be a non-empty mapping")

    rows: list[dict[str, str | float]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for concept_name, point_cloud in concept_clouds.items():
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ValueError("every concept name must be a non-empty string")
        try:
            result = extract_geometry_features(
                point_cloud,
                concept_name,
                **feature_kwargs,
            )
        except Exception as exc:
            raise ValueError(
                f"geometry feature extraction failed for concept {concept_name!r}: {exc}"
            ) from exc

        rows.append(
            {
                "concept": result.concept_name,
                **{
                    column: float(result.features[column])
                    for column in GEOMETRY_FEATURE_COLUMNS
                },
            }
        )
        metadata[result.concept_name] = result.metadata

    feature_table = pd.DataFrame(
        rows,
        columns=["concept", *GEOMETRY_FEATURE_COLUMNS],
    )
    numeric_values = feature_table.loc[:, GEOMETRY_FEATURE_COLUMNS].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric_values).all():
        raise RuntimeError("concept feature table contains NaN or infinite values")
    return feature_table, metadata


def save_concept_feature_outputs(
    feature_table: pd.DataFrame,
    metadata: Mapping[str, Any],
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Save the feature table as CSV and its non-feature metadata as JSON."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "geometry_features.csv"
    metadata_path = output_path / "geometry_metadata.json"
    feature_table.to_csv(csv_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return csv_path, metadata_path


# ---------------------------------------------------------------------------
# 8. Feature-group selection
# ---------------------------------------------------------------------------


def select_feature_columns(
    feature_table: pd.DataFrame,
    feature_groups: Sequence[str] = ("ph", "ricci"),
) -> list[str]:
    """Select PH, Ricci, or combined columns in canonical deterministic order."""

    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError("feature_table must be a pandas DataFrame")
    groups = tuple(feature_groups)
    if not groups:
        raise ValueError("feature_groups must select at least one feature group")
    unknown_groups = set(groups) - {"ph", "ricci"}
    if unknown_groups:
        raise ValueError(
            "unknown feature groups: " + ", ".join(sorted(unknown_groups))
        )

    requested_columns: list[str] = []
    if "ph" in groups:
        requested_columns.extend(PH_FEATURE_COLUMNS)
    if "ricci" in groups:
        requested_columns.extend(RICCI_FEATURE_COLUMNS)

    missing_columns = [
        column for column in requested_columns if column not in feature_table.columns
    ]
    if missing_columns:
        raise ValueError(
            "feature_table is missing selected columns: " + ", ".join(missing_columns)
        )
    return requested_columns


# ---------------------------------------------------------------------------
# 9. Clustering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptClusteringResult:
    """Artifacts from clustering one fingerprint row per concept dataset."""

    original_feature_table: pd.DataFrame
    selected_feature_columns: list[str]
    standardized_feature_matrix: np.ndarray
    concept_names: list[str]
    linkage_matrix: np.ndarray
    flat_cluster_labels: np.ndarray | None
    pca_coordinates: np.ndarray
    scaler: StandardScaler
    pca_model: PCA


def cluster_concepts(
    feature_table: pd.DataFrame,
    feature_groups: Sequence[str] = ("ph", "ricci"),
    n_clusters: int | None = None,
    linkage_method: str = "ward",
) -> ConceptClusteringResult:
    """Cluster concepts using their full standardized geometry fingerprints."""

    if "concept" not in feature_table.columns:
        raise ValueError("feature_table must contain a 'concept' column")
    concept_names = feature_table["concept"].astype(str).tolist()
    if len(concept_names) < 2:
        raise ValueError("hierarchical clustering requires at least two concepts")
    if len(set(concept_names)) != len(concept_names):
        raise ValueError("feature_table concept names must be unique")

    selected_columns = select_feature_columns(feature_table, feature_groups)
    selected_values = feature_table.loc[:, selected_columns].to_numpy(dtype=float)
    if not np.isfinite(selected_values).all():
        raise ValueError("selected clustering features must contain only finite values")
    if not np.any(np.std(selected_values, axis=0) > 0.0):
        raise ValueError("at least one selected feature must vary across concepts")

    scaler = StandardScaler()
    standardized = scaler.fit_transform(selected_values)
    linkage_matrix = linkage(standardized, method=linkage_method)

    flat_labels: np.ndarray | None = None
    if n_clusters is not None:
        if not isinstance(n_clusters, int) or isinstance(n_clusters, bool):
            raise ValueError("n_clusters must be an integer or None")
        if not 1 <= n_clusters <= len(concept_names):
            raise ValueError(
                "n_clusters must be between 1 and the number of concepts"
            )
        flat_labels = fcluster(
            linkage_matrix,
            t=n_clusters,
            criterion="maxclust",
        ).astype(int) - 1

    pca_model = PCA(n_components=2)
    pca_coordinates = pca_model.fit_transform(standardized)
    return ConceptClusteringResult(
        original_feature_table=feature_table.copy(),
        selected_feature_columns=list(selected_columns),
        standardized_feature_matrix=standardized,
        concept_names=concept_names,
        linkage_matrix=linkage_matrix,
        flat_cluster_labels=flat_labels,
        pca_coordinates=pca_coordinates,
        scaler=scaler,
        pca_model=pca_model,
    )


# ---------------------------------------------------------------------------
# 10. Plotting
# ---------------------------------------------------------------------------


def _save_or_show_figure(
    figure: Figure,
    *,
    save_path: str | Path | None,
    show: bool,
) -> None:
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    if show:
        plt.show()


def plot_concept_dendrogram(
    result: ConceptClusteringResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot the hierarchical clustering tree with concept-labeled leaves."""

    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    dendrogram(
        result.linkage_matrix,
        labels=result.concept_names,
        leaf_rotation=30.0,
        leaf_font_size=10.0,
        ax=axis,
    )
    axis.set_title("Hierarchical clustering of concept geometry fingerprints")
    axis.set_xlabel("Concept dataset")
    axis.set_ylabel("Linkage distance")
    figure.tight_layout()
    _save_or_show_figure(figure, save_path=save_path, show=show)
    return figure


def plot_concept_feature_pca(
    result: ConceptClusteringResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot a labeled 2D PCA view; clustering still uses all selected features."""

    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    color_values: np.ndarray | str
    if result.flat_cluster_labels is None:
        color_values = "#2563eb"
    else:
        color_values = result.flat_cluster_labels
    scatter = axis.scatter(
        result.pca_coordinates[:, 0],
        result.pca_coordinates[:, 1],
        c=color_values,
        cmap="tab10" if result.flat_cluster_labels is not None else None,
        s=72,
        edgecolor="white",
        linewidth=0.7,
        zorder=2,
    )
    for concept_name, (x_coordinate, y_coordinate) in zip(
        result.concept_names, result.pca_coordinates
    ):
        axis.annotate(
            concept_name,
            (x_coordinate, y_coordinate),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    explained = result.pca_model.explained_variance_ratio_ * 100.0
    axis.set_xlabel(f"PC1 ({explained[0]:.1f}% explained variance)")
    axis.set_ylabel(f"PC2 ({explained[1]:.1f}% explained variance)")
    axis.set_title(
        "Geometry fingerprint PCA (visualization only; clustering uses full space)"
    )
    axis.grid(alpha=0.2)
    if result.flat_cluster_labels is not None:
        handles, _ = scatter.legend_elements()
        neutral_labels = [
            f"cluster {cluster_id}"
            for cluster_id in sorted(set(result.flat_cluster_labels.tolist()))
        ]
        axis.legend(handles, neutral_labels, title="Neutral cluster ID")
    figure.tight_layout()
    _save_or_show_figure(figure, save_path=save_path, show=show)
    return figure


def plot_feature_heatmap(
    result: ConceptClusteringResult,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot standardized geometry features with concepts as rows."""

    feature_count = len(result.selected_feature_columns)
    figure_width = max(10.0, 0.48 * feature_count)
    figure_height = max(4.0, 0.55 * len(result.concept_names) + 2.0)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    image = axis.imshow(
        result.standardized_feature_matrix,
        aspect="auto",
        cmap="coolwarm",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(feature_count))
    axis.set_xticklabels(
        result.selected_feature_columns,
        rotation=55,
        ha="right",
        fontsize=8,
    )
    axis.set_yticks(np.arange(len(result.concept_names)))
    axis.set_yticklabels(result.concept_names)
    axis.set_xlabel("Geometry feature (standardized)")
    axis.set_ylabel("Concept dataset")
    axis.set_title("Standardized concept geometry fingerprints")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Standardized value")
    figure.tight_layout()
    _save_or_show_figure(figure, save_path=save_path, show=show)
    return figure


# Existing callers used this longer name; an alias preserves that API without
# maintaining a second plotting implementation.
plot_standardized_feature_heatmap = plot_feature_heatmap


# ---------------------------------------------------------------------------
# 11. Demo data
# ---------------------------------------------------------------------------


def create_demo_concept_clouds(seed: int = 7) -> dict[str, np.ndarray]:
    """Create small neutral fixtures that exercise different cloud structures.

    The fixture generator knows how the clouds were made, but those labels and
    generation parameters are not given to the clustering algorithm and do not
    constitute a predefined-shape classifier.
    """

    rng = np.random.default_rng(seed)
    sample_count = 28
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)

    demo_01 = np.column_stack((np.cos(angles), np.sin(angles)))
    demo_01 += rng.normal(scale=0.045, size=demo_01.shape)

    shifted_angles = angles + 0.12
    demo_02 = 1.15 * np.column_stack(
        (np.cos(shifted_angles), np.sin(shifted_angles))
    )
    demo_02 += rng.normal(scale=0.065, size=demo_02.shape)

    coordinate = np.linspace(-1.0, 1.0, sample_count)
    demo_03 = np.column_stack(
        (coordinate, 0.05 * rng.normal(size=sample_count))
    )

    demo_04 = rng.uniform(-1.0, 1.0, size=(sample_count, 2))
    demo_05 = rng.normal(size=(sample_count, 4))

    return {
        "demo_01": demo_01,
        "demo_02": demo_02,
        "demo_03": demo_03,
        "demo_04": demo_04,
        "demo_05": demo_05,
    }


# ---------------------------------------------------------------------------
# 12. Command-line interface
# ---------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one PH-plus-Ricci fingerprint per concept and cluster the rows."
        )
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the bundled small synthetic demonstration",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="number of neutral flat cluster IDs to report (default: 3)",
    )
    parser.add_argument(
        "--graph-density",
        type=float,
        default=0.10,
        help="shared matched-density graph target for every concept (default: 0.10)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "concept_clustering",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.demo:
        raise SystemExit(
            "No activation-file loader is connected yet. Re-run with --demo."
        )

    concept_clouds = create_demo_concept_clouds(seed=args.seed)
    warnings.warn(
        "Clustering only a few concept datasets is exploratory; expand and "
        "repeat the analysis before drawing conclusions.",
        stacklevel=1,
    )
    feature_table, metadata = build_concept_feature_table(
        concept_clouds,
        graph_density=args.graph_density,
    )
    result = cluster_concepts(
        feature_table,
        n_clusters=args.n_clusters,
    )

    output_directory = args.output_dir.resolve()
    save_concept_feature_outputs(feature_table, metadata, output_directory)
    figures = [
        plot_concept_dendrogram(
            result, save_path=output_directory / "dendrogram.png"
        ),
        plot_concept_feature_pca(
            result, save_path=output_directory / "feature_pca.png"
        ),
        plot_feature_heatmap(
            result, save_path=output_directory / "feature_heatmap.png"
        ),
    ]

    print("\nGeometry feature table (one row per complete concept dataset):")
    print(feature_table.to_string(index=False))
    print("\nNeutral flat cluster assignments:")
    if result.flat_cluster_labels is None:
        print("  no flat labels requested")
    else:
        for concept_name, cluster_id in zip(
            result.concept_names, result.flat_cluster_labels
        ):
            print(f"  {concept_name}: cluster {cluster_id}")
    print(f"\nSaved outputs to {output_directory}")

    for figure in figures:
        plt.close(figure)
    return 0


__all__ = [
    "ConceptClusteringResult",
    "GEOMETRY_FEATURE_COLUMNS",
    "H0_FEATURE_COLUMNS",
    "H1_FEATURE_COLUMNS",
    "PH_FEATURE_COLUMNS",
    "RICCI_FEATURE_COLUMNS",
    "GeometryFeatureResult",
    "build_concept_feature_table",
    "build_epsilon_graph",
    "cluster_concepts",
    "compute_ollivier_ricci_curvatures",
    "compute_persistence_diagrams",
    "create_demo_concept_clouds",
    "extract_geometry_features",
    "extract_persistence_features",
    "get_finite_persistences",
    "main",
    "normalize_point_cloud_diameter",
    "plot_concept_dendrogram",
    "plot_concept_feature_pca",
    "plot_feature_heatmap",
    "plot_standardized_feature_heatmap",
    "reduce_point_cloud_pca",
    "save_concept_feature_outputs",
    "select_feature_columns",
    "select_matched_density_epsilon",
    "summarize_curvature_values",
    "summarize_persistence_diagram",
    "validate_point_cloud",
]


if __name__ == "__main__":
    raise SystemExit(main())
