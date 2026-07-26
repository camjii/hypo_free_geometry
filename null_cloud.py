"""
Compare a concept manifold against its own null.

    manifold = Manifold(pipeline, activations, seed=42)
    result = ManifoldComparator().compare_against_nulls(
        manifold, kind="covariance_gaussian", n_nulls=30, base_seed=100,
    )

The null manifolds have the same sample count and activation dimension as the
observed manifold. Each null is generated from statistics estimated from the
observed activations and is measured using the same selected metric pipeline.

The covariance-Gaussian null preserves the estimated mean and covariance. The
comparison therefore tests whether the observed cloud contains structure beyond
what a Gaussian model with the same first- and second-order statistics explains.

Intrinsic dimension is descriptive by default: TwoNN is high-variance at the
sample sizes typical for activation clouds, so ID enters a significance test
only when ``infer_intrinsic_dimension=True``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import persim
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.covariance import EmpiricalCovariance, LedoitWolf


SUPPORTED_METRICS = (
    "intrinsic_dimension",
    "topology",
    "curvature",
)
DEFAULT_METRICS = SUPPORTED_METRICS
COVARIANCE_ESTIMATORS = (
    "ledoit_wolf",
    "regularized_empirical",
)


def _finite(dgm):
    """Strip the [0, inf] H0 bar -- persim returns inf otherwise."""
    d = np.asarray(dgm, dtype=float)
    if d.size == 0:
        return np.empty((0, 2), dtype=float)

    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError(
            "persistence diagram must have shape [number_of_bars, 2]"
        )

    return d[np.isfinite(d).all(axis=1)]


def _normalize_metrics(metrics: Sequence[str] | None) -> tuple[str, ...]:
    if metrics is None:
        return DEFAULT_METRICS
    if isinstance(metrics, str):
        raise TypeError("metrics must be a sequence of metric names, not a string")
    selected = tuple(dict.fromkeys(str(name) for name in metrics))
    if not selected:
        raise ValueError("metrics must select at least one of "
                         + ", ".join(SUPPORTED_METRICS))
    unknown = [name for name in selected if name not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(
            "unknown metrics: "
            + ", ".join(unknown)
            + ". Supported: "
            + ", ".join(SUPPORTED_METRICS)
        )
    return selected


def _effective_rank(covariance: np.ndarray) -> float:
    eigenvalues = np.clip(np.linalg.eigvalsh(np.asarray(covariance, dtype=float)), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    probs = eigenvalues / total
    return float(np.exp(-np.sum(probs * np.log(probs + 1e-300))))


def fit_null_gaussian(
    cloud: np.ndarray,
    *,
    kind: str,
    covariance_estimator: str = "ledoit_wolf",
) -> dict[str, Any]:
    """Fit null parameters and return sampling diagnostics."""
    x = np.asarray(cloud, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("cloud must have shape [n_samples, n_features] with n_samples >= 2")

    mean = x.mean(axis=0)
    empir = EmpiricalCovariance(store_precision=False).fit(x)
    empir_cov = np.asarray(empir.covariance_, dtype=float)

    if kind == "noise":
        kind = "covariance_gaussian"
    if kind == "isotropic_gaussian":
        average_variance = float(np.var(x, axis=0, ddof=1).mean())
        if average_variance <= 0:
            raise ValueError("isotropic Gaussian null requires positive variance")
        cov = average_variance * np.eye(x.shape[1], dtype=float)
        return {
            "kind": kind,
            "mean": mean,
            "covariance": cov,
            "covariance_estimator": "isotropic_average_variance",
            "average_variance": average_variance,
            "diagnostics": {
                "mean_l2_norm": float(np.linalg.norm(mean)),
                "empirical_cov_frobenius": float(np.linalg.norm(empir_cov)),
                "null_cov_frobenius": float(np.linalg.norm(cov)),
                "covariance_frobenius_difference": float(np.linalg.norm(cov - empir_cov)),
                "empirical_effective_rank": _effective_rank(empir_cov),
                "null_effective_rank": _effective_rank(cov),
                "top_empirical_eigenvalues": np.sort(np.linalg.eigvalsh(empir_cov))[::-1][:5].tolist(),
                "top_null_eigenvalues": np.sort(np.linalg.eigvalsh(cov))[::-1][:5].tolist(),
            },
        }

    if kind != "covariance_gaussian":
        raise ValueError(
            f"unknown null kind: {kind}. Use 'covariance_gaussian', "
            "'isotropic_gaussian', or 'noise'."
        )

    if covariance_estimator not in COVARIANCE_ESTIMATORS:
        raise ValueError(
            f"unknown covariance_estimator: {covariance_estimator}. "
            f"Use one of {COVARIANCE_ESTIMATORS}."
        )

    if covariance_estimator == "ledoit_wolf":
        estimator = LedoitWolf(store_precision=False).fit(x)
        cov = np.asarray(estimator.covariance_, dtype=float)
        location = np.asarray(estimator.location_, dtype=float)
        extras = {
            "shrinkage": float(getattr(estimator, "shrinkage_", float("nan"))),
        }
    else:
        # Ridge-stabilised empirical covariance for finite-sample / d ≳ n cases.
        location = mean
        eigenvalues = np.linalg.eigvalsh(empir_cov)
        ridge = max(1e-6, 1e-3 * float(np.mean(np.clip(eigenvalues, 0.0, None)) + 1e-12))
        cov = empir_cov + ridge * np.eye(empir_cov.shape[0], dtype=float)
        extras = {"ridge": ridge}

    return {
        "kind": kind,
        "mean": location,
        "covariance": cov,
        "covariance_estimator": covariance_estimator,
        "diagnostics": {
            "mean_l2_norm": float(np.linalg.norm(location)),
            "mean_difference_l2": float(np.linalg.norm(location - mean)),
            "empirical_cov_frobenius": float(np.linalg.norm(empir_cov)),
            "null_cov_frobenius": float(np.linalg.norm(cov)),
            "covariance_frobenius_difference": float(np.linalg.norm(cov - empir_cov)),
            "relative_covariance_frobenius_difference": float(
                np.linalg.norm(cov - empir_cov) / (np.linalg.norm(empir_cov) + 1e-12)
            ),
            "empirical_effective_rank": _effective_rank(empir_cov),
            "null_effective_rank": _effective_rank(cov),
            "effective_rank_difference": _effective_rank(cov) - _effective_rank(empir_cov),
            "top_empirical_eigenvalues": np.sort(np.linalg.eigvalsh(empir_cov))[::-1][:5].tolist(),
            "top_null_eigenvalues": np.sort(np.linalg.eigvalsh(cov))[::-1][:5].tolist(),
            **extras,
        },
    }


def sample_null_cloud(
    fit: Mapping[str, Any],
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.multivariate_normal(
        mean=np.asarray(fit["mean"], dtype=float),
        cov=np.asarray(fit["covariance"], dtype=float),
        size=int(sample_count),
    )


class Manifold:
    """A measured cloud: ID, persistence diagrams, curvature signature.

    pipeline: exposes get_intrinsic_dim, reduce_pca, create_persistence_diagram,
              create_epsilon_graph, compute_ollivier_ricci
    """

    def __init__(
        self,
        pipeline,
        opt_activations,
        cloud=None,
        label="concept",
        seed=0,
        eps_density=0.10,
        var_threshold=0.95,
        metrics: Sequence[str] | None = None,
        covariance_estimator: str = "ledoit_wolf",
    ):
        self.pipeline = pipeline
        self.opt = np.asarray(opt_activations, dtype=float)
        self.label = label
        self.rng = np.random.default_rng(seed)
        self.eps_density = eps_density
        self.var_threshold = var_threshold
        self.metrics = _normalize_metrics(metrics)
        self.covariance_estimator = str(covariance_estimator)
        self.cloud = self.opt if cloud is None else np.asarray(cloud, dtype=float)
        self.null_fit_diagnostics: dict[str, Any] | None = None
        self.intrinsic_dim = float("nan")
        self.intrinsic_dim_error: str | None = None
        self.diameter = float("nan")
        self.m = 0
        self.dgms: list[np.ndarray] = []
        self.eps = float("nan")
        self.curvature_values = np.empty(0, dtype=float)
        self.curvature_error: str | None = None
        self._measure()

    def _measure(self) -> None:
        if "intrinsic_dimension" in self.metrics:
            try:
                value = float(self.pipeline.get_intrinsic_dim(self.cloud))
                if not np.isfinite(value):
                    raise ValueError("intrinsic-dimension estimate is non-finite")
                self.intrinsic_dim = value
                self.intrinsic_dim_error = None
            except Exception as exc:  # estimator failures must not abort topology
                self.intrinsic_dim = float("nan")
                self.intrinsic_dim_error = f"{type(exc).__name__}: {exc}"

        needs_projection = ("topology" in self.metrics) or ("curvature" in self.metrics)
        if not needs_projection:
            return

        # REFIT per cloud: PCA can manufacture structure from high-dim noise,
        # so freezing the concept's basis onto the null would never test it.
        projected = self.pipeline.reduce_pca(self.cloud, self.var_threshold)
        distances = pdist(projected)
        if distances.size == 0 or not np.isfinite(distances).all() or float(np.max(distances)) <= 0.0:
            raise ValueError("projected cloud must contain distinct finite points")

        # SCALE-NORMALISE: a loop is a shape property, not a size property.
        self.diameter = float(np.max(distances))
        projected = projected / self.diameter
        self.m = int(projected.shape[1])

        if "topology" in self.metrics:
            self.dgms = self.pipeline.create_persistence_diagram(projected)["dgms"]
        else:
            self.dgms = []

        sorted_distances = np.sort(distances / self.diameter)
        self.eps = float(
            sorted_distances[
                min(int(self.eps_density * len(sorted_distances)), len(sorted_distances) - 1)
            ]
        )

        if "curvature" in self.metrics:
            try:
                graph = self.pipeline.create_epsilon_graph(projected, self.eps)
                curv = self.pipeline.compute_ollivier_ricci(graph)
                values = np.asarray(curv["raw_values"], dtype=float)
                if values.size == 0 or not np.isfinite(values).all():
                    self.curvature_values = np.empty(0, dtype=float)
                    self.curvature_error = "empty_or_nonfinite_curvature"
                else:
                    self.curvature_values = values
                    self.curvature_error = None
            except Exception as exc:
                self.curvature_values = np.empty(0, dtype=float)
                self.curvature_error = f"{type(exc).__name__}: {exc}"

    def null(self, kind="covariance_gaussian", seed=None, covariance_estimator=None):
        """Generate and measure a matched Gaussian null.

        covariance_gaussian:
            Preserves the fitted mean and covariance. Default estimator is
            Ledoit-Wolf; ``regularized_empirical`` is available when less
            shrinkage is desired.

        isotropic_gaussian:
            Preserves the fitted mean and average variance, but removes covariance
            directions and anisotropy.

        noise:
            Backward-compatible alias for covariance_gaussian.

        shuffled:
            Not available because separate positive and negative activations are
            not stored.
        """
        if kind == "shuffled":
            raise NotImplementedError(
                "kind='shuffled' requires separate positive and negative "
                "activations. Permuting rows of one completed point cloud "
                "would not change its geometry."
            )
        if kind == "noise":
            kind = "covariance_gaussian"

        estimator = (
            self.covariance_estimator
            if covariance_estimator is None
            else str(covariance_estimator)
        )
        fit = fit_null_gaussian(
            self.cloud,
            kind=kind,
            covariance_estimator=estimator,
        )
        null_seed = (
            int(self.rng.integers(1 << 30))
            if seed is None
            else int(seed)
        )
        cloud = sample_null_cloud(fit, sample_count=len(self.cloud), seed=null_seed)
        measured = Manifold(
            pipeline=self.pipeline,
            opt_activations=cloud,
            label=f"null:{kind}",
            seed=null_seed,
            eps_density=self.eps_density,
            var_threshold=self.var_threshold,
            metrics=self.metrics,
            covariance_estimator=estimator,
        )
        measured.null_fit_diagnostics = fit["diagnostics"]
        return measured

    def __repr__(self):
        return (
            f"Manifold({self.label}: ID={self.intrinsic_dim:.2f}, m={self.m}, "
            f"{len(self.curvature_values)} edges, metrics={self.metrics})"
        )


class ManifoldComparator:
    def diagram_distance(self, m1, m2, max_dim=1):
        """Per homology dimension. Never combined -- H0 and H1 stay separate."""
        out = {}
        for k in range(min(max_dim + 1, len(m1.dgms), len(m2.dgms))):
            a, b = _finite(m1.dgms[k]), _finite(m2.dgms[k])
            try:
                wasserstein = float(persim.wasserstein(a, b))
            except Exception:
                wasserstein = float("nan")
            try:
                bottleneck = float(persim.bottleneck(a, b))
            except Exception:
                bottleneck = float("nan")
            out[f"H{k}"] = {
                "wasserstein": wasserstein,
                "bottleneck": bottleneck,
            }
        return out

    def curvature_difference(self, m1, m2):
        """Signed and absolute curvature distribution comparisons."""
        c1 = np.asarray(m1.curvature_values, dtype=float)
        c2 = np.asarray(m2.curvature_values, dtype=float)
        if c1.size == 0 or c2.size == 0 or not np.isfinite(c1).all() or not np.isfinite(c2).all():
            return {
                "distribution_distance": float("nan"),
                "mean_difference": float("nan"),
                "negative_fraction_difference": float("nan"),
                "absolute_negative_fraction_difference": float("nan"),
                "frac_negative_difference": float("nan"),
            }
        negative_difference = float((c1 < 0).mean() - (c2 < 0).mean())
        return {
            "distribution_distance": float(wasserstein_distance(c1, c2)),
            "mean_difference": float(c1.mean() - c2.mean()),
            "negative_fraction_difference": negative_difference,
            "absolute_negative_fraction_difference": abs(negative_difference),
            # Backward-compatible alias used by older callers.
            "frac_negative_difference": abs(negative_difference),
        }

    def compare(self, m1, m2, max_dim=1):
        return {
            "id_difference": abs(m1.intrinsic_dim - m2.intrinsic_dim)
            if np.isfinite(m1.intrinsic_dim) and np.isfinite(m2.intrinsic_dim)
            else float("nan"),
            "diagram_distance": self.diagram_distance(m1, m2, max_dim),
            "curvature": self.curvature_difference(m1, m2),
        }

    def _flatten_distances(
        self,
        m1,
        m2,
        max_dim=1,
        *,
        metrics: Sequence[str],
        infer_intrinsic_dimension: bool,
    ):
        """Named scalar distances. Metrics stay independent."""
        distances: dict[str, float] = {}
        comparison = self.compare(m1, m2, max_dim=max_dim)

        if "intrinsic_dimension" in metrics and infer_intrinsic_dimension:
            distances["id_difference"] = float(comparison["id_difference"])

        if "topology" in metrics:
            for homology_dimension, values in comparison["diagram_distance"].items():
                distances[f"{homology_dimension}_wasserstein"] = float(values["wasserstein"])
                distances[f"{homology_dimension}_bottleneck"] = float(values["bottleneck"])

        if "curvature" in metrics:
            curv = comparison["curvature"]
            distances["curvature_wasserstein"] = float(curv["distribution_distance"])
            distances["curvature_mean_difference"] = float(curv["mean_difference"])
            distances["curvature_negative_fraction_difference"] = float(
                curv["negative_fraction_difference"]
            )
        return distances

    @staticmethod
    def _metric_direction(name: str) -> str:
        if name in {
            "curvature_mean_difference",
            "curvature_negative_fraction_difference",
        }:
            return "two_sided"
        return "greater"

    @staticmethod
    def _score_from_row(values: np.ndarray, direction: str) -> float:
        arr = np.asarray(values, dtype=float)
        if direction == "two_sided":
            arr = np.abs(arr)
        return float(np.nanmedian(arr))

    def _exchangeable_loo_result(
        self,
        matrix: np.ndarray,
        *,
        n_nulls: int,
        direction: str,
        calibration_method: str,
    ) -> dict[str, Any]:
        """Observed median distance vs leave-one-out null medians."""
        observed_values = matrix[0, 1:]
        observed_score = self._score_from_row(observed_values, direction)
        null_matrix = matrix[1:, 1:]
        null_scores = np.array(
            [
                self._score_from_row(np.delete(null_matrix[index], index), direction)
                for index in range(n_nulls)
            ],
            dtype=float,
        )
        null_median = float(np.nanmedian(null_scores))
        null_mad = float(np.nanmedian(np.abs(null_scores - null_median)))
        finite_null = null_scores[np.isfinite(null_scores)]
        if np.isfinite(observed_score) and finite_null.size:
            empirical_rank = int(1 + np.sum(finite_null >= observed_score))
            monte_carlo_pvalue = float(empirical_rank / (n_nulls + 1))
        else:
            empirical_rank = None
            monte_carlo_pvalue = float("nan")
        # Clamp into the attainable Monte Carlo interval when finite.
        if np.isfinite(monte_carlo_pvalue):
            lower = 1.0 / (n_nulls + 1.0)
            monte_carlo_pvalue = float(min(1.0, max(lower, monte_carlo_pvalue)))
        return {
            "observed_score": observed_score,
            "null_scores": null_scores.tolist(),
            "null_median": null_median,
            "null_mad": null_mad,
            "robust_z": float(
                (observed_score - null_median) / (1.4826 * null_mad + 1e-12)
            )
            if np.isfinite(observed_score) and np.isfinite(null_median)
            else float("nan"),
            "empirical_rank": empirical_rank,
            "monte_carlo_pvalue": monte_carlo_pvalue,
            "n_nulls": int(n_nulls),
            "metric_direction": direction,
            "calibration_method": calibration_method,
            "calibration_status": "inferential",
        }

    def _descriptive_id_result(self, manifold: Manifold, nulls: Sequence[Manifold]) -> dict[str, Any]:
        observed = float(manifold.intrinsic_dim)
        null_ids = np.asarray([float(null.intrinsic_dim) for null in nulls], dtype=float)
        finite_null = null_ids[np.isfinite(null_ids)]
        abs_diffs = (
            np.abs(observed - finite_null)
            if np.isfinite(observed) and finite_null.size
            else np.empty(0, dtype=float)
        )
        return {
            "observed_intrinsic_dimension": observed,
            "null_intrinsic_dimensions": null_ids.tolist(),
            "absolute_observed_to_null_difference": float(np.nanmedian(abs_diffs))
            if abs_diffs.size
            else float("nan"),
            "null_median": float(np.nanmedian(finite_null)) if finite_null.size else float("nan"),
            "null_mad": float(np.nanmedian(np.abs(finite_null - np.nanmedian(finite_null))))
            if finite_null.size
            else float("nan"),
            "observed_score": float(np.nanmedian(abs_diffs)) if abs_diffs.size else float("nan"),
            "null_scores": abs_diffs.tolist(),
            "robust_z": float("nan"),
            "empirical_rank": None,
            "monte_carlo_pvalue": None,
            "n_nulls": int(len(nulls)),
            "metric_direction": "greater",
            "calibration_method": "descriptive_twoNN",
            "calibration_status": "descriptive",
            "reason": (
                "TwoNN intrinsic dimension is high-variance at typical activation "
                "sample sizes; reported descriptively unless infer_intrinsic_dimension=True."
            ),
            "estimator_error": manifold.intrinsic_dim_error,
        }

    def _inferential_id_result(self, manifold: Manifold, nulls: Sequence[Manifold]) -> dict[str, Any]:
        """Calibrated ID test: deviation from the null ID center, LOO null scores."""
        observed = float(manifold.intrinsic_dim)
        null_ids = np.asarray([float(null.intrinsic_dim) for null in nulls], dtype=float)
        n_nulls = len(null_ids)
        if not np.isfinite(observed) or not np.isfinite(null_ids).any():
            result = self._descriptive_id_result(manifold, nulls)
            result["calibration_status"] = "failed"
            result["monte_carlo_pvalue"] = float("nan")
            return result

        center = float(np.nanmedian(null_ids))
        observed_score = float(abs(observed - center))
        null_scores = np.array(
            [
                abs(null_ids[index] - float(np.nanmedian(np.delete(null_ids, index))))
                for index in range(n_nulls)
            ],
            dtype=float,
        )
        null_median = float(np.nanmedian(null_scores))
        null_mad = float(np.nanmedian(np.abs(null_scores - null_median)))
        empirical_rank = int(1 + np.sum(null_scores >= observed_score))
        pvalue = float(empirical_rank / (n_nulls + 1))
        lower = 1.0 / (n_nulls + 1.0)
        pvalue = float(min(1.0, max(lower, pvalue)))
        return {
            "observed_intrinsic_dimension": observed,
            "null_intrinsic_dimensions": null_ids.tolist(),
            "absolute_observed_to_null_difference": float(
                np.nanmedian(np.abs(observed - null_ids))
            ),
            "observed_score": observed_score,
            "null_scores": null_scores.tolist(),
            "null_median": null_median,
            "null_mad": null_mad,
            "robust_z": float((observed_score - null_median) / (1.4826 * null_mad + 1e-12)),
            "empirical_rank": empirical_rank,
            "monte_carlo_pvalue": pvalue,
            "n_nulls": int(n_nulls),
            "metric_direction": "greater",
            "calibration_method": "loo_null_center_deviation",
            "calibration_status": "inferential",
            "estimator_error": manifold.intrinsic_dim_error,
        }

    def compare_against_nulls(
        self,
        manifold,
        kind="covariance_gaussian",
        n_nulls=30,
        base_seed=0,
        max_dim=1,
        metrics: Sequence[str] | None = None,
        infer_intrinsic_dimension: bool = False,
        covariance_estimator: str | None = None,
    ):
        """Compare a manifold against repeated samples from one null model.

        Topology and curvature use an exchangeable leave-one-out median pairwise
        distance calibration. Intrinsic dimension is descriptive by default.
        """
        if n_nulls < 3:
            raise ValueError("n_nulls must be at least 3")

        selected = _normalize_metrics(metrics if metrics is not None else manifold.metrics)
        estimator = (
            getattr(manifold, "covariance_estimator", "ledoit_wolf")
            if covariance_estimator is None
            else str(covariance_estimator)
        )

        # Remeasure only when the caller requests a different metric subset so
        # topology-only comparisons can skip Ricci on every null draw.
        if set(getattr(manifold, "metrics", ())) != set(selected):
            observed = Manifold(
                pipeline=manifold.pipeline,
                opt_activations=manifold.cloud,
                label=getattr(manifold, "label", "concept"),
                seed=0,
                eps_density=manifold.eps_density,
                var_threshold=manifold.var_threshold,
                metrics=selected,
                covariance_estimator=estimator,
            )
        else:
            observed = manifold
            observed.covariance_estimator = estimator

        nulls = [
            observed.null(
                kind=kind,
                seed=base_seed + index,
                covariance_estimator=estimator,
            )
            for index in range(n_nulls)
        ]

        results: dict[str, Any] = {}

        if "intrinsic_dimension" in selected:
            if infer_intrinsic_dimension:
                results["id_difference"] = self._inferential_id_result(observed, nulls)
            else:
                results["id_difference"] = self._descriptive_id_result(observed, nulls)

        distance_metrics = [
            name
            for name in self._flatten_distances(
                observed,
                nulls[0],
                max_dim=max_dim,
                metrics=selected,
                infer_intrinsic_dimension=False,
            )
        ]
        if distance_metrics:
            objects = [observed, *nulls]
            n_objects = len(objects)
            matrices = {
                name: np.full((n_objects, n_objects), np.nan, dtype=float)
                for name in distance_metrics
            }
            for i in range(n_objects):
                for j in range(i + 1, n_objects):
                    distances = self._flatten_distances(
                        objects[i],
                        objects[j],
                        max_dim=max_dim,
                        metrics=selected,
                        infer_intrinsic_dimension=False,
                    )
                    for name, value in distances.items():
                        matrices[name][i, j] = value
                        matrices[name][j, i] = value

            for name, matrix in matrices.items():
                results[name] = self._exchangeable_loo_result(
                    matrix,
                    n_nulls=n_nulls,
                    direction=self._metric_direction(name),
                    calibration_method="exchangeable_loo_median_distance",
                )

        fit_diagnostics = None
        if nulls and getattr(nulls[0], "null_fit_diagnostics", None) is not None:
            fit_diagnostics = nulls[0].null_fit_diagnostics

        return {
            "null_kind": "covariance_gaussian" if kind == "noise" else kind,
            "n_nulls": n_nulls,
            "base_seed": base_seed,
            "max_dim": max_dim,
            "metrics_requested": list(selected),
            "infer_intrinsic_dimension": bool(infer_intrinsic_dimension),
            "covariance_estimator": estimator,
            "null_fit_diagnostics": fit_diagnostics,
            "metrics": results,
        }

    def compare_both_nulls(
        self,
        manifold,
        n_nulls=30,
        base_seed=100,
        max_dim=1,
        metrics: Sequence[str] | None = None,
        infer_intrinsic_dimension: bool = False,
        covariance_estimator: str | None = None,
    ):
        """Evaluate the manifold against both Gaussian null models."""
        return {
            "covariance_gaussian": self.compare_against_nulls(
                manifold,
                kind="covariance_gaussian",
                n_nulls=n_nulls,
                base_seed=base_seed,
                max_dim=max_dim,
                metrics=metrics,
                infer_intrinsic_dimension=infer_intrinsic_dimension,
                covariance_estimator=covariance_estimator,
            ),
            "isotropic_gaussian": self.compare_against_nulls(
                manifold,
                kind="isotropic_gaussian",
                n_nulls=n_nulls,
                base_seed=base_seed + n_nulls,
                max_dim=max_dim,
                metrics=metrics,
                infer_intrinsic_dimension=infer_intrinsic_dimension,
                covariance_estimator=covariance_estimator,
            ),
        }
