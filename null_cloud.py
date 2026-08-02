"""Compare a concept manifold against its own null.

    manifold = Manifold(pipeline, activations, seed=42)
    result = ManifoldComparator().compare_against_nulls(
        manifold, kind="covariance_gaussian", n_nulls=30, base_seed=100,
    )

The null manifolds have the same sample count and activation dimension as the
observed manifold. Each null is generated from statistics estimated from the
observed activations and is measured through the same selected metric pipeline.

What the nulls preserve and destroy
-----------------------------------
covariance_gaussian
    Preserves the fitted mean and a *regularized* covariance estimate -- not the
    empirical covariance. Both estimators shrink: Ledoit-Wolf toward a scaled
    identity by an amount that grows with n_features / n_samples, and
    ``regularized_empirical`` by a small ridge. Destroys every higher-order
    property: multimodality, curvature, and nonlinear manifold structure.

    In the d >> n regime the shrinkage is not a detail. On a 12 x 2304
    activation cloud the Ledoit-Wolf intensity is ~0.63 and the null's effective
    rank is ~61x the empirical one, so anisotropy, eigenspectrum and the induced
    distance structure are all substantially altered and the draw is closer to
    isotropic than to covariance-matched. ``fit_null_gaussian`` reports a
    ``covariance_match`` block and ``compare_against_nulls`` warns once per
    ensemble when the match is poor. Read a rejection against a poorly matched
    null as "not this near-isotropic Gaussian", not "not any Gaussian with my
    covariance".

isotropic_gaussian
    Preserves the fitted mean and the average per-feature variance. Destroys
    anisotropy and every covariance direction, so a rejection is also driven by
    ordinary linear correlation and says nothing on its own about nonlinear
    structure.

noise
    Backward-compatible alias for covariance_gaussian.

Inference
---------
Every empirical p-value in this module goes through :func:`empirical_pvalue`.
A metric whose observed statistic is non-finite, or whose null ensemble has too
few finite statistics, reports ``inference_available=False`` and a NaN p-value
with a ``failure_reason``. Unavailable inference is never reported as a
rejection.

Intrinsic dimension is descriptive by default: TwoNN is high-variance at the
sample sizes typical for activation clouds, so ID enters a significance test
only when ``infer_intrinsic_dimension=True``.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

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

# Fewer finite null statistics than this and the ensemble cannot support any
# inference worth reporting.
MINIMUM_VALID_NULLS = 3

# A Monte Carlo p-value cannot fall below 1 / (valid_nulls + 1), so alpha = 0.05
# is unreachable with fewer than 19 usable null draws.
MINIMUM_NULLS_FOR_ALPHA_05 = 19

# Above either threshold the fitted covariance is far enough from the empirical
# one that "covariance-matched" overstates what the null preserves.
_COVARIANCE_MISMATCH_FROBENIUS = 0.30
_COVARIANCE_MISMATCH_RANK_INFLATION = 2.0


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------


def validate_cloud(cloud: Any, *, name: str = "cloud", minimum_samples: int = 2) -> np.ndarray:
    """Return a validated finite ``[n_samples, n_features]`` float array.

    Raises ValueError naming the offending argument rather than letting NaN or
    a wrong shape surface several frames deep inside sklearn.
    """
    array = np.asarray(cloud, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional array shaped [n_samples, n_features]; "
            f"received shape {array.shape}"
        )
    if array.shape[0] < minimum_samples:
        raise ValueError(
            f"{name} must contain at least {minimum_samples} samples; "
            f"received {array.shape[0]}"
        )
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature dimension")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")
    return array


def _validate_unit_interval(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must be a finite number in the interval (0, 1]")
    return number


def _normalize_metrics(metrics: Sequence[str] | None) -> tuple[str, ...]:
    """Deduplicate and validate a metric selection."""
    if metrics is None:
        return DEFAULT_METRICS
    if isinstance(metrics, str):
        raise TypeError("metrics must be a sequence of metric names, not a string")
    selected = tuple(dict.fromkeys(str(name) for name in metrics))
    if not selected:
        raise ValueError(
            "metrics must select at least one of " + ", ".join(SUPPORTED_METRICS)
        )
    unknown = [name for name in selected if name not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(
            "unknown metrics: "
            + ", ".join(unknown)
            + ". Supported: "
            + ", ".join(SUPPORTED_METRICS)
        )
    return selected


def _finite(dgm):
    """Strip the [0, inf] H0 bar -- persim returns inf otherwise."""
    d = np.asarray(dgm, dtype=float)
    if d.size == 0:
        return np.empty((0, 2), dtype=float)
    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError("persistence diagram must have shape [number_of_bars, 2]")
    return d[np.isfinite(d).all(axis=1)]


# ---------------------------------------------------------------------------
# 2. The one empirical-inference calculation
# ---------------------------------------------------------------------------


def empirical_pvalue(
    observed_score: float,
    null_scores: Sequence[float],
    *,
    direction: str = "greater",
    minimum_valid_nulls: int = MINIMUM_VALID_NULLS,
) -> dict[str, Any]:
    """Rank one observed statistic against a null ensemble.

    The p-value is ``(1 + #{valid null >= observed}) / (valid nulls + 1)``, the
    standard plus-one corrected Monte Carlo estimate. Only *finite* null
    statistics are valid, and the count of valid nulls -- not the number
    requested -- sets the denominator: counting a rank over survivors while
    dividing by the requested draws biases every p-value downward and caps the
    reachable p-value below 1.

    Inference is unavailable, and ``pvalue`` is NaN, when the observed statistic
    is non-finite or fewer than ``minimum_valid_nulls`` null statistics are
    finite. Unavailable is never the same as non-significant: callers must test
    ``inference_available`` before acting on ``pvalue``.

    ``direction="two_sided"`` ranks absolute deviations; ``"greater"`` ranks the
    statistic as given, which is the correct choice for a distance.
    """
    if direction not in {"greater", "two_sided"}:
        raise ValueError("direction must be 'greater' or 'two_sided'")

    scores = np.asarray(list(null_scores), dtype=float)
    observed = float(observed_score)
    if direction == "two_sided":
        scores = np.abs(scores)
        observed = abs(observed)

    valid = scores[np.isfinite(scores)]
    n_valid = int(valid.size)
    null_median = float(np.median(valid)) if n_valid else float("nan")
    null_mad = float(np.median(np.abs(valid - null_median))) if n_valid else float("nan")

    failure_reason: str | None = None
    if not np.isfinite(observed):
        failure_reason = "observed statistic is not finite"
    elif n_valid < minimum_valid_nulls:
        failure_reason = (
            f"only {n_valid} of {scores.size} null statistics are finite; "
            f"at least {minimum_valid_nulls} are required"
        )

    if failure_reason is not None:
        return {
            "observed_score": float(observed_score),
            "null_scores": [float(value) for value in null_scores],
            "n_valid_nulls": n_valid,
            "pvalue": float("nan"),
            "minimum_attainable_pvalue": float("nan"),
            "empirical_rank": None,
            "inference_available": False,
            "failure_reason": failure_reason,
            "null_median": null_median,
            "null_mad": null_mad,
            "robust_z": float("nan"),
            "direction": direction,
        }

    empirical_rank = int(1 + np.sum(valid >= observed))
    pvalue = float(empirical_rank / (n_valid + 1))
    return {
        "observed_score": float(observed_score),
        "null_scores": [float(value) for value in null_scores],
        "n_valid_nulls": n_valid,
        "pvalue": pvalue,
        # No ensemble can report significance finer than this, however far the
        # observed statistic sits from the null cloud.
        "minimum_attainable_pvalue": float(1.0 / (n_valid + 1)),
        "empirical_rank": empirical_rank,
        "inference_available": True,
        "failure_reason": None,
        "null_median": null_median,
        "null_mad": null_mad,
        "robust_z": float((observed - null_median) / (1.4826 * null_mad + 1e-12)),
        "direction": direction,
    }


def unavailable_result(reason: str, *, direction: str = "greater") -> dict[str, Any]:
    """An inference result for a metric that could not be measured at all."""
    return empirical_pvalue(float("nan"), [], direction=direction) | {
        "failure_reason": reason
    }


# ---------------------------------------------------------------------------
# 3. Fitting and sampling a null distribution
# ---------------------------------------------------------------------------


def _effective_rank(covariance: np.ndarray) -> float:
    """Roy-Vetterli effective rank: exp(entropy of the eigenvalue spectrum)."""
    eigenvalues = np.clip(np.linalg.eigvalsh(np.asarray(covariance, dtype=float)), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    probs = eigenvalues / total
    return float(np.exp(-np.sum(probs * np.log(probs + 1e-300))))


def _covariance_diagnostics(
    *,
    mean: np.ndarray,
    location: np.ndarray,
    covariance: np.ndarray,
    empirical_covariance: np.ndarray,
    isotropic: bool,
) -> dict[str, Any]:
    """Describe how far the sampled covariance sits from the empirical one.

    ``covariance_match.is_matched`` is the flag callers should read before
    describing a covariance null as covariance-matched. It is always False for
    the isotropic null, which discards covariance structure by design.
    """
    empirical_rank = _effective_rank(empirical_covariance)
    null_rank = _effective_rank(covariance)
    relative = float(
        np.linalg.norm(covariance - empirical_covariance)
        / (np.linalg.norm(empirical_covariance) + 1e-12)
    )
    inflation = float(null_rank / empirical_rank) if empirical_rank > 0.0 else float("nan")
    matched = (
        not isotropic
        and relative <= _COVARIANCE_MISMATCH_FROBENIUS
        and (not np.isfinite(inflation) or inflation <= _COVARIANCE_MISMATCH_RANK_INFLATION)
    )
    return {
        "mean_l2_norm": float(np.linalg.norm(location)),
        "mean_difference_l2": float(np.linalg.norm(location - mean)),
        "empirical_cov_frobenius": float(np.linalg.norm(empirical_covariance)),
        "null_cov_frobenius": float(np.linalg.norm(covariance)),
        "empirical_effective_rank": empirical_rank,
        "null_effective_rank": null_rank,
        "top_empirical_eigenvalues": np.sort(np.linalg.eigvalsh(empirical_covariance))[::-1][:5].tolist(),
        "top_null_eigenvalues": np.sort(np.linalg.eigvalsh(covariance))[::-1][:5].tolist(),
        "covariance_match": {
            "relative_frobenius_difference": relative,
            "effective_rank_inflation": inflation,
            "is_matched": bool(matched),
            "note": "isotropic null intentionally discards covariance structure"
            if isotropic
            else None,
        },
    }


def fit_null_gaussian(
    cloud: np.ndarray,
    *,
    kind: str,
    covariance_estimator: str = "ledoit_wolf",
) -> dict[str, Any]:
    """Fit the null's mean and covariance once for a whole ensemble.

    The fit depends only on ``cloud``, so an ensemble must call this once and
    reuse the result. ``diagnostics["covariance_match"]`` records whether the
    fitted covariance is close enough to the empirical one for a
    "covariance-matched" description to hold; see the module docstring.
    """
    x = validate_cloud(cloud)
    if kind == "noise":
        kind = "covariance_gaussian"

    mean = x.mean(axis=0)
    empirical_covariance = np.asarray(
        EmpiricalCovariance(store_precision=False).fit(x).covariance_, dtype=float
    )

    if kind == "isotropic_gaussian":
        average_variance = float(np.var(x, axis=0, ddof=1).mean())
        if average_variance <= 0:
            raise ValueError("isotropic Gaussian null requires positive variance")
        location = mean
        covariance = average_variance * np.eye(x.shape[1], dtype=float)
        estimator_name = "isotropic_average_variance"
        extras: dict[str, Any] = {"average_variance": average_variance}
    elif kind == "covariance_gaussian":
        if covariance_estimator not in COVARIANCE_ESTIMATORS:
            raise ValueError(
                f"unknown covariance_estimator: {covariance_estimator}. "
                f"Use one of {COVARIANCE_ESTIMATORS}."
            )
        estimator_name = covariance_estimator
        if covariance_estimator == "ledoit_wolf":
            estimator = LedoitWolf(store_precision=False).fit(x)
            covariance = np.asarray(estimator.covariance_, dtype=float)
            location = np.asarray(estimator.location_, dtype=float)
            extras = {"shrinkage": float(getattr(estimator, "shrinkage_", float("nan")))}
        else:
            # Ridge-stabilised empirical covariance for finite-sample / d >= n.
            location = mean
            eigenvalues = np.linalg.eigvalsh(empirical_covariance)
            ridge = max(
                1e-6, 1e-3 * float(np.mean(np.clip(eigenvalues, 0.0, None)) + 1e-12)
            )
            covariance = empirical_covariance + ridge * np.eye(
                empirical_covariance.shape[0], dtype=float
            )
            extras = {"ridge": ridge}
    else:
        raise ValueError(
            f"unknown null kind: {kind}. Use 'covariance_gaussian', "
            "'isotropic_gaussian', or 'noise'."
        )

    diagnostics = _covariance_diagnostics(
        mean=mean,
        location=location,
        covariance=covariance,
        empirical_covariance=empirical_covariance,
        isotropic=kind == "isotropic_gaussian",
    )
    diagnostics.update(extras)
    return {
        "kind": kind,
        "mean": location,
        "covariance": covariance,
        "covariance_estimator": estimator_name,
        "diagnostics": diagnostics,
        **extras,
    }


def sample_null_cloud(
    fit: Mapping[str, Any],
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Draw one null cloud from a fit produced by :func:`fit_null_gaussian`."""
    rng = np.random.default_rng(int(seed))
    return rng.multivariate_normal(
        mean=np.asarray(fit["mean"], dtype=float),
        cov=np.asarray(fit["covariance"], dtype=float),
        size=int(sample_count),
    )


# ---------------------------------------------------------------------------
# 4. Measuring one cloud
# ---------------------------------------------------------------------------


class Manifold:
    """A measured cloud: ID, persistence diagrams, curvature signature.

    pipeline: exposes get_intrinsic_dim, reduce_pca, create_persistence_diagram,
              create_epsilon_graph, compute_ollivier_ricci

    Intrinsic dimension and curvature record their own failures in
    ``intrinsic_dim_error`` / ``curvature_error`` and leave the other metrics
    measurable. A cloud that cannot be projected or that yields a non-positive
    epsilon is a hard measurement failure and raises.
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
        self.opt = validate_cloud(opt_activations, name="opt_activations")
        self.label = label
        self.rng = np.random.default_rng(seed)
        self.eps_density = _validate_unit_interval(eps_density, "eps_density")
        self.var_threshold = _validate_unit_interval(var_threshold, "var_threshold")
        self.metrics = _normalize_metrics(metrics)
        self.covariance_estimator = str(covariance_estimator)
        self.cloud = self.opt if cloud is None else validate_cloud(cloud, name="cloud")

        self.intrinsic_dim = float("nan")
        self.intrinsic_dim_error: str | None = None
        self.diameter = float("nan")
        self.m = 0
        self.dgms: list[np.ndarray] = []
        self.eps = float("nan")
        self.curvature_values = np.empty(0, dtype=float)
        self.curvature_error: str | None = None
        self.graph_diagnostics: dict[str, Any] | None = None
        self._measure()

    def _measure(self) -> None:
        if "intrinsic_dimension" in self.metrics:
            try:
                value = float(self.pipeline.get_intrinsic_dim(self.cloud))
                if not np.isfinite(value):
                    raise ValueError("intrinsic-dimension estimate is non-finite")
                self.intrinsic_dim = value
            except Exception as exc:  # estimator failures must not abort topology
                self.intrinsic_dim = float("nan")
                self.intrinsic_dim_error = f"{type(exc).__name__}: {exc}"

        if not ({"topology", "curvature"} & set(self.metrics)):
            return

        # REFIT per cloud: PCA can manufacture structure from high-dim noise,
        # so freezing the concept's basis onto the null would never test it.
        projected = self.pipeline.reduce_pca(self.cloud, self.var_threshold)
        distances = pdist(projected)
        if (
            distances.size == 0
            or not np.isfinite(distances).all()
            or float(np.max(distances)) <= 0.0
        ):
            raise ValueError("projected cloud must contain distinct finite points")

        # SCALE-NORMALISE: a loop is a shape property, not a size property.
        self.diameter = float(np.max(distances))
        projected = projected / self.diameter
        self.m = int(projected.shape[1])

        if "topology" in self.metrics:
            self.dgms = self.pipeline.create_persistence_diagram(projected)["dgms"]

        self.eps = self._select_epsilon(distances / self.diameter)

        if "curvature" in self.metrics:
            self._measure_curvature(projected)

    def _select_epsilon(self, normalized_distances: np.ndarray) -> float:
        """Epsilon at the ``eps_density`` quantile of the pairwise distances.

        A non-positive or non-finite epsilon means graph geometry is not
        measurable -- enough duplicate points that the graph would be empty.
        concept_geometry.select_matched_density_epsilon refuses the same
        condition; refuse it here too rather than building a degenerate graph.
        """
        ordered = np.sort(normalized_distances)
        index = min(int(self.eps_density * len(ordered)), len(ordered) - 1)
        epsilon = float(ordered[index])
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(
                f"matched-density epsilon selection produced a non-positive radius "
                f"(eps={epsilon!r}); the cloud has too many duplicate points at "
                f"eps_density={self.eps_density}"
            )
        return epsilon

    def _measure_curvature(self, projected: np.ndarray) -> None:
        try:
            graph = self.pipeline.create_epsilon_graph(projected, self.eps)
            self.graph_diagnostics = _graph_diagnostics(graph)
            values = np.asarray(
                self.pipeline.compute_ollivier_ricci(graph)["raw_values"], dtype=float
            )
            if values.size == 0 or not np.isfinite(values).all():
                self.curvature_error = "empty_or_nonfinite_curvature"
            else:
                self.curvature_values = values
        except Exception as exc:
            self.curvature_values = np.empty(0, dtype=float)
            self.curvature_error = f"{type(exc).__name__}: {exc}"

    def null(self, kind="covariance_gaussian", seed=None, covariance_estimator=None, fit=None):
        """Generate and measure one null draw.

        See the module docstring for what each ``kind`` preserves and destroys.
        ``fit`` reuses a previously computed :func:`fit_null_gaussian` result;
        ensembles must pass it, because the fit depends only on ``self.cloud``
        and refitting per draw costs roughly three quarters of the runtime on a
        wide activation cloud.
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
        if fit is None:
            fit = fit_null_gaussian(self.cloud, kind=kind, covariance_estimator=estimator)
        null_seed = int(self.rng.integers(1 << 30)) if seed is None else int(seed)

        return Manifold(
            pipeline=self.pipeline,
            opt_activations=sample_null_cloud(
                fit, sample_count=len(self.cloud), seed=null_seed
            ),
            label=f"null:{kind}",
            seed=null_seed,
            eps_density=self.eps_density,
            var_threshold=self.var_threshold,
            metrics=self.metrics,
            covariance_estimator=estimator,
        )

    def __repr__(self):
        return (
            f"Manifold({self.label}: ID={self.intrinsic_dim:.2f}, m={self.m}, "
            f"{len(self.curvature_values)} edges, metrics={self.metrics})"
        )


def _graph_diagnostics(graph) -> dict[str, Any]:
    """Epsilon-graph size and connectivity, recorded next to the curvature.

    Ollivier-Ricci is per edge, so curvature distributions from graphs with very
    different edge or component counts are not directly comparable.
    """
    try:
        import networkx as nx

        degrees = [degree for _, degree in graph.degree()]
        return {
            "node_count": int(graph.number_of_nodes()),
            "edge_count": int(graph.number_of_edges()),
            "connected_components": int(nx.number_connected_components(graph)),
            "isolated_nodes": int(sum(1 for degree in degrees if degree == 0)),
            "mean_degree": float(np.mean(degrees)) if degrees else float("nan"),
        }
    except Exception as exc:  # diagnostics must never break measurement
        return {"error": f"{type(exc).__name__}: {exc}"}


def build_null_ensemble(
    observed: Manifold,
    *,
    kind: str,
    n_nulls: int,
    base_seed: int,
    covariance_estimator: str,
    fit: Mapping[str, Any],
) -> tuple[list[Manifold], list[dict[str, Any]]]:
    """Draw ``n_nulls`` nulls, keeping going past individual failures.

    Seeds are ``base_seed + index``, so a fixed ``base_seed`` reproduces the
    whole ensemble. Returns the measured nulls and one record per failure; the
    caller decides whether enough survived.
    """
    nulls: list[Manifold] = []
    failures: list[dict[str, Any]] = []
    for index in range(n_nulls):
        seed = base_seed + index
        try:
            nulls.append(
                observed.null(
                    kind=kind,
                    seed=seed,
                    covariance_estimator=covariance_estimator,
                    fit=fit,
                )
            )
        except Exception as exc:
            failures.append(
                {"index": index, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            )
    return nulls, failures


# ---------------------------------------------------------------------------
# 5. Comparison
# ---------------------------------------------------------------------------


_TWO_SIDED_METRICS = frozenset(
    {"curvature_mean_difference", "curvature_negative_fraction_difference"}
)

CURVATURE_UNAVAILABLE = {
    "distribution_distance": float("nan"),
    "mean_difference": float("nan"),
    "negative_fraction_difference": float("nan"),
    "absolute_negative_fraction_difference": float("nan"),
    "frac_negative_difference": float("nan"),
}


def diagram_distance_pair(dgm1, dgm2) -> dict[str, float]:
    """Bottleneck and Wasserstein between two diagrams, NaN when persim refuses.

    Infinite bars are stripped first; persim would otherwise warn and drop them
    itself. A NaN here means the distance could not be computed, and callers
    must treat it as an unmeasured metric rather than as a zero distance.
    """
    a, b = _finite(dgm1), _finite(dgm2)
    values: dict[str, float] = {}
    for name, function in (
        ("wasserstein", persim.wasserstein),
        ("bottleneck", persim.bottleneck),
    ):
        try:
            values[name] = float(function(a, b))
        except Exception:
            values[name] = float("nan")
    return values


def curvature_distribution_difference(values1, values2) -> dict[str, float]:
    """Signed and absolute curvature-distribution comparisons.

    ``negative_fraction_difference`` is signed: positive means the first cloud
    has the larger negative-curvature fraction. ``frac_negative_difference`` is
    the backward-compatible *absolute* alias read by topology_metric; it carries
    no direction. All-NaN when either side has no usable edge curvatures.
    """
    c1 = np.asarray(values1, dtype=float)
    c2 = np.asarray(values2, dtype=float)
    if c1.size == 0 or c2.size == 0 or not np.isfinite(c1).all() or not np.isfinite(c2).all():
        return dict(CURVATURE_UNAVAILABLE)
    signed = float((c1 < 0).mean() - (c2 < 0).mean())
    return {
        "distribution_distance": float(wasserstein_distance(c1, c2)),
        "mean_difference": float(c1.mean() - c2.mean()),
        "negative_fraction_difference": signed,
        "absolute_negative_fraction_difference": abs(signed),
        "frac_negative_difference": abs(signed),
    }


class ManifoldComparator:
    def diagram_distance(self, m1, m2, max_dim=1):
        """Per homology dimension. Never combined -- H0 and H1 stay separate."""
        return {
            f"H{k}": diagram_distance_pair(m1.dgms[k], m2.dgms[k])
            for k in range(min(max_dim + 1, len(m1.dgms), len(m2.dgms)))
        }

    def curvature_difference(self, m1, m2):
        """Curvature comparison between two measured manifolds.

        See :func:`curvature_distribution_difference` for the signed/absolute
        contract.
        """
        return curvature_distribution_difference(m1.curvature_values, m2.curvature_values)

    def compare(self, m1, m2, max_dim=1):
        return {
            "id_difference": abs(m1.intrinsic_dim - m2.intrinsic_dim)
            if np.isfinite(m1.intrinsic_dim) and np.isfinite(m2.intrinsic_dim)
            else float("nan"),
            "diagram_distance": self.diagram_distance(m1, m2, max_dim),
            "curvature": self.curvature_difference(m1, m2),
        }

    def _flatten_distances(self, m1, m2, max_dim, metrics: Sequence[str]) -> dict[str, float]:
        """Named scalar distances between two manifolds. Metrics stay independent."""
        distances: dict[str, float] = {}
        comparison = self.compare(m1, m2, max_dim=max_dim)
        if "topology" in metrics:
            for homology_dimension, values in comparison["diagram_distance"].items():
                distances[f"{homology_dimension}_wasserstein"] = float(values["wasserstein"])
                distances[f"{homology_dimension}_bottleneck"] = float(values["bottleneck"])
        if "curvature" in metrics:
            curvature = comparison["curvature"]
            distances["curvature_wasserstein"] = float(curvature["distribution_distance"])
            distances["curvature_mean_difference"] = float(curvature["mean_difference"])
            distances["curvature_negative_fraction_difference"] = float(
                curvature["negative_fraction_difference"]
            )
        return distances

    @staticmethod
    def _median_or_nan(values: np.ndarray, two_sided: bool) -> float:
        arr = np.asarray(values, dtype=float)
        if two_sided:
            arr = np.abs(arr)
        arr = arr[np.isfinite(arr)]
        # An all-NaN row is how "this metric could not be measured for this
        # object" is represented; it is not an anomaly worth warning about.
        return float(np.median(arr)) if arr.size else float("nan")

    def _loo_result(self, matrix: np.ndarray, name: str) -> dict[str, Any]:
        """Observed median distance to the nulls vs leave-one-out null medians.

        Under the null every object is exchangeable, so each object's median
        distance to the others is an exchangeable statistic and ranking the
        observed one against the nulls gives a calibrated p-value.
        """
        two_sided = name in _TWO_SIDED_METRICS
        null_matrix = matrix[1:, 1:]
        result = empirical_pvalue(
            self._median_or_nan(matrix[0, 1:], two_sided),
            [
                self._median_or_nan(np.delete(null_matrix[index], index), two_sided)
                for index in range(null_matrix.shape[0])
            ],
            direction="greater",
        )
        result["metric_direction"] = "two_sided" if two_sided else "greater"
        result["calibration_method"] = "exchangeable_loo_median_distance"
        return result

    def _pairwise_matrices(
        self, objects: Sequence[Manifold], names: Sequence[str], max_dim: int, metrics: Sequence[str]
    ) -> dict[str, np.ndarray]:
        n = len(objects)
        matrices = {name: np.full((n, n), np.nan, dtype=float) for name in names}
        for i in range(n):
            for j in range(i + 1, n):
                for name, value in self._flatten_distances(
                    objects[i], objects[j], max_dim, metrics
                ).items():
                    matrices[name][i, j] = value
                    matrices[name][j, i] = value
        return matrices

    def _id_result(
        self, observed: Manifold, nulls: Sequence[Manifold], *, inferential: bool
    ) -> dict[str, Any]:
        """Intrinsic-dimension comparison, descriptive unless asked otherwise.

        Descriptive mode reports the observed ID next to the null IDs and never
        produces a p-value. Inferential mode ranks |observed - null centre|
        against leave-one-out null deviations.
        """
        observed_id = float(observed.intrinsic_dim)
        null_ids = np.asarray([float(null.intrinsic_dim) for null in nulls], dtype=float)
        finite_ids = null_ids[np.isfinite(null_ids)]

        if not inferential:
            differences = (
                np.abs(observed_id - finite_ids)
                if np.isfinite(observed_id) and finite_ids.size
                else np.empty(0, dtype=float)
            )
            return {
                "observed_intrinsic_dimension": observed_id,
                "null_intrinsic_dimensions": null_ids.tolist(),
                "absolute_observed_to_null_difference": float(np.median(differences))
                if differences.size
                else float("nan"),
                "n_valid_nulls": int(finite_ids.size),
                "pvalue": float("nan"),
                "inference_available": False,
                "failure_reason": (
                    "TwoNN intrinsic dimension is high-variance at typical activation "
                    "sample sizes; reported descriptively unless "
                    "infer_intrinsic_dimension=True"
                ),
                "calibration_method": "descriptive_twoNN",
                "estimator_error": observed.intrinsic_dim_error,
            }

        if np.isfinite(observed_id) and finite_ids.size:
            centre = float(np.median(finite_ids))
            null_scores = [
                abs(null_ids[index] - float(np.median(np.delete(finite_ids, index))))
                if np.isfinite(null_ids[index]) and finite_ids.size > 1
                else float("nan")
                for index in range(null_ids.size)
            ]
            result = empirical_pvalue(abs(observed_id - centre), null_scores)
        else:
            result = unavailable_result(
                "intrinsic dimension is non-finite for the observed cloud "
                "or for every null draw"
            )
        result.update(
            {
                "observed_intrinsic_dimension": observed_id,
                "null_intrinsic_dimensions": null_ids.tolist(),
                "calibration_method": "loo_null_center_deviation",
                "estimator_error": observed.intrinsic_dim_error,
            }
        )
        return result

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
        """Compare a manifold against repeated draws from one null model.

        Pure with respect to ``manifold``: nothing is written back onto it.
        The Gaussian fit is computed once and reused for every draw. Draws that
        fail to measure are recorded in ``null_failures`` and excluded from the
        statistics instead of aborting the ensemble. Every metric block carries
        ``inference_available``, ``n_valid_nulls`` and
        ``minimum_attainable_pvalue``, so a p-value is always readable against
        the ensemble that produced it.
        """
        if n_nulls < MINIMUM_VALID_NULLS:
            raise ValueError(f"n_nulls must be at least {MINIMUM_VALID_NULLS}")
        if n_nulls < MINIMUM_NULLS_FOR_ALPHA_05:
            warnings.warn(
                f"n_nulls={n_nulls} caps every empirical p-value at a minimum of "
                f"{1.0 / (n_nulls + 1):.3f}; alpha=0.05 is unreachable below "
                f"n_nulls={MINIMUM_NULLS_FOR_ALPHA_05}.",
                stacklevel=2,
            )

        selected = _normalize_metrics(metrics if metrics is not None else manifold.metrics)
        estimator = (
            getattr(manifold, "covariance_estimator", "ledoit_wolf")
            if covariance_estimator is None
            else str(covariance_estimator)
        )

        # Re-measure into a new object rather than mutating the caller's: a
        # different metric subset or estimator is a different measurement.
        if set(manifold.metrics) != set(selected) or manifold.covariance_estimator != estimator:
            observed = Manifold(
                pipeline=manifold.pipeline,
                opt_activations=manifold.cloud,
                label=manifold.label,
                seed=0,
                eps_density=manifold.eps_density,
                var_threshold=manifold.var_threshold,
                metrics=selected,
                covariance_estimator=estimator,
            )
        else:
            observed = manifold

        resolved_kind = "covariance_gaussian" if kind == "noise" else kind
        fit = fit_null_gaussian(
            observed.cloud, kind=resolved_kind, covariance_estimator=estimator
        )
        match = fit["diagnostics"]["covariance_match"]
        if resolved_kind == "covariance_gaussian" and not match["is_matched"]:
            # Once per ensemble, not once per draw.
            warnings.warn(
                "the fitted covariance null is not covariance-matched "
                f"(relative Frobenius difference {match['relative_frobenius_difference']:.2f}, "
                f"effective-rank inflation x{match['effective_rank_inflation']:.1f}). "
                "With n_features close to or above n_samples the estimator shrinks "
                "toward isotropy, so a rejection is partly a rejection of isotropy "
                "rather than evidence against all Gaussians with this covariance.",
                stacklevel=2,
            )

        nulls, null_failures = build_null_ensemble(
            observed,
            kind=resolved_kind,
            n_nulls=n_nulls,
            base_seed=base_seed,
            covariance_estimator=estimator,
            fit=fit,
        )
        if null_failures:
            warnings.warn(
                f"{len(null_failures)} of {n_nulls} null draws failed to measure; "
                "statistics use the surviving draws only (see 'null_failures' and "
                "each metric's 'n_valid_nulls').",
                stacklevel=2,
            )

        results: dict[str, Any] = {}
        if len(nulls) < MINIMUM_VALID_NULLS:
            reason = (
                f"only {len(nulls)} of {n_nulls} null draws could be measured; "
                f"at least {MINIMUM_VALID_NULLS} are required"
            )
            # Comparing the observed manifold with itself yields exactly the
            # metric names a real comparison would have produced.
            names = ["id_difference"] if "intrinsic_dimension" in selected else []
            names += list(self._flatten_distances(observed, observed, max_dim, selected))
            results = {name: unavailable_result(reason) for name in names}
        else:
            if "intrinsic_dimension" in selected:
                results["id_difference"] = self._id_result(
                    observed, nulls, inferential=infer_intrinsic_dimension
                )
            objects = [observed, *nulls]
            names = list(self._flatten_distances(observed, nulls[0], max_dim, selected))
            for name, matrix in self._pairwise_matrices(
                objects, names, max_dim, selected
            ).items():
                results[name] = self._loo_result(matrix, name)

        return {
            "null_kind": resolved_kind,
            "n_nulls": n_nulls,
            "n_nulls_measured": len(nulls),
            "null_failures": null_failures,
            "base_seed": base_seed,
            "max_dim": max_dim,
            "metrics_requested": list(selected),
            "infer_intrinsic_dimension": bool(infer_intrinsic_dimension),
            "covariance_estimator": estimator,
            "null_fit_diagnostics": fit["diagnostics"],
            "observed_graph_diagnostics": observed.graph_diagnostics,
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
        """Evaluate the manifold against both Gaussian null models.

        The two ensembles use disjoint seed ranges so their draws are independent.
        """
        return {
            kind: self.compare_against_nulls(
                manifold,
                kind=kind,
                n_nulls=n_nulls,
                base_seed=base_seed + offset,
                max_dim=max_dim,
                metrics=metrics,
                infer_intrinsic_dimension=infer_intrinsic_dimension,
                covariance_estimator=covariance_estimator,
            )
            for offset, kind in ((0, "covariance_gaussian"), (n_nulls, "isotropic_gaussian"))
        }
