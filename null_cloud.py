"""Compare a concept manifold against a low-rank Gaussian null.

    manifold = Manifold(pipeline, activations, seed=42)
    result = ManifoldComparator().compare_against_nulls(
        manifold, n_nulls=30, base_seed=100,
    )

The null model
--------------
One null: ``low_rank_gaussian``. It is fitted in the observed cloud's own
empirical SVD subspace and sampled many times.

Let ``X`` be ``(n_samples, n_features)`` with mean ``mu`` and centered SVD
``X - mu = U S V^T``. Keep the ``r`` numerically nonzero directions, where
``r <= min(n_samples - 1, n_features)``, and let ``lambda = s^2 / (n - 1)`` be
the matching covariance eigenvalues. A null draw is

    Z = Q diag(sqrt(lambda))          with Q centered and Q^T Q = (n - 1) I
    X_null = Z V_r + mu

for a Haar-uniform centered orthonormal frame ``Q``. Every draw therefore
reproduces the observed mean, the observed principal subspace, and the observed
nonzero covariance spectrum *exactly*, not merely in expectation, while its
orientation inside that subspace is uniformly random.

Preserved: sample count, ambient dimension, mean, low-rank subspace, nonzero
covariance eigenvalues, effective rank, anisotropy.
Destroyed: every higher-order and nonlinear property -- curved trajectories,
loops, clusters, and any non-Gaussian latent organisation not implied by the
second moments.

Why the spectrum is matched exactly rather than in expectation. Independent
Gaussian latents would match it only in expectation, and at this project's
sample sizes the realised spectrum wanders far. On the repository's 12 x 2304
layer-6 activation cloud, independent latents give a relative spectrum error of
0.52 and an effective rank of 5.85 +/- 0.66 against an observed 9.36, retaining
6.7 +/- 0.5 PCA components against the observed 10. The exact construction gives
0.00 error, effective rank 9.36, and exactly 10 components. PCA dimension feeds
every downstream topology statistic, so matching in expectation would leave a
systematic linear-structure mismatch inside the comparison. This is a
moment-constrained surrogate -- a matrix-valued low-rank adaptation of
covariance-constrained surrogate testing -- not an i.i.d. Gaussian sample.

Interpreting a rejection
------------------------
A small p-value means the observed statistic is unusual relative to clouds
sharing its mean and linear second-order structure and nothing else. It does not
show the recovered structure is semantically meaningful, and it does not rule out
explanations outside this particular linear Gaussian control. With n_samples in
the tens, both topology and the covariance spectrum remain uncertain.

Inference
---------
Every p-value goes through :func:`empirical_pvalue`. A metric whose observed
statistic is non-finite, or whose ensemble has too few finite statistics, reports
``inference_available=False`` with a ``failure_reason`` and a NaN p-value.
Unavailable inference is never reported as a rejection.

Intrinsic dimension is descriptive by default: TwoNN is high-variance at these
sample sizes, so it enters a test only when ``infer_intrinsic_dimension=True``.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import persim
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance


SUPPORTED_METRICS = ("intrinsic_dimension", "topology", "curvature")
DEFAULT_METRICS = SUPPORTED_METRICS

#: The one null model. Older names resolve here with a deprecation warning.
NULL_KIND = "low_rank_gaussian"
_DEPRECATED_KINDS = {
    "noise": "an alias that never named a model",
    "covariance_gaussian": "a Ledoit-Wolf null that did not preserve the observed spectrum",
    "isotropic_gaussian": "a null that discarded anisotropy entirely",
}

#: Fewer finite null statistics than this and no inference is reported.
MINIMUM_VALID_NULLS = 3
#: A p-value cannot fall below 1 / (valid + 1), so alpha = 0.05 needs this many.
MINIMUM_NULLS_FOR_ALPHA_05 = 19

CURVATURE_UNAVAILABLE = {
    "distribution_distance": float("nan"),
    "mean_difference": float("nan"),
    "negative_fraction_difference": float("nan"),
    "absolute_negative_fraction_difference": float("nan"),
    "frac_negative_difference": float("nan"),
}

_TWO_SIDED_METRICS = frozenset(
    {"curvature_mean_difference", "curvature_negative_fraction_difference"}
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_cloud(cloud: Any, *, name: str = "cloud") -> np.ndarray:
    """Return a validated finite ``[n_samples, n_features]`` float array.

    Raises ValueError naming the offending argument rather than letting a wrong
    shape or a NaN surface several frames deep inside sklearn.
    """
    array = np.asarray(cloud, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional [n_samples, n_features]; got shape {array.shape}"
        )
    if array.shape[0] < 2:
        raise ValueError(f"{name} needs at least 2 samples; got {array.shape[0]}")
    if array.shape[1] < 1:
        raise ValueError(f"{name} needs at least one feature")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")
    return array


def _unit_interval(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must be a finite number in (0, 1]")
    return number


def _normalize_metrics(metrics: Sequence[str] | None) -> tuple[str, ...]:
    if metrics is None:
        return DEFAULT_METRICS
    if isinstance(metrics, str):
        raise TypeError("metrics must be a sequence of metric names, not a string")
    selected = tuple(dict.fromkeys(str(name) for name in metrics))
    if not selected:
        raise ValueError("metrics must select at least one of " + ", ".join(SUPPORTED_METRICS))
    unknown = [name for name in selected if name not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(
            f"unknown metrics: {', '.join(unknown)}. Supported: {', '.join(SUPPORTED_METRICS)}"
        )
    return selected


def resolve_null_kind(kind: str | None) -> str:
    """Map a legacy null name onto the single supported model."""
    if kind is None or kind == NULL_KIND:
        return NULL_KIND
    if kind in _DEPRECATED_KINDS:
        warnings.warn(
            f"null kind {kind!r} is deprecated ({_DEPRECATED_KINDS[kind]}) and now "
            f"resolves to {NULL_KIND!r}; pass {NULL_KIND!r} explicitly.",
            DeprecationWarning,
            stacklevel=3,
        )
        return NULL_KIND
    raise ValueError(f"unknown null kind: {kind!r}. The only model is {NULL_KIND!r}.")


def _finite(dgm):
    """Strip the [0, inf] H0 bar -- persim returns inf otherwise."""
    d = np.asarray(dgm, dtype=float)
    if d.size == 0:
        return np.empty((0, 2), dtype=float)
    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError("persistence diagram must have shape [number_of_bars, 2]")
    return d[np.isfinite(d).all(axis=1)]


# ---------------------------------------------------------------------------
# The null model
# ---------------------------------------------------------------------------


def effective_rank(eigenvalues: Any) -> float:
    """Roy-Vetterli effective rank: exp(entropy of the eigenvalue spectrum)."""
    values = np.clip(np.asarray(eigenvalues, dtype=float), 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probs = values / total
    return float(np.exp(-np.sum(probs * np.log(probs + 1e-300))))


def fit_low_rank_gaussian(cloud: Any) -> dict[str, Any]:
    """Fit the null once per observed cloud, from the thin centered SVD.

    Never forms a dense ``n_features x n_features`` covariance: only the
    ``rank x n_features`` right-singular basis and the ``rank`` nonzero
    covariance eigenvalues are kept. Rank is the number of singular values above
    ``max(n, d) * eps * s_max``, the usual numerical-rank tolerance.
    """
    x = validate_cloud(cloud)
    n_samples, n_features = x.shape
    mean = x.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(x - mean, full_matrices=False)

    largest = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = max(n_samples, n_features) * np.finfo(float).eps * largest
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < 1:
        raise ValueError(
            "cloud has zero centered rank: every sample is identical, so there is "
            "no covariance structure to match"
        )

    return {
        "kind": NULL_KIND,
        "mean": mean,
        "basis": np.ascontiguousarray(vt[:rank]),
        "eigenvalues": singular_values[:rank] ** 2 / (n_samples - 1),
        "rank": rank,
        "n_samples": n_samples,
        "n_features": n_features,
    }


def sample_low_rank_gaussian(fit: Mapping[str, Any], *, seed: int) -> np.ndarray:
    """Draw one null cloud whose nonzero covariance spectrum matches the fit exactly.

    A Gaussian score matrix is centered and replaced by its orthogonal polar
    factor, which is Haar-uniform over centered orthonormal frames. Scaling that
    frame by ``sqrt(eigenvalues)`` makes the draw's latent Gram matrix equal
    ``(n - 1) diag(eigenvalues)`` to numerical precision, so every draw carries
    the observed spectrum while its orientation is uniformly random.
    """
    rng = np.random.default_rng(int(seed))
    n_samples, rank = int(fit["n_samples"]), int(fit["rank"])

    scores = rng.normal(size=(n_samples, rank))
    scores -= scores.mean(axis=0)
    left, _, right = np.linalg.svd(scores, full_matrices=False)
    frame = (left @ right) * np.sqrt(n_samples - 1)

    latent = frame * np.sqrt(np.asarray(fit["eigenvalues"], dtype=float))
    return latent @ np.asarray(fit["basis"], dtype=float) + np.asarray(fit["mean"], dtype=float)


def null_diagnostics(fit: Mapping[str, Any], reference: Any) -> dict[str, Any]:
    """Check one representative draw against the fitted target. Computed once."""
    target = np.asarray(fit["eigenvalues"], dtype=float)
    drawn = np.asarray(reference, dtype=float)
    singular = np.linalg.svd(drawn - drawn.mean(axis=0), compute_uv=False)
    sample = (singular**2 / (drawn.shape[0] - 1))[: target.size]
    return {
        "rank": int(fit["rank"]),
        "mean_error": float(
            np.linalg.norm(drawn.mean(axis=0) - fit["mean"])
            / (np.linalg.norm(fit["mean"]) + 1e-12)
        ),
        "target_eigenvalues": target.tolist(),
        "sample_eigenvalues": sample.tolist(),
        "relative_spectrum_error": float(
            np.linalg.norm(sample - target) / (np.linalg.norm(target) + 1e-12)
        ),
        "target_effective_rank": effective_rank(target),
        "sample_effective_rank": effective_rank(sample),
    }


# ---------------------------------------------------------------------------
# The one empirical-inference calculation
# ---------------------------------------------------------------------------


def empirical_pvalue(
    observed: float,
    null_values: Sequence[float],
    *,
    direction: str = "greater",
    minimum_valid_nulls: int = MINIMUM_VALID_NULLS,
) -> dict[str, Any]:
    """Rank one observed statistic against a null ensemble.

    ``pvalue = (1 + #{valid null >= observed}) / (valid nulls + 1)`` -- the
    plus-one corrected Monte Carlo estimate. Only *finite* null statistics are
    valid, and their count, not the number requested, sets the denominator:
    ranking over survivors while dividing by the request biases every p-value
    downward and caps the reachable value below 1.

    Inference is unavailable, and ``pvalue`` is NaN, when ``observed`` is
    non-finite or fewer than ``minimum_valid_nulls`` nulls are finite.
    Unavailable is not the same as non-significant: check ``inference_available``
    before acting on ``pvalue``.
    """
    if direction not in {"greater", "two_sided"}:
        raise ValueError("direction must be 'greater' or 'two_sided'")

    values = np.asarray(list(null_values), dtype=float)
    statistic = float(observed)
    if direction == "two_sided":
        values, statistic = np.abs(values), abs(statistic)
    valid = values[np.isfinite(values)]

    if not np.isfinite(statistic):
        reason = "observed statistic is not finite"
    elif valid.size < minimum_valid_nulls:
        reason = (
            f"only {valid.size} of {values.size} null statistics are finite; "
            f"at least {minimum_valid_nulls} are required"
        )
    else:
        reason = None

    result: dict[str, Any] = {
        "observed": float(observed),
        "null_values": [float(value) for value in null_values],
        "n_valid_nulls": int(valid.size),
        "pvalue": float("nan"),
        "minimum_attainable_pvalue": float("nan"),
        "inference_available": False,
        "failure_reason": reason,
    }
    if reason is None:
        result["pvalue"] = float((1 + np.sum(valid >= statistic)) / (valid.size + 1))
        result["minimum_attainable_pvalue"] = float(1.0 / (valid.size + 1))
        result["inference_available"] = True
    return result


def unavailable(reason: str) -> dict[str, Any]:
    """An inference result for a metric that could not be measured at all."""
    return empirical_pvalue(float("nan"), []) | {"failure_reason": reason}


# ---------------------------------------------------------------------------
# Measuring one cloud
# ---------------------------------------------------------------------------


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


class Manifold:
    """A measured cloud: intrinsic dimension, persistence diagrams, curvature.

    pipeline: exposes get_intrinsic_dim, reduce_pca, create_persistence_diagram,
              create_epsilon_graph, compute_ollivier_ricci

    Intrinsic dimension and curvature record their own failures in
    ``intrinsic_dim_error`` / ``curvature_error`` and leave the other metrics
    measurable. A cloud that cannot be projected, or whose matched-density
    epsilon is not positive, is a hard measurement failure and raises.
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
    ):
        self.pipeline = pipeline
        self.opt = validate_cloud(opt_activations, name="opt_activations")
        self.label = label
        self.rng = np.random.default_rng(seed)
        self.eps_density = _unit_interval(eps_density, "eps_density")
        self.var_threshold = _unit_interval(var_threshold, "var_threshold")
        self.metrics = _normalize_metrics(metrics)
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
            except Exception as exc:  # must not abort topology
                self.intrinsic_dim_error = f"{type(exc).__name__}: {exc}"

        if not ({"topology", "curvature"} & set(self.metrics)):
            return

        # REFIT per cloud: PCA can manufacture structure from high-dim noise,
        # so freezing the concept's basis onto the null would never test it.
        projected = self.pipeline.reduce_pca(self.cloud, self.var_threshold)
        distances = pdist(projected)
        if distances.size == 0 or not np.isfinite(distances).all() or distances.max() <= 0.0:
            raise ValueError("projected cloud must contain distinct finite points")

        # SCALE-NORMALISE: a loop is a shape property, not a size property.
        self.diameter = float(distances.max())
        projected = projected / self.diameter
        self.m = int(projected.shape[1])

        if "topology" in self.metrics:
            self.dgms = self.pipeline.create_persistence_diagram(projected)["dgms"]

        # Epsilon at the eps_density quantile of the pairwise distances. A
        # non-positive radius means enough duplicate points that the graph would
        # be empty; concept_geometry refuses the same condition.
        ordered = np.sort(distances / self.diameter)
        self.eps = float(ordered[min(int(self.eps_density * len(ordered)), len(ordered) - 1)])
        if not np.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(
                f"matched-density epsilon selection produced a non-positive radius "
                f"(eps={self.eps!r}); too many duplicate points at "
                f"eps_density={self.eps_density}"
            )

        if "curvature" in self.metrics:
            self._measure_curvature(projected)

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

    def null(self, kind: str | None = None, seed: int | None = None, fit=None) -> "Manifold":
        """Draw and measure one null cloud. See the module docstring for the model.

        ``fit`` reuses a :func:`fit_low_rank_gaussian` result; ensembles must pass
        it, because the fit depends only on ``self.cloud``.
        """
        resolve_null_kind(kind)
        if fit is None:
            fit = fit_low_rank_gaussian(self.cloud)
        null_seed = int(self.rng.integers(1 << 30)) if seed is None else int(seed)
        return Manifold(
            pipeline=self.pipeline,
            opt_activations=sample_low_rank_gaussian(fit, seed=null_seed),
            label=f"null:{NULL_KIND}",
            seed=null_seed,
            eps_density=self.eps_density,
            var_threshold=self.var_threshold,
            metrics=self.metrics,
        )

    def __repr__(self):
        return (
            f"Manifold({self.label}: ID={self.intrinsic_dim:.2f}, m={self.m}, "
            f"{len(self.curvature_values)} edges, metrics={self.metrics})"
        )


def build_null_ensemble(
    observed: Manifold,
    *,
    n_nulls: int,
    base_seed: int,
    fit: Mapping[str, Any],
    **null_kwargs: Any,
) -> tuple[list[Manifold], list[dict[str, Any]]]:
    """Draw ``n_nulls`` nulls, continuing past individual failures.

    Seeds are ``base_seed + index``, so a fixed ``base_seed`` reproduces the whole
    ensemble. Returns the measured nulls and one record per failure; the caller
    decides whether enough survived.
    """
    nulls: list[Manifold] = []
    failures: list[dict[str, Any]] = []
    for index in range(n_nulls):
        seed = base_seed + index
        try:
            nulls.append(observed.null(seed=seed, fit=fit, **null_kwargs))
        except Exception as exc:
            failures.append(
                {"index": index, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            )
    return nulls, failures


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def diagram_distance_pair(dgm1, dgm2) -> dict[str, float]:
    """Bottleneck and Wasserstein between two diagrams, NaN when persim refuses.

    Infinite bars are stripped first. A NaN means the distance could not be
    computed and must be treated as unmeasured, not as a zero distance.
    """
    a, b = _finite(dgm1), _finite(dgm2)
    values: dict[str, float] = {}
    for name, function in (("wasserstein", persim.wasserstein), ("bottleneck", persim.bottleneck)):
        try:
            values[name] = float(function(a, b))
        except Exception:
            values[name] = float("nan")
    return values


def curvature_distribution_difference(values1, values2) -> dict[str, float]:
    """Signed and absolute curvature-distribution comparisons.

    ``negative_fraction_difference`` is signed: positive means the first cloud has
    the larger negative-curvature fraction. ``frac_negative_difference`` is the
    backward-compatible *absolute* alias read by topology_metric; it carries no
    direction. All-NaN when either side has no usable edge curvatures.
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
        """See :func:`curvature_distribution_difference` for the signed contract."""
        return curvature_distribution_difference(m1.curvature_values, m2.curvature_values)

    def compare(self, m1, m2, max_dim=1):
        return {
            "id_difference": abs(m1.intrinsic_dim - m2.intrinsic_dim)
            if np.isfinite(m1.intrinsic_dim) and np.isfinite(m2.intrinsic_dim)
            else float("nan"),
            "diagram_distance": self.diagram_distance(m1, m2, max_dim),
            "curvature": self.curvature_difference(m1, m2),
        }

    def _distances(self, m1, m2, max_dim, metrics) -> dict[str, float]:
        """Named scalar distances between two manifolds. Metrics stay independent."""
        distances: dict[str, float] = {}
        comparison = self.compare(m1, m2, max_dim=max_dim)
        if "topology" in metrics:
            for dimension, values in comparison["diagram_distance"].items():
                distances[f"{dimension}_wasserstein"] = float(values["wasserstein"])
                distances[f"{dimension}_bottleneck"] = float(values["bottleneck"])
        if "curvature" in metrics:
            curvature = comparison["curvature"]
            distances["curvature_wasserstein"] = float(curvature["distribution_distance"])
            distances["curvature_mean_difference"] = float(curvature["mean_difference"])
            distances["curvature_negative_fraction_difference"] = float(
                curvature["negative_fraction_difference"]
            )
        return distances

    @staticmethod
    def _median(values, two_sided: bool) -> float:
        arr = np.asarray(values, dtype=float)
        if two_sided:
            arr = np.abs(arr)
        arr = arr[np.isfinite(arr)]
        # An all-NaN row means "not measurable for this object", not an anomaly.
        return float(np.median(arr)) if arr.size else float("nan")

    def _loo_result(self, matrix: np.ndarray, name: str) -> dict[str, Any]:
        """Observed median distance to the nulls vs leave-one-out null medians.

        Under the null all objects are exchangeable, so each object's median
        distance to the others is exchangeable and ranking the observed one gives
        a calibrated p-value.
        """
        two_sided = name in _TWO_SIDED_METRICS
        nulls = matrix[1:, 1:]
        return empirical_pvalue(
            self._median(matrix[0, 1:], two_sided),
            [
                self._median(np.delete(nulls[index], index), two_sided)
                for index in range(nulls.shape[0])
            ],
        )

    def _id_result(self, observed, nulls, *, inferential: bool) -> dict[str, Any]:
        """Intrinsic-dimension comparison, descriptive unless asked otherwise.

        Descriptive mode reports the observed dimension in ``observed`` and the
        null dimensions in ``null_values``, with no p-value. Inferential mode puts
        ``|observed - null centre|`` in ``observed`` and ranks it against
        leave-one-out null deviations.
        """
        observed_id = float(observed.intrinsic_dim)
        null_ids = np.asarray([float(null.intrinsic_dim) for null in nulls], dtype=float)
        finite = null_ids[np.isfinite(null_ids)]

        if not inferential:
            result = empirical_pvalue(observed_id, null_ids.tolist())
            result.update(
                pvalue=float("nan"),
                minimum_attainable_pvalue=float("nan"),
                inference_available=False,
                failure_reason=(
                    "TwoNN intrinsic dimension is high-variance at these sample "
                    "sizes; reported descriptively unless infer_intrinsic_dimension=True"
                ),
            )
        elif np.isfinite(observed_id) and finite.size:
            centre = float(np.median(finite))
            result = empirical_pvalue(
                abs(observed_id - centre),
                [
                    abs(value - float(np.median(np.delete(finite, index))))
                    if np.isfinite(value) and finite.size > 1
                    else float("nan")
                    for index, value in enumerate(null_ids)
                ],
            )
        else:
            result = unavailable(
                "intrinsic dimension is non-finite for the observed cloud or every null"
            )
        result["estimator_error"] = observed.intrinsic_dim_error
        return result

    def compare_against_nulls(
        self,
        manifold,
        kind: str | None = None,
        n_nulls=30,
        base_seed=0,
        max_dim=1,
        metrics: Sequence[str] | None = None,
        infer_intrinsic_dimension: bool = False,
    ):
        """Compare a manifold against repeated draws from the low-rank null.

        Pure with respect to ``manifold``: nothing is written back onto it. The
        null is fitted once and reused for every draw. Draws that fail to measure
        are recorded in ``failures`` and excluded rather than aborting.
        """
        resolve_null_kind(kind)
        if n_nulls < MINIMUM_VALID_NULLS:
            raise ValueError(f"n_nulls must be at least {MINIMUM_VALID_NULLS}")
        if n_nulls < MINIMUM_NULLS_FOR_ALPHA_05:
            warnings.warn(
                f"n_nulls={n_nulls} caps every p-value at a minimum of "
                f"{1.0 / (n_nulls + 1):.3f}; alpha=0.05 needs at least "
                f"{MINIMUM_NULLS_FOR_ALPHA_05}.",
                stacklevel=2,
            )

        selected = _normalize_metrics(metrics if metrics is not None else manifold.metrics)
        # Re-measure into a new object rather than mutating the caller's: a
        # different metric subset is a different measurement.
        observed = manifold
        if set(manifold.metrics) != set(selected):
            observed = Manifold(
                pipeline=manifold.pipeline,
                opt_activations=manifold.cloud,
                label=manifold.label,
                seed=0,
                eps_density=manifold.eps_density,
                var_threshold=manifold.var_threshold,
                metrics=selected,
            )

        fit = fit_low_rank_gaussian(observed.cloud)
        nulls, failures = build_null_ensemble(
            observed, n_nulls=n_nulls, base_seed=base_seed, fit=fit
        )
        if failures:
            warnings.warn(
                f"{len(failures)} of {n_nulls} null draws failed to measure; "
                "statistics use the surviving draws only (see 'failures' and each "
                "metric's 'n_valid_nulls').",
                stacklevel=2,
            )

        if len(nulls) < MINIMUM_VALID_NULLS:
            reason = (
                f"only {len(nulls)} of {n_nulls} null draws could be measured; "
                f"at least {MINIMUM_VALID_NULLS} are required"
            )
            # A self-comparison yields exactly the names a real comparison would.
            names = ["id_difference"] if "intrinsic_dimension" in selected else []
            names += list(self._distances(observed, observed, max_dim, selected))
            results = {name: unavailable(reason) for name in names}
            diagnostics = None
        else:
            results = {}
            if "intrinsic_dimension" in selected:
                results["id_difference"] = self._id_result(
                    observed, nulls, inferential=infer_intrinsic_dimension
                )
            objects = [observed, *nulls]
            size = len(objects)
            matrices = {
                name: np.full((size, size), np.nan)
                for name in self._distances(observed, nulls[0], max_dim, selected)
            }
            for i in range(size):
                for j in range(i + 1, size):
                    for name, value in self._distances(
                        objects[i], objects[j], max_dim, selected
                    ).items():
                        matrices[name][i, j] = matrices[name][j, i] = value
            results.update(
                {name: self._loo_result(matrix, name) for name, matrix in matrices.items()}
            )
            diagnostics = null_diagnostics(fit, nulls[0].cloud)

        return {
            "null_kind": NULL_KIND,
            "n_requested": n_nulls,
            "n_drawn": len(nulls),
            "failures": failures,
            "base_seed": base_seed,
            "max_dim": max_dim,
            "metrics_requested": list(selected),
            "infer_intrinsic_dimension": bool(infer_intrinsic_dimension),
            "null_diagnostics": diagnostics,
            "observed_graph_diagnostics": observed.graph_diagnostics,
            "metrics": results,
        }
