"""Behavioral tests for null_cloud: the low-rank null, inference, failure handling.

Grouped by behavior. Everything uses the in-process ``FastPipeline`` double, so
nothing here downloads a model or computes curvature.
"""

from __future__ import annotations

import functools
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import skdim
from ripser import ripser
from sklearn.decomposition import PCA

import null_cloud
from null_cloud import (
    MINIMUM_VALID_NULLS,
    NULL_KIND,
    Manifold,
    ManifoldComparator,
    build_null_ensemble,
    effective_rank,
    empirical_pvalue,
    fit_low_rank_gaussian,
    null_diagnostics,
    resolve_null_kind,
    sample_low_rank_gaussian,
    unavailable,
    validate_cloud,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FastPipeline:
    """Minimal measurement backend. Curvature is never expected here."""

    def get_intrinsic_dim(self, cloud):
        return float(skdim.id.TwoNN().fit(np.asarray(cloud, dtype=float)).dimension_)

    def reduce_pca(self, cloud, var_threshold=0.95):
        x = np.asarray(cloud, dtype=float)
        pca = PCA(n_components=min(x.shape))
        full = pca.fit_transform(x)
        count = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_threshold) + 1)
        return full[:, :count]

    def create_persistence_diagram(self, projected):
        return ripser(np.asarray(projected, dtype=float), maxdim=1)

    def create_epsilon_graph(self, projected, eps):
        raise AssertionError("curvature was requested unexpectedly")

    def compute_ollivier_ricci(self, graph):
        raise AssertionError("curvature was requested unexpectedly")


def _isotropic(n=48, d=8, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


def _low_rank(n=48, d=64, rank=4, seed=1):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(d, rank)))
    return (rng.normal(size=(n, rank)) * np.linspace(3.0, 0.5, rank)) @ basis.T


def _circle(n=48, d=8, seed=2):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    basis, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    return np.column_stack((np.cos(angles), np.sin(angles))) @ basis[:, :2].T


def _line(n=48, d=8, seed=3):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    latent = np.column_stack((np.linspace(-1.0, 1.0, n), 0.02 * rng.normal(size=n)))
    return latent @ basis[:, :2].T


def _helix(n=48, d=8, seed=4):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0 * np.pi, n)
    latent = np.column_stack((np.cos(t), np.sin(t), 0.30 * t))
    basis, _ = np.linalg.qr(rng.normal(size=(d, 3)))
    return latent @ basis[:, :3].T


def _swiss_roll(n=48, d=8, seed=5):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.uniform(size=n))
    latent = np.column_stack((t * np.cos(t), 12.0 * rng.uniform(size=n), t * np.sin(t)))
    basis, _ = np.linalg.qr(rng.normal(size=(d, 3)))
    return latent @ basis[:, :3].T


def _curvatures(values):
    return type("C", (), {"curvature_values": np.asarray(values, dtype=float)})()


def _manifold(cloud, seed=0, metrics=("topology",)):
    return Manifold(FastPipeline(), cloud, metrics=metrics, seed=seed)


def _spectrum(cloud):
    x = np.asarray(cloud, dtype=float)
    singular = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    return singular**2 / (x.shape[0] - 1)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_rejects_non_two_dimensional_input(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            validate_cloud(np.arange(10.0))

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            validate_cloud(np.arange(8.0).reshape(1, 8))

    def test_rejects_zero_features(self):
        with pytest.raises(ValueError, match="at least one feature"):
            validate_cloud(np.empty((5, 0)))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_activations(self, bad):
        cloud = _isotropic(seed=0).copy()
        cloud[2, 3] = bad
        with pytest.raises(ValueError, match="only finite values"):
            _manifold(cloud)

    @pytest.mark.parametrize("eps_density", [0.0, -0.1, 1.5, np.nan])
    def test_rejects_invalid_eps_density(self, eps_density):
        with pytest.raises(ValueError, match="eps_density"):
            Manifold(FastPipeline(), _isotropic(seed=0), eps_density=eps_density)

    @pytest.mark.parametrize("var_threshold", [0.0, 1.5, np.inf])
    def test_rejects_invalid_var_threshold(self, var_threshold):
        with pytest.raises(ValueError, match="var_threshold"):
            Manifold(FastPipeline(), _isotropic(seed=0), var_threshold=var_threshold)

    def test_rejects_unknown_metrics(self):
        with pytest.raises(ValueError, match="unknown metrics"):
            Manifold(FastPipeline(), _isotropic(seed=0), metrics=("topology", "bogus"))

    def test_rejects_metrics_given_as_a_bare_string(self):
        with pytest.raises(TypeError, match="not a string"):
            Manifold(FastPipeline(), _isotropic(seed=0), metrics="topology")

    def test_constant_cloud_is_a_measurement_failure(self):
        with pytest.raises(ValueError, match="distinct finite points"):
            _manifold(np.ones((20, 5)))

    def test_zero_epsilon_is_rejected_rather_than_building_an_empty_graph(self):
        # 19 duplicates plus one distinct point put the 10th percentile of the
        # pairwise distances at exactly zero.
        duplicated = np.vstack([np.ones((19, 5)), np.full((1, 5), 2.0)])
        with pytest.raises(ValueError, match="non-positive radius"):
            _manifold(duplicated)

    def test_rank_zero_cloud_is_rejected_clearly(self):
        with pytest.raises(ValueError, match="zero centered rank"):
            fit_low_rank_gaussian(np.ones((6, 4)))


# ---------------------------------------------------------------------------
# The low-rank null: fit and sampling
# ---------------------------------------------------------------------------


class TestLowRankNull:
    def test_draw_shape_matches_the_observed_cloud(self):
        cloud = _low_rank(n=20, d=64, rank=5)
        drawn = sample_low_rank_gaussian(fit_low_rank_gaussian(cloud), seed=0)
        assert drawn.shape == cloud.shape

    def test_rank_never_exceeds_the_centered_maximum(self):
        for n, d, rank in ((12, 2304, 8), (20, 6, 6), (40, 10, 10)):
            fit = fit_low_rank_gaussian(_low_rank(n=n, d=d, rank=rank))
            assert fit["rank"] <= min(n - 1, d)
            assert fit["rank"] >= 1

    def test_no_dense_feature_by_feature_covariance_is_built(self):
        """Only a rank x d basis is stored, never a d x d matrix."""
        fit = fit_low_rank_gaussian(_low_rank(n=12, d=2304, rank=8))
        assert fit["basis"].shape == (fit["rank"], 2304)
        for value in fit.values():
            if isinstance(value, np.ndarray):
                assert value.shape != (2304, 2304)
                assert value.size <= fit["rank"] * 2304

    def test_fixed_seed_reproduces_the_null_cloud(self):
        fit = fit_low_rank_gaussian(_low_rank())
        assert np.allclose(
            sample_low_rank_gaussian(fit, seed=7), sample_low_rank_gaussian(fit, seed=7)
        )

    def test_different_seeds_give_different_null_clouds(self):
        fit = fit_low_rank_gaussian(_low_rank())
        assert not np.allclose(
            sample_low_rank_gaussian(fit, seed=1), sample_low_rank_gaussian(fit, seed=2)
        )

    def test_mean_is_preserved(self):
        cloud = _low_rank(n=30, d=40, rank=6)
        fit = fit_low_rank_gaussian(cloud)
        for seed in range(5):
            drawn = sample_low_rank_gaussian(fit, seed=seed)
            assert np.allclose(drawn.mean(axis=0), cloud.mean(axis=0), atol=1e-8)

    def test_draws_lie_in_the_observed_principal_subspace(self):
        cloud = _low_rank(n=24, d=50, rank=5)
        fit = fit_low_rank_gaussian(cloud)
        drawn = sample_low_rank_gaussian(fit, seed=0)
        centered = drawn - drawn.mean(axis=0)
        # Projecting onto the fitted basis and back must be a no-op.
        basis = fit["basis"]
        reprojected = (centered @ basis.T) @ basis
        assert np.allclose(centered, reprojected, atol=1e-8)

    def test_nonzero_covariance_spectrum_is_matched_exactly(self):
        cloud = _low_rank(n=18, d=90, rank=7)
        fit = fit_low_rank_gaussian(cloud)
        target = fit["eigenvalues"]
        for seed in range(5):
            sample = _spectrum(sample_low_rank_gaussian(fit, seed=seed))[: target.size]
            assert np.allclose(sample, target, rtol=1e-8, atol=1e-10)

    def test_effective_rank_is_matched_exactly(self):
        cloud = _low_rank(n=16, d=120, rank=6)
        fit = fit_low_rank_gaussian(cloud)
        observed = effective_rank(_spectrum(cloud)[: fit["rank"]])
        for seed in range(5):
            drawn = sample_low_rank_gaussian(fit, seed=seed)
            assert effective_rank(_spectrum(drawn)[: fit["rank"]]) == pytest.approx(
                observed, rel=1e-6
            )

    def test_orientation_actually_varies_between_draws(self):
        """Matching the spectrum exactly must not freeze the point configuration."""
        fit = fit_low_rank_gaussian(_low_rank(n=20, d=40, rank=5))
        from scipy.spatial.distance import pdist

        first = pdist(sample_low_rank_gaussian(fit, seed=0))
        second = pdist(sample_low_rank_gaussian(fit, seed=1))
        assert not np.allclose(first, second)

    def test_diagnostics_confirm_the_match_and_stay_finite(self):
        fit = fit_low_rank_gaussian(_low_rank(n=14, d=200, rank=6))
        diagnostics = null_diagnostics(fit, sample_low_rank_gaussian(fit, seed=0))
        assert set(diagnostics) == {
            "rank",
            "mean_error",
            "target_eigenvalues",
            "sample_eigenvalues",
            "relative_spectrum_error",
            "target_effective_rank",
            "sample_effective_rank",
        }
        assert diagnostics["relative_spectrum_error"] < 1e-10
        assert diagnostics["mean_error"] < 1e-10
        assert diagnostics["target_effective_rank"] == pytest.approx(
            diagnostics["sample_effective_rank"], rel=1e-8
        )
        assert all(
            np.isfinite(value)
            for value in (
                diagnostics["mean_error"],
                diagnostics["relative_spectrum_error"],
                diagnostics["target_effective_rank"],
                diagnostics["sample_effective_rank"],
            )
        )


class TestRealShapedCloud:
    """The regime the project actually runs in: n = 12, d = 2304."""

    CLOUD = None

    @pytest.fixture(scope="class")
    def cloud(self):
        rng = np.random.default_rng(11)
        basis, _ = np.linalg.qr(rng.normal(size=(2304, 8)))
        return (rng.normal(size=(12, 8)) * np.linspace(3.0, 0.3, 8)) @ basis.T

    def test_fit_returns_a_rank_within_the_centered_maximum(self, cloud):
        fit = fit_low_rank_gaussian(cloud)
        assert fit["rank"] <= 11
        assert fit["basis"].shape == (fit["rank"], 2304)

    def test_sampling_is_deterministic_and_shaped_correctly(self, cloud):
        fit = fit_low_rank_gaussian(cloud)
        first = sample_low_rank_gaussian(fit, seed=3)
        assert first.shape == (12, 2304)
        assert np.array_equal(first, sample_low_rank_gaussian(fit, seed=3))

    def test_spectrum_diagnostics_are_finite_and_exact(self, cloud):
        fit = fit_low_rank_gaussian(cloud)
        diagnostics = null_diagnostics(fit, sample_low_rank_gaussian(fit, seed=0))
        assert np.isfinite(diagnostics["relative_spectrum_error"])
        assert diagnostics["relative_spectrum_error"] < 1e-10

    def test_fitting_once_is_much_cheaper_than_refitting_per_draw(self, cloud):
        import time

        n_draws = 10
        start = time.perf_counter()
        fit = fit_low_rank_gaussian(cloud)
        for seed in range(n_draws):
            sample_low_rank_gaussian(fit, seed=seed)
        once = time.perf_counter() - start

        start = time.perf_counter()
        for seed in range(n_draws):
            sample_low_rank_gaussian(fit_low_rank_gaussian(cloud), seed=seed)
        per_draw = time.perf_counter() - start
        assert once < per_draw


# ---------------------------------------------------------------------------
# Null naming
# ---------------------------------------------------------------------------


class TestNullNaming:
    def test_canonical_name_resolves_to_itself(self):
        assert resolve_null_kind(None) == NULL_KIND
        assert resolve_null_kind(NULL_KIND) == NULL_KIND

    @pytest.mark.parametrize(
        "legacy", ["noise", "covariance_gaussian", "isotropic_gaussian"]
    )
    def test_legacy_names_warn_and_resolve_to_the_one_model(self, legacy):
        with pytest.deprecated_call():
            assert resolve_null_kind(legacy) == NULL_KIND

    def test_unknown_name_is_an_error(self):
        with pytest.raises(ValueError, match="unknown null kind"):
            resolve_null_kind("shuffled")

    def test_manifold_null_draws_the_canonical_model(self):
        null = _manifold(_circle(seed=40), seed=40).null(seed=7)
        assert null.label == f"null:{NULL_KIND}"
        assert null.cloud.shape == _circle(seed=40).shape


# ---------------------------------------------------------------------------
# Empirical p-value correctness
# ---------------------------------------------------------------------------


class TestEmpiricalPvalue:
    def test_observed_larger_than_every_null_hits_the_floor(self):
        result = empirical_pvalue(10.0, [1.0, 2.0, 3.0, 4.0])
        assert result["pvalue"] == pytest.approx(1.0 / 5.0)
        assert result["minimum_attainable_pvalue"] == pytest.approx(1.0 / 5.0)
        assert result["inference_available"] is True

    def test_observed_smaller_than_every_null_gives_one(self):
        assert empirical_pvalue(0.0, [1.0, 2.0, 3.0, 4.0])["pvalue"] == pytest.approx(1.0)

    def test_plus_one_correction_is_retained(self):
        # Two of four nulls are >= observed, so the rank is 1 + 2 = 3.
        assert empirical_pvalue(2.5, [1.0, 2.0, 3.0, 4.0])["pvalue"] == pytest.approx(3.0 / 5.0)

    @pytest.mark.parametrize("observed", [np.nan, np.inf, -np.inf])
    def test_non_finite_observed_score_is_never_significant(self, observed):
        result = empirical_pvalue(observed, [1.0, 2.0, 3.0, 4.0])
        assert np.isnan(result["pvalue"])
        assert result["inference_available"] is False
        assert "not finite" in result["failure_reason"]

    def test_non_finite_nulls_are_excluded_from_the_denominator(self):
        result = empirical_pvalue(0.5, [1.0, np.nan, 2.0, np.inf, 3.0, np.nan])
        assert result["n_valid_nulls"] == 3
        # All three survivors exceed the observed value, so p is 1.0 against
        # three nulls -- not 4/7 against the six requested.
        assert result["pvalue"] == pytest.approx(1.0)
        assert result["minimum_attainable_pvalue"] == pytest.approx(0.25)

    def test_too_few_valid_nulls_marks_inference_unavailable(self):
        result = empirical_pvalue(0.5, [1.0, np.nan, np.nan, np.nan])
        assert result["inference_available"] is False
        assert np.isnan(result["pvalue"])
        assert "at least" in result["failure_reason"]

    def test_two_sided_direction_ranks_absolute_deviations(self):
        nulls = [-3.0, -1.0, 1.0, 3.0]
        assert (
            empirical_pvalue(-2.0, nulls, direction="two_sided")["pvalue"]
            == empirical_pvalue(2.0, nulls, direction="two_sided")["pvalue"]
        )

    def test_rejects_unknown_direction(self):
        with pytest.raises(ValueError, match="direction"):
            empirical_pvalue(1.0, [1.0, 2.0, 3.0], direction="less")

    def test_unavailable_carries_the_reason(self):
        result = unavailable("backend exploded")
        assert result["inference_available"] is False
        assert result["failure_reason"] == "backend exploded"
        assert np.isnan(result["pvalue"])

    def test_result_schema_is_the_canonical_one(self):
        assert set(empirical_pvalue(1.0, [0.5, 0.6, 0.7])) == {
            "observed",
            "null_values",
            "pvalue",
            "inference_available",
            "n_valid_nulls",
            "minimum_attainable_pvalue",
            "failure_reason",
        }

    def test_is_calibrated_under_exchangeability(self):
        from scipy.spatial.distance import pdist, squareform

        comparator = ManifoldComparator()
        rng = np.random.default_rng(4242)
        pvalues = []
        for _ in range(1500):
            matrix = squareform(pdist(rng.normal(size=(20, 4))))
            np.fill_diagonal(matrix, np.nan)
            pvalues.append(comparator._loo_result(matrix, "H1_bottleneck")["pvalue"])
        pvalues = np.asarray(pvalues, dtype=float)
        assert abs(np.mean(pvalues <= 0.10) - 0.10) < 0.03
        assert abs(np.mean(pvalues <= 0.20) - 0.20) < 0.04


# ---------------------------------------------------------------------------
# Ensembles: reproducibility, failure handling, mutation safety
# ---------------------------------------------------------------------------


class _FailAtCall(FastPipeline):
    """Raise on selected reduce_pca calls, otherwise behave normally."""

    def __init__(self, fail_on=(), always=False):
        self.calls = 0
        self.fail_on = set(fail_on)
        self.always = always

    def reduce_pca(self, cloud, var_threshold=0.95):
        self.calls += 1
        if self.always or self.calls in self.fail_on:
            raise RuntimeError("simulated backend failure")
        return super().reduce_pca(cloud, var_threshold)


class TestEnsembleBehavior:
    def test_fixed_base_seed_gives_identical_results(self):
        cloud, comparator = _circle(seed=32), ManifoldComparator()
        runs = [
            comparator.compare_against_nulls(
                _manifold(cloud, seed=32), n_nulls=19, base_seed=555, metrics=("topology",)
            )
            for _ in range(2)
        ]
        assert {k: v["pvalue"] for k, v in runs[0]["metrics"].items()} == {
            k: v["pvalue"] for k, v in runs[1]["metrics"].items()
        }

    def test_null_is_fitted_once_per_ensemble(self, monkeypatch):
        calls = {"n": 0}
        original = null_cloud.fit_low_rank_gaussian

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(null_cloud, "fit_low_rank_gaussian", counting)
        ManifoldComparator().compare_against_nulls(
            _manifold(_isotropic(seed=25), seed=25),
            n_nulls=19,
            base_seed=25,
            metrics=("topology",),
        )
        assert calls["n"] == 1

    def test_one_failed_draw_does_not_abort_the_ensemble(self):
        manifold = _manifold(_circle(seed=23), seed=23)
        manifold.pipeline = _FailAtCall(fail_on={5})
        with pytest.warns(UserWarning, match="failed to measure"):
            result = ManifoldComparator().compare_against_nulls(
                manifold, n_nulls=19, base_seed=23, metrics=("topology",)
            )
        assert len(result["failures"]) == 1
        assert result["n_requested"] == 19
        assert result["n_drawn"] == 18
        assert result["metrics"]["H1_bottleneck"]["n_valid_nulls"] == 18
        assert result["metrics"]["H1_bottleneck"]["inference_available"] is True

    def test_failed_draws_do_not_bias_the_denominator(self):
        manifold = _manifold(_circle(seed=44), seed=44)
        manifold.pipeline = _FailAtCall(fail_on={2, 3, 4, 5, 6})
        with pytest.warns(UserWarning):
            result = ManifoldComparator().compare_against_nulls(
                manifold, n_nulls=19, base_seed=44, metrics=("topology",)
            )
        block = result["metrics"]["H1_bottleneck"]
        assert result["n_drawn"] == 14
        assert block["n_valid_nulls"] == 14
        assert block["minimum_attainable_pvalue"] == pytest.approx(1.0 / 15.0)

    def test_all_failed_draws_produce_an_explicit_unavailable_result(self):
        manifold = _manifold(_circle(seed=45), seed=45)
        manifold.pipeline = _FailAtCall(always=True)
        with pytest.warns(UserWarning):
            result = ManifoldComparator().compare_against_nulls(
                manifold, n_nulls=19, base_seed=45, metrics=("topology",)
            )
        assert result["n_drawn"] == 0
        assert len(result["failures"]) == 19
        for block in result["metrics"].values():
            assert block["inference_available"] is False
            assert np.isnan(block["pvalue"])
            assert "could be measured" in block["failure_reason"]

    def test_build_null_ensemble_records_each_failure_with_its_seed(self):
        manifold = _manifold(_circle(seed=46), seed=46)
        manifold.pipeline = _FailAtCall(fail_on={1, 3})
        nulls, failures = build_null_ensemble(
            manifold, n_nulls=5, base_seed=900, fit=fit_low_rank_gaussian(manifold.cloud)
        )
        assert len(nulls) == 3
        assert [failure["seed"] for failure in failures] == [900, 902]

    def test_too_few_nulls_warns_that_alpha_005_is_unreachable(self):
        with pytest.warns(UserWarning, match="alpha=0.05"):
            ManifoldComparator().compare_against_nulls(
                _manifold(_circle(seed=22), seed=22), n_nulls=5, base_seed=22, metrics=("topology",)
            )

    def test_n_nulls_below_the_minimum_is_a_hard_error(self):
        with pytest.raises(ValueError, match="n_nulls must be at least"):
            ManifoldComparator().compare_against_nulls(
                _manifold(_circle(seed=47), seed=47),
                n_nulls=MINIMUM_VALID_NULLS - 1,
                metrics=("topology",),
            )

    def test_comparison_leaves_the_observed_manifold_unchanged(self):
        manifold = _manifold(_isotropic(seed=24), seed=24)
        before = {
            "metrics": manifold.metrics,
            "eps": manifold.eps,
            "diameter": manifold.diameter,
            "m": manifold.m,
            "cloud": manifold.cloud.copy(),
            "dgms": [d.copy() for d in manifold.dgms],
            "rng_state": manifold.rng.bit_generator.state,
        }
        ManifoldComparator().compare_against_nulls(
            manifold, n_nulls=19, base_seed=24, metrics=("topology",)
        )
        assert manifold.metrics == before["metrics"]
        assert manifold.eps == before["eps"]
        assert manifold.diameter == before["diameter"]
        assert manifold.m == before["m"]
        assert np.array_equal(manifold.cloud, before["cloud"])
        assert all(np.array_equal(a, b) for a, b in zip(manifold.dgms, before["dgms"]))
        assert manifold.rng.bit_generator.state == before["rng_state"]

    def test_result_reports_the_null_diagnostics_once(self):
        result = ManifoldComparator().compare_against_nulls(
            _manifold(_low_rank(n=30, d=60, rank=5), seed=8),
            n_nulls=19,
            base_seed=8,
            metrics=("topology",),
        )
        assert result["null_kind"] == NULL_KIND
        assert result["null_diagnostics"]["relative_spectrum_error"] < 1e-10
        assert isinstance(result["null_diagnostics"]["rank"], int)


class TestMetricIndependence:
    def test_intrinsic_dimension_failure_leaves_topology_measurable(self):
        class NoIntrinsicDim(FastPipeline):
            def get_intrinsic_dim(self, cloud):
                raise RuntimeError("estimator unavailable")

        manifold = Manifold(
            NoIntrinsicDim(), _circle(seed=48), metrics=("intrinsic_dimension", "topology"), seed=48
        )
        assert np.isnan(manifold.intrinsic_dim)
        assert "estimator unavailable" in manifold.intrinsic_dim_error
        assert len(manifold.dgms) >= 2

    def test_selecting_only_topology_skips_curvature_and_id(self):
        manifold = Manifold(
            FastPipeline(), _circle(seed=3), metrics=("intrinsic_dimension", "topology"), seed=3
        )
        result = ManifoldComparator().compare_against_nulls(
            manifold, n_nulls=19, base_seed=10, metrics=("topology",)
        )
        assert "H1_bottleneck" in result["metrics"]
        assert "curvature_wasserstein" not in result["metrics"]
        assert "id_difference" not in result["metrics"]

    def test_descriptive_id_reports_no_pvalue(self):
        block = ManifoldComparator().compare_against_nulls(
            _manifold(_isotropic(seed=4), seed=4, metrics=("intrinsic_dimension",)),
            n_nulls=19,
            base_seed=20,
            metrics=("intrinsic_dimension",),
        )["metrics"]["id_difference"]
        assert block["inference_available"] is False
        assert np.isnan(block["pvalue"])
        assert "descriptively" in block["failure_reason"]

    def test_inferential_id_reports_a_bounded_pvalue(self):
        block = ManifoldComparator().compare_against_nulls(
            _manifold(_isotropic(seed=4), seed=4, metrics=("intrinsic_dimension",)),
            n_nulls=19,
            base_seed=20,
            metrics=("intrinsic_dimension",),
            infer_intrinsic_dimension=True,
        )["metrics"]["id_difference"]
        assert block["inference_available"] is True
        assert block["minimum_attainable_pvalue"] <= block["pvalue"] <= 1.0


class TestCurvatureComparison:
    def test_signed_difference_reverses_with_direction(self):
        forward = ManifoldComparator().curvature_difference(
            _curvatures([-0.9, -0.8]), _curvatures([0.9, 0.8])
        )
        backward = ManifoldComparator().curvature_difference(
            _curvatures([0.9, 0.8]), _curvatures([-0.9, -0.8])
        )
        assert forward["negative_fraction_difference"] == pytest.approx(1.0)
        assert backward["negative_fraction_difference"] == pytest.approx(-1.0)
        assert forward["mean_difference"] == pytest.approx(-backward["mean_difference"])

    def test_absolute_difference_is_unchanged_when_direction_reverses(self):
        forward = ManifoldComparator().curvature_difference(
            _curvatures([-0.9, -0.8]), _curvatures([0.9, 0.8])
        )
        backward = ManifoldComparator().curvature_difference(
            _curvatures([0.9, 0.8]), _curvatures([-0.9, -0.8])
        )
        for key in ("absolute_negative_fraction_difference", "frac_negative_difference"):
            assert forward[key] == pytest.approx(backward[key]) == pytest.approx(1.0)

    def test_empty_curvature_yields_nan_rather_than_raising(self):
        result = ManifoldComparator().curvature_difference(
            _curvatures([]), _curvatures([0.1, 0.2])
        )
        assert all(np.isnan(value) for value in result.values())


# ---------------------------------------------------------------------------
# Scientific behavior
# ---------------------------------------------------------------------------


TOPOLOGY_AXES = ("H0_bottleneck", "H0_wasserstein", "H1_bottleneck", "H1_wasserstein")


@functools.lru_cache(maxsize=None)
def _reject_rates(maker, *, trials=8, n_nulls=19, alpha=0.05):
    """Rejection rate per topology axis over ``trials`` independent seeds.

    Memoised: several tests compare the same shapes, and each call runs
    trials x n_nulls measurements.
    """
    comparator, pipeline = ManifoldComparator(), FastPipeline()
    counts = dict.fromkeys(TOPOLOGY_AXES, 0)
    for trial in range(trials):
        manifold = Manifold(
            pipeline, maker(seed=2000 + trial), metrics=("topology",), seed=2000 + trial
        )
        metrics = comparator.compare_against_nulls(
            manifold, n_nulls=n_nulls, base_seed=8000 + trial * 40, metrics=("topology",)
        )["metrics"]
        for axis in TOPOLOGY_AXES:
            block = metrics[axis]
            if block["inference_available"] and block["pvalue"] <= alpha:
                counts[axis] += 1
    return tuple(sorted((axis, count / trials) for axis, count in counts.items()))


def _rates(maker):
    return dict(_reject_rates(maker))


class TestScientificBehavior:
    """Several seeds per shape. These are calibration checks, not proofs.

    Measured rejection rates at alpha = 0.05 over 8 seeds, n_nulls = 19:

        shape        H0_bn  H0_wass  H1_bn  H1_wass
        isotropic     0.00     0.00   0.00     0.12
        low_rank      0.12     0.25   0.00     0.00
        line          0.00     1.00   0.00     0.00
        circle        0.00     0.62   1.00     1.00
        helix         0.75     1.00   0.00     0.00
        swiss_roll    0.12     0.25   1.00     1.00

    Loops (circle, Swiss roll) are caught by H1; curves (line, helix) have no
    loop and are caught by the H0 merge structure instead. Which axis fires is a
    property of the shape, so the tests below assert per-shape rather than
    forcing every shape through H1.
    """

    @pytest.mark.parametrize("maker", [_isotropic, _low_rank])
    def test_linear_gaussian_clouds_are_not_systematically_rejected(self, maker):
        """A linear-Gaussian cloud is exactly what this null explains."""
        rates = _rates(maker)
        assert max(rates.values()) <= 0.30, rates

    @pytest.mark.parametrize("maker", [_circle, _swiss_roll])
    def test_shapes_with_a_loop_are_caught_by_h1(self, maker):
        assert _rates(maker)["H1_bottleneck"] >= 0.5

    @pytest.mark.parametrize("maker", [_line, _helix])
    def test_curves_without_a_loop_are_caught_by_h0(self, maker):
        rates = _rates(maker)
        assert rates["H0_wasserstein"] >= 0.5, rates
        # A contractible curve genuinely has no 1-cycle, so H1 should stay quiet.
        assert rates["H1_bottleneck"] <= 0.25, rates

    @pytest.mark.parametrize("maker", [_circle, _helix, _line, _swiss_roll])
    def test_nonlinear_shapes_outrank_matched_gaussian_controls(self, maker):
        """The contrast against the controls, not any absolute rate, is the claim."""
        nonlinear = max(_rates(maker).values())
        control = max(max(_rates(_isotropic).values()), max(_rates(_low_rank).values()))
        assert nonlinear > control

    def test_rejection_is_not_driven_by_spectrum_or_rank_mismatch(self):
        """Every draw carries the observed spectrum, so rank cannot be the signal."""
        diagnostics = ManifoldComparator().compare_against_nulls(
            _manifold(_circle(seed=77), seed=77), n_nulls=19, base_seed=77, metrics=("topology",)
        )["null_diagnostics"]
        assert diagnostics["relative_spectrum_error"] < 1e-10
        assert diagnostics["target_effective_rank"] == pytest.approx(
            diagnostics["sample_effective_rank"], rel=1e-8
        )

    def test_twonn_variability_is_large_relative_to_signal(self):
        """Why intrinsic dimension stays descriptive by default."""
        ids = [FastPipeline().get_intrinsic_dim(_isotropic(n=36, d=12, seed=i)) for i in range(30)]
        assert np.std(ids) > 1.0
