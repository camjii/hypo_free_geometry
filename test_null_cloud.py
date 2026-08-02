"""Behavioral tests for null_cloud: inference, failure handling, determinism.

Grouped by behavior rather than by function. Every test uses the in-process
``FastPipeline`` double, so nothing here downloads a model or runs curvature.
"""

from __future__ import annotations

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
    Manifold,
    ManifoldComparator,
    build_null_ensemble,
    empirical_pvalue,
    fit_null_gaussian,
    sample_null_cloud,
    unavailable_result,
    validate_cloud,
)


# ---------------------------------------------------------------------------
# Fixtures and test doubles
# ---------------------------------------------------------------------------


class FastPipeline:
    """Minimal measurement backend. Curvature is never expected here."""

    def get_intrinsic_dim(self, point_cloud):
        return float(skdim.id.TwoNN().fit(np.asarray(point_cloud, dtype=float)).dimension_)

    def reduce_pca(self, point_cloud, var_threshold=0.95):
        x = np.asarray(point_cloud, dtype=float)
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


def _spiked(n=48, d=8, seed=1):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 2))
    loadings = np.zeros((d, 2))
    loadings[0, 0] = 3.0
    loadings[1, 1] = 2.0
    return factors @ loadings.T + 0.05 * rng.normal(size=(n, d))


def _circle(n=48, d=8, seed=2):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    latent = np.column_stack((np.cos(angles), np.sin(angles)))
    basis = rng.normal(size=(d, 2))
    q, _ = np.linalg.qr(basis)
    return latent @ q[:, :2].T


def _curvature_double(values):
    return type("C", (), {"curvature_values": np.asarray(values, dtype=float)})()


def _topology_manifold(cloud, seed=0):
    return Manifold(FastPipeline(), cloud, metrics=("topology",), seed=seed)


@pytest.fixture
def comparator():
    return ManifoldComparator()


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
            Manifold(FastPipeline(), cloud, metrics=("topology",))

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

    def test_rejects_unknown_null_kind(self):
        with pytest.raises(ValueError, match="unknown null kind"):
            fit_null_gaussian(_isotropic(seed=0), kind="bogus")


class TestDegenerateGeometry:
    def test_constant_cloud_is_a_measurement_failure(self):
        with pytest.raises(ValueError, match="distinct finite points"):
            Manifold(FastPipeline(), np.ones((20, 5)), metrics=("topology",))

    def test_zero_epsilon_is_rejected_rather_than_building_an_empty_graph(self):
        # 19 duplicates plus one distinct point puts the 10th percentile of the
        # pairwise distances at exactly zero.
        duplicated = np.vstack([np.ones((19, 5)), np.full((1, 5), 2.0)])
        with pytest.raises(ValueError, match="non-positive radius"):
            Manifold(FastPipeline(), duplicated, metrics=("topology",))


# ---------------------------------------------------------------------------
# Empirical p-value correctness
# ---------------------------------------------------------------------------


class TestEmpiricalPvalue:
    def test_observed_larger_than_every_null_hits_the_floor(self):
        result = empirical_pvalue(10.0, [1.0, 2.0, 3.0, 4.0])
        assert result["pvalue"] == pytest.approx(1.0 / 5.0)
        assert result["minimum_attainable_pvalue"] == pytest.approx(1.0 / 5.0)
        assert result["empirical_rank"] == 1
        assert result["inference_available"] is True

    def test_observed_smaller_than_every_null_gives_one(self):
        assert empirical_pvalue(0.0, [1.0, 2.0, 3.0, 4.0])["pvalue"] == pytest.approx(1.0)

    def test_plus_one_correction_is_retained(self):
        # Two of four nulls are >= observed, so the rank is 1 + 2 = 3.
        result = empirical_pvalue(2.5, [1.0, 2.0, 3.0, 4.0])
        assert result["empirical_rank"] == 3
        assert result["pvalue"] == pytest.approx(3.0 / 5.0)

    def test_nan_observed_score_is_not_significant(self):
        result = empirical_pvalue(float("nan"), [1.0, 2.0, 3.0, 4.0])
        assert np.isnan(result["pvalue"])
        assert result["inference_available"] is False
        assert "not finite" in result["failure_reason"]
        assert result["empirical_rank"] is None

    @pytest.mark.parametrize("observed", [np.inf, -np.inf])
    def test_infinite_observed_score_is_not_significant(self, observed):
        result = empirical_pvalue(observed, [1.0, 2.0, 3.0, 4.0])
        assert np.isnan(result["pvalue"])
        assert result["inference_available"] is False

    def test_non_finite_nulls_are_excluded_from_the_denominator(self):
        result = empirical_pvalue(0.5, [1.0, np.nan, 2.0, np.inf, 3.0, np.nan])
        assert result["n_valid_nulls"] == 3
        # All three survivors exceed the observed score, so the p-value is 1.0
        # against three nulls -- not 4/7 against the requested six.
        assert result["pvalue"] == pytest.approx(1.0)
        assert result["minimum_attainable_pvalue"] == pytest.approx(0.25)

    def test_too_few_valid_nulls_marks_inference_unavailable(self):
        result = empirical_pvalue(0.5, [1.0, np.nan, np.nan, np.nan])
        assert result["n_valid_nulls"] == 1
        assert result["inference_available"] is False
        assert np.isnan(result["pvalue"])
        assert "at least" in result["failure_reason"]

    def test_two_sided_direction_ranks_absolute_deviations(self):
        nulls = [-3.0, -1.0, 1.0, 3.0]
        assert empirical_pvalue(-2.0, nulls, direction="two_sided")["pvalue"] == (
            empirical_pvalue(2.0, nulls, direction="two_sided")["pvalue"]
        )
        assert empirical_pvalue(-4.0, nulls, direction="two_sided")["pvalue"] == pytest.approx(0.2)

    def test_rejects_unknown_direction(self):
        with pytest.raises(ValueError, match="direction"):
            empirical_pvalue(1.0, [1.0, 2.0, 3.0], direction="less")

    def test_unavailable_result_carries_the_reason(self):
        result = unavailable_result("backend exploded")
        assert result["inference_available"] is False
        assert result["failure_reason"] == "backend exploded"
        assert np.isnan(result["pvalue"])

    def test_is_calibrated_under_exchangeability(self, comparator):
        """Exchangeable objects must reject at about alpha."""
        from scipy.spatial.distance import pdist, squareform

        n_nulls = 19
        rng = np.random.default_rng(4242)
        pvalues = []
        for _ in range(1500):
            matrix = squareform(pdist(rng.normal(size=(n_nulls + 1, 4))))
            np.fill_diagonal(matrix, np.nan)
            pvalues.append(comparator._loo_result(matrix, "H1_bottleneck")["pvalue"])
        pvalues = np.asarray(pvalues, dtype=float)
        assert abs(np.mean(pvalues <= 0.10) - 0.10) < 0.03
        assert abs(np.mean(pvalues <= 0.20) - 0.20) < 0.04


# ---------------------------------------------------------------------------
# Sampling and covariance fitting
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sampled_cloud_recovers_a_known_covariance(self):
        rng = np.random.default_rng(29)
        dimension = 6
        rotation, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        target = rotation @ np.diag([9.0, 4.0, 2.0, 1.0, 0.5, 0.25]) @ rotation.T
        observed = rng.multivariate_normal(np.zeros(dimension), target, size=4000)

        fit = fit_null_gaussian(observed, kind="covariance_gaussian")
        drawn = sample_null_cloud(fit, sample_count=20000, seed=3)
        relative = np.linalg.norm(np.cov(drawn, rowvar=False) - fit["covariance"]) / np.linalg.norm(
            fit["covariance"]
        )
        assert relative < 0.05
        assert np.allclose(drawn.mean(axis=0), fit["mean"], atol=0.1)

    def test_isotropic_null_preserves_average_variance_and_drops_anisotropy(self):
        cloud = _spiked(n=200, d=8, seed=30)
        covariance = fit_null_gaussian(cloud, kind="isotropic_gaussian")["covariance"]
        assert np.allclose(covariance, np.diag(np.diag(covariance)))
        assert np.allclose(np.diag(covariance), covariance[0, 0])
        assert covariance[0, 0] == pytest.approx(float(np.var(cloud, axis=0, ddof=1).mean()))

    def test_noise_alias_resolves_to_the_covariance_null(self):
        cloud = _isotropic(seed=15)
        assert fit_null_gaussian(cloud, kind="noise")["kind"] == "covariance_gaussian"
        manifold = _topology_manifold(cloud, seed=15)
        with pytest.warns(UserWarning):
            result = ManifoldComparator().compare_against_nulls(
                manifold, kind="noise", n_nulls=4, base_seed=16, metrics=("topology",)
            )
        assert result["null_kind"] == "covariance_gaussian"

    @pytest.mark.parametrize("kind", ["covariance_gaussian", "isotropic_gaussian", "noise"])
    def test_supported_null_kinds_still_produce_a_measurable_draw(self, kind):
        manifold = _topology_manifold(_circle(seed=40), seed=40)
        null = manifold.null(kind=kind, seed=7)
        assert null.cloud.shape == manifold.cloud.shape
        assert len(null.dgms) >= 2

    def test_shuffled_kind_is_explicitly_unsupported(self):
        manifold = _topology_manifold(_circle(seed=41), seed=41)
        with pytest.raises(NotImplementedError, match="shuffled"):
            manifold.null(kind="shuffled")


class TestCovarianceDiagnostics:
    def test_wide_clouds_are_flagged_as_not_covariance_matched(self):
        rng = np.random.default_rng(27)
        wide = rng.normal(size=(36, 768)) @ np.diag(np.linspace(3.0, 0.1, 768))
        match = fit_null_gaussian(wide, kind="covariance_gaussian")["diagnostics"]["covariance_match"]
        assert match["is_matched"] is False
        assert match["relative_frobenius_difference"] > 0.30
        assert match["effective_rank_inflation"] > 2.0

    def test_well_sampled_clouds_are_flagged_as_matched(self):
        narrow = np.random.default_rng(27).normal(size=(4000, 6))
        match = fit_null_gaussian(narrow, kind="covariance_gaussian")["diagnostics"]["covariance_match"]
        assert match["is_matched"] is True

    def test_isotropic_null_is_never_claimed_to_be_matched(self):
        match = fit_null_gaussian(_spiked(seed=9), kind="isotropic_gaussian")["diagnostics"][
            "covariance_match"
        ]
        assert match["is_matched"] is False
        assert "discards covariance structure" in match["note"]

    def test_both_estimators_expose_their_regularisation(self):
        spiked = _spiked(seed=9)
        lw = fit_null_gaussian(spiked, kind="covariance_gaussian", covariance_estimator="ledoit_wolf")
        reg = fit_null_gaussian(
            spiked, kind="covariance_gaussian", covariance_estimator="regularized_empirical"
        )
        assert "shrinkage" in lw["diagnostics"]
        assert "ridge" in reg["diagnostics"]
        assert (
            lw["diagnostics"]["null_effective_rank"]
            >= reg["diagnostics"]["null_effective_rank"] - 1e-6
        )

    def test_mismatch_warns_exactly_once_per_ensemble(self, recwarn):
        rng = np.random.default_rng(28)
        wide = rng.normal(size=(30, 400)) @ np.diag(np.linspace(3.0, 0.1, 400))
        manifold = _topology_manifold(wide, seed=28)
        ManifoldComparator().compare_against_nulls(
            manifold, kind="covariance_gaussian", n_nulls=19, base_seed=28, metrics=("topology",)
        )
        mismatch = [w for w in recwarn if "not covariance-matched" in str(w.message)]
        assert len(mismatch) == 1

    def test_diagnostics_are_reported_once_on_the_result(self):
        manifold = _topology_manifold(_circle(seed=42), seed=42)
        result = ManifoldComparator().compare_against_nulls(
            manifold, kind="isotropic_gaussian", n_nulls=19, base_seed=42, metrics=("topology",)
        )
        assert "covariance_match" in result["null_fit_diagnostics"]
        assert isinstance(result["null_fit_diagnostics"]["covariance_match"], dict)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_fixed_base_seed_gives_identical_results(self):
        cloud = _circle(seed=32)
        pipeline, comparator = FastPipeline(), ManifoldComparator()
        outputs = []
        for _ in range(2):
            manifold = Manifold(pipeline, cloud, metrics=("topology",), seed=32)
            result = comparator.compare_against_nulls(
                manifold, kind="covariance_gaussian", n_nulls=19, base_seed=555, metrics=("topology",)
            )
            outputs.append({k: v["pvalue"] for k, v in result["metrics"].items()})
        assert outputs[0] == outputs[1]

    def test_different_seeds_produce_different_sampled_clouds(self):
        fit = fit_null_gaussian(_isotropic(seed=33), kind="covariance_gaussian")
        first = sample_null_cloud(fit, sample_count=20, seed=1)
        second = sample_null_cloud(fit, sample_count=20, seed=2)
        assert not np.allclose(first, second)
        assert np.allclose(first, sample_null_cloud(fit, sample_count=20, seed=1))

    def test_reusing_a_fit_does_not_change_the_drawn_cloud(self):
        cloud = _isotropic(n=40, d=10, seed=31)
        manifold = _topology_manifold(cloud, seed=31)
        fit = fit_null_gaussian(cloud, kind="covariance_gaussian")
        assert np.allclose(
            manifold.null(kind="covariance_gaussian", seed=77).cloud,
            manifold.null(kind="covariance_gaussian", seed=77, fit=fit).cloud,
        )

    def test_both_nulls_use_disjoint_seed_ranges(self):
        manifold = _topology_manifold(_circle(seed=12), seed=12)
        both = ManifoldComparator().compare_both_nulls(
            manifold, n_nulls=19, base_seed=13, metrics=("topology",)
        )
        assert set(both) == {"covariance_gaussian", "isotropic_gaussian"}
        assert both["covariance_gaussian"]["base_seed"] == 13
        assert both["isotropic_gaussian"]["base_seed"] == 13 + 19


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class _FailAtCall(FastPipeline):
    """Raise on the nth reduce_pca call, otherwise behave normally."""

    def __init__(self, fail_on=(), total_failures=None):
        self.calls = 0
        self.fail_on = set(fail_on)
        self.total_failures = total_failures

    def reduce_pca(self, point_cloud, var_threshold=0.95):
        self.calls += 1
        if self.total_failures is not None or self.calls in self.fail_on:
            raise RuntimeError("simulated backend failure")
        return super().reduce_pca(point_cloud, var_threshold)


class TestFailureHandling:
    def test_one_failed_draw_does_not_abort_the_ensemble(self):
        manifold = _topology_manifold(_circle(seed=23), seed=23)
        manifold.pipeline = _FailAtCall(fail_on={5})
        with pytest.warns(UserWarning, match="failed to measure"):
            result = ManifoldComparator().compare_against_nulls(
                manifold, kind="isotropic_gaussian", n_nulls=19, base_seed=23, metrics=("topology",)
            )
        assert len(result["null_failures"]) == 1
        assert "simulated backend failure" in result["null_failures"][0]["error"]
        assert result["n_nulls_measured"] == 18
        assert result["metrics"]["H1_bottleneck"]["n_valid_nulls"] == 18
        assert result["metrics"]["H1_bottleneck"]["inference_available"] is True

    def test_failed_draws_do_not_bias_the_denominator(self):
        """The p-value must be ranked against survivors, not requested draws."""
        manifold = _topology_manifold(_circle(seed=44), seed=44)
        manifold.pipeline = _FailAtCall(fail_on={2, 3, 4, 5, 6})
        with pytest.warns(UserWarning):
            result = ManifoldComparator().compare_against_nulls(
                manifold, kind="isotropic_gaussian", n_nulls=19, base_seed=44, metrics=("topology",)
            )
        block = result["metrics"]["H1_bottleneck"]
        assert result["n_nulls_measured"] == 14
        assert block["n_valid_nulls"] == 14
        assert block["minimum_attainable_pvalue"] == pytest.approx(1.0 / 15.0)
        assert block["pvalue"] >= block["minimum_attainable_pvalue"]

    def test_all_failed_draws_produce_an_explicit_unavailable_result(self):
        manifold = _topology_manifold(_circle(seed=45), seed=45)
        manifold.pipeline = _FailAtCall(total_failures=True)
        with pytest.warns(UserWarning):
            result = ManifoldComparator().compare_against_nulls(
                manifold, kind="isotropic_gaussian", n_nulls=19, base_seed=45, metrics=("topology",)
            )
        assert result["n_nulls_measured"] == 0
        assert len(result["null_failures"]) == 19
        for name in ("H0_bottleneck", "H1_bottleneck", "H0_wasserstein", "H1_wasserstein"):
            block = result["metrics"][name]
            assert block["inference_available"] is False
            assert np.isnan(block["pvalue"])
            assert "could be measured" in block["failure_reason"]

    def test_build_null_ensemble_records_each_failure_with_its_seed(self):
        manifold = _topology_manifold(_circle(seed=46), seed=46)
        manifold.pipeline = _FailAtCall(fail_on={1, 3})
        fit = fit_null_gaussian(manifold.cloud, kind="isotropic_gaussian")
        nulls, failures = build_null_ensemble(
            manifold,
            kind="isotropic_gaussian",
            n_nulls=5,
            base_seed=900,
            covariance_estimator="ledoit_wolf",
            fit=fit,
        )
        assert len(nulls) == 3
        assert [failure["seed"] for failure in failures] == [900, 902]

    def test_too_few_nulls_warns_that_alpha_005_is_unreachable(self):
        manifold = _topology_manifold(_circle(seed=22), seed=22)
        with pytest.warns(UserWarning, match="alpha=0.05 is unreachable"):
            ManifoldComparator().compare_against_nulls(
                manifold, kind="isotropic_gaussian", n_nulls=5, base_seed=22, metrics=("topology",)
            )

    def test_n_nulls_below_the_minimum_is_a_hard_error(self):
        manifold = _topology_manifold(_circle(seed=47), seed=47)
        with pytest.raises(ValueError, match="n_nulls must be at least"):
            ManifoldComparator().compare_against_nulls(
                manifold, n_nulls=MINIMUM_VALID_NULLS - 1, metrics=("topology",)
            )

    def test_intrinsic_dimension_failure_leaves_topology_measurable(self):
        class NoIntrinsicDim(FastPipeline):
            def get_intrinsic_dim(self, point_cloud):
                raise RuntimeError("estimator unavailable")

        manifold = Manifold(
            NoIntrinsicDim(), _circle(seed=48), metrics=("intrinsic_dimension", "topology"), seed=48
        )
        assert np.isnan(manifold.intrinsic_dim)
        assert "estimator unavailable" in manifold.intrinsic_dim_error
        assert len(manifold.dgms) >= 2


# ---------------------------------------------------------------------------
# Unavailable inference and metric independence
# ---------------------------------------------------------------------------


class TestUnavailableInference:
    def test_descriptive_id_reports_no_pvalue(self):
        manifold = Manifold(
            FastPipeline(), _isotropic(seed=4), metrics=("intrinsic_dimension", "topology"), seed=4
        )
        block = ManifoldComparator().compare_against_nulls(
            manifold,
            kind="isotropic_gaussian",
            n_nulls=19,
            base_seed=20,
            metrics=("intrinsic_dimension",),
            infer_intrinsic_dimension=False,
        )["metrics"]["id_difference"]
        assert block["inference_available"] is False
        assert np.isnan(block["pvalue"])
        assert block["calibration_method"] == "descriptive_twoNN"
        assert "observed_intrinsic_dimension" in block

    def test_inferential_id_reports_a_bounded_pvalue(self):
        manifold = Manifold(
            FastPipeline(), _isotropic(seed=4), metrics=("intrinsic_dimension", "topology"), seed=4
        )
        block = ManifoldComparator().compare_against_nulls(
            manifold,
            kind="isotropic_gaussian",
            n_nulls=19,
            base_seed=20,
            metrics=("intrinsic_dimension",),
            infer_intrinsic_dimension=True,
        )["metrics"]["id_difference"]
        assert block["inference_available"] is True
        assert block["calibration_method"] == "loo_null_center_deviation"
        assert block["minimum_attainable_pvalue"] <= block["pvalue"] <= 1.0

    def test_selecting_only_topology_skips_curvature_and_id(self):
        manifold = Manifold(
            FastPipeline(), _circle(seed=3), metrics=("intrinsic_dimension", "topology"), seed=3
        )
        assert manifold.curvature_values.size == 0
        result = ManifoldComparator().compare_against_nulls(
            manifold, kind="isotropic_gaussian", n_nulls=19, base_seed=10, metrics=("topology",)
        )
        assert "H1_bottleneck" in result["metrics"]
        assert "curvature_wasserstein" not in result["metrics"]
        assert "id_difference" not in result["metrics"]

    def test_every_metric_block_reports_its_ensemble_size(self):
        manifold = _topology_manifold(_circle(seed=21), seed=21)
        result = ManifoldComparator().compare_against_nulls(
            manifold, kind="isotropic_gaussian", n_nulls=19, base_seed=21, metrics=("topology",)
        )
        for block in result["metrics"].values():
            assert block["n_valid_nulls"] == 19
            assert block["minimum_attainable_pvalue"] == pytest.approx(1.0 / 20.0)
            assert block["calibration_method"] == "exchangeable_loo_median_distance"
            assert block["metric_direction"] in {"greater", "two_sided"}


# ---------------------------------------------------------------------------
# Mutation safety
# ---------------------------------------------------------------------------


class TestMutationSafety:
    def test_comparison_leaves_the_observed_manifold_unchanged(self):
        manifold = Manifold(
            FastPipeline(),
            _isotropic(seed=24),
            metrics=("topology",),
            seed=24,
            covariance_estimator="ledoit_wolf",
        )
        before = {
            "covariance_estimator": manifold.covariance_estimator,
            "metrics": manifold.metrics,
            "eps": manifold.eps,
            "diameter": manifold.diameter,
            "m": manifold.m,
            "cloud": manifold.cloud.copy(),
            "dgms": [d.copy() for d in manifold.dgms],
            "curvature_values": manifold.curvature_values.copy(),
        }
        ManifoldComparator().compare_against_nulls(
            manifold,
            kind="covariance_gaussian",
            n_nulls=19,
            base_seed=24,
            metrics=("topology",),
            covariance_estimator="regularized_empirical",
        )
        assert manifold.covariance_estimator == before["covariance_estimator"]
        assert manifold.metrics == before["metrics"]
        assert manifold.eps == before["eps"]
        assert manifold.diameter == before["diameter"]
        assert manifold.m == before["m"]
        assert np.array_equal(manifold.cloud, before["cloud"])
        assert all(np.array_equal(a, b) for a, b in zip(manifold.dgms, before["dgms"]))
        assert np.array_equal(manifold.curvature_values, before["curvature_values"])

    def test_gaussian_is_fitted_once_per_ensemble(self, monkeypatch):
        calls = {"n": 0}
        original = null_cloud.fit_null_gaussian

        def counting_fit(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(null_cloud, "fit_null_gaussian", counting_fit)
        manifold = _topology_manifold(_isotropic(seed=25), seed=25)
        ManifoldComparator().compare_against_nulls(
            manifold, kind="covariance_gaussian", n_nulls=19, base_seed=25, metrics=("topology",)
        )
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Curvature comparison semantics
# ---------------------------------------------------------------------------


class TestCurvatureComparison:
    def test_signed_difference_reverses_with_direction(self):
        negative, positive = _curvature_double([-0.9, -0.8]), _curvature_double([0.9, 0.8])
        forward = ManifoldComparator().curvature_difference(negative, positive)
        backward = ManifoldComparator().curvature_difference(positive, negative)
        assert forward["negative_fraction_difference"] == pytest.approx(1.0)
        assert backward["negative_fraction_difference"] == pytest.approx(-1.0)
        assert forward["mean_difference"] == pytest.approx(-backward["mean_difference"])

    def test_absolute_difference_is_unchanged_when_direction_reverses(self):
        negative, positive = _curvature_double([-0.9, -0.8]), _curvature_double([0.9, 0.8])
        forward = ManifoldComparator().curvature_difference(negative, positive)
        backward = ManifoldComparator().curvature_difference(positive, negative)
        for key in ("absolute_negative_fraction_difference", "frac_negative_difference"):
            assert forward[key] == pytest.approx(backward[key]) == pytest.approx(1.0)

    def test_empty_curvature_yields_nan_rather_than_raising(self):
        result = ManifoldComparator().curvature_difference(
            _curvature_double([]), _curvature_double([0.1, 0.2])
        )
        assert all(np.isnan(value) for value in result.values())


# ---------------------------------------------------------------------------
# Scientific sanity checks
# ---------------------------------------------------------------------------


class TestScientificBehavior:
    def test_twonn_variability_is_large_relative_to_signal(self):
        ids = [FastPipeline().get_intrinsic_dim(_isotropic(n=36, d=12, seed=i)) for i in range(30)]
        assert np.std(ids) > 1.0

    def test_circle_rejects_through_h1_and_gaussian_does_not(self):
        def reject_rate(maker):
            comparator, pipeline, rejects = ManifoldComparator(), FastPipeline(), 0
            for trial in range(10):
                manifold = Manifold(
                    pipeline, maker(seed=2000 + trial), metrics=("topology",), seed=2000 + trial
                )
                result = comparator.compare_against_nulls(
                    manifold,
                    kind="isotropic_gaussian",
                    n_nulls=19,
                    base_seed=8000 + trial * 40,
                    metrics=("topology",),
                )
                block = result["metrics"]["H1_bottleneck"]
                if block["inference_available"] and block["pvalue"] <= 0.05:
                    rejects += 1
            return rejects / 10

        assert reject_rate(_circle) >= 0.5
        assert reject_rate(_isotropic) <= 0.25

    def test_inferential_id_false_positive_rate_stays_near_alpha(self):
        comparator, pipeline, rejects = ManifoldComparator(), FastPipeline(), 0
        trials = 40
        for trial in range(trials):
            manifold = Manifold(
                pipeline,
                _isotropic(n=40, d=8, seed=1000 + trial),
                metrics=("intrinsic_dimension",),
                seed=1000 + trial,
            )
            block = comparator.compare_against_nulls(
                manifold,
                kind="isotropic_gaussian",
                n_nulls=19,
                base_seed=5000 + trial * 50,
                metrics=("intrinsic_dimension",),
                infer_intrinsic_dimension=True,
            )["metrics"]["id_difference"]
            if block["inference_available"] and block["pvalue"] <= 0.05:
                rejects += 1
        # Slack for a small Monte Carlo experiment.
        assert rejects / trials <= 0.20
