"""Behavioral tests for the null-calibration benchmark: inference, scoring, report."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / "null_calibration.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("null_calibration", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["null_calibration"] = module
    spec.loader.exec_module(module)
    return module


nc = _load_module()


def _base_row(**overrides):
    row = {column: np.nan for column in nc.RUN_COLUMNS}
    row.update(
        {
            "seed": 0,
            "sample_size": 36,
            "ambient_dim": 12,
            "noise_level": 0.0,
            "pca_mode": "variance_95",
            "pca_components": 2,
            "h0_max_persistence": 0.1,
            "h1_max_persistence": 0.0,
            "intrinsic_dimension": 1.0,
            "intrinsic_dimension_error": 0.1,
            "expected_intrinsic_dimension": 1.0,
            "h0_pvalue": 0.5,
            "h1_pvalue": 0.5,
            "id_pvalue": 0.5,
            "curvature_enabled": False,
            "failure": "",
        }
    )
    row.update(overrides)
    return row


def _fixture_runs(*, n_seeds: int = 2, include_oracle_boost: bool = True) -> pd.DataFrame:
    rows = []
    datasets = {
        "circle": {"h0": 0.28, "h1": 0.55, "id_err": 0.12, "expected": 1.0},
        "two_clusters": {"h0": 0.74, "h1": 0.01, "id_err": 0.10, "expected": 3.0},
        "line": {"h0": 0.10, "h1": 0.00, "id_err": 0.15, "expected": 1.0},
        "y_tree": {"h0": 0.15, "h1": 0.00, "id_err": 0.20, "expected": 1.0},
        "swiss_roll": {"h0": 0.30, "h1": 0.12, "id_err": 0.40, "expected": 2.0},
        "isotropic_gaussian": {"h0": 0.35, "h1": 0.05, "id_err": 0.40, "expected": 3.0},
        "spiked_gaussian": {"h0": 0.30, "h1": 0.04, "id_err": 0.15, "expected": 2.0},
    }
    for seed in range(n_seeds):
        for noise in (0.0, 0.15):
            for mode in nc.PCA_MODES:
                for dataset, spec in datasets.items():
                    h1 = spec["h1"]
                    h0 = spec["h0"]
                    comps = 2.0
                    if dataset == "circle" and mode == "parallel_analysis":
                        h1 = 0.0
                        comps = 1.0
                    if dataset == "circle" and mode == "none":
                        h1 = 0.55
                    if include_oracle_boost and dataset == "circle" and mode == "oracle":
                        # Better than production on purpose; must not raise overall score.
                        h1 = 0.90
                    if noise > 0:
                        h1 = h1 * 0.7
                        h0 = h0 * 0.8
                        id_err = spec["id_err"] + 2.0
                    else:
                        id_err = spec["id_err"]
                    rows.append(
                        _base_row(
                            dataset=dataset,
                            seed=seed,
                            noise_level=noise,
                            pca_mode=mode,
                            pca_components=comps,
                            h0_max_persistence=h0,
                            h1_max_persistence=h1,
                            intrinsic_dimension_error=id_err,
                            expected_intrinsic_dimension=spec["expected"],
                            # The Gaussian datasets must not reject their own
                            # matched null; the structured ones should.
                            h0_pvalue=0.05 if dataset == "two_clusters" else 0.4,
                            h1_pvalue=0.05 if dataset == "circle" else 0.5,
                            id_pvalue=0.4,
                            curvature_enabled=False,
                        )
                    )
    return pd.DataFrame(rows)


def test_deterministic_generation():
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    a = nc.generate_circle(40, rng_a)
    b = nc.generate_circle(40, rng_b)
    assert np.allclose(a, b)
    cloud_a, _ = nc.embed_with_noise(a, ambient_dim=16, noise_level=0.05, seed=3)
    cloud_b, _ = nc.embed_with_noise(a, ambient_dim=16, noise_level=0.05, seed=3)
    assert np.allclose(cloud_a, cloud_b)


def test_orthonormal_embedding_preserves_pairwise_distances():
    rng = np.random.default_rng(11)
    latent = nc.generate_line(30, rng)
    embedded, basis = nc.embed_with_noise(latent, ambient_dim=20, noise_level=0.0, seed=5)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-8)
    assert np.allclose(pdist(latent), pdist(embedded), atol=1e-8)


def test_score_values_within_bounds_and_exclude_unmeasured():
    runs = _fixture_runs(n_seeds=2)
    result = nc.compute_pipeline_benchmark(runs, production_pca_mode="variance_95")
    assert 0.0 <= result["overall_score"] <= 100.0
    assert result["available_points"] == pytest.approx(80.0)
    assert result["evidence_coverage"] == pytest.approx(80.0)
    assert result["components"]["null_calibration"]["available"] == 0.0
    assert result["components"]["null_calibration"]["status"] == "N/A"
    assert result["components"]["curvature"]["available"] == 0.0
    assert result["components"]["curvature"]["status"] == "N/A"
    for key in ("topology_recovery", "pca_preservation", "noise_robustness", "intrinsic_dimension"):
        comp = result["components"][key]
        assert 0.0 <= comp["earned"] <= comp["available"]


def test_oracle_pca_does_not_affect_production_score():
    runs = _fixture_runs(n_seeds=2, include_oracle_boost=True)
    boosted = nc.compute_pipeline_benchmark(runs, production_pca_mode="variance_95")
    # Zero out oracle circle H1; production score must stay identical.
    muted = runs.copy()
    mask = (muted["dataset"] == "circle") & (muted["pca_mode"] == "oracle")
    muted.loc[mask, "h1_max_persistence"] = 0.01
    control = nc.compute_pipeline_benchmark(muted, production_pca_mode="variance_95")
    assert boosted["components"]["pca_preservation"]["earned"] == pytest.approx(
        control["components"]["pca_preservation"]["earned"]
    )
    assert boosted["overall_score"] == pytest.approx(control["overall_score"])


def test_null_calibration_scored_with_enough_seeds():
    runs = _fixture_runs(n_seeds=10)
    result = nc.compute_pipeline_benchmark(runs, production_pca_mode="variance_95")
    assert result["components"]["null_calibration"]["available"] == pytest.approx(20.0)
    assert result["available_points"] == pytest.approx(100.0)
    assert result["confidence"] in {"MEDIUM", "HIGH", "LOW"}


def test_confidence_rules():
    low = nc.compute_pipeline_benchmark(_fixture_runs(n_seeds=2), production_pca_mode="variance_95")
    assert low["confidence"] == "LOW"
    medium = nc.compute_pipeline_benchmark(_fixture_runs(n_seeds=10), production_pca_mode="variance_95")
    assert medium["confidence"] in {"MEDIUM", "LOW"}  # LOW only if coverage < 80, here 100
    assert medium["confidence"] == "MEDIUM"
    high_runs = _fixture_runs(n_seeds=30)
    # Still no curvature => cannot be HIGH
    high = nc.compute_pipeline_benchmark(high_runs, production_pca_mode="variance_95")
    assert high["confidence"] == "MEDIUM"
    # Add enough curvature rows to satisfy HIGH rule.
    extra = []
    for seed in range(30):
        extra.append(
            _base_row(
                dataset="circle",
                seed=seed,
                noise_level=0.0,
                pca_mode="oracle",
                curvature_enabled=True,
                curvature_wasserstein=0.2,
                curvature_mean_difference=0.1,
                curvature_negative_fraction_difference=-0.05,
                h1_max_persistence=0.5,
                h0_max_persistence=0.2,
            )
        )
    with_curv = pd.concat([high_runs, pd.DataFrame(extra)], ignore_index=True)
    scored = nc.compute_pipeline_benchmark(with_curv, production_pca_mode="variance_95")
    assert scored["confidence"] == "HIGH"
    assert scored["components"]["curvature"]["status"] == "EXPLORATORY"
    assert scored["components"]["curvature"]["available"] == 0.0


def test_report_html_is_sole_output(tmp_path):
    runs = _fixture_runs(n_seeds=2)
    config = {
        "datasets": ["circle", "two_clusters"],
        "sample_size": 36,
        "ambient_dim": 12,
        "noise_levels": [0.0, 0.15],
        "seeds": [0, 1],
        "pca_modes": list(nc.PCA_MODES),
        "production_pca_mode": "variance_95",
        "n_nulls": 19,
        "significance_alpha": 0.05,
        "output_dir": str(tmp_path),
    }
    # Plant legacy files that must be removed.
    (tmp_path / "scorecard.csv").write_text("x", encoding="utf-8")
    (tmp_path / "raw_results.csv").write_text("x", encoding="utf-8")
    (tmp_path / "report.md").write_text("x", encoding="utf-8")
    fig = tmp_path / "figures"
    fig.mkdir()
    (fig / "old.png").write_bytes(b"x")

    report_path = nc.save_report(runs, config, tmp_path, runtime_seconds=1.23)
    assert report_path.name == "report.html"
    assert report_path.exists()
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["report.html"]

    html = report_path.read_text(encoding="utf-8")
    assert "Pipeline benchmark score" in html
    assert "Evidence coverage" in html
    assert "LOW CONFIDENCE" in html
    for name in (
        "Topology recovery",
        "PCA preservation",
        "Noise robustness",
        "Null calibration",
        "Intrinsic dimension",
        "Curvature",
    ):
        assert name in html
    assert "Main findings" in html or "3. Main findings" in html
    assert "Configuration" in html
    assert "Limitations" in html
    assert "recommendations" in html.lower() or "Limitations" in html
    assert "<style>" in html
    assert "http://" not in html and "https://" not in html


def test_empty_diagrams_and_metric_failures_are_graceful():
    empty = np.empty((0, 2), dtype=float)
    distances = nc._safe_diagram_distance(empty, empty)
    assert np.isfinite(distances["wasserstein"]) or np.isnan(distances["wasserstein"])
    m1 = type("M", (), {})()
    m2 = type("M", (), {})()
    m1.curvature_skipped = True
    m2.curvature_skipped = True
    m1.curvature_values = np.empty(0)
    m2.curvature_values = np.empty(0)
    curv = nc._safe_curvature_difference(m1, m2)
    assert np.isnan(curv["distribution_distance"])


# ---------------------------------------------------------------------------
# Empirical inference: unavailable must never read as a detection
# ---------------------------------------------------------------------------


def _distance_matrix(n_nulls: int, seed: int = 0) -> np.ndarray:
    """Symmetric (1 + n_nulls) matrix of null-like distances, NaN on the diagonal."""
    rng = np.random.default_rng(seed)
    values = rng.random((n_nulls + 1, n_nulls + 1))
    matrix = (values + values.T) / 2.0
    np.fill_diagonal(matrix, np.nan)
    return matrix


def test_nan_observed_score_is_not_significant():
    """Regression: a wholly unmeasurable metric used to report the p-value floor."""
    matrix = _distance_matrix(19, seed=0)
    matrix[0, 1:] = np.nan
    matrix[1:, 0] = np.nan
    result = nc.scores_from_matrix(matrix)
    assert np.isnan(result["pvalue"])
    assert result["inference_available"] is False
    assert "not finite" in result["failure_reason"]


def test_infinite_observed_score_is_not_significant():
    matrix = _distance_matrix(19, seed=1)
    matrix[0, 1:] = np.inf
    matrix[1:, 0] = np.inf
    result = nc.scores_from_matrix(matrix)
    assert np.isnan(result["pvalue"])
    assert result["inference_available"] is False


def test_nan_null_scores_are_excluded_from_the_denominator():
    matrix = _distance_matrix(9, seed=2)
    # Observed sits below every null, so an intact ensemble reports p = 1.0.
    matrix[0, 1:] = 0.0
    matrix[1:, 0] = 0.0
    intact = nc.scores_from_matrix(matrix)
    assert intact["n_valid_nulls"] == 9
    assert intact["pvalue"] == pytest.approx(1.0)

    damaged = matrix.copy()
    for index in range(1, 7):
        damaged[index, :] = np.nan
        damaged[:, index] = np.nan
    damaged[0, 7:] = 0.0
    damaged[7:, 0] = 0.0
    result = nc.scores_from_matrix(damaged)
    assert result["n_valid_nulls"] == 3
    assert result["pvalue"] == pytest.approx(1.0)
    assert result["minimum_attainable_pvalue"] == pytest.approx(0.25)


def test_too_few_valid_null_scores_marks_inference_unavailable():
    matrix = _distance_matrix(9, seed=3)
    for index in range(1, 9):
        matrix[index, :] = np.nan
        matrix[:, index] = np.nan
    result = nc.scores_from_matrix(matrix)
    assert result["inference_available"] is False
    assert np.isnan(result["pvalue"])


def test_minimum_attainable_pvalue_is_reported():
    result = nc.scores_from_matrix(_distance_matrix(19, seed=4))
    assert result["n_valid_nulls"] == 19
    assert result["minimum_attainable_pvalue"] == pytest.approx(1.0 / 20.0)
    assert result["pvalue"] >= result["minimum_attainable_pvalue"]


def test_empirical_pvalue_uses_the_valid_null_count():
    matrix = _distance_matrix(9, seed=5)
    matrix[0, 1:] = 10.0      # observed exceeds every null
    matrix[1:, 0] = 10.0
    for index in (1, 2, 3, 4):
        matrix[index, :] = np.nan
        matrix[:, index] = np.nan
    matrix[0, 5:] = 10.0
    matrix[5:, 0] = 10.0
    result = nc.scores_from_matrix(matrix)
    assert result["n_valid_nulls"] == 5
    assert result["pvalue"] == pytest.approx(1.0 / 6.0)


# ---------------------------------------------------------------------------
# Rejection annotation
# ---------------------------------------------------------------------------


def test_rejection_annotation_ignores_unavailable_pvalues():
    runs = pd.DataFrame(
        {
            "h0_pvalue": [0.01, np.nan, 0.90],
            "h1_pvalue": [np.nan, 0.02, 0.50],
            "id_pvalue": [np.nan, np.nan, np.nan],
        }
    )
    annotated = nc.annotate_rejections(runs, alpha=0.05)
    assert annotated["reject_h0"].tolist() == [True, False, False]
    assert annotated["reject_h1"].tolist() == [False, True, False]
    # A metric that was never measurable contributes no detections at all,
    # and is flagged as unavailable rather than as a passed test.
    assert annotated["reject_id"].tolist() == [False, False, False]
    assert annotated["id_pvalue_available"].tolist() == [False, False, False]
    assert annotated["h0_pvalue_available"].tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# Null ensembles: failure handling and fit reuse
# ---------------------------------------------------------------------------


class _StubPipeline:
    """In-process measurement backend; no model download, no curvature."""

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.calls = 0

    def get_intrinsic_dim(self, cloud):
        return float(np.asarray(cloud).shape[1])

    def reduce_pca(self, cloud, var_threshold=0.95):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError("simulated backend failure")
        x = np.asarray(cloud, dtype=float)
        return x[:, :3]

    def create_persistence_diagram(self, projected):
        from ripser import ripser

        return ripser(np.asarray(projected, dtype=float), maxdim=1)

    def create_epsilon_graph(self, projected, eps):
        import networkx as nx
        from sklearn.neighbors import radius_neighbors_graph

        return nx.Graph(
            radius_neighbors_graph(
                np.asarray(projected, float), radius=float(eps), mode="distance",
                include_self=False,
            )
        )

    def compute_ollivier_ricci(self, graph):
        return {"graph": graph, "mean_curvature": float("nan"), "raw_values": []}


def _stub_manifold(pipeline=None, seed=0, n=24, d=6):
    cloud = np.random.default_rng(seed).normal(size=(n, d))
    return nc.CalibrationManifold(
        pipeline or _StubPipeline(), cloud, seed=seed, enable_curvature=False
    )


def test_one_failed_null_draw_does_not_abort_the_ensemble():
    manifold = _stub_manifold()
    manifold.pipeline.fail_on = {manifold.pipeline.calls + 3}
    result = nc.robust_null_comparison(
        manifold, n_nulls=8, base_seed=5, include_curvature=False
    )
    assert len(result["failures"]) == 1
    assert result["n_drawn"] == 7
    assert result["metrics"]["H1_bottleneck"]["n_valid_nulls"] == 7


def test_all_failed_draws_produce_an_explicit_unavailable_result():
    manifold = _stub_manifold(seed=1)
    manifold.pipeline.fail_on = set(range(manifold.pipeline.calls + 1, 100))
    result = nc.robust_null_comparison(
        manifold, n_nulls=8, base_seed=6, include_curvature=False
    )
    assert result["n_drawn"] == 0
    for block in result["metrics"].values():
        assert block["inference_available"] is False
        assert np.isnan(block["pvalue"])
        assert "could be measured" in block["failure_reason"]


def test_gaussian_is_fitted_once_per_ensemble(monkeypatch):
    calls = {"n": 0}
    original = nc.fit_low_rank_gaussian

    def counting_fit(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(nc, "fit_low_rank_gaussian", counting_fit)
    nc.robust_null_comparison(
        _stub_manifold(seed=2), n_nulls=6, base_seed=7, include_curvature=False,
    )
    assert calls["n"] == 1


def test_fixed_base_seed_gives_identical_results():
    def run():
        return nc.robust_null_comparison(
            _stub_manifold(seed=3), n_nulls=6, base_seed=11, include_curvature=False,
        )["metrics"]["H0_bottleneck"]["pvalue"]

    assert run() == run()


def test_different_base_seeds_change_the_sampled_nulls():
    def scores(base_seed):
        return nc.robust_null_comparison(
            _stub_manifold(seed=3), n_nulls=6, base_seed=base_seed, include_curvature=False,
        )["metrics"]["H0_bottleneck"]["null_values"]

    assert scores(11) != scores(4242)


def test_observed_manifold_is_not_mutated():
    manifold = _stub_manifold(seed=4)
    before = {
        "enable_curvature": manifold.enable_curvature,
        "eps": manifold.eps,
        "diameter": manifold.diameter,
        "intrinsic_dim": manifold.intrinsic_dim,
        "cloud": manifold.cloud.copy(),
    }
    nc.robust_null_comparison(
        manifold, n_nulls=6, base_seed=12, include_curvature=False
    )
    assert manifold.enable_curvature == before["enable_curvature"]
    assert manifold.eps == before["eps"]
    assert manifold.diameter == before["diameter"]
    assert manifold.intrinsic_dim == before["intrinsic_dim"]
    assert np.array_equal(manifold.cloud, before["cloud"])


def test_null_diagnostics_are_included_once_and_confirm_the_match():
    result = nc.robust_null_comparison(
        _stub_manifold(seed=5), n_nulls=6, base_seed=13, include_curvature=False,
    )
    diagnostics = result["null_diagnostics"]
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


@pytest.mark.parametrize("legacy", ["covariance_gaussian", "isotropic_gaussian", "noise"])
def test_legacy_null_names_still_work_with_a_deprecation_warning(legacy):
    manifold = _stub_manifold(seed=6)
    with pytest.deprecated_call():
        null = manifold.null(kind=legacy, seed=3)
    assert null.cloud.shape == manifold.cloud.shape
    assert null.enable_curvature is manifold.enable_curvature


def test_canonical_null_name_needs_no_warning():
    manifold = _stub_manifold(seed=6)
    null = manifold.null(kind=nc.NULL_KIND, seed=3)
    assert null.cloud.shape == manifold.cloud.shape


def test_invalid_input_shape_raises_a_concise_error():
    with pytest.raises(ValueError, match="two-dimensional"):
        nc.CalibrationManifold(_StubPipeline(), np.arange(10.0))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_activations_are_rejected(bad):
    cloud = np.random.default_rng(0).normal(size=(20, 6))
    cloud[1, 1] = bad
    with pytest.raises(ValueError, match="only finite values"):
        nc.CalibrationManifold(_StubPipeline(), cloud)


def test_zero_epsilon_is_rejected():
    duplicated = np.vstack([np.ones((19, 6)), np.full((1, 6), 2.0)])
    with pytest.raises(ValueError, match="non-positive radius"):
        nc.CalibrationManifold(_StubPipeline(), duplicated)

