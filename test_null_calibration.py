"""Tests for the synthetic null-calibration benchmark and HTML report."""

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
            "iso_h0_pvalue": 0.5,
            "iso_h1_pvalue": 0.5,
            "iso_id_pvalue": 0.5,
            "cov_h0_pvalue": 0.5,
            "cov_h1_pvalue": 0.5,
            "cov_id_pvalue": 0.5,
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
                            iso_h0_pvalue=0.4 if dataset != "two_clusters" else 0.05,
                            iso_h1_pvalue=0.05 if dataset == "circle" else 0.5,
                            iso_id_pvalue=0.05 if dataset in {"spiked_gaussian", "isotropic_gaussian"} else 0.4,
                            cov_h0_pvalue=0.5,
                            cov_h1_pvalue=0.5,
                            cov_id_pvalue=0.4 if dataset == "spiked_gaussian" else 0.5,
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
