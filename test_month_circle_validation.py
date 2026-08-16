import numpy as np
import pytest

from month_circle_validation import (
    MONTHS,
    circulant_r2,
    fourier_alignment,
    h1_max_persistence,
    load_month_layers,
    neighbor_gap,
    run_null_cloud_test,
    validate_karkada_null,
    validate_layer,
)


def _circle(d=16):
    theta = 2.0 * np.pi * np.arange(12) / 12
    latent = np.column_stack((np.cos(theta), np.sin(theta)))
    basis, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(d, 2)))
    return latent @ basis.T


def test_ideal_circle_has_exact_first_harmonic_and_circulant_gram():
    cloud = _circle()
    assert fourier_alignment(cloud) == pytest.approx(1.0)
    assert circulant_r2(cloud) == pytest.approx(1.0)
    assert neighbor_gap(cloud) > 0.3
    assert h1_max_persistence(cloud) > 0.2


def test_month_order_metrics_drop_after_label_permutation():
    cloud = _circle()
    permuted = cloud[np.random.default_rng(8).permutation(12)]
    assert fourier_alignment(cloud) > fourier_alignment(permuted)
    assert circulant_r2(cloud) > circulant_r2(permuted)
    assert neighbor_gap(cloud) > neighbor_gap(permuted)
    assert h1_max_persistence(cloud) == pytest.approx(h1_max_persistence(permuted))


def test_loader_restores_calendar_order(tmp_path):
    vectors = {
        f"The month of the year is {month}": np.full(3, index)
        for index, month in reversed(list(enumerate(MONTHS)))
    }
    path = tmp_path / "months.npy"
    np.save(path, {"layer_0": vectors}, allow_pickle=True)
    loaded = load_month_layers(path)["layer_0"]
    assert np.array_equal(loaded[:, 0], np.arange(12))


def test_validation_is_deterministic_and_uses_finite_null_denominator():
    first = validate_layer(_circle(), n_nulls=19, base_seed=22)
    second = validate_layer(_circle(), n_nulls=19, base_seed=22)
    assert first == second
    for result in first["inference"].values():
        assert result["n_valid_nulls"] == 19
        assert result["minimum_attainable_pvalue"] == pytest.approx(0.05)


def test_validation_refuses_too_few_nulls():
    with pytest.raises(ValueError, match="at least 19"):
        validate_layer(_circle(), n_nulls=18, base_seed=0)


def test_karkada_positive_control_uses_null_cloud_public_topology_schema():
    result = run_null_cloud_test(_circle(), n_nulls=19, base_seed=10)
    assert set(result["metrics"]) == {
        "H0_wasserstein",
        "H0_bottleneck",
        "H1_wasserstein",
        "H1_bottleneck",
    }
    assert result["null_kind"] == "low_rank_gaussian"
    assert result["n_requested"] == result["n_drawn"] == 19
    assert all(block["n_valid_nulls"] == 19 for block in result["metrics"].values())


def test_karkada_specific_null_test_is_deterministic_and_pre_registered():
    first = validate_karkada_null(_circle(), n_nulls=19, base_seed=31)
    second = validate_karkada_null(_circle(), n_nulls=19, base_seed=31)
    assert first == second
    assert first["statistic"] == "circulant_r2"
    assert first["inference"]["n_valid_nulls"] == 19
    assert first["inference"]["pvalue"] == pytest.approx(0.05)
