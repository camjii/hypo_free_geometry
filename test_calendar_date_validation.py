import numpy as np
import pytest

from calendar_date_validation import (
    PRIMARY_METRIC,
    PROMPT_TEMPLATE,
    build_parser,
    calendar_dates,
    calendar_prompts,
    load_activations,
    max_h1_persistence,
    run_topology_validation,
)


def _circle(n=32, d=8):
    theta = 2.0 * np.pi * np.arange(n) / n
    latent = np.column_stack((np.cos(theta), np.sin(theta)))
    return np.column_stack((latent, np.zeros((n, d - 2))))


def test_calendar_contract_has_exactly_one_non_leap_year():
    dates = calendar_dates()
    assert len(dates) == len(set(dates)) == 365
    assert dates[0] == "January 1"
    assert dates[-1] == "December 31"
    assert "February 29" not in dates


def test_prompt_contract_is_locked():
    assert PROMPT_TEMPLATE == "The calendar date is {month} {day}"
    prompts = calendar_prompts(("January 1", "December 31"))
    assert prompts == (
        "The calendar date is January 1",
        "The calendar date is December 31",
    )


def test_extraction_defaults_to_cpu_for_current_environment():
    assert build_parser().parse_args(["prepare"]).device == "cpu"


def test_loader_rejects_noncanonical_or_incomplete_date_cloud(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, activations=np.ones((364, 3)), dates=calendar_dates()[:-1])
    with pytest.raises(ValueError, match=r"\[365, features\]"):
        load_activations(path)


def test_label_free_circle_control_rejects_exact_spectrum_null():
    result = run_topology_validation(_circle(), n_nulls=19, base_seed=31)
    assert result["primary_metric"] == PRIMARY_METRIC
    assert result["primary"]["inference_available"]
    assert result["primary"]["pvalue"] == pytest.approx(0.05)
    assert result["hypothesis_supported"]
    assert result["n_drawn"] == 19


def test_max_h1_persistence_detects_an_ideal_loop():
    assert max_h1_persistence(_circle()) > 0.2


def test_inference_is_invariant_to_input_row_order():
    cloud = _circle()
    permutation = np.random.default_rng(8).permutation(len(cloud))
    first = run_topology_validation(cloud, n_nulls=19, base_seed=17)
    second = run_topology_validation(cloud[permutation], n_nulls=19, base_seed=17)
    assert second["primary"]["observed"] == pytest.approx(
        first["primary"]["observed"], rel=1e-10, abs=1e-12
    )
    assert second["primary"]["pvalue"] == first["primary"]["pvalue"]
