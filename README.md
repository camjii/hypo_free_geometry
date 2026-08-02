# hypo_free_geometry

## Setup

This repo is set up as an editable-installable package via `pyproject.toml`, so root-level
modules (`pipeline_draft`, `concept_geometry`, `clustering_algo`, `compare`,
`ground_truth_comp`, `null_cloud`, `topology_metric`) can be imported from anywhere in the
repo (e.g. notebooks in `ground_truth_verification/`) without `sys.path` hacks.

After creating/activating your own virtual environment, install the repo in editable mode:

```bash
uv pip install -e .
# or, without uv:
pip install -e .
```

Then any script or notebook can do e.g. `from pipeline_draft import Pipeline` directly.

If you add a new root-level module that other files need to import, add it to the
`py-modules` list in `pyproject.toml` and re-run the install command above.

## The null model

The default null is a **low-rank Gaussian surrogate fitted in the observed
empirical SVD subspace**. It preserves the observed mean and the nonzero
covariance spectrum while removing higher-order and nonlinear organization.

Given `X` of shape `(n_samples, n_features)`, the centered SVD
`X - mean = U S V^T` is truncated to its numerically nonzero rank
`r <= min(n_samples - 1, n_features)`. Each draw replaces the latent scores with
a Haar-uniform centered orthonormal frame rescaled to the observed covariance
eigenvalues, then maps it back through `V_r` and re-adds the mean. No dense
`n_features x n_features` covariance is ever formed.

The **empirical sample spectrum is preserved exactly**, not only in expectation.
Independent Gaussian latents would match it only in expectation, and at this
project's sample sizes (`n_samples` 12–100, `n_features` 768–3000) the realised
spectrum wanders far enough to shift the retained PCA dimension, which feeds
every downstream topology statistic. On the repository's 12 x 2304 layer-6
activation cloud, independent latents give a relative spectrum error of 0.52 and
an effective rank of 5.85 against an observed 9.36; the exact construction gives
0.00 and 9.36.

Preserved: sample count, ambient dimension, mean, low-rank subspace, nonzero
covariance eigenvalues, effective rank, anisotropy.
Destroyed: non-Gaussian latent organization, nonlinear topology, curved
trajectories, and any loops or clusters not implied by the linear covariance.

Method class: a matrix-valued low-rank adaptation of covariance-constrained
surrogate testing. It is a moment-constrained surrogate, not an i.i.d. Gaussian
sample.

### What a rejection does and does not mean

- It means the observed statistic is **unusual under this particular linear
  Gaussian control** — clouds sharing the observed mean and second-order
  structure and nothing else.
- It does **not** show that a recovered structure is semantically meaningful.
- With very small sample counts, both the topology estimates and the covariance
  spectrum itself remain uncertain, so treat single-dataset results as
  provisional.

Legacy null names (`noise`, `covariance_gaussian`, `isotropic_gaussian`) still
resolve to this model but emit a `DeprecationWarning`; pass
`low_rank_gaussian` explicitly.

## Null-calibration benchmark

From the repository root:

```bash
pytest -q test_null_calibration.py
python null_calibration.py --config null_calibration.yaml
```

The sole generated artifact is `outputs/null_calibration/report.html`, a
self-contained pipeline benchmark score report.