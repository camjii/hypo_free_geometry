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

## Gemma 7B activation extraction

`activation_extraction.py` extracts the final non-padding token from a selected
zero-based decoder layer of the base `google/gemma-7b` model. Prompt wording is
centralized in `prompt_templates.py`; the extractor includes fixed days, months,
and chess-square prompt sets and can apply any shared template to a text file of
custom values.

```bash
# Install only the model-extraction dependencies.
uv pip install -e '.[activation]'

# Inspect the built-in prompt sets and shared templates without loading a model.
python activation_extraction.py --list

# Extract the 12 month activations from decoder layer 6.
python activation_extraction.py --prompt-set months --layer 6

# Or render one value per line with another shared template.
python activation_extraction.py \
  --values-file years.txt \
  --template year \
  --layer 6
```

Each run writes an NPZ containing `activations`, `labels`, and `prompts`, plus a
JSON provenance manifest containing the immutable Hugging Face revision,
layer/token contract, resolved device, library versions, artifact checksum, and
runtime. The extractor refuses to replace an existing activation archive unless
`--overwrite` is passed.

CPU is the reproducible default. Select `--device cuda` or `--device mps` only
after validating that backend in the target environment; `--device auto` is
available for exploratory runs.

Gemma 7B is a gated, multi-gigabyte checkpoint. Accept its Hugging Face license,
run `hf auth login`, and confirm that the target machine has enough memory and
disk before starting the model-backed command.

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

## Karkada month-circle validation

Following the calendar-geometry experiment in [Karkada et al.](https://arxiv.org/abs/2602.15029),
the checked-in Gemma-2-2B activation snapshots can be tested against the
fixed-spectrum null without loading a model:

```bash
hypo311/bin/python month_circle_validation.py
```

The primary analysis is pre-specified at `layer_6` and uses contextualized
prompts of the form `The month of the year is {month}`. The paper predicts that
month similarity depends on cyclic calendar separation, so the paper-derived
primary statistic is circulant R² of the centered Gram matrix. It is ranked against
999 draws from the unchanged exact-spectrum null. The existing unlabeled H0/H1
Wasserstein and bottleneck comparison runs over 99 draws as a secondary generic
topology diagnostic; Fourier alignment, neighbor separation, bare tokens, and
all-layer curves are also secondary.

Because this activation snapshot was inspected before the statistic was made
primary, the result is paper-derived reanalysis rather than a formal
preregistration. A fresh held-out activation extraction is needed for a
confirmatory claim.
Raw metrics, a resolved provenance manifest, and figures are written beneath
`outputs/month_circle_validation/`.

## Held-out 365-date validation

`calendar_date_validation.py` is the confirmatory, hypothesis-free follow-up.
It fixes 365 non-leap-year dates, the prompt
`The calendar date is {month} {day}`, Gemma-2-2B `layer_6`, the final prompt
token, seed 260215, and one primary statistic before activation extraction.
Calendar labels and ordering are hidden from inference and restored only for the
final PCA visualization.

The primary statistic is maximum finite H1 persistence after the same per-cloud
95%-variance PCA and diameter normalization used by `null_cloud.py`. It is
ranked against 99 unchanged exact-spectrum null draws. This generic statistic
tests for a persistent loop without encoding which dates should be neighbors.
The pre-data manifest records why the full leave-one-out diagram-distance test
was replaced after a synthetic timing check exposed quadratic scaling at 365
points.

```bash
# Write the analysis contract without loading the model.
python calendar_date_validation.py prepare

# Requires Hugging Face access to google/gemma-2-2b and TransformerLens.
python calendar_date_validation.py extract

# Uses the repository's topology environment.
hypo311/bin/python calendar_date_validation.py analyze
```

This is a Karkada-inspired Gemma extension, not an exact reproduction of the
paper's 365-date word embeddings. Outputs are written beneath
`outputs/calendar_date_validation/`.

Extraction defaults to CPU because the currently installed TransformerLens
flags this machine's PyTorch 2.12.1 MPS backend as potentially silently
incorrect. Do not override `--device` to `mps` until that compatibility warning
is resolved.
