# hypo_free_geometry — PCA plot package

Self-contained snapshot of everything used to produce the figures in `plots/`
(gemma-2-2b, layer 6, final-token `resid_post` activations). These are the
**frozen versions** of the pipeline files as of the plot runs — the main repo's
`pipeline_draft.py` / `null_cloud.py` / `topology_metric.py` have since evolved
and are **not** interchangeable with these.

## Contents

| File | Purpose |
|---|---|
| `run_ground_truths.py` | 17 ground-truth concept sets → 34 plots (`plots/<concept>_pca{2d,3d}.png`) |
| `run_concepts.py` | days / months / chess → 6 plots |
| `main.py` | days-of-week per-layer PCA grid (`layer_grid.png`) |
| `pipeline_draft.py` | model loading + activation extraction (frozen) |
| `null_cloud.py`, `topology_metric.py` | imported by the pipeline; topology-vs-null metric (frozen) |
| `requirements.txt` | Python deps (see pinning note inside re: GraphRicciCurvature) |
| `plots/` | all 46 output figures |

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # GraphRicciCurvature may need --no-deps, see file
python run_ground_truths.py              # caches activations in acts/, resumable
python run_concepts.py
```

Notes:
- Model weights come from `unsloth/gemma-2-2b` (ungated mirror of the
  licence-gated `google/gemma-2-2b`), loaded in bfloat16 (~5 GB download,
  fits a 16 GB machine). Swap to the official repo in `pipeline_draft.py` if
  you have HF access.
- On macOS the pipeline forces single-threaded OpenMP and `fork`
  multiprocessing — see comments at the top of `pipeline_draft.py`.
- `run_ground_truths.py` caches per-concept activations in `acts/` (not
  included here; regenerated on first run).

## Reading the plots

- Plot titles show the variance captured by the displayed PCs.
- Gray edges are the *expected ground-truth structure* (cyclic neighbors,
  chains, tree/grid adjacency) drawn in representation space — if the model
  matches the ground truth, edges connect nearby points without crossings.
- Known caveats found in the results: surface-form artifacts dominate some
  sets (musical keys split natural-vs-sharp/flat; compass splits hyphenated
  vs plain; century years are tokenization outliers; key "A" suffers article
  polysemy). See team notes for the full scorecard.
