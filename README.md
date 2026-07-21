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