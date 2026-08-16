"""Held-out, label-free topology validation on 365 calendar dates.

This is a Karkada-inspired extension, not an exact reproduction of the paper's
365-date word-embedding experiment.  Calendar order is never passed to the
inferential code; it is used only after inference to draw the PCA trajectory.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "main"
PRIMARY_LAYER = 6
PROMPT_TEMPLATE = "The calendar date is {month} {day}"
BASE_SEED = 260215
DEFAULT_OUTPUT = Path("outputs/calendar_date_validation")
DEFAULT_ACTIVATIONS = DEFAULT_OUTPUT / "calendar_dates_layer6.npz"
PRIMARY_METRIC = "max_h1_persistence"


class TopologyPipeline:
    """Minimal adapter for null_cloud's production topology path."""

    def reduce_pca(self, cloud: np.ndarray, var_threshold: float = 0.95) -> np.ndarray:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(cloud.shape))
        projected = pca.fit_transform(cloud)
        count = int(
            np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_threshold) + 1
        )
        return projected[:, :count]

    def create_persistence_diagram(self, projected: np.ndarray) -> dict[str, Any]:
        from ripser import ripser

        return ripser(projected, maxdim=1)


def calendar_dates() -> tuple[str, ...]:
    """All dates in a fixed non-leap year, in calendar order."""
    return tuple(
        f"{calendar.month_name[month]} {day}"
        for month in range(1, 13)
        for day in range(1, calendar.monthrange(2001, month)[1] + 1)
    )


def calendar_prompts(
    dates: Sequence[str] | None = None, template: str = PROMPT_TEMPLATE
) -> tuple[str, ...]:
    return tuple(
        template.format(month=date.rsplit(" ", 1)[0], day=date.rsplit(" ", 1)[1])
        for date in (calendar_dates() if dates is None else dates)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")


def design_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """The analysis contract, written before model loading or inference."""
    return {
        "experiment": {
            "id": args.run_id,
            "question": (
                "Can the unchanged null-cloud topology test detect non-null geometry "
                "in an unordered cloud of 365 calendar-date activations?"
            ),
            "falsifiable_hypothesis": (
                "At fixed layer 6, maximum finite H1 persistence rejects the "
                "exact-spectrum null at alpha=0.05."
            ),
            "source_revision": _git_revision(),
            "commands": {
                "extract": "python calendar_date_validation.py extract",
                "analyze": "python calendar_date_validation.py analyze",
            },
        },
        "model": {
            "identifier": args.model,
            "revision": args.model_revision,
            "framework": "TransformerLens",
            "precision": args.dtype,
            "device": args.device,
            "activation_extraction": {
                "hook": f"blocks.{args.layer}.hook_resid_post",
                "layer": args.layer,
                "token_position": "final non-padding prompt token",
                "aggregation": "none",
                "expected_shape": [365, 2304],
            },
        },
        "inputs": {
            "dates": "365 dates of non-leap year 2001, January 1 through December 31",
            "prompt_template": args.prompt_template,
            "labels_available_to_inference": False,
            "paper": "https://arxiv.org/abs/2602.15029",
        },
        "method": {
            "primary_metric": PRIMARY_METRIC,
            "secondary_metrics": [],
            "alpha": 0.05,
            "n_nulls": args.n_nulls,
            "seed": args.seed,
            "null": "unchanged exact-spectrum low_rank_gaussian from null_cloud.py",
            "pca": "refit independently per cloud to 95% explained variance",
            "row_order": "deterministically shuffled before inference",
            "calendar_order_use": "visualization only, after metrics are saved",
            "success_criterion": (
                "primary inference is available and max_h1_persistence p <= 0.05"
            ),
            "protocol_amendment": (
                "Before extracting or inspecting 365-date activations, the primary "
                "metric was changed from the full-diagram leave-one-out H1 "
                "Wasserstein comparison to maximum finite H1 persistence. A "
                "synthetic 365-point timing run showed that the former did not "
                "finish 19 nulls in 120 seconds because its null-to-null distance "
                "matrix scales quadratically. The replacement is label-free, uses "
                "the same PCA and persistence pipeline, and scales linearly."
            ),
        },
        "claims": {
            "non_claims": [
                "This is not an exact reproduction of the paper's tokenized word embeddings.",
                "Rejection does not identify a circle by name or establish semantic meaning.",
                "Secondary metrics are descriptive and not alternate routes to success.",
            ]
        },
    }


def extract_activations(args: argparse.Namespace) -> Path:
    """Extract one fixed residual-stream layer for the locked prompt set."""
    import torch
    from huggingface_hub import HfApi
    from transformer_lens import HookedTransformer

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "design_manifest.json", design_manifest(args))
    destination = Path(args.activations)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite held-out activations at {destination}"
        )
    try:
        resolved_revision = (
            HfApi().model_info(args.model, revision=args.model_revision).sha
        )
    except Exception as exc:
        raise RuntimeError(
            f"cannot access {args.model}; accept its Hugging Face license and run "
            "`hf auth login` before extraction"
        ) from exc
    if not resolved_revision:
        raise RuntimeError(f"could not resolve an immutable revision for {args.model}")

    try:
        model = HookedTransformer.from_pretrained(
            args.model,
            revision=resolved_revision,
            device=args.device,
            dtype=args.dtype,
            first_n_layers=args.layer + 1,
        )
    except OSError as exc:
        raise RuntimeError(
            f"cannot download {args.model}; accept its Hugging Face license and "
            "run `hf auth login` before extraction"
        ) from exc
    dates = calendar_dates()
    prompts = calendar_prompts(dates, args.prompt_template)
    hook = f"blocks.{args.layer}.hook_resid_post"
    batches: list[np.ndarray] = []
    for start in range(0, len(prompts), args.batch_size):
        batch = list(prompts[start : start + args.batch_size])
        tokens = model.to_tokens(batch, padding_side="right")
        pad_token_id = model.tokenizer.pad_token_id
        if pad_token_id is None:
            raise RuntimeError("tokenizer must define pad_token_id")
        final_positions = (tokens != pad_token_id).sum(dim=1) - 1
        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=[hook])
        activation = cache[hook]
        row = torch.arange(len(batch), device=activation.device)
        batches.append(activation[row, final_positions].float().cpu().numpy())
        del cache, activation

    values = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if values.shape[0] != 365 or not np.isfinite(values).all():
        raise RuntimeError(f"invalid activation matrix: {values.shape}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            activations=values,
            dates=np.asarray(dates),
            prompts=np.asarray(prompts),
            layer=np.asarray(args.layer),
            model=np.asarray(args.model),
            model_revision=np.asarray(resolved_revision),
        )
    os.replace(temporary, destination)
    _write_json(
        output / "extraction.json",
        {
            "path": str(destination),
            "sha256": _sha256(destination),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "model_revision": resolved_revision,
        },
    )
    return destination


def load_activations(path: str | Path) -> tuple[np.ndarray, tuple[str, ...]]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        cloud = np.asarray(archive["activations"], dtype=float)
        dates = tuple(str(value) for value in archive["dates"])
    if cloud.ndim != 2 or cloud.shape[0] != 365:
        raise ValueError(
            f"activations must have shape [365, features]; got {cloud.shape}"
        )
    if dates != calendar_dates():
        raise ValueError("date labels must be the canonical 365-day calendar")
    if not np.isfinite(cloud).all():
        raise ValueError("activations contain non-finite values")
    return cloud, dates


def max_h1_persistence(cloud: np.ndarray, variance_threshold: float = 0.95) -> float:
    """Longest finite H1 bar after per-cloud PCA and diameter normalization."""
    from scipy.spatial.distance import pdist

    projected = TopologyPipeline().reduce_pca(
        np.asarray(cloud, dtype=float), variance_threshold
    )
    distances = pdist(projected)
    diameter = float(distances.max()) if distances.size else float("nan")
    if not np.isfinite(diameter) or diameter <= 0.0:
        return float("nan")
    diagrams = TopologyPipeline().create_persistence_diagram(projected / diameter)[
        "dgms"
    ]
    if len(diagrams) < 2 or not len(diagrams[1]):
        return 0.0
    bars = np.asarray(diagrams[1], dtype=float)
    persistence = bars[:, 1] - bars[:, 0]
    persistence = persistence[np.isfinite(persistence)]
    return float(persistence.max()) if persistence.size else 0.0


def run_topology_validation(
    cloud: np.ndarray, *, n_nulls: int, base_seed: int
) -> dict[str, Any]:
    """Run label-free inference after hiding the source row order."""
    from null_cloud import (
        empirical_pvalue,
        fit_low_rank_gaussian,
        sample_low_rank_gaussian,
    )

    if n_nulls < 19:
        raise ValueError("n_nulls must be at least 19")
    order = np.random.default_rng(base_seed + 10_000_000).permutation(len(cloud))
    unordered = np.asarray(cloud, dtype=float)[order]
    fit = fit_low_rank_gaussian(unordered)
    observed = max_h1_persistence(unordered)
    null_values = [
        max_h1_persistence(sample_low_rank_gaussian(fit, seed=base_seed + index))
        for index in range(n_nulls)
    ]
    primary = empirical_pvalue(observed, null_values)
    return {
        "primary_metric": PRIMARY_METRIC,
        "primary": primary,
        "hypothesis_supported": bool(
            primary["inference_available"] and primary["pvalue"] <= 0.05
        ),
        "null_kind": "low_rank_gaussian",
        "n_requested": n_nulls,
        "n_drawn": len(null_values),
        "null_rank": int(fit["rank"]),
    }


def render_figure(
    cloud: np.ndarray, dates: Sequence[str], result: dict[str, Any], output: Path
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hypo-free-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    coordinates = PCA(n_components=3).fit_transform(cloud)
    colors = plt.colormaps["hsv"](np.arange(len(cloud)) / len(cloud))
    primary = result["primary"]
    fig = plt.figure(figsize=(12, 5.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*coordinates.T, color="#8a8a8a", linewidth=0.7, alpha=0.7)
    ax.scatter(*coordinates.T, c=colors, s=8)
    for index in (0, 90, 181, 273, 364):
        ax.text(*coordinates[index], dates[index], fontsize=6)
    ax.set_title("Calendar order revealed after inference")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    ax = fig.add_subplot(1, 2, 2)
    ax.hist(primary["null_values"], bins=20, color="#b8c4d8", edgecolor="white")
    ax.axvline(primary["observed"], color="#d62728", linewidth=2, label="observed")
    ax.set_title(f"Label-free {PRIMARY_METRIC.replace('_', ' ')}")
    ax.set_xlabel("Maximum finite H1 persistence")
    ax.set_ylabel("Null objects")
    ax.legend(frameon=False)
    ax.text(
        0.97,
        0.95,
        f"p={primary['pvalue']:.3g}\n{primary['n_valid_nulls']} valid nulls",
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    fig.suptitle("365-date held-out null-cloud validation")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "design_manifest.json"
    if not manifest_path.exists():
        _write_json(manifest_path, design_manifest(args))
    cloud, dates = load_activations(args.activations)
    result = run_topology_validation(cloud, n_nulls=args.n_nulls, base_seed=args.seed)
    _write_json(output / "metrics.json", result)
    render_figure(cloud, dates, result, output / "validation.png")
    _write_json(
        output / "run_metadata.json",
        {
            "activation_sha256": _sha256(Path(args.activations)),
            "elapsed_seconds": time.time() - started,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "os": platform.platform(),
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "extract", "analyze"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument("--prompt-template", default=PROMPT_TEMPLATE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-nulls", type=int, default=99)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--run-id", default="karkada-inspired-365-dates-layer6-v1")
    parser.add_argument("--activations", default=str(DEFAULT_ACTIVATIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        destination = output / "design_manifest.json"
        _write_json(destination, design_manifest(args))
        print(destination)
    if args.command == "extract":
        extract_activations(args)
    if args.command == "analyze":
        result = analyze(args)
        print(
            json.dumps(
                {
                    "primary_metric": PRIMARY_METRIC,
                    "pvalue": result["primary"]["pvalue"],
                    "hypothesis_supported": result["hypothesis_supported"],
                    "output": args.output,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
