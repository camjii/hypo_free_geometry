"""Extract one activation manifold per concept set under all five templates.

Reads ``concept_sets.json``, renders every value under the five universal
templates in :mod:`templates`, and writes one NPZ per set holding an
``(n_values, n_templates, hidden_size)`` array.  Flattening the first two axes
gives the ``n x 5`` point cloud the preprocessing stage consumes; keeping them
separate lets that stage estimate a per-template offset without re-deriving
which point came from which template.

Activations are read at the concept itself rather than at the final sequence
token, because four of the five templates place the value mid-sentence or end
in punctuation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from activation_extraction import (
    _git_revision,
    _package_version,
    _resolve_model_revision,
    _sha256,
    _write_json_atomic,
    hidden_state_index,
    value_token_indices,
)
from activation_extraction import _load_model
from templates import TEMPLATE_ORDER, UNIVERSAL_TEMPLATES

DEFAULT_MODEL_ID = "google/gemma-2-2b"
DEFAULT_MODEL_REVISION = "main"
DEFAULT_LAYER = 6
DEFAULT_SEED = 260215
DEFAULT_BATCH_SIZE = 16
DEFAULT_DTYPE = "float32"
DEFAULT_DEVICE = "auto"
CONCEPT_SETS = Path("concept_sets.json")
DEFAULT_OUTPUT_ROOT = Path("manifolds")


def load_concept_sets(path: Path) -> dict[str, dict[str, Any]]:
    """Read the consolidated concept-set file, rejecting a stale schema."""
    document = json.loads(Path(path).read_text())
    version = document.get("schema_version")
    if version != 1:
        raise ValueError(f"{path}: unsupported schema_version {version!r}")
    sets = document["sets"]
    for name, entry in sets.items():
        values = entry["values"]
        if len(values) != len(set(values)):
            raise ValueError(
                f"{name}: repeated values render to identical prompts; rebuild "
                "concept_sets.json with scripts/build_concept_sets.py"
            )
        if entry["n_values"] != len(values):
            raise ValueError(f"{name}: n_values disagrees with the value list")
    return sets


def render_set(values: Sequence[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """Render every (value, template) pair in row-major value-then-template order."""
    prompts: list[str] = []
    spans: list[tuple[int, int]] = []
    for value in values:
        for template in UNIVERSAL_TEMPLATES:
            prompts.append(template.render(value))
            spans.append(template.value_span(value))
    return prompts, spans


def extract_concept_activations(
    *,
    prompts: Sequence[str],
    spans: Sequence[tuple[int, int]],
    torch: Any,
    tokenizer: Any,
    model: Any,
    layer: int,
    batch_size: int,
) -> np.ndarray:
    """Return one residual vector per prompt, read at the concept's last token."""
    if len(prompts) != len(spans):
        raise ValueError("one value span is required per prompt")
    device = str(next(model.parameters()).device)
    state_index = hidden_state_index(layer, int(model.config.num_hidden_layers))

    collected: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        batch_spans = list(spans[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        positions = value_token_indices(
            encoded["offset_mapping"].cpu().numpy(),
            encoded["special_tokens_mask"].cpu().numpy(),
            batch_spans,
        )
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.inference_mode():
            output = model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        if output.hidden_states is None:
            raise RuntimeError("model did not return hidden states")
        hidden = output.hidden_states[state_index]
        rows = torch.arange(len(batch), device=hidden.device)
        columns = torch.as_tensor(positions, device=hidden.device)
        collected.append(hidden[rows, columns].detach().float().cpu().numpy())
        del output, hidden

    activations = np.concatenate(collected, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(activations).all():
        raise RuntimeError("activation matrix contains non-finite values")
    return activations


def save_manifold(
    *,
    path: Path,
    activations: np.ndarray,
    values: Sequence[str],
    prompts: Sequence[str],
    entry: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """Write one manifold NPZ plus a JSON sidecar recording its provenance."""
    n_values, n_templates = len(values), len(TEMPLATE_ORDER)
    expected = (n_values, n_templates, activations.shape[-1])
    cloud = activations.reshape(expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        activations=cloud,
        labels=np.asarray(values, dtype=np.str_),
        templates=np.asarray(TEMPLATE_ORDER, dtype=np.str_),
        prompts=np.asarray(prompts, dtype=np.str_).reshape(n_values, n_templates),
        family=np.asarray(entry["family"] or "", dtype=np.str_),
        layer=np.asarray(metadata["layer"], dtype=np.int64),
        model=np.asarray(metadata["model"], dtype=np.str_),
        model_revision=np.asarray(metadata["model_revision"], dtype=np.str_),
    )
    sidecar = path.with_suffix(".json")
    _write_json_atomic(
        sidecar,
        {
            **metadata,
            "concept_set": path.stem,
            "family": entry["family"],
            "expected_topology": entry["expected_topology"],
            "value_source": entry["source"],
            "n_values": n_values,
            "n_templates": n_templates,
            "templates": list(TEMPLATE_ORDER),
            "shape": list(expected),
            "npz_sha256": _sha256(path),
        },
    )
    return sidecar


def dry_run(sets: dict[str, dict[str, Any]], names: Sequence[str]) -> int:
    """Render and validate every prompt without loading the model."""
    total = 0
    for name in names:
        values = sets[name]["values"]
        prompts, spans = render_set(values)
        for index, value in enumerate(values):
            for offset in range(len(TEMPLATE_ORDER)):
                row = index * len(TEMPLATE_ORDER) + offset
                start, end = spans[row]
                if prompts[row][start:end] != str(value).strip():
                    raise RuntimeError(
                        f"{name}: span mismatch for {value!r} under "
                        f"{TEMPLATE_ORDER[offset]!r}"
                    )
        total += len(prompts)
        print(f"  {name:<28} {len(values):>3} values -> {len(prompts):>4} prompts")
    print(f"\n{len(names)} sets, {total} prompts, all value spans verified")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept-sets", type=Path, default=CONCEPT_SETS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sets",
        nargs="+",
        help="extract only these concept sets (default: every set)",
    )
    parser.add_argument(
        "--skip-unassigned",
        action="store_true",
        help="skip sets whose family label has not been decided",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render and validate prompts without loading the model",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sets = load_concept_sets(args.concept_sets)

    names = list(args.sets) if args.sets else list(sets)
    unknown = [name for name in names if name not in sets]
    if unknown:
        raise SystemExit(f"unknown concept set(s): {', '.join(sorted(unknown))}")
    if args.skip_unassigned:
        skipped = [name for name in names if sets[name]["family"] is None]
        names = [name for name in names if sets[name]["family"] is not None]
        if skipped:
            print(f"skipping {len(skipped)} set(s) without a family label")

    if args.dry_run:
        print(f"dry run over {len(names)} concept set(s):")
        return dry_run(sets, names)

    revision = _resolve_model_revision(args.model, args.revision)
    torch, tokenizer, model = _load_model(
        model_id=args.model,
        revision=revision,
        device=args.device,
        dtype_name=args.dtype,
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_layers = int(model.config.num_hidden_layers)
    if args.layer == DEFAULT_LAYER and n_layers != 28:
        print(
            f"note: layer {args.layer} was chosen for a 28-layer model; "
            f"{args.model} has {n_layers} layers",
            file=sys.stderr,
        )
    metadata = {
        "model": args.model,
        "model_revision": revision,
        "layer": args.layer,
        "n_layers": n_layers,
        "hidden_size": int(model.config.hidden_size),
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "git_revision": _git_revision(),
        "numpy": _package_version("numpy"),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
    }

    for name in names:
        entry = sets[name]
        prompts, spans = render_set(entry["values"])
        activations = extract_concept_activations(
            prompts=prompts,
            spans=spans,
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            layer=args.layer,
            batch_size=args.batch_size,
        )
        path = args.output_root / f"{name}.npz"
        save_manifold(
            path=path,
            activations=activations,
            values=entry["values"],
            prompts=prompts,
            entry=entry,
            metadata=metadata,
        )
        print(
            f"  {name:<28} {entry['n_values']:>3} x {len(TEMPLATE_ORDER)} "
            f"x {metadata['hidden_size']} -> {path}"
        )

    print(f"\nwrote {len(names)} manifold(s) to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
