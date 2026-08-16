"""Extract final-token residual activations from the base Gemma 7B model.

The output is an NPZ archive with an ``activations`` matrix shaped
``(n_prompts, hidden_size)`` and aligned ``labels`` and ``prompts`` arrays.  It
therefore matches the shared activation-dataset contract used by the manifold
recovery experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from prompt_templates import (
    PROMPT_SETS,
    PROMPT_TEMPLATES,
    PromptSet,
    get_prompt_set,
    get_prompt_template,
    render_prompts,
)


DEFAULT_MODEL_ID = "google/gemma-7b"
DEFAULT_MODEL_REVISION = "main"
DEFAULT_LAYER = 6
DEFAULT_SEED = 260215
DEFAULT_PROMPT_SET = "months"
DEFAULT_DEVICE = "cpu"
DEFAULT_OUTPUT_ROOT = Path("outputs/activations")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_prompt_values(path: str | Path) -> tuple[str, ...]:
    """Load ordered, unique prompt values from a UTF-8 text file."""
    source = Path(path)
    values = tuple(
        line.strip()
        for line in source.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not values:
        raise ValueError(f"prompt-values file is empty: {source}")
    if len(values) != len(set(values)):
        raise ValueError(f"prompt-values file contains duplicate labels: {source}")
    return values


def final_token_indices(attention_mask: np.ndarray) -> np.ndarray:
    """Return final non-padding indices for a right-padded attention mask."""
    mask = np.asarray(attention_mask)
    if mask.ndim != 2:
        raise ValueError(f"attention mask must be 2D; got shape {mask.shape}")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("attention mask must contain only 0 and 1")
    lengths = mask.sum(axis=1, dtype=np.int64)
    if np.any(lengths == 0):
        raise ValueError("each prompt must contain at least one non-padding token")
    expected = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    if not np.array_equal(mask.astype(bool), expected):
        raise ValueError("attention mask must use right padding")
    return lengths - 1


def value_token_indices(
    offset_mapping: np.ndarray,
    special_tokens_mask: np.ndarray,
    spans: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Return the last token index overlapping each prompt's value span.

    Templates may end in punctuation or place the value mid-sentence, so the
    final sequence token is not always the concept.  This reads the activation
    at the concept itself, which coincides with the final non-padding token
    whenever the value ends the prompt.
    """
    offsets = np.asarray(offset_mapping)
    special = np.asarray(special_tokens_mask)
    if offsets.ndim != 3 or offsets.shape[2] != 2:
        raise ValueError(f"offset mapping must be (batch, seq, 2); got {offsets.shape}")
    if special.shape != offsets.shape[:2]:
        raise ValueError("special-token mask must match the offset mapping")
    if len(spans) != offsets.shape[0]:
        raise ValueError("one value span is required per prompt")

    indices = np.empty(len(spans), dtype=np.int64)
    for row, (start, end) in enumerate(spans):
        if start >= end:
            raise ValueError(f"value span must be non-empty; got ({start}, {end})")
        token_start = offsets[row, :, 0]
        token_end = offsets[row, :, 1]
        overlaps = (
            (special[row] == 0)
            & (token_end > token_start)
            & (token_start < end)
            & (token_end > start)
        )
        matched = np.flatnonzero(overlaps)
        if not matched.size:
            raise ValueError(f"no token overlaps value span ({start}, {end})")
        indices[row] = matched[-1]
    return indices


def hidden_state_index(layer: int, n_layers: int) -> int:
    """Map a zero-based decoder layer to Hugging Face's hidden-state tuple."""
    if layer < 0 or layer >= n_layers:
        raise ValueError(
            f"layer must be in [0, {n_layers - 1}] for this model; got {layer}"
        )
    # hidden_states[0] is the embedding output.  Entry layer + 1 is the
    # residual stream after decoder block ``layer``.
    return layer + 1


def _resolve_prompt_input(
    *,
    prompt_set_name: str | None,
    template_name: str | None,
    values_file: str | Path | None,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    if values_file is None:
        if template_name is not None:
            raise ValueError("--template requires --values-file")
        prompt_set: PromptSet = get_prompt_set(prompt_set_name or DEFAULT_PROMPT_SET)
        values = prompt_set.values
        prompts = prompt_set.prompts
        return prompt_set.name, prompt_set.template_name, values, prompts

    if prompt_set_name is not None:
        raise ValueError("choose either --prompt-set or --values-file, not both")
    if template_name is None:
        raise ValueError("--values-file requires --template")
    values = load_prompt_values(values_file)
    prompts = render_prompts(template_name, values)
    return Path(values_file).stem, template_name, values, prompts


def _slug(value: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "-" for character in value
    )
    return "-".join(part for part in normalized.split("-") if part)


def default_output_path(model_id: str, prompt_set_name: str, layer: int) -> Path:
    """Build a stable portable output path for one activation cloud."""
    return (
        DEFAULT_OUTPUT_ROOT
        / _slug(model_id.split("/")[-1])
        / f"{_slug(prompt_set_name)}_layer_{layer}.npz"
    )


def resolve_device(torch_module: Any, requested: str) -> str:
    """Resolve ``auto`` without requiring Accelerate or a device map."""
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(torch_module: Any, dtype_name: str) -> Any:
    """Translate the CLI dtype name to a value accepted by Transformers."""
    if dtype_name == "auto":
        return "auto"
    return {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }[dtype_name]


def transformers_dtype_keyword(version_string: str) -> str:
    """Use the loader keyword supported by the installed Transformers major."""
    try:
        major = int(version_string.split(".", 1)[0])
    except ValueError as error:
        raise ValueError(
            f"cannot parse Transformers version {version_string!r}"
        ) from error
    return "dtype" if major >= 5 else "torch_dtype"


def _resolve_model_revision(model_id: str, revision: str) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "activation extraction requires the optional dependencies; run "
            "`pip install -e '.[activation]'`"
        ) from error

    try:
        resolved = HfApi().model_info(model_id, revision=revision).sha
    except Exception as error:
        raise RuntimeError(
            f"cannot access {model_id!r}; accept the Gemma license on Hugging "
            "Face and run `hf auth login` before extraction"
        ) from error
    if not resolved:
        raise RuntimeError(
            f"Hugging Face did not resolve {model_id!r} at revision {revision!r}"
        )
    return str(resolved)


def _load_model(
    *, model_id: str, revision: str, device: str, dtype_name: str
) -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "activation extraction requires the optional dependencies; run "
            "`pip install -e '.[activation]'`"
        ) from error

    resolved_device = resolve_device(torch, device)
    dtype = resolve_dtype(torch, dtype_name)
    dtype_keyword = transformers_dtype_keyword(transformers.__version__)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            low_cpu_mem_usage=True,
            **{dtype_keyword: dtype},
        )
    except OSError as error:
        raise RuntimeError(
            f"could not load {model_id!r}; verify Hugging Face access and local "
            "memory before retrying"
        ) from error
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("the tokenizer defines neither a pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.to(resolved_device)
    model.eval()
    return torch, tokenizer, model


def extract_final_token_activations(
    *,
    prompts: Sequence[str],
    model_id: str,
    revision: str,
    layer: int,
    batch_size: int,
    device: str,
    dtype_name: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load Gemma and extract one residual layer at each prompt's final token."""
    if not prompts:
        raise ValueError("at least one prompt is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    resolved_revision = _resolve_model_revision(model_id, revision)
    torch, tokenizer, model = _load_model(
        model_id=model_id,
        revision=resolved_revision,
        device=device,
        dtype_name=dtype_name,
    )
    resolved_device = str(next(model.parameters()).device)
    n_layers = int(model.config.num_hidden_layers)
    state_index = hidden_state_index(layer, n_layers)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    batches: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        encoded = tokenizer(batch, padding=True, return_tensors="pt")
        positions = final_token_indices(encoded["attention_mask"].cpu().numpy())
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        if output.hidden_states is None:
            raise RuntimeError("model did not return hidden states")
        hidden = output.hidden_states[state_index]
        row_indices = torch.arange(len(batch), device=hidden.device)
        token_indices = torch.as_tensor(positions, device=hidden.device)
        selected = hidden[row_indices, token_indices]
        batches.append(selected.detach().float().cpu().numpy())
        del output, hidden, selected

    activations = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if activations.ndim != 2 or activations.shape[0] != len(prompts):
        raise RuntimeError(f"unexpected activation shape: {activations.shape}")
    if not np.isfinite(activations).all():
        raise RuntimeError("activation matrix contains non-finite values")
    return activations, {
        "model_revision": resolved_revision,
        "device": resolved_device,
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "n_layers": n_layers,
        "hidden_size": int(model.config.hidden_size),
    }


def save_activation_archive(
    destination: str | Path,
    *,
    activations: np.ndarray,
    labels: Sequence[str],
    prompts: Sequence[str],
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Atomically persist the activation contract and its provenance manifest."""
    path = Path(destination)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"activation output must use the .npz extension: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing activations: {path}")
    values = np.asarray(activations, dtype=np.float32)
    label_array = np.asarray(labels)
    prompt_array = np.asarray(prompts)
    if values.ndim != 2:
        raise ValueError(f"activations must be 2D; got shape {values.shape}")
    if len(values) != len(label_array) or len(values) != len(prompt_array):
        raise ValueError("activations, labels, and prompts must have equal row counts")
    if not np.isfinite(values).all():
        raise ValueError("activations contain non-finite values")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            activations=values,
            labels=label_array,
            prompts=prompt_array,
            layer=np.asarray(metadata["layer"], dtype=np.int64),
            model=np.asarray(metadata["model_id"]),
            model_revision=np.asarray(metadata["model_revision"]),
            template_name=np.asarray(metadata["template_name"]),
            template=np.asarray(metadata["template"]),
        )
    os.replace(temporary, path)
    manifest_path = path.with_suffix(".json")
    manifest = {
        **metadata,
        "artifact": {
            "path": str(path),
            "sha256": _sha256(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return path, manifest_path


def _list_registry() -> None:
    print("Built-in prompt sets:")
    for name, prompt_set in PROMPT_SETS.items():
        print(
            f"  {name:<16} {len(prompt_set.values):>3} values  "
            f"template={prompt_set.template_name}"
        )
    print("\nShared templates:")
    for name, template in PROMPT_TEMPLATES.items():
        print(f"  {name:<20} {template.text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list prompt inputs")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt-set", choices=tuple(PROMPT_SETS))
    source.add_argument("--values-file", type=Path)
    parser.add_argument("--template", choices=tuple(PROMPT_TEMPLATES))
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default=DEFAULT_DEVICE
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        _list_registry()
        return 0
    try:
        prompt_set_name, template_name, labels, prompts = _resolve_prompt_input(
            prompt_set_name=args.prompt_set,
            template_name=args.template,
            values_file=args.values_file,
        )
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))

    template = get_prompt_template(template_name)
    destination = args.output or default_output_path(
        args.model, prompt_set_name, args.layer
    )
    started_at = time.time()
    activations, runtime = extract_final_token_activations(
        prompts=prompts,
        model_id=args.model,
        revision=args.model_revision,
        layer=args.layer,
        batch_size=args.batch_size,
        device=args.device,
        dtype_name=args.dtype,
        seed=args.seed,
    )
    elapsed_seconds = time.time() - started_at
    metadata = {
        "experiment": "gemma-final-token-activation-extraction-v1",
        "source_revision": _git_revision(),
        "command": " ".join([sys.executable, *sys.argv]),
        "model_id": args.model,
        "model_revision": runtime["model_revision"],
        "framework": "Hugging Face Transformers",
        "layer": args.layer,
        "layer_indexing": "zero-based decoder block",
        "activation": "decoder-block output residual stream",
        "token_position": "final non-padding prompt token",
        "prompt_set": prompt_set_name,
        "template_name": template_name,
        "template": template.text,
        "n_prompts": len(prompts),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "requested_dtype": args.dtype,
        "resolved_device": runtime["device"],
        "resolved_dtype": runtime["dtype"],
        "n_model_layers": runtime["n_layers"],
        "hidden_size": runtime["hidden_size"],
        "started_unix": started_at,
        "elapsed_seconds": elapsed_seconds,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "huggingface_hub": _package_version("huggingface-hub"),
        },
        "limitations": [
            "Floating-point results can vary across hardware and accelerator kernels.",
            "This artifact contains one layer and one token position per prompt.",
        ],
    }
    archive_path, manifest_path = save_activation_archive(
        destination,
        activations=activations,
        labels=labels,
        prompts=prompts,
        metadata=metadata,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "activations": str(archive_path),
                "manifest": str(manifest_path),
                "shape": list(activations.shape),
                "model_revision": runtime["model_revision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
