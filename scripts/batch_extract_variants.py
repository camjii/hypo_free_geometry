"""Extract layer-6 concept-token activations for every family x 10 paraphrases."""

# ruff: noqa: E402  (sys.path setup must precede sibling imports)

import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from concept_families import FAMILIES
from prompt_templates import get_template_variants
from activation_extraction import (
    _load_model,
    _resolve_model_revision,
    hidden_state_index,
    load_prompt_values,
    save_activation_archive,
    value_token_indices,
)

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-7b"
LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SEED = 260215
BATCH = 12
from activation_extraction import _slug

OUT = REPO / "outputs/activations" / _slug(MODEL_ID.split("/")[-1]) / "variants"
VALUES = REPO / "prompt_values"
OUT.mkdir(parents=True, exist_ok=True)

# value-set name -> canonical template family
SETS = {f["name"]: f["template"] for f in FAMILIES}
SETS.update(
    {
        "days": "day_of_week",
        "months": "month_of_year",
        "chess_squares": "chess_square",
        "calendar_dates": "calendar_date",
    }
)

# built-in sets have no value file; materialize them from the registry
from prompt_templates import CHESS_SQUARES, DAYS, MONTHS

BUILTIN = {"days": DAYS, "months": MONTHS, "chess_squares": CHESS_SQUARES}


def values_for(name):
    if name in BUILTIN:
        return tuple(BUILTIN[name])
    return load_prompt_values(VALUES / f"{name}.txt")


total_prompts = sum(len(values_for(n)) * 10 for n in SETS)
print(
    f"planned: {len(SETS)} value sets x 10 variants = {total_prompts} prompts",
    flush=True,
)

print("resolving revision + loading model once...", flush=True)
revision = _resolve_model_revision(MODEL_ID, "main")
torch, tokenizer, model = _load_model(
    model_id=MODEL_ID, revision=revision, device="cpu", dtype_name="auto"
)
device = str(next(model.parameters()).device)
n_layers = int(model.config.num_hidden_layers)
state_index = hidden_state_index(LAYER, n_layers)
hidden_size = int(model.config.hidden_size)
dtype_str = str(next(model.parameters()).dtype).removeprefix("torch.")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def extract(prompts, spans):
    out = []
    for s in range(0, len(prompts), BATCH):
        bp, bs = prompts[s : s + BATCH], spans[s : s + BATCH]
        enc = tokenizer(
            bp,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        pos = value_token_indices(
            enc["offset_mapping"].cpu().numpy(),
            enc["special_tokens_mask"].cpu().numpy(),
            bs,
        )
        feed = {
            k: v.to(device)
            for k, v in enc.items()
            if k in ("input_ids", "attention_mask")
        }
        with torch.inference_mode():
            res = model(
                **feed, output_hidden_states=True, return_dict=True, use_cache=False
            )
        hid = res.hidden_states[state_index]
        rows = torch.arange(len(bp), device=hid.device)
        cols = torch.as_tensor(pos, device=hid.device)
        out.append(hid[rows, cols].detach().float().cpu().numpy())
        del res, hid
    a = np.concatenate(out, 0).astype(np.float32)
    assert np.isfinite(a).all(), "non-finite activations"
    return a


started = time.time()
done = 0
for set_name, family in SETS.items():
    values = values_for(set_name)
    for variant in get_template_variants(family):
        dest = OUT / f"{set_name}__{variant.name}_layer_{LAYER}.npz"
        if dest.exists():
            done += len(values)
            continue
        prompts = [variant.render(v) for v in values]
        spans = [variant.value_span(v) for v in values]
        t0 = time.time()
        acts = extract(prompts, spans)
        meta = {
            "experiment": "gemma-concept-token-activation-extraction-v2",
            "model_id": MODEL_ID,
            "model_revision": revision,
            "framework": "Hugging Face Transformers",
            "layer": LAYER,
            "layer_indexing": "zero-based decoder block",
            "activation": "decoder-block output residual stream",
            "token_position": "last token of the concept value span",
            "prompt_set": set_name,
            "template_name": variant.name,
            "template": variant.text,
            "template_family": family,
            "n_prompts": len(prompts),
            "seed": SEED,
            "resolved_device": device,
            "resolved_dtype": dtype_str,
            "n_model_layers": n_layers,
            "hidden_size": hidden_size,
            "environment": {"python": platform.python_version()},
        }
        save_activation_archive(
            dest,
            activations=acts,
            labels=values,
            prompts=prompts,
            metadata=meta,
            overwrite=True,
        )
        done += len(values)
        rate = (time.time() - started) / max(done, 1)
        eta = (total_prompts - done) * rate / 60
        print(
            f"  {variant.name:<28} {str(acts.shape):<11} "
            f"{time.time() - t0:5.1f}s  [{done}/{total_prompts}] eta {eta:.0f}m",
            flush=True,
        )

print(f"\nDONE {done} prompts in {(time.time() - started) / 60:.1f} min")
