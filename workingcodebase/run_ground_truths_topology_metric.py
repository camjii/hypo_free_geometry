"""
Run the topology metric on every ground-truth concept set. No plots.

Concept definitions (words, prompt templates, expected structure) come from
ground_truths.py. Activations are gemma-2-2b layer-6 final-token residuals,
cached per concept in acts/ -- the same cache files the old plotting runner
used -- so finished concepts are never re-extracted and the model is only
loaded if some concept is missing from the cache.

Ground truths are raw activation clouds with no contrastive pos/neg pairing,
so the metric is computed against the "noise" null: a Gaussian matched to each
cloud's own mean and covariance. The "shuffled" null needs a pairing to break
and would reproduce the concept cloud exactly here.

Per-concept results print as they finish and are all written to
ground_truth_topology_metrics.json at the end.
"""

import os
import json
import numpy as np

from pipeline_draft import Pipeline
from null_cloud import Manifold
from topology_metric import TopologyMetric
from ground_truths import GROUND_TRUTHS

LAYER = 'layer_6'
ACTDIR = 'acts'
RESULTS_PATH = 'ground_truth_topology_metrics.json'
NULL_KIND = 'noise'
N_NULL = 3   # average the metric over independent null draws
SEED = 0


class LeanPipeline(Pipeline):
    """Pipeline with a lower-peak model load. The frozen Pipeline.__init__
    uses from_pretrained, whose weight *processing* (fold_ln etc.) casts the
    state dict to fp32 and holds several copies -- ~35GB peak, OOM-killed on
    a 16GB machine. no_processing keeps bf16 end-to-end; the residual stream
    itself is unaffected by the skipped folding."""

    def __init__(self, pos_prompts):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformer_lens import HookedTransformer
        hf_model = AutoModelForCausalLM.from_pretrained(
            'unsloth/gemma-2-2b', torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained('unsloth/gemma-2-2b')
        self.model = HookedTransformer.from_pretrained_no_processing(
            'gemma-2-2b', hf_model=hf_model, tokenizer=tokenizer,
            dtype=torch.bfloat16)
        del hf_model
        self.pos_prompts = pos_prompts


class Measurer:
    """Measurement-only facade over Pipeline: Manifold needs these five
    methods, none of which touch the model, so cached concepts can be
    measured without loading gemma."""
    get_intrinsic_dim = Pipeline.get_intrinsic_dim
    reduce_pca = Pipeline.reduce_pca
    create_persistence_diagram = Pipeline.create_persistence_diagram
    compute_ollivier_ricci = Pipeline.compute_ollivier_ricci

    def create_epsilon_graph(self, projected, eps):
        # Manifold picks eps as an exact pdist value; sklearn recomputes the
        # distances and can land 1 ulp above it, dropping boundary edges. At
        # n=4 (seasons) the boundary edge is the ONLY edge, and an empty edge
        # list crashes GraphRicciCurvature (pool chunksize 0). The relative
        # nudge is far below any meaningful density change.
        return Pipeline.create_epsilon_graph(self, projected, eps * (1 + 1e-6))


def load_activations(name, words, template, pipeline):
    """Cached [n, d] activations for a concept, extracting (and caching) if
    needed. Returns (activations, pipeline) so the model loads at most once."""
    cache = f'{ACTDIR}/{name}.npy'
    if os.path.exists(cache):
        return np.load(cache), pipeline

    prompts = [template.format(w) for w in words]
    if pipeline is None:
        pipeline = LeanPipeline(prompts)
    else:
        pipeline.pos_prompts = prompts
    act = pipeline.collect_activations()[LAYER]
    np.save(cache, act)
    return act, pipeline


if __name__ == '__main__':
    os.makedirs(ACTDIR, exist_ok=True)

    # phase 1: make sure every concept's activations are cached, then drop
    # the model so the metric phase has the machine's memory to itself
    pipeline = None
    activations = {}
    for name, (words, template, edges, cyclic) in GROUND_TRUTHS.items():
        act, pipeline = load_activations(name, words, template, pipeline)
        activations[name] = act
        print(f'{name}: activations {act.shape}')
    del pipeline

    # phase 2: topology metric per concept, no model needed. Saved after
    # every concept, and already-saved concepts are skipped, so a crash
    # costs one concept, not the run.
    measurer = Measurer()
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f'resuming: {len(results)} concepts already in {RESULTS_PATH}')

    failed = []
    for name, (words, template, edges, cyclic) in GROUND_TRUTHS.items():
        if name in results:
            continue
        act = activations[name]
        print(f'\n{name}: n={act.shape[0]}, d={act.shape[1]}')

        try:
            manifold = Manifold(measurer, act, np.zeros_like(act), cloud=act,
                                label=name, seed=SEED)
            tm = TopologyMetric(manifold, kind=NULL_KIND, n_null=N_NULL)
        except Exception as e:
            print(f'{name}: FAILED -- {type(e).__name__}: {e}')
            failed.append(name)
            continue
        print(tm)

        results[name] = {
            'n_points': int(act.shape[0]),
            'cyclic': cyclic,
            'dimension': tm.dimension,
            'topology': {'H0_bottleneck': tm.topology[0],
                         'H1_bottleneck': tm.topology[1],
                         'min_bottleneck': tm.topology[2]},
            'curvature': tm.curvature,
        }
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nsaved metrics for {len(results)} concepts to {RESULTS_PATH}'
          + (f' ({len(failed)} failed: {", ".join(failed)})' if failed else ''))
