'''
1. Run ground truth experiment from paper (e.g. karkada historical years "The year is x", gemma2-2b)
2. Run our pipeline with same prompt package
3. Validate our Ricci curvature code against GraphRicciCurvature (OTD/ATD) on a known graph (e.g. karate club) before trusting it
4. Get point clouds, ph graph, id + ricci curvature
5. align clouds with geometric representation alignment, track mean + median edge curvature
6. boostrap point clouds (or activations with replacement) and rerun to get confidence intervals
'''

import os
import sys

# this file lives one directory below the flat top-level modules
# (pipeline_draft, null_cloud, compare) -- put repo root on the path so the
# same `from x import Y` imports used everywhere else still resolve here too.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pipeline_draft import Pipeline
from null_cloud import Manifold
from compare import ManifoldComparator


class GroundTruthComp():

    def __init__(self, ground_truth_acts, pos_prompts, pipeline=None, label="concept"):
        self.ground_truth_acts = np.asarray(ground_truth_acts)  # act1: paper's activations
        self.pos_prompts = pos_prompts                          # step 2: same prompt package
        self.pipeline = pipeline or Pipeline(pos_prompts)
        self.label = label

    def get_pipeline_cloud(self):
        """ O step 2-3: run our pipeline, return its point cloud at the selected layer."""
        pass

    def build_manifolds(self, pipeline_acts):
        """ step 3: wrap ground_truth_acts and pipeline_acts as Manifolds so
        they're measured identically (ID, persistence diagrams, ricci curvature)."""
        pass

    def align(self, ground_truth_manifold, recovered_manifold):
        """ step 4: align clouds with geometric representation alignment, then
        track mean + median edge curvature on the aligned pair. Pick a method --
        Procrustes assumes paired points (same prompt order in both clouds); CKA-style
        subspace alignment doesn't. Depends on whether the ground-truth paper's
        activations line up 1:1 with pos_prompts."""
        pass

    def compare(self, ground_truth_manifold, recovered_manifold):
        """TODO step 3: ID, persistence diagram, and curvature distances between
        ground truth and recovered manifolds, via ManifoldComparator().compare(...)."""
        pass

    def bootstrap(self, pipeline_acts, n_boot=100, frac=0.8, ci=0.95):
        """TODO step 5: subsample ground_truth_acts and pipeline_acts (without
        replacement) and rerun compare() each draw to get a distribution of
        each scalar metric, then reduce with confidence_interval() below.

        pipeline_acts is a param, not fetched via self.get_pipeline_cloud(),
        so this doesn't depend on that method being implemented yet.

        Subsample, don't resample with replacement: duplicate points sit at
        distance 0 from themselves, which breaks skdim's TwoNN (undefined
        nearest-neighbor ratio) and can collapse the eps percentile used for
        the epsilon graph toward 0, degenerating the curvature signal.
        """
        pass

    def run(self):
        """TODO: full pipeline, steps 1-5 end to end."""
        pass


def confidence_interval(values, ci=0.95):
    """Percentile bootstrap CI over a list of per-replicate metric values.

    Standalone (not tied to GroundTruthComp) so it can summarize any list of
    bootstrap replicates -- e.g. id_diffs / h0 / h1 / curv_dist / curv_neg
    collected inside bootstrap() above.
    """
    values = np.asarray(values, dtype=float)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(values, [100 * alpha, 100 * (1 - alpha)])
    return {
        "mean": float(values.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }
