"""
Three numbers classifying a manifold against its own null.

    TopologyMetric(m1).metric  ->  (dimension, topology, curvature)

One axis per pipeline measurement:
    dimension   |ID_concept - ID_null|
    topology    H{max_dim} bottleneck vs null, z-scored against null-vs-null spread
    curvature   distribution_distance * (frac_negative_difference - 0.25)

Not a point in a metric space: the three axes are in different units
(dimensions, normalised distance, curvature), so ||metric|| is meaningless.
Read them separately; never sum, average, or norm them.

topology is the one axis built as an actual test rather than a single-sample
distance: n_nulls independent null draws give a null-vs-null bottleneck
distribution (pure sampling noise, no real signal by construction), and the
concept's own bottleneck-to-null is z-scored against that distribution. A
single-draw distance can't tell "genuinely far from null" apart from "this
particular null draw happened to land oddly" -- z-scoring against repeated
draws can.
"""

import numpy as np
import persim
from collections import namedtuple
from null_cloud import ManifoldComparator, _finite
Metric = namedtuple("Metric", ["dimension", "topology", "curvature"])


class TopologyMetric:

    def __init__(self, manifold, kind="noise", max_dim=1, n_nulls=5):
        comparator = ManifoldComparator()
        nulls = [manifold.null(kind) for _ in range(n_nulls)]

        r = comparator.compare(manifold, nulls[0], max_dim)
        self.dimension = float(r["id_difference"])

        # z-score, not a single bottleneck: nulls[0] is the anchor for the
        # null-vs-null baseline, nulls[1:] are compared against both the
        # concept and the anchor so the two distributions are measured against
        # the exact same set of comparison draws.
        anchor, comparisons = nulls[0], nulls[1:]
        self.topology = []
        for k in range(min(max_dim + 1, len(manifold.dgms))):
            concept_dgm = _finite(manifold.dgms[k])
            anchor_dgm = _finite(anchor.dgms[k])
            real_dists = np.array([
                persim.bottleneck(concept_dgm, _finite(n.dgms[k])) for n in comparisons
            ])
            null_dists = np.array([
                persim.bottleneck(anchor_dgm, _finite(n.dgms[k])) for n in comparisons
            ])
            z = (real_dists.mean() - null_dists.mean()) / (null_dists.std() + 1e-9)
            self.topology.append(float(z))
        self.topology.append(min(self.topology)) #metric for H0, H1, and lowest z-score irrespective of feature separately

        c = r["curvature"]
        self.curvature = float(c["distribution_distance"]
                               * (c["frac_negative_difference"] - 0.25))  # < 0 means more negative curvature than null --> more clustered than null
        # > 0 means more positive curvature than null --> more bridges/tree structure than null

        self.metric = Metric(self.dimension, self.topology, self.curvature)
        self.label = manifold.label

    def getMetric(self):
        return self.metric

    def __iter__(self):
        return iter(self.metric)

    def __repr__(self):
        return (f"TopologyMetric({self.label}: dimension={self.dimension:.3f}, "
                f"topology={self.topology}, curvature={self.curvature:.3f})")