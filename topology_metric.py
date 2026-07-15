"""
Three numbers classifying a manifold against its own null.

    TopologyMetric(m1).metric  ->  (dimension, topology, curvature)

(0, 0, 0) means indistinguishable from the null: no structure the pairing
produced. Each axis is a distance, so each is >= 0 and larger = further from
null.

One axis per pipeline measurement:
    dimension  |ID_concept - ID_null|              how much more/less complex
    topology   H{max_dim} bottleneck vs null       features the null lacks
    curvature  Wasserstein between curvature dists how differently organised

Two things the vector is NOT:
  - It is not a point in a metric space. The three axes are in different units
    (dimensions, normalised distance, curvature), so ||metric|| is meaningless.
    Read the three separately; never sum, average, or norm them.
  - It is not a significance test. These are distances from ONE null draw
    (or the median of n_null draws). They say how far from null, not whether
    that distance is large. Raise n_null to reduce variance; that still is not
    a p-value.
"""

import numpy as np
from collections import namedtuple

Metric = namedtuple("Metric", ["dimension", "topology", "curvature"])


class TopologyMetric:

    def __init__(self, manifold, kind="shuffled", n_null=1, max_dim=1):
        cmp = ManifoldComparator()
        runs = [cmp.compare(manifold, manifold.null(kind), max_dim)
                for _ in range(n_null)]

        self.dimension = float(np.median(
            [r["id_difference"] for r in runs]))

        # bottleneck, not wasserstein: the null's mismatched subtraction smears
        # variance and generates a thicket of short noise bars. Wasserstein sums
        # them all, so it mostly measures the null's noise volume. Bottleneck
        # reports only the single worst-matched feature -- the concept's real
        # structure, which is what classifies.
        self.topology = float(np.median(
            [r["diagram_distance"][f"H{max_dim}"]["bottleneck"] for r in runs]))

        self.curvature = float((0.3 - runs["curvature"]["frac_negative_difference"]) * np.median(
            [r["curvature"]["distribution_distance"] for r in runs]))

        self.metric = Metric(self.dimension, self.topology, self.curvature)
        self.label = manifold.label
        self.n_null = n_null

    def getMetric(self):
        return self.metric

    def __repr__(self):
        return (f"TopologyMetric({self.label}: dimension={self.dimension:.3f}, "
                f"topology={self.topology:.3f}, curvature={self.curvature:.3f})")