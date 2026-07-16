"""
Three numbers classifying a manifold against its own null.

    TopologyMetric(m1).metric  ->  (dimension, topology, curvature)

One axis per pipeline measurement:
    dimension   |ID_concept - ID_null|
    topology    [H0, H1] bottleneck vs null
    curvature   distribution_distance * (0.3 - frac_negative_difference)

Not a point in a metric space: the three axes are in different units
(dimensions, normalised distance, curvature), so ||metric|| is meaningless.
Read them separately; never sum, average, or norm them.

Not a significance test: these are distances from one null draw. They say how
far from null, not whether that distance is large.
"""

import numpy as np
from collections import namedtuple

from compare import ManifoldComparator

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
        self.topology = [
            float(np.median([r["diagram_distance"]["H0"]["bottleneck"] for r in runs])),
            float(np.median([r["diagram_distance"]["H1"]["bottleneck"] for r in runs])),
        ]  # metric for H0 and H1 separately

        frac_negative_difference = np.median(
            [r["curvature"]["frac_negative_difference"] for r in runs])
        self.curvature = float((0.3 - frac_negative_difference) * np.median(
            [r["curvature"]["distribution_distance"] for r in runs]))

        self.metric = Metric(self.dimension, self.topology, self.curvature)
        self.label = manifold.label

    def getMetric(self):
        return self.metric

    def __iter__(self):
        return iter(self.metric)

    def __repr__(self):
        return (f"TopologyMetric({self.label}: dimension={self.dimension:.3f}, "
                f"topology={self.topology}, curvature={self.curvature:.3f})")