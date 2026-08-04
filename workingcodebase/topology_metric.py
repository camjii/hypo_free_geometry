"""
Three numbers classifying a manifold against its own null.

    TopologyMetric(m1).metric  ->  (dimension, topology, curvature)

One axis per pipeline measurement:
    dimension   |ID_concept - ID_null|
    topology    H{max_dim} bottleneck vs null
    curvature   distribution_distance * (frac_negative_difference - 0.25)

Not a point in a metric space: the three axes are in different units
(dimensions, normalised distance, curvature), so ||metric|| is meaningless.
Read them separately; never sum, average, or norm them.

Not a significance test: these are distances from one null draw. They say how
far from null, not whether that distance is large.

NOTE: frozen version that produced everything in plots/ -- the main repo's
topology_metric.py has since evolved.
"""

import numpy as np
from collections import namedtuple

from null_cloud import ManifoldComparator

Metric = namedtuple("Metric", ["dimension", "topology", "curvature"])


class TopologyMetric:

    def __init__(self, manifold, kind="noise", n_null=1, max_dim=1):
        # n_null > 1: average over independent null draws -- each is one draw
        # from the null distribution, so averaging reduces single-draw variance.
        # manifold.null() reseeds from the manifold's rng, so draws differ.
        results = [ManifoldComparator().compare(manifold, manifold.null(kind), max_dim)
                   for _ in range(n_null)]

        self.dimension = float(np.mean([r["id_difference"] for r in results]))

        # bottleneck, not wasserstein: the null's mismatched subtraction smears
        # variance and generates a thicket of short noise bars. Wasserstein sums
        # them all, so it mostly measures the null's noise volume. Bottleneck
        # reports only the single worst-matched feature -- the concept's real
        # structure, which is what classifies.
        H0_bottleneck = float(np.mean([r["diagram_distance"]["H0"]["bottleneck"] for r in results]))
        H1_bottleneck = float(np.mean([r["diagram_distance"]["H1"]["bottleneck"] for r in results]))
        min_bottleneck = min(H0_bottleneck, H1_bottleneck)
        self.topology = [H0_bottleneck, H1_bottleneck, min_bottleneck] #metric for H0, H1, and lowest bottleneck irresepctive of feature seperately


        self.curvature = float(np.mean([c["distribution_distance"]
                                        * (c["frac_negative_difference"] - 0.25)
                                        for c in (r["curvature"] for r in results)]))  # < 0 means more negative curvature than null --> more clustered than null
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
