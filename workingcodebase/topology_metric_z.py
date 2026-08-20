"""
Every axis is now a z-score in null-sigma units 
rather than a raw distance. This makes axes comparable in scale 
(which is what lets you Euclidean-combine H0 and H1) while still measuring different things; and that topology_metric.py is deliberately untouched because it produced plots/ and the existing JSON.
"""

"""
This essentially tells us how far the manifold sits from it's own nulls.
What this does is draws n null clouds and sets an anchor to the first
null drawn. 


This also mitigates the failure mode of similar shapes
not grouping simply because the dimensionalities differ


"""



import numpy as np
import persim
from collections import namedtuple
from scipy.stats import wasserstein_distance

from null_cloud import _finite

from null_cloud import Manifold
from run_ground_truths_topology_metric import Measurer


def _z(real, null):
        if np.std(null) ==0:
            assert ZeroDivisionError, 'cannot divide by 0'
        else:
            x = (np.mean(real) - np.mean(null)) / np.std(null)
            return x


class TopologyMetricZ():
    def __init__(self, manifold, kind = 'noise', n_nulls = 10, max_dim=1):
        self.manifold = manifold # of type 'Manifold'
        nulls = [manifold.null(kind) for _ in range(n_nulls)]
        anchor, comparisons = nulls[0], nulls[1:]
        real = [abs(manifold.intrinsic_dim - n.intrinsic_dim) for n in comparisons]
        null = [abs(anchor.intrinsic_dim   - n.intrinsic_dim) for n in comparisons]
        self.dimension = TopologyMetricZ._z(real, null)
