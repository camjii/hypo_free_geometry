'''
Run TopologyMetric for every concept (n_null >= 10) -> z-score dimension and
curvature against their null distributions, not just the null mean -> collect
into {concept: (dim_z, topology, curv_z)} -> split into per-axis arrays ->
build a distance matrix per axis -> cluster each axis independently (HDBSCAN,
precomputed) -> zip per-axis labels back onto concepts -> report joint
cluster membership per concept
'''

import numpy as np
import persim
from collections import namedtuple
from scipy.stats import wasserstein_distance

from null_cloud import _finite
import os
import json
import numpy as np

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN

from null_cloud import Manifold
from run_ground_truths_topology_metric import Measurer




class ClusteringAlgo():
    def __init__(self, prompts):
        self.acts = np.load('opt_acts_concept.npy') #placeholder
        self.measurer = Measurer()
        self.manifold = Manifold(self.measurer, self.acts)

        

   