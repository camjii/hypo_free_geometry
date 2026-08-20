"""Concept -> feature row -> clusters. Minimal path, no null sweep.

Only scale-free features: n runs 4..64 across concepts, so bar counts and raw
entropies would cluster by sample size instead of shape. H0 entropy is divided
by its log(n-1) ceiling, which is what makes it comparable across concepts.
"""

import os
import sys
import io
import contextlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ground_truths import GROUND_TRUTHS
from null_cloud import Manifold, _finite
from run_ground_truths_topology_metric import Measurer

ACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'acts')

FEATURES = ['intrinsic_dim', 'H1_max_pers', 'H0_entropy_norm',
            'curv_mean', 'curv_std', 'curv_frac_neg']


def _row(m, n):
    L0 = (lambda b: b[:, 1] - b[:, 0])(_finite(m.dgms[0]))
    L1 = (lambda b: b[:, 1] - b[:, 0])(_finite(m.dgms[1]))
    c = np.asarray(m.curvature_values, dtype=float)
    p = L0 / L0.sum() if L0.sum() > 0 else np.array([1.0])
    return {
        'intrinsic_dim': float(m.intrinsic_dim),
        'H1_max_pers': float(L1.max()) if len(L1) else 0.0,
        'H0_entropy_norm': float(-(p * np.log(p)).sum() / np.log(max(n - 1, 2))),
        'curv_mean': float(c.mean()) if len(c) else 0.0,
        'curv_std': float(c.std()) if len(c) else 0.0,
        'curv_frac_neg': float((c < 0).mean()) if len(c) else 0.0,
        'n_points': int(n),          # carried for diagnostics, never clustered on
    }


def build_features(acts=ACTS, seed=0):
    """One row per ground-truth concept with cached activations."""
    measurer, rows, index = Measurer(), [], []
    for name in GROUND_TRUTHS:
        path = os.path.join(acts, f'{name}.npy')
        if not os.path.exists(path):
            continue
        act = np.load(path)
        with contextlib.redirect_stdout(io.StringIO()):   # Manifold prints per call
            m = Manifold(measurer, act, label=name, seed=seed)
        rows.append(_row(m, act.shape[0]))
        index.append(name)
    return pd.DataFrame(rows, index=pd.Index(index, name='concept'))


def cluster(df, k=4, seed=0):
    """Add a `cluster` column. Standardised so no single feature dominates."""
    X = StandardScaler().fit_transform(df[FEATURES].values)
    return df.assign(cluster=KMeans(k, n_init=20, random_state=seed).fit_predict(X))


if __name__ == '__main__':
    out = cluster(build_features(), k=4)
    print(out.sort_values('cluster').round(3).to_string())
    print('\nmean n_points by cluster (if this tracks the clusters, they are size bins):')
    print(out.groupby('cluster')['n_points'].agg(['mean', 'count']).round(1).to_string())
