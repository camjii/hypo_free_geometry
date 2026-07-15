"""
Compare a concept manifold against its own null.

    m1 = Manifold(pipeline, pos_acts, neg_acts)
    ManifoldComparator().compare(m1, m1.null())

A manifold carries the activations it was built from, so it can generate its own
null: same prompts, same model, same layer, same n -- only the pos/neg pairing
broken. Both are measured by the same method, so the comparison reflects the
data rather than two code paths.
"""

import numpy as np
import persim
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance


def _finite(dgm):
    """Strip the [0, inf] H0 bar -- persim returns inf otherwise."""
    d = np.asarray(dgm, dtype=float)
    return d[np.isfinite(d).all(axis=1)] if d.size else np.empty((0, 2))


class Manifold:
    """A measured cloud: ID, persistence diagrams, curvature signature.

    pipeline: exposes get_intrinsic_dim, reduce_pca, create_persistence_diagram,
              create_epsilon_graph, compute_ollivier_ricci
    pos_acts,
    neg_acts: cached [n, d] activations at the chosen layer. Kept so the
              manifold can build its own null with no forward passes.
    """

    def __init__(self, pipeline, pos_acts, neg_acts, cloud=None,
                 label="concept", seed=0, eps_density=0.10, var_threshold=0.95):
        self.pipeline = pipeline
        self.pos = np.asarray(pos_acts)
        self.neg = np.asarray(neg_acts)
        self.label = label
        self.rng = np.random.default_rng(seed)
        self.eps_density = eps_density
        self.var_threshold = var_threshold
        self.cloud = self.pos - self.neg if cloud is None else np.asarray(cloud)
        self._measure()

    def _measure(self):
        self.intrinsic_dim = float(self.pipeline.get_intrinsic_dim(self.cloud))

        # REFIT per cloud: PCA can manufacture structure from high-dim noise,
        # so freezing the concept's basis onto the null would never test it.
        projected = self.pipeline.reduce_pca(self.cloud, self.var_threshold)

        # SCALE-NORMALISE: mismatched subtraction adds variance, so nulls spread
        # wider and their bars run longer in raw units. A loop is a shape
        # property, not a size property.
        self.diameter = float(pdist(projected).max())
        projected = projected / self.diameter
        self.m = int(projected.shape[1])

        self.dgms = self.pipeline.create_persistence_diagram(projected)["dgms"]

        # eps from matched DENSITY, not a fixed value: clouds of different spread
        # give different densities at the same eps, and density drives curvature
        # independently of shape.
        d = np.sort(pdist(projected))
        self.eps = float(d[min(int(self.eps_density * len(d)), len(d) - 1)])

        graph = self.pipeline.create_epsilon_graph(projected, self.eps)
        curv = self.pipeline.compute_ollivier_ricci(graph)
        self.curvature_values = np.asarray(curv["raw_values"], dtype=float)

    def null(self, kind="shuffled"):
        """A null of this manifold, measured identically.

        shuffled: break the pos/neg pairing only -- everything else identical.
                  The tightest counterfactual, and the only null that tests
                  whether the pairing did any work.
        noise:    Gaussian with this cloud's own mean and covariance.
        """
        n = len(self.cloud)

        if kind == "shuffled":
            perm = self.rng.permutation(n)
            while np.any(perm == np.arange(n)):        # no accidental true pairs
                perm = self.rng.permutation(n)
            cloud = self.pos - self.neg[perm]
        elif kind == "noise":
            cov = np.cov(self.cloud.T) + 1e-6 * np.eye(self.cloud.shape[1])
            cloud = self.rng.multivariate_normal(self.cloud.mean(0), cov, n)
        else:
            raise ValueError(f"unknown null kind: {kind}")

        return Manifold(self.pipeline, self.pos, self.neg, cloud=cloud,
                        label=f"null:{kind}", seed=int(self.rng.integers(1 << 30)),
                        eps_density=self.eps_density, var_threshold=self.var_threshold)  #null(manifold) returns manifold of its null

    def __repr__(self):
        return (f"Manifold({self.label}: ID={self.intrinsic_dim:.2f}, m={self.m}, "
                f"{len(self.curvature_values)} edges)")


class ManifoldComparator:

    def diagram_distance(self, m1, m2, max_dim=1):
        """Per homology dimension. Never combined -- H0 measures merge scales,
        H1 measures loop lifespans, different units."""
        out = {}
        for k in range(min(max_dim + 1, len(m1.dgms), len(m2.dgms))):
            a, b = _finite(m1.dgms[k]), _finite(m2.dgms[k])
            out[f"H{k}"] = {
                "wasserstein": float(persim.wasserstein(a, b)),  # sums all gaps --> total difference in Betti features
                "bottleneck": float(persim.bottleneck(a, b)),    # worst gap only
            }
        return out

    def curvature_difference(self, m1, m2):
        """Distributions"""
        c1 = np.asarray(m1.curvature_values, dtype=float)
        c2 = np.asarray(m2.curvature_values, dtype=float)
        return {
            "distribution_distance": float(wasserstein_distance(c1, c2)),  # difference in curvature distributions at the selected epsilon
            "frac_negative_difference": float(abs((c1 < 0).mean() - (c2 < 0).mean())),  # difference in fraction of negative curvatures
            # fraction of negative curvatures, semantic meaning:
            # 0          --> tight blob
            # 0.05-0.15  --> mostly clustered with few bridge edges
            # 0.3-0.5    --> substantial branching / tree-like organization
            # > 0.5      --> predominantly hyperbolic / mostly bridges / very sparse
        }

    def compare(self, m1, m2, max_dim=1):
        return {
            "id_difference": abs(m1.intrinsic_dim - m2.intrinsic_dim),   # difference in complexity of structures
            "diagram_distance": self.diagram_distance(m1, m2, max_dim),  # difference in topological features (holes, voids)
            "curvature": self.curvature_difference(m1, m2),              # difference in curvature features
        }