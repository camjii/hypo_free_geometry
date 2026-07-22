"""
Compare a concept manifold against its own null.

    manifold = Manifold(pipeline, activations, seed=42)

    result = ManifoldComparator().compare_against_nulls(manifold,kind="covariance_gaussian" ,n_nulls=30 ,base_seed=100,)

The null manifolds have the same sample count and activation dimension as the
observed manifold. Each null is generated from statistics estimated from the
observed activations and is measured using the same intrinsic-dimension,
persistent-homology, and curvature pipeline.

The covariance-Gaussian null preserves the estimated mean and covariance. The
comparison therefore tests whether the observed cloud contains structure beyond
what a Gaussian model with the same first- and second-order statistics explains.
"""


import numpy as np
import persim
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.covariance import LedoitWolf


def _finite(dgm):
    """Strip the [0, inf] H0 bar -- persim returns inf otherwise."""
    d = np.asarray(dgm, dtype=float)
    if d.size == 0:
        return np.empty((0, 2), dtype=float)
    
    # fix: catches nonempty diagrams that don't have shape [n, 2]
    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError(
            "persistence diagram must have shape [number_of_bars, 2]"
        )

    return d[np.isfinite(d).all(axis=1)] 



class Manifold:
    """A measured cloud: ID, persistence diagrams, curvature signature.

    pipeline: exposes get_intrinsic_dim, reduce_pca, create_persistence_diagram,
              create_epsilon_graph, compute_ollivier_ricci
    pos_acts,
    neg_acts: cached [n, d] activations at the chosen layer. Kept so the
              manifold can build its own null with no forward passes.
    """

    def __init__(self, pipeline, opt_activations, cloud=None,
                 label="concept", seed=0, eps_density=0.10, var_threshold=0.95):
        self.pipeline = pipeline
        self.opt = np.asarray(opt_activations, dtype=float)
        self.label = label
        self.rng = np.random.default_rng(seed)
        self.eps_density = eps_density
        self.var_threshold = var_threshold
        self.cloud = (self.opt if cloud is None else np.asarray(cloud, dtype=float))
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

    def null(self, kind="covariance_gaussian", seed = None):
        """Generate and measure a matched Gaussian null.

        covariance_gaussian:
            Preserves the fitted mean and covariance using Ledoit-Wolf estimation.
            This tests for nonlinear or non-Gaussian structure beyond first- and
            second-order statistics.

        isotropic_gaussian:
            Preserves the fitted mean and average variance, but removes covariance
            directions and anisotropy. This is a stronger null for detecting linear
            or directional structure.

        noise:
            Backward-compatible alias for covariance_gaussian.

        shuffled:
            Not available because separate positive and negative activations are
            not stored. Permuting rows of one completed point cloud would leave its
            geometry unchanged.
        """
        n = len(self.cloud)

        if kind == "shuffled":
            raise NotImplementedError(
                "kind='shuffled' requires separate positive and negative "
                "activations. Permuting rows of one completed point cloud "
                "would not change its geometry."
            )
    # Keep old code that calls manifold.null(kind="noise") working.
        if kind == "noise":
            kind = "covariance_gaussian"

        if kind not in {
            "covariance_gaussian",
            "isotropic_gaussian",
        }:
            raise ValueError(
                f"unknown null kind: {kind}. Use 'covariance_gaussian', "
                "'isotropic_gaussian', or 'noise'."
            )

        null_seed = (
            int(self.rng.integers(1 << 30))
            if seed is None
            else int(seed)
        )
        rng = np.random.default_rng(null_seed)

        if kind == "covariance_gaussian":
            # More stable than the raw empirical covariance when dimensions
            # greatly exceed the number of activation samples.
            estimator = LedoitWolf(
                store_precision=False
            ).fit(self.cloud)

            cloud = rng.multivariate_normal(
                mean=estimator.location_,
                cov=estimator.covariance_,
                size=n,
            )

        elif kind == "isotropic_gaussian":
            mean = self.cloud.mean(axis=0)

            # Average empirical variance across activation dimensions.
            average_variance = float(
                np.var(
                    self.cloud,
                    axis=0,
                    ddof=1,
                ).mean()
            )

            if average_variance <= 0:
                raise ValueError(
                    "isotropic Gaussian null requires positive variance"
                )

            cloud = mean + rng.normal(
                loc=0.0,
                scale=np.sqrt(average_variance),
                size=self.cloud.shape,
            )

        return Manifold(
            pipeline=self.pipeline,
            opt_activations=cloud,
            label=f"null:{kind}",
            seed=null_seed,
            eps_density=self.eps_density,
            var_threshold=self.var_threshold,
        )

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
        negative_difference = float((c1 < 0).mean() - (c2 < 0).mean())
        return {
            "distribution_distance": float(wasserstein_distance(c1, c2)),  # difference in curvature distributions at the selected epsilon
            "mean_difference": float(c1.mean() - c2.mean()), # signed difference in average curvature; positive means the concept has higher average curvature than the null, negative means lower
            "negative_fraction_difference": negative_difference, # signed difference in the proportion of negative-curvature edges; positive means the concept has more negative edges than the null, negative means fewer
            "absolute_negative_fraction_difference": abs(negative_difference), # size of the difference in negative-edge proportions, ignoring which manifold has more negative edges
            "frac_negative_difference": float(abs((c1 < 0).mean() - (c2 < 0).mean())),  # difference in fraction of negative curvatures

            #need to recheck this, thresholds arn't unibersal

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
    
    def _flatten_distances(self, m1, m2, max_dim=1):
        """convert one manifold comparison into named scalar distances."""
        comparison = self.compare(m1, m2, max_dim=max_dim)

        distances = {
            "id_difference": float(comparison["id_difference"]),
            "curvature_wasserstein": float(comparison["curvature"]["distribution_distance"]),
            "curvature_negative_fraction_difference": float(
                comparison["curvature"]["frac_negative_difference"]
            ),
        }

        for homology_dimension, values in comparison["diagram_distance"].items():
            distances[f"{homology_dimension}_wasserstein"] = float(values["wasserstein"])
            distances[f"{homology_dimension}_bottleneck"] = float(values["bottleneck"])

        return distances


    def compare_against_nulls(self, manifold, kind="covariance_gaussian", n_nulls=30, base_seed=0, max_dim=1,
    ):
        """compare a manifold against repeated samples from one null model."""
        if n_nulls < 3:
            raise ValueError("n_nulls must be at least 3")

        nulls = [
            manifold.null(kind=kind, seed=base_seed + index)
            for index in range(n_nulls)
        ]

        objects = [manifold, *nulls]
        number_of_objects = len(objects)

        metric_names = self._flatten_distances(
            objects[0],
            objects[1],
            max_dim=max_dim,
        ).keys()

        distance_matrices = {
            name: np.zeros((number_of_objects, number_of_objects), dtype=float)
            for name in metric_names
        }

        for i in range(number_of_objects):
            for j in range(i + 1, number_of_objects):
                distances = self._flatten_distances(
                    objects[i],
                    objects[j],
                    max_dim=max_dim,
                )

                for name, value in distances.items():
                    distance_matrices[name][i, j] = value
                    distance_matrices[name][j, i] = value

        results = {}

        for name, matrix in distance_matrices.items():
            observed_score = float(np.median(matrix[0, 1:]))
            null_matrix = matrix[1:, 1:]

            null_scores = np.array([
                np.median(np.delete(null_matrix[i], i))
                for i in range(n_nulls)
            ], dtype=float)

            null_median = float(np.median(null_scores))
            null_mad = float(np.median(np.abs(null_scores - null_median)))

            results[name] = {
                "observed_score": observed_score,
                "null_median": null_median,
                "null_mad": null_mad,
                "robust_z": float(
                    (observed_score - null_median)
                    / (1.4826 * null_mad + 1e-12)
                ),
                "monte_carlo_pvalue": float(
                    (1 + np.sum(null_scores >= observed_score))
                    / (n_nulls + 1)
                ),
                "null_scores": null_scores.tolist(),
            }

        return {
            "null_kind": kind,
            "n_nulls": n_nulls,
            "base_seed": base_seed,
            "max_dim": max_dim,
            "metrics": results,
        }
    

    def compare_both_nulls(self, manifold, n_nulls=30, base_seed=100, max_dim=1):
        """Evaluate the manifold against both Gaussian null models."""
        return {
            "covariance_gaussian": self.compare_against_nulls(manifold, kind="covariance_gaussian", n_nulls=n_nulls, base_seed=base_seed, max_dim=max_dim),
            "isotropic_gaussian": self.compare_against_nulls(manifold, kind="isotropic_gaussian", n_nulls=n_nulls, base_seed=base_seed + n_nulls, max_dim=max_dim),
        }