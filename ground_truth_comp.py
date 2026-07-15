"""
compare pipeline results against ground-truth results


"""

import numpy as np
from persim import bottleneck
from scipy.stats import wasserstein_distance

class GroundTruthComparator:
    """compare recovered and ground truth manifold measurements."""

    def compare_intrinsic_dimension(
        self,
        recovered_dimension,
        ground_truth_dimension,
    ):
        """compare intrinsic dimensions."""
        return abs(recovered_dimension - ground_truth_dimension)

    def compare_persistent_homology(
        self,
        recovered_ph,
        ground_truth_ph,
    ):
        """compare H0 and H1 persistence diagrams."""

        recovered_diagrams = recovered_ph["dgms"]
        ground_truth_diagrams = ground_truth_ph["dgms"]
        """
        following depends on number of max dim of risper in the pipeline
        
        """
        h0_error = bottleneck(
            recovered_diagrams[0],
            ground_truth_diagrams[0],
        )

        h1_error = bottleneck(
            recovered_diagrams[1],
            ground_truth_diagrams[1],
        )

        return {
            "h0_bottleneck_distance": float(h0_error),
            "h1_bottleneck_distance": float(h1_error),
        }

    def compare_curvature(
        self,
        recovered_curvature,
        ground_truth_curvature,
    ):
        """compare mean curvature and full curvature distributions."""

        recovered_mean = recovered_curvature["mean_curvature"]
        ground_truth_mean = ground_truth_curvature["mean_curvature"]

        mean_error = abs(recovered_mean - ground_truth_mean)

        recovered_values = recovered_curvature["raw_values"]
        ground_truth_values = ground_truth_curvature["raw_values"]

        distribution_error = wasserstein_distance(
            recovered_values,
            ground_truth_values,
        )

        return {
            "mean_curvature_error": float(mean_error),
            "curvature_distribution_error": float(distribution_error),
        }

    def compare(
        self,
        recovered_dimension,
        ground_truth_dimension,
        recovered_ph,
        ground_truth_ph,
        recovered_curvature,
        ground_truth_curvature,
    ):
        """run all ground-truth comparisons."""

        dimension_error = self.compare_intrinsic_dimension(
            recovered_dimension,
            ground_truth_dimension,
        )

        ph_results = self.compare_persistent_homology(
            recovered_ph,
            ground_truth_ph,
        )

        curvature_results = self.compare_curvature(
            recovered_curvature,
            ground_truth_curvature,
        )

        return {
            "intrinsic_dimension_error": float(dimension_error),
            "h0_bottleneck_distance": ph_results[
                "h0_bottleneck_distance"
            ],
            "h1_bottleneck_distance": ph_results[
                "h1_bottleneck_distance"
            ],
            "mean_curvature_error": curvature_results[
                "mean_curvature_error"
            ],
            "curvature_distribution_error": curvature_results[
                "curvature_distribution_error"
            ],
        }
    

class GroundTruth():
    """
    comparison of olivier ricci curv?
    
    
    
    """
    pass