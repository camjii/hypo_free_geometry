"""
Compare two concept manifolds.
 
Descriptive only: returns how different m1 and m2 are on three axes[ID, PD, CURV]
 
"""
 
import numpy as np
import persim
from scipy.stats import wasserstein_distance
 
 
def _finite(dgm):
    """Strip the [0, inf] H0 bar -- persim returns inf otherwise."""
    d = np.asarray(dgm, dtype=float)
    return d[np.isfinite(d).all(axis=1)] if d.size else np.empty((0, 2))
 
 
class ManifoldComparator:
 
    def diagram_distance(self, m1, m2, max_dim=1):
        """Per homology dimension. Never combined -- H0 measures merge scales,
        H1 measures loop lifespans, different units."""
        out = {}
        for k in range(min(max_dim + 1, len(m1.dgms), len(m2.dgms))):
            a, b = _finite(m1.dgms[k]), _finite(m2.dgms[k])
            out[f"H{k}"] = {
                "wasserstein": float(persim.wasserstein(a, b)),  # sums all gaps --> total difference Betti Features
                "bottleneck": float(persim.bottleneck(a, b)),    # worst gap only
            }
        return out
 
    def curvature_difference(self, m1, m2):
        """Distributions"""
        c1 = np.asarray(m1.curvature_values, dtype=float)
        c2 = np.asarray(m2.curvature_values, dtype=float)
        return {
            "distribution_distance": float(wasserstein_distance(c1, c2)),  # measures difference in curvature distributions of a single selected epsilon in Ollivier Ricci Function
            "frac_negative_difference": float(abs((c1 < 0).mean() - (c2 < 0).mean())),  # measures difference in fraction of negative curvatures
            #fraction of negative curvatures semantic meaning
            # 0 --> tight blob
            # 0.05 - 0.15 --> mostly clustered with few bridge edges
            # 0.3 - 0.5 --> substantial branching/tree-like organization
            # > 0.5 --> predominantly hyperbolic/mostly bridges/very sparse
        }
 
    def compare(self, m1, m2, max_dim=1):
        return {
            "id_difference": abs(m1.intrinsic_dim - m2.intrinsic_dim),   # measures difference in complexity of structures
            "diagram_distance": self.diagram_distance(m1, m2, max_dim),   # measures difference in topological features (holes, and voids)
            "curvature": self.curvature_difference(m1, m2),     # measures difference in curvature features
        }
    
