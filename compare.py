class ManifoldComparator:
    def __init__(self):
        pass

    def compare_manifolds(self, m1, m2):
        ID_diff = abs(m1.get_intrinsic_dim() - m2.get_intrinsic_dim())
        maxCurve = max(m1.compute_ollivier_ricci()['mean_curvature'], m2.compute_ollivier_ricci()['mean_curvature'])
        minCurve = min(m1.compute_ollivier_ricci()['mean_curvature'], m2.compute_ollivier_ricci()['mean_curvature'])
        curv_diff = 100 * (maxCurve - minCurve) / maxCurve if maxCurve != 0 else 0

        return ID_diff, curv_diff

