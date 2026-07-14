class ManifoldComparator:
    def __init__(self):
        pass

    def compare_manifolds(self, m1, m2):
        ID_diff = abs(m1.get_intrinsic_dim() - m2.get_intrinsic_dim())
        curv_diff = 0
        for i, j in [m1.compute_ollivier_ricci()[ 'raw_values' ], m2.compute_ollivier_ricci()[ 'raw_values' ]]:
            curv_diff += (i - j)**2
        
        return ID_diff, curv_diff