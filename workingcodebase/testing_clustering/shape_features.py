import os,sys,io,time,glob,json,contextlib,warnings
warnings.filterwarnings('ignore')
import numpy as np, networkx as nx
sys.path.insert(0,os.path.abspath('..'))
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from ripser import ripser
import skdim
from GraphRicciCurvature.OllivierRicci import OllivierRicci

M, K, N_NULLS, MAXDIM = 3, 4, 20, 2

def project(X, m=None):
    m = M if m is None else m
    P = PCA(n_components=min(m, min(X.shape))).fit_transform(X)
    return P / pdist(P).max()

def bars(dgms, k):
    if k >= len(dgms): return np.empty(0)
    d = np.asarray(dgms[k], float)
    d = d[np.isfinite(d).all(axis=1)] if d.size else np.empty((0,2))
    return d[:,1]-d[:,0] if len(d) else np.empty(0)

def diagram(P):
    return ripser(P, maxdim=MAXDIM)['dgms']

def mst_degrees(P):
    T = minimum_spanning_tree(squareform(pdist(P))).toarray()
    A = (T + T.T) > 0
    return A.sum(1)

def intrinsic_dim(P, n):
    """TwoNN on the PROJECTION, capped at what is attainable.

    The pipeline default runs TwoNN on the raw 2304-d cloud, where n=12 points
    yields values like 39 -- above the n-1 ceiling and therefore meaningless.
    """
    try:
        d = float(skdim.id.TwoNN().fit(P).dimension_)
    except Exception:
        return np.nan
    ceiling = min(n-1, P.shape[1])
    return np.nan if not np.isfinite(d) else min(d, ceiling)

_ORDER_NULL = {}


def order_score(P, n):
    """Fraction of angular neighbours that are also neighbours in the value order.

    Persistence says a loop exists; this says it is THE loop. 1.0 means the cycle
    traverses in the concept's own order (up to rotation and reflection). The
    distinction is not academic: at layer 14 `years` shows H1 at 5.9x its null
    while ordering at 0.11, i.e. a closed cycle unrelated to chronology.
    """
    c = P.mean(axis=0)
    o = np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    return sum(1 for i in range(n) if abs(o[i] - o[(i + 1) % n]) in (1, n - 1)) / n


def order_p(score, n, draws=4000, seed=0):
    """Empirical p-value for an order score, from random clouds of the same n.

    Calibrated ONCE per n and cached, rather than per manifold: the null depends
    only on the point count, so 20 diagram computations per concept buy nothing
    that one shared table does not already give. A perfect score has p < 1/4000
    for every n >= 12.
    """
    if n not in _ORDER_NULL:
        rng = np.random.default_rng(seed)
        _ORDER_NULL[n] = np.array([
            order_score(PCA(n_components=2).fit_transform(rng.normal(size=(n, n - 1))), n)
            for _ in range(draws)])
    return float((_ORDER_NULL[n] >= score).mean())


def null_bar_threshold(X, n, rng, k, q=95):
    """p95 of pooled null bar lengths in dimension k, from Gaussians matched to
    this cloud's own mean and covariance (drawn inside its (n-1)-dim span)."""
    U,S,Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    sd = S/np.sqrt(max(n-1,1))
    pool=[]
    for _ in range(N_NULLS):
        Y = rng.normal(size=(n,len(sd)))*sd
        pool.append(bars(diagram(project(Y)), k))
    pool = np.concatenate([p for p in pool if len(p)]) if any(len(p) for p in pool) else np.array([0.0])
    return float(np.percentile(pool, q))

def cycle_density(g):
    """Independent cycles per node: (E - V + components) / V.

    The direct measure of how tree-like a graph is, and unlike mst_max_deg it is
    graded rather than a single integer. A tree has no independent cycles at all
    (0.0); a planar patch or lattice has many. This is what separates `tree` from
    `plane`, which both branch and are therefore identical under mst_max_deg.
    """
    import networkx as nx
    V, E = g.number_of_nodes(), g.number_of_edges()
    return (E - V + nx.number_connected_components(g)) / max(V, 1)


def local_dimension(P, k=6):
    """Mean and spread of the intrinsic dimension estimated in each point's own
    neighbourhood.

    A tree is locally 1-D everywhere (its branches are curves) with a few
    branch points; a plane or sphere is locally 2-D throughout. The global
    estimator cannot see that difference because it averages over the whole
    cloud, so a branchy 1-D object and a genuine surface both come out near 2.
    """
    n = len(P)
    k = min(k, n - 1)
    if k < 2:
        return np.nan, np.nan
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    dims = []
    for i in range(n):
        nb = P[np.argsort(D[i])[1:k + 1]]
        nb = nb - nb.mean(axis=0)
        if not np.isfinite(nb).all() or np.allclose(nb, 0):
            continue
        ev = np.linalg.svd(nb, compute_uv=False) ** 2
        if ev.sum() <= 0:
            continue
        # components needed for 90% of local variance
        dims.append(int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.90)) + 1)
    return (float(np.mean(dims)), float(np.std(dims))) if dims else (np.nan, np.nan)


def shape_features(X, seed=0, use_nulls=True):
    n = X.shape[0]
    P = project(X)
    dg = diagram(P)
    rng = np.random.default_rng(seed)

    L1, L2 = bars(dg,1), bars(dg,2)
    # Per-manifold nulls cost 2 * N_NULLS diagram computations each. They are
    # optional because the order test below is calibrated globally and is the
    # stricter gate anyway -- it rejects the spurious late-layer `years` loop
    # that the null threshold passes.
    if use_nulls:
        t1 = null_bar_threshold(X, n, rng, 1)
        t2 = null_bar_threshold(X, n, np.random.default_rng(seed + 1), 2)
    else:
        t1 = t2 = np.inf

    deg = mst_degrees(P)
    A = kneighbors_graph(P, min(K, n-1), mode='distance')
    g = nx.Graph(A.maximum(A.T))
    orc = OllivierRicci(g, alpha=0.5, proc=1, verbose='ERROR').compute_ricci_curvature()
    c = np.array([e[-1] for e in orc.edges(data='ricciCurvature')], float)

    ldm, lds = local_dimension(P)
    srt = np.sort(L1)[::-1]
    osc = order_score(P, n)
    op = order_p(osc, n)
    return {
        'id_proj':        intrinsic_dim(P, n),
        'b1_sig':         int((L1 > t1).sum()),      # loops above the null p95
        'b2_sig':         int((L2 > t2).sum()),      # voids -- needs maxdim=2
        'h1_gap':         float(srt[0]/srt[1]) if len(srt) > 1 and srt[1] > 0 else float(len(srt) == 1),
        'h1_top':         float(srt[0]) if len(srt) else 0.0,
        'h1_thresh':      t1,
        'order':          osc,
        'order_p':        op,
        'loop':           int(op < 0.01 and len(srt) > 0 and srt[0] > 0),
        # A circle is a loop whose MST is a PATH. The order test alone over-fires
        # on trees: `kinship` and `taxonomy` pass it because their value lists are
        # tree traversals, which project to a plausible angular sweep. Requiring
        # max degree 2 separates "closed cycle" from "branching object that
        # happens to sweep".
        'circle':         int(op < 0.01 and len(srt) > 0 and srt[0] > 0
                              and int(deg.max()) <= 2),
        'mst_max_deg':    int(deg.max()),
        'cyc_density':    cycle_density(g),
        'local_dim':      ldm,
        'local_dim_std':  lds,
        'mst_frac_leaf':  float((deg == 1).mean()),
        'mst_frac_deg2':  float((deg == 2).mean()),
        'curv_mean':      float(c.mean()), 'curv_std': float(c.std()),
        'curv_frac_neg':  float((c < 0).mean()),
        'n_points':       n,
    }

if __name__ == '__main__':
    V='../../outputs/activations/gemma-2-2b/variants'
    for c in ('seasons','months','chess_squares'):
        f=sorted(glob.glob(f'{V}/{c}__*.npz'))[0]
        X=np.load(f,allow_pickle=True)['activations'].astype(np.float64)
        t=time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            r=shape_features(X)
        print(f'{c:15s} n={X.shape[0]:3d} {time.time()-t:5.1f}s  '
              f"id={r['id_proj']:.2f} b1={r['b1_sig']} b2={r['b2_sig']} "
              f"mst_maxdeg={r['mst_max_deg']} frac_deg2={r['mst_frac_deg2']:.2f}")
