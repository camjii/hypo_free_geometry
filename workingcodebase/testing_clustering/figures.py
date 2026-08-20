"""The three paper figures. Self-contained: reads shape_features.csv and the
activation cache directly, so it survives the notebook being overwritten.

    from figures import fig_loop_significance, fig_positive_control, fig_method_vs_baseline
    fig_loop_significance(); fig_positive_control(); fig_method_vs_baseline()

Each figure answers one question a reviewer will ask, in order: does any concept
have a loop its own noise cannot explain; is a null result a blind detector or an
absent signal; and does the topological signature beat using no topology at all.
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ripser import ripser
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VARIANTS = os.path.join(REPO, 'outputs/activations/gemma-2-2b/variants')
CACHE = os.path.join(HERE, 'shape_features.csv')
SEED, N_GROUPS = 0, 6

FEATURES = ['id_proj', 'order', 'loop', 'h1_gap', 'mst_max_deg',
            'mst_frac_leaf', 'mst_frac_deg2',
            'curv_mean', 'curv_std', 'curv_frac_neg']

SURFACE, INK, INK_MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#dcdcd8'
S1, S2, MUTED = '#2a78d6', '#eb6834', '#c2c6c9'

variant_index = lambda s: int(re.search(r'_v(\d+)$', s).group(1))


def load():
    return pd.read_csv(CACHE)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _group(D, k):
    return fcluster(linkage(squareform(D, checks=False), 'average'), k, 'maxclust')


def _D_topology(sub):
    X = StandardScaler().fit_transform(
        np.asarray(sub.groupby('concept')[FEATURES].mean().values, float))
    return squareform(pdist(X))


def _D_centroid(sub, _cache={}):
    """No topology at all -- just where each concept's mean activation points."""
    if not _cache:
        for f in sorted(glob.glob(os.path.join(VARIANTS, '*.npz'))):
            s = os.path.basename(f)[:-4]
            _cache[(s.split('__')[0], s.split('__')[1].rsplit('_layer', 1)[0])] = \
                np.load(f, allow_pickle=True)['activations'].astype(np.float64)
    cs = sorted(sub.concept.unique())
    M = np.stack([np.vstack([_cache[(c, v)] for v in sub[sub.concept == c].variant]).mean(0)
                  for c in cs])
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    return np.clip(1.0 - M @ M.T, 0, None)


def _D_shuffled(sub, seed):
    """Floor. Each half gets its OWN permutation, or the shuffle cancels."""
    s = sub.groupby('concept')[FEATURES].mean()
    X = StandardScaler().fit_transform(
        np.random.default_rng(seed).permutation(np.asarray(s.values, float)))
    return squareform(pdist(X))


def fig_loop_significance(df=None, save=None):
    """FIGURE 1 -- loop structure against each concept's own null.

    x is the p95 of that concept's matched-null H1 bars, y its longest observed
    bar, so a point above the identity line has a loop its own noise cannot
    explain. Every circularity claim reduces to which side of the line a point
    sits on, which the reader can check directly.
    """
    df = load() if df is None else df
    g = df.groupby('concept')
    obs, thr, fam = g['h1_top'].mean(), g['h1_thresh'].mean(), g['family'].first()
    sig = obs > thr

    fig, ax = plt.subplots(figsize=(7.6, 7), facecolor=SURFACE)
    _style(ax)
    hi = max(obs.max(), thr.max()) * 1.12
    ax.plot([0, hi], [0, hi], color=INK, lw=1.2, ls='--', zorder=2)
    ax.fill_between([0, hi], [0, hi], [hi, hi], color=S2, alpha=0.06, zorder=0)
    ax.text(hi * 0.60, hi * 0.655, 'loop = own noise', fontsize=8.5,
            color=INK_MUTED, rotation=39)

    for flag, colour, marker, lab in ((False, S1, 'o', 'no significant loop'),
                                      (True, S2, 's', 'loop beats own null')):
        m = sig.values == flag
        ax.scatter(thr[m], obs[m], s=74, c=colour, marker=marker,
                   edgecolors=SURFACE, linewidths=1.6, zorder=3, label=lab)

    placed = []                       # nudge labels apart when two land together
    for c in obs.index[sig.values]:
        dy = 2
        while any(abs(obs[c] - py) < hi * 0.04 and abs(dy - d) < 12 for py, d in placed):
            dy += 13
        placed.append((obs[c], dy))
        ax.annotate(f'{c}  [{fam[c]}]', (thr[c], obs[c]), fontsize=8.5, color=INK,
                    xytext=(9, dy), textcoords='offset points', zorder=4)

    ax.set_xlabel("matched-null H1 bar length, p95", fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel('longest observed H1 bar', fontsize=9.5, color=INK_MUTED)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK)
    nc = int((fam == 'circle').sum()); ns = int(((fam == 'circle') & sig).sum())
    ax.set_title("Loop structure against each concept's own null\n"
                 f'{ns} of {nc} concepts labelled "circle" clear their null',
                 fontsize=12, color=INK)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.show()


def fig_positive_control(df=None, save=None):
    """FIGURE 2 -- synthetic circles through the identical measurement path.

    If the method finds a 12-point circle at 0.61 while the corpus sits near
    0.02, the detector is not the limiting factor, which is what makes Figure 1
    a claim about the representations rather than about the tool.
    """
    df = load() if df is None else df
    rng = np.random.default_rng(SEED)

    def maxh1(P):
        P = P / pdist(P).max()
        d = np.asarray(ripser(P, maxdim=1)['dgms'][1])
        return float((d[:, 1] - d[:, 0]).max()) if d.size else 0.0

    ns = [4, 6, 8, 10, 12, 16, 20, 24, 32]
    clean = [maxh1(np.c_[np.cos(t), np.sin(t)])
             for t in (np.linspace(0, 2 * np.pi, n, endpoint=False) for n in ns)]
    noises = np.linspace(0, 0.6, 9)
    t12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    base = np.c_[np.cos(t12), np.sin(t12)]
    noisy = [np.mean([maxh1(base + rng.normal(0, s, base.shape)) for _ in range(12)])
             for s in noises]

    real = df.groupby('concept')['h1_top'].mean()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE)
    for a in (ax, bx):
        _style(a)
        a.axhspan(real.min(), real.max(), color=MUTED, alpha=0.5, zorder=0)
        a.axhline(real.median(), color=INK_MUTED, lw=1.1, ls=':', zorder=1)

    ax.plot(ns, clean, color=S1, lw=2, marker='o', ms=5, zorder=3)
    ax.set_xlabel('points on a perfect circle', fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel('longest H1 bar', fontsize=9.5, color=INK_MUTED)
    ax.set_title('Sensitivity vs sample size', fontsize=11, color=INK)
    ax.text(ns[-1], real.median(), ' corpus median ', fontsize=8,
            color=INK_MUTED, ha='right', va='bottom')

    bx.plot(noises, noisy, color=S2, lw=2, marker='s', ms=5, zorder=3)
    bx.set_xlabel('noise SD added to a 12-point circle', fontsize=9.5, color=INK_MUTED)
    bx.set_ylabel('longest H1 bar', fontsize=9.5, color=INK_MUTED)
    bx.set_title('Sensitivity vs noise', fontsize=11, color=INK)
    fig.suptitle('Positive control: grey band spans the 24 measured concepts',
                 fontsize=12, color=INK)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.show()


def fig_method_vs_baseline(df=None, ks=range(3, 11), save=None):
    """FIGURE 3 -- split-half reproducibility across every k.

    The largest-cluster annotation is load-bearing: a partition of 18-plus-
    singletons reproduces easily while grouping nothing, so a high ARI without
    that number next to it is not interpretable.
    """
    df = load() if df is None else df
    v = df.variant.map(variant_index)
    lo, hi = df[v <= 5], df[v > 5]

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    _style(ax)
    for name, fn, colour, mk in (('topological signature', _D_topology, S1, 'o'),
                                 ('centroid cosine (no topology)', _D_centroid, S2, 's')):
        Da, Db = fn(lo), fn(hi)
        ax.plot(list(ks), [adjusted_rand_score(_group(Da, k), _group(Db, k)) for k in ks],
                color=colour, lw=2, marker=mk, ms=5, label=name)
    sa, sb = _D_shuffled(lo, 1), _D_shuffled(hi, 2)
    ax.plot(list(ks), [adjusted_rand_score(_group(sa, k), _group(sb, k)) for k in ks],
            color=INK_MUTED, lw=1.4, ls='--', label='shuffled features (floor)')

    full = _D_topology(df)
    for k in ks:
        ax.annotate(f'{max(np.bincount(_group(full, k))[1:])}/24', (k, -0.12),
                    fontsize=7.5, color=INK_MUTED, ha='center')
    ax.annotate('largest cluster:', (list(ks)[0] - 0.55, -0.12), fontsize=7.5,
                color=INK_MUTED, ha='right')
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel('number of groups (k)', fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel('split-half ARI', fontsize=9.5, color=INK_MUTED)
    ax.set_title('Reproducibility across disjoint prompt-template halves',
                 fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.show()


def fig_point_cloud(df=None, k=N_GROUPS, save=None):
    """The clustering as a point cloud: what the algorithm is actually looking at.

    PCA of the ten diagnostic features, fitted on all 240 manifolds so the small
    points (individual prompt templates) and the large ones (concept means) share
    a space. Reading it: the spread around a labelled point is how much rewording
    the prompt moves that concept; the distance between labelled points is what
    the clustering operates on.

    Colour marks the split the algorithm actually finds -- one dense undivided
    mass plus a few separated concepts -- rather than assigning a hue per cluster,
    which would imply the mass is subdivided when it is not.
    """
    df = load() if df is None else df
    X = StandardScaler().fit_transform(np.asarray(df[FEATURES].values, float))
    emb = PCA(n_components=2).fit_transform(X)

    sig = df.groupby('concept')[FEATURES].mean()
    lab = pd.Series(_group(_D_topology(df), k), index=sig.index)
    biggest = lab.value_counts().idxmax()
    outlier = set(lab.index[lab != biggest])

    e = pd.DataFrame(emb, columns=['x', 'y'])
    e['concept'] = df.concept.values

    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)

    for c, gp in e.groupby('concept'):
        out = c in outlier
        colour = S2 if out else MUTED
        cx, cy = gp.x.mean(), gp.y.mean()
        ax.scatter(gp.x, gp.y, s=16, color=colour, alpha=0.45, linewidths=0, zorder=2)
        ax.scatter([cx], [cy], s=110 if out else 62, color=S2 if out else S1,
                   edgecolors=SURFACE, linewidths=1.8, zorder=4,
                   marker='s' if out else 'o')
        ax.annotate(c, (cx, cy), fontsize=8.5 if out else 7.5,
                    color=INK if out else INK_MUTED, zorder=5,
                    xytext=(0, 11 if out else 9), textcoords='offset points',
                    ha='center', weight='bold' if out else 'normal')

    handles = [
        plt.Line2D([], [], marker='o', ls='', color=S1, ms=8,
                   label=f'in the undivided mass ({int((lab == biggest).sum())} concepts)'),
        plt.Line2D([], [], marker='s', ls='', color=S2, ms=8,
                   label=f'separated by the clustering ({len(outlier)} concepts)'),
        plt.Line2D([], [], marker='o', ls='', color=MUTED, ms=5,
                   label='individual prompt templates'),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK, loc='best')
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_xlabel('PC1 of the diagnostic features', fontsize=9.5, color=INK_MUTED)
    ax.set_ylabel('PC2', fontsize=9.5, color=INK_MUTED)
    ax.set_title('24 concepts and their 240 manifolds in feature space',
                 fontsize=12, color=INK)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, facecolor=SURFACE, bbox_inches='tight')
    plt.show()
