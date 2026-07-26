"""
Extract gemma-2-2b layer-6 final-token activations for three concept sets and
save 2D + 3D PCA plots for each (6 plots total, in plots/):

  days    -- expected ring   (7 points, cyclic)
  months  -- expected ring   (12 points, cyclic)
  chess   -- expected planar 8x8 lattice, H1 = 0 (64 points, grid)

Ground-truth structure is drawn as light edges: cyclic neighbors for days and
months, rank/file adjacency for chess.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pipeline_draft import Pipeline

LAYER = 'layer_6'   # chosen by both our topology-vs-null metric and the paper
OUTDIR = 'plots'

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
FILES, RANKS = 'ABCDEFGH', range(1, 9)
CHESS = [f'{f}{r}' for f in FILES for r in RANKS]


def cyclic_edges(n):
    return [(i, (i + 1) % n) for i in range(n)]


def grid_edges():
    """Rank/file adjacency of the 8x8 board, indices into CHESS (file-major)."""
    idx = {sq: i for i, sq in enumerate(CHESS)}
    edges = []
    for fi, f in enumerate(FILES):
        for r in RANKS:
            if r < 8:
                edges.append((idx[f'{f}{r}'], idx[f'{f}{r + 1}']))
            if fi < 7:
                edges.append((idx[f'{f}{r}'], idx[f'{FILES[fi + 1]}{r}']))
    return edges


CONCEPTS = {
    'days':   (DAYS,   'The day of the week is {}',    cyclic_edges(len(DAYS))),
    'months': (MONTHS, 'The month of the year is {}',  cyclic_edges(len(MONTHS))),
    'chess':  (CHESS,  'The chess board square is {}', grid_edges()),
}


def plot_pca(X, labels, edges, colors, path, title, dim):
    fig = plt.figure(figsize=(9, 8))
    if dim == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()

    for i, j in edges:
        seg = X[[i, j]]
        ax.plot(*seg.T, color='gray', alpha=0.35, lw=0.8, zorder=1)

    ax.scatter(*X.T, c=colors, cmap='hsv' if len(labels) < 20 else 'viridis',
               s=45, zorder=2)
    fontsize = 9 if len(labels) < 20 else 6
    for point, lab in zip(X, labels):
        if dim == 3:
            ax.text(*point, lab, fontsize=fontsize, alpha=0.85)
        else:
            ax.annotate(lab, point, fontsize=fontsize, alpha=0.85,
                        textcoords='offset points', xytext=(4, 4))

    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    if dim == 3:
        ax.set_zlabel('PC3')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {path}')


if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)

    pipeline = None
    for name, (words, template, edges) in CONCEPTS.items():
        prompts = [template.format(w) for w in words]
        if pipeline is None:
            pipeline = Pipeline(prompts)      # loads the model once
        else:
            pipeline.pos_prompts = prompts

        act = pipeline.collect_activations()[LAYER]

        pca = PCA(n_components=3)
        X3 = pca.fit_transform(act)
        evr = pca.explained_variance_ratio_
        print(f'{name}: n={len(words)}, top-3 PC variance = '
              f'{evr[0]:.2f}/{evr[1]:.2f}/{evr[2]:.2f}')

        labels = [w[:3] for w in words] if name != 'chess' else words
        colors = np.arange(len(words))     # cyclic position / file-major index

        plot_pca(X3[:, :2], labels, edges, colors,
                 f'{OUTDIR}/{name}_pca2d.png',
                 f'{name} -- gemma-2-2b {LAYER} (PC1-2: {evr[:2].sum():.0%} var)', 2)
        plot_pca(X3, labels, edges, colors,
                 f'{OUTDIR}/{name}_pca3d.png',
                 f'{name} -- gemma-2-2b {LAYER} (PC1-3: {evr.sum():.0%} var)', 3)

    print('done')
