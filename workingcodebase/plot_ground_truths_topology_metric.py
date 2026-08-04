"""
Plot every ground-truth concept's topology metric as a point in 3D metric
space, in three side-by-side 3D scatter panels. Two axes are shared:

    x = dimension   |ID_concept - ID_null|
    z = curvature   distribution_distance * (frac_negative_difference - 0.25)

Only the topology (y) axis differs between panels:

    panel 1: H0 bottleneck vs null
    panel 2: H1 bottleneck vs null
    panel 3: min(H0, H1) bottleneck vs null

Points are colored by the concept's expected ground-truth structure class and
each point is labeled with its concept name, so identity never rides on color
alone.

Reads ground_truth_topology_metrics.json (written by
run_ground_truths_topology_metric.py) -- no model, no recomputation.
Usage: python plot_ground_truths_topology_metric.py [results.json] [out.png]
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ground_truths import GROUND_TRUTHS

RESULTS_PATH = 'ground_truth_topology_metrics.json'
OUTPATH = 'plots/ground_truth_topology_metrics_3d.png'

# chart chrome (light surface)
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
MUTED = '#898781'
GRID = '#e1e0d9'

# expected structure class per concept (see ground_truths.py docstring)
STRUCTURE = {
    'years': 'chain', 'log_numbers': 'chain', 'planets': 'chain',
    'chess_pieces': 'chain',
    'days': 'cycle', 'months': 'cycle', 'colors': 'cycle', 'emotions': 'cycle',
    'notes': 'cycle', 'fifths': 'cycle', 'hours': 'cycle', 'compass': 'cycle',
    'seasons': 'cycle', 'vowels': 'cycle',
    'taxonomy': 'tree', 'kinship': 'tree',
    'chess': 'lattice',
    'us_cities': 'cloud', 'global_cities': 'cloud', 'amino_acids': 'cloud',
    'political': 'cloud', 'directions_3d': 'cloud',
    'elements': 'spiral',
}

CLASS_COLORS = {          # fixed categorical slot order, never cycled
    'chain':   '#2a78d6',
    'cycle':   '#eb6834',
    'tree':    '#1baf7a',
    'lattice': '#eda100',
    'cloud':   '#e87ba4',
    'spiral':  '#008300',
}

TOPOLOGY_AXES = [
    ('H0_bottleneck',  'H0 bottleneck vs null'),
    ('H1_bottleneck',  'H1 bottleneck vs null'),
    ('min_bottleneck', 'min(H0, H1) bottleneck vs null'),
]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(SURFACE)
        axis.pane.set_edgecolor(GRID)
        axis._axinfo['grid']['color'] = GRID
        axis._axinfo['grid']['linewidth'] = 0.6
    ax.tick_params(colors=MUTED, labelsize=7)


def main(results_path=RESULTS_PATH, outpath=OUTPATH):
    if not os.path.exists(results_path):
        sys.exit(f'{results_path} not found -- run '
                 'run_ground_truths_topology_metric.py first')
    with open(results_path) as f:
        results = json.load(f)

    missing = [n for n in GROUND_TRUTHS if n not in results]
    if missing:
        print(f'warning: no metrics for {", ".join(missing)} -- plotting the rest')
    unknown = [n for n in results if n not in STRUCTURE]
    if unknown:
        sys.exit(f'no structure class for: {", ".join(unknown)} -- add to STRUCTURE')

    names = list(results)
    dims = [results[n]['dimension'] for n in names]
    curvs = [results[n]['curvature'] for n in names]
    colors = [CLASS_COLORS[STRUCTURE[n]] for n in names]

    fig = plt.figure(figsize=(19, 7))
    fig.patch.set_facecolor(SURFACE)

    for panel, (key, axis_label) in enumerate(TOPOLOGY_AXES, start=1):
        topos = [results[n]['topology'][key] for n in names]

        ax = fig.add_subplot(1, 3, panel, projection='3d')
        style_axes(ax)
        ax.scatter(dims, topos, curvs, c=colors, s=55, alpha=0.9,
                   edgecolors=SURFACE, linewidths=1)
        for x, y, z, name in zip(dims, topos, curvs, names):
            ax.text(x, y, z, f' {name}', fontsize=6.5, color=INK, alpha=0.85)

        ax.set_xlabel('dimension  |ID − ID null|', fontsize=8, color=MUTED)
        ax.set_ylabel(f'topology  {axis_label}', fontsize=8, color=MUTED)
        ax.set_zlabel('curvature vs null', fontsize=8, color=MUTED)
        ax.set_title(axis_label, fontsize=10, color=INK)

    handles = [Line2D([], [], linestyle='', marker='o', markersize=8,
                      markerfacecolor=c, markeredgecolor=SURFACE, label=cls)
               for cls, c in CLASS_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=len(CLASS_COLORS),
               frameon=False, fontsize=9, labelcolor=INK,
               title='expected ground-truth structure', title_fontsize=9)

    fig.suptitle('Ground-truth concepts in topology-metric space '
                 '(gemma-2-2b layer 6, vs noise null)', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {outpath}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
