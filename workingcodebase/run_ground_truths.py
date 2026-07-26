"""
Extract gemma-2-2b layer-6 final-token activations for 17 ground-truth concept
sets and save 2D + 3D PCA plots for each (34 plots, in plots/).

Expected ground truths (see literature refs in the team notes):
  years, log_numbers, planets, chess_pieces  -- 1D chains
  colors, emotions, notes, fifths, hours, compass, seasons -- S^1 cycles
  taxonomy, kinship                          -- trees (H1 = 0)
  us_cities, global_cities, amino_acids, political, vowels, directions_3d
                                             -- 2D/3D clouds / sphere / plane

Activations are cached per concept in acts/ so an interrupted run resumes
without re-running finished concepts.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pipeline_draft import Pipeline

LAYER = 'layer_6'
OUTDIR, ACTDIR = 'plots', 'acts'


def cycle(n):
    return [(i, (i + 1) % n) for i in range(n)]


def chain(n):
    return [(i, i + 1) for i in range(n - 1)]


# ---- concept definitions: name -> (words, prompt template, edges, cyclic) ----

YEARS = [str(y) for y in range(1700, 2021, 20)]
LOG_NUMBERS = ['1', '10', '100', '1000', '10000', '100000', '1000000']
PLANETS = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
CHESS_PIECES = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']

COLORS = ['red', 'orange', 'yellow', 'lime', 'green', 'teal',
          'cyan', 'blue', 'indigo', 'purple', 'magenta', 'pink']
EMOTIONS = ['happy', 'delighted', 'excited', 'tense', 'angry', 'frustrated',
            'sad', 'depressed', 'bored', 'tired', 'calm', 'relaxed']  # circumplex order
NOTES = ['C', 'C sharp', 'D', 'D sharp', 'E', 'F',
         'F sharp', 'G', 'G sharp', 'A', 'A sharp', 'B']
FIFTHS = ['C', 'G', 'D', 'A', 'E', 'B',
          'F sharp', 'D flat', 'A flat', 'E flat', 'B flat', 'F']
HOURS = [f'{h}:00' for h in range(24)]
COMPASS = ['north', 'north-east', 'east', 'south-east',
           'south', 'south-west', 'west', 'north-west']
SEASONS = ['spring', 'summer', 'autumn', 'winter']

TAXONOMY = ['animal', 'mammal', 'bird', 'fish', 'dog', 'cat', 'horse',
            'eagle', 'sparrow', 'penguin', 'shark', 'salmon', 'trout']
TAXONOMY_EDGES = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6),
                  (2, 7), (2, 8), (2, 9), (3, 10), (3, 11), (3, 12)]

KINSHIP = ['grandfather', 'grandmother', 'father', 'mother', 'uncle', 'aunt',
           'brother', 'sister', 'cousin', 'son', 'daughter', 'nephew', 'niece',
           'grandson', 'granddaughter']
KINSHIP_EDGES = [(0, 2), (1, 3), (2, 9), (3, 10), (2, 4), (3, 5), (4, 8), (5, 8),
                 (6, 7), (6, 11), (7, 12), (9, 13), (10, 14)]

US_CITIES = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
             'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'Seattle',
             'Denver', 'Boston', 'Miami', 'Atlanta', 'Detroit', 'Minneapolis',
             'New Orleans', 'Las Vegas', 'Portland', 'Kansas City']
GLOBAL_CITIES = ['Tokyo', 'London', 'Sydney', 'Cairo', 'Buenos Aires', 'Honolulu',
                 'New York', 'Cape Town', 'Moscow', 'Beijing', 'Mumbai', 'Lagos',
                 'Rio de Janeiro', 'Mexico City', 'Reykjavik', 'Singapore']
AMINO_ACIDS = ['Alanine', 'Arginine', 'Asparagine', 'Aspartate', 'Cysteine',
               'Glutamine', 'Glutamate', 'Glycine', 'Histidine', 'Isoleucine',
               'Leucine', 'Lysine', 'Methionine', 'Phenylalanine', 'Proline',
               'Serine', 'Threonine', 'Tryptophan', 'Tyrosine', 'Valine']
POLITICAL = ['Bernie Sanders', 'Elizabeth Warren', 'Alexandria Ocasio-Cortez',
             'Joe Biden', 'Barack Obama', 'Bill Clinton', 'Joe Manchin',
             'Mitt Romney', 'John McCain', 'George W. Bush', 'Ted Cruz',
             'Donald Trump']  # roughly ordered left -> right (DW-NOMINATE style)
VOWELS = ['i', 'e', 'ɛ', 'a', 'ɑ', 'ɔ', 'o', 'u']  # peripheral cardinal loop
DIRECTIONS_3D = ['forward', 'backward', 'left', 'right',
                 'up', 'down', 'zenith', 'nadir']

ELEMENTS = ['Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron', 'Carbon',
            'Nitrogen', 'Oxygen', 'Fluorine', 'Neon', 'Sodium', 'Magnesium',
            'Aluminium', 'Silicon', 'Phosphorus', 'Sulfur', 'Chlorine', 'Argon',
            'Potassium', 'Calcium', 'Scandium', 'Titanium', 'Vanadium',
            'Chromium', 'Manganese', 'Iron', 'Cobalt', 'Nickel', 'Copper',
            'Zinc', 'Gallium', 'Germanium', 'Arsenic', 'Selenium', 'Bromine',
            'Krypton']
# atomic-number chain + same-group links (the spiral's rungs)
ELEMENT_EDGES = chain(len(ELEMENTS)) + [
    (0, 2), (2, 10), (10, 18),   # H-Li-Na-K
    (3, 11), (11, 19),           # Be-Mg-Ca
    (4, 12), (12, 30),           # B-Al-Ga
    (5, 13), (13, 31),           # C-Si-Ge
    (6, 14), (14, 32),           # N-P-As
    (7, 15), (15, 33),           # O-S-Se
    (8, 16), (16, 34),           # F-Cl-Br
    (1, 9), (9, 17), (17, 35),   # He-Ne-Ar-Kr
]

CONCEPTS = {
    'years':         (YEARS, 'In the year {}', chain(len(YEARS)), False),
    'log_numbers':   (LOG_NUMBERS, 'The quantity is {}', chain(7), False),
    'planets':       (PLANETS, 'The planet in the solar system is {}', chain(8), False),
    'chess_pieces':  (CHESS_PIECES, 'The chess piece is the {}', chain(6), False),
    'colors':        (COLORS, 'The color of the object is {}', cycle(12), True),
    'emotions':      (EMOTIONS, 'The emotion they are feeling is {}', cycle(12), True),
    'notes':         (NOTES, 'The musical note being played is {}', cycle(12), True),
    'fifths':        (FIFTHS, 'The song is written in the key of {} major', cycle(12), True),
    'hours':         (HOURS, 'The time of day is {}', cycle(24), True),
    'compass':       (COMPASS, 'The compass direction is {}', cycle(8), True),
    'seasons':       (SEASONS, 'The season of the year is {}', cycle(4), True),
    'taxonomy':      (TAXONOMY, 'The concept being discussed is the {}', TAXONOMY_EDGES, False),
    'kinship':       (KINSHIP, 'Their family relation is the {}', KINSHIP_EDGES, False),
    'us_cities':     (US_CITIES, 'The location of the city {}', [], False),
    'global_cities': (GLOBAL_CITIES, 'The location of the city {}', [], False),
    'amino_acids':   (AMINO_ACIDS, 'The amino acid in the protein is {}', [], False),
    'political':     (POLITICAL, 'The political views of {}', [], False),
    'vowels':        (VOWELS, 'The vowel sound is pronounced "{}"', cycle(8), True),
    'directions_3d': (DIRECTIONS_3D, 'The direction of movement is {}', [], False),
    'elements':      (ELEMENTS, 'The chemical element is {}', ELEMENT_EDGES, False),
}


def plot_pca(X, labels, edges, colors, cmap, path, title, dim):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection='3d') if dim == 3 else fig.add_subplot()

    for i, j in edges:
        ax.plot(*X[[i, j]].T, color='gray', alpha=0.35, lw=0.8, zorder=1)

    ax.scatter(*X.T, c=colors, cmap=cmap, s=45, zorder=2)
    fontsize = 9 if len(labels) <= 20 else 6
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
    os.makedirs(ACTDIR, exist_ok=True)

    pipeline = None
    for name, (words, template, edges, cyclic) in CONCEPTS.items():
        cache = f'{ACTDIR}/{name}.npy'
        if os.path.exists(cache):
            act = np.load(cache)
            print(f'{name}: loaded cached activations {act.shape}')
        else:
            prompts = [template.format(w) for w in words]
            if pipeline is None:
                pipeline = Pipeline(prompts)      # loads the model once
            else:
                pipeline.pos_prompts = prompts
            act = pipeline.collect_activations()[LAYER]
            np.save(cache, act)

        n_comp = min(3, len(words) - 1)
        pca = PCA(n_components=n_comp)
        X3 = pca.fit_transform(act)
        evr = pca.explained_variance_ratio_
        print(f'{name}: n={len(words)}, top PC variance = '
              + '/'.join(f'{v:.2f}' for v in evr))

        labels = words
        colors = np.arange(len(words))
        cmap = 'hsv' if cyclic else 'viridis'

        plot_pca(X3[:, :2], labels, edges, colors, cmap,
                 f'{OUTDIR}/{name}_pca2d.png',
                 f'{name} -- gemma-2-2b {LAYER} (PC1-2: {evr[:2].sum():.0%} var)', 2)
        if n_comp == 3:
            X3d = X3
        else:  # seasons has only 3 points' worth of rank; pad for a flat 3D view
            X3d = np.column_stack([X3, np.zeros(len(words))])
        plot_pca(X3d, labels, edges, colors, cmap,
                 f'{OUTDIR}/{name}_pca3d.png',
                 f'{name} -- gemma-2-2b {LAYER} (top PCs: {evr.sum():.0%} var)', 3)

    print('done')
