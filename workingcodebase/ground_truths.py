"""
Reference data: every ground-truth concept set with its prompt template and
expected structure. Import-only -- nothing here loads a model or runs.

Each entry: name -> (words, prompt_template, edges, cyclic)
  words     surface forms substituted into the template
  template  '...{}...' formatted with each word to build the prompt
  edges     index pairs encoding the expected ground-truth structure
  cyclic    True when the expected structure is a closed loop (S^1)

Expected structures (literature refs in the team notes):
  years, log_numbers, planets, chess_pieces      -- 1D chains
  days, months, colors, emotions, notes, fifths,
  hours, compass, seasons, vowels                -- S^1 cycles
  taxonomy, kinship                              -- trees (H1 = 0)
  chess                                          -- planar 8x8 lattice (H1 = 0)
  us_cities, global_cities, amino_acids, political, directions_3d
                                                 -- 2D/3D clouds / sphere / plane
  elements                                       -- atomic-number spiral
"""


def cycle(n):
    return [(i, (i + 1) % n) for i in range(n)]


def chain(n):
    return [(i, i + 1) for i in range(n - 1)]


# ---- 1D chains ----

YEARS = [str(y) for y in range(1700, 2021, 20)]
LOG_NUMBERS = ['1', '10', '100', '1000', '10000', '100000', '1000000']
PLANETS = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
CHESS_PIECES = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']

# ---- S^1 cycles ----

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
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
VOWELS = ['i', 'e', 'ɛ', 'a', 'ɑ', 'ɔ', 'o', 'u']  # peripheral cardinal loop

# ---- trees (H1 = 0) ----

TAXONOMY = ['animal', 'mammal', 'bird', 'fish', 'dog', 'cat', 'horse',
            'eagle', 'sparrow', 'penguin', 'shark', 'salmon', 'trout']
TAXONOMY_EDGES = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6),
                  (2, 7), (2, 8), (2, 9), (3, 10), (3, 11), (3, 12)]

KINSHIP = ['grandfather', 'grandmother', 'father', 'mother', 'uncle', 'aunt',
           'brother', 'sister', 'cousin', 'son', 'daughter', 'nephew', 'niece',
           'grandson', 'granddaughter']
KINSHIP_EDGES = [(0, 2), (1, 3), (2, 9), (3, 10), (2, 4), (3, 5), (4, 8), (5, 8),
                 (6, 7), (6, 11), (7, 12), (9, 13), (10, 14)]

# ---- planar 8x8 lattice ----

CHESS_FILES, CHESS_RANKS = 'ABCDEFGH', range(1, 9)
CHESS = [f'{f}{r}' for f in CHESS_FILES for r in CHESS_RANKS]


def chess_grid_edges():
    """Rank/file adjacency of the 8x8 board, indices into CHESS (file-major)."""
    idx = {sq: i for i, sq in enumerate(CHESS)}
    edges = []
    for fi, f in enumerate(CHESS_FILES):
        for r in CHESS_RANKS:
            if r < 8:
                edges.append((idx[f'{f}{r}'], idx[f'{f}{r + 1}']))
            if fi < 7:
                edges.append((idx[f'{f}{r}'], idx[f'{CHESS_FILES[fi + 1]}{r}']))
    return edges


# ---- 2D/3D clouds / sphere / plane ----

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
DIRECTIONS_3D = ['forward', 'backward', 'left', 'right',
                 'up', 'down', 'zenith', 'nadir']

# ---- atomic-number spiral ----

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


GROUND_TRUTHS = {
    'years':         (YEARS, 'In the year {}', chain(len(YEARS)), False),
    'log_numbers':   (LOG_NUMBERS, 'The quantity is {}', chain(7), False),
    'planets':       (PLANETS, 'The planet in the solar system is {}', chain(8), False),
    'chess_pieces':  (CHESS_PIECES, 'The chess piece is the {}', chain(6), False),
    'days':          (DAYS, 'The day of the week is {}', cycle(7), True),
    'months':        (MONTHS, 'The month of the year is {}', cycle(12), True),
    'colors':        (COLORS, 'The color of the object is {}', cycle(12), True),
    'emotions':      (EMOTIONS, 'The emotion they are feeling is {}', cycle(12), True),
    'notes':         (NOTES, 'The musical note being played is {}', cycle(12), True),
    'fifths':        (FIFTHS, 'The song is written in the key of {} major', cycle(12), True),
    'hours':         (HOURS, 'The time of day is {}', cycle(24), True),
    'compass':       (COMPASS, 'The compass direction is {}', cycle(8), True),
    'seasons':       (SEASONS, 'The season of the year is {}', cycle(4), True),
    'vowels':        (VOWELS, 'The vowel sound is pronounced "{}"', cycle(8), True),
    'taxonomy':      (TAXONOMY, 'The concept being discussed is the {}', TAXONOMY_EDGES, False),
    'kinship':       (KINSHIP, 'Their family relation is the {}', KINSHIP_EDGES, False),
    'chess':         (CHESS, 'The chess board square is {}', chess_grid_edges(), False),
    'us_cities':     (US_CITIES, 'The location of the city {}', [], False),
    'global_cities': (GLOBAL_CITIES, 'The location of the city {}', [], False),
    'amino_acids':   (AMINO_ACIDS, 'The amino acid in the protein is {}', [], False),
    'political':     (POLITICAL, 'The political views of {}', [], False),
    'directions_3d': (DIRECTIONS_3D, 'The direction of movement is {}', [], False),
    'elements':      (ELEMENTS, 'The chemical element is {}', ELEMENT_EDGES, False),
}
