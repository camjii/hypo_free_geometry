"""Consolidate every concept source into a single ``concept_sets.json``.

Values currently live in three places that partly overlap: the curated
``concept_families.FAMILIES`` table, the built-in ``PROMPT_SETS`` in
``prompt_templates`` (days, months, chess squares), and the mathematical
series under ``ground_truth_verification/``.  The topology pipeline needs one
file, so this script merges them and fails loudly on any disagreement rather
than silently preferring one source.

Family labels are carried over from the existing ``expected`` strings via an
explicit table below.  Sets whose family has never been decided are emitted
with ``"family": null`` and reported, so an unreviewed set can never be
mistaken for a reviewed one.
"""

# ruff: noqa: E402  (sys.path setup must precede sibling imports)

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from concept_families import FAMILIES
from proposed_families import PROPOSED
from prompt_templates import PROMPT_SETS

OUT = REPO / "concept_sets.json"
SERIES_FILE = REPO / "ground_truth_verification" / "complex_series_50_each.json"

# Family label for each curated set, read off the existing expected-topology
# string.  This preserves the earlier judgement instead of re-deciding it.
FAMILY_BY_SET = {
    "years": "line",
    "quantities": "line",
    "planets": "line",
    "chess_pieces": "line",
    "colors": "circle",
    "emotions": "circle",
    "musical_notes": "circle",
    "circle_of_fifths": "circle",
    "clock_hours": "circle",
    "compass": "circle",
    "seasons": "circle",
    "days": "circle",
    "months": "circle",
    "amino_acids": "plane",
    "cities_us": "plane",
    "vowels": "plane",
    "political": "plane",
    "cities_global": "sphere",
    "directions_3d": "sphere",
    "taxonomy": "tree",
    "kinship": "tree",
    "elements": "spiral",
    "chess_squares": "grid",
    # The prompt carries a series value but never its index, so the model is
    # shown a bare number and these sample a magnitude line rather than any
    # sequence structure.  See notes in expectations_v2.json.
    "apery_numbers": "line",
    "bell_numbers": "line",
    "catalan_numbers": "line",
    "fibonacci_numbers": "line",
    "lucas_numbers": "line",
    "partition_numbers": "line",
    "prime_numbers": "line",
    "ramanujan_tau": "line",
    "recaman_sequence": "line",
    "riemann_zeta_zero_ordinates": "line",
    # Two dates per month across one year: the annual cycle is sampled, the
    # within-month coordinate is not, so this is a circle and not a torus.
    "calendar_dates": "circle",
}

# Expected-topology strings for sets that predate concept_families.FAMILIES.
EXPECTED_BY_SET = {
    "days": "weekday circle (S1)",
    "months": "calendar circle (S1)",
    "chess_squares": "8x8 grid (2D)",
    "calendar_dates": "annual circle (S1)",
    "apery_numbers": "1D magnitude line",
    "bell_numbers": "1D magnitude line",
    "catalan_numbers": "1D magnitude line",
    "fibonacci_numbers": "1D magnitude line",
    "lucas_numbers": "1D magnitude line",
    "partition_numbers": "1D magnitude line",
    "prime_numbers": "1D magnitude line",
    "ramanujan_tau": "1D magnitude line",
    "recaman_sequence": "1D magnitude line",
    "riemann_zeta_zero_ordinates": "1D magnitude line",
}


def _add(
    sets, name, values, *, kind, source, expected=None, notes=None, math_series=False
):
    """Register one set, refusing to overwrite a name from another source.

    Repeated values render to identical prompts and therefore to coincident
    activations, whose zero nearest-neighbour distance divides by zero in the
    TwoNN ratio.  In a curated set that is a curation bug and hard-fails here.
    In a mathematical series it is a property of the sequence (F_1 = F_2 = 1),
    so the later occurrence is dropped and the discarded indices recorded: the
    prompt carries only the value, never the index, so a repeat contributes no
    information a single occurrence does not already carry.
    """
    values = [str(value).strip() for value in values]
    if not values:
        raise ValueError(f"{name}: empty value list")
    if any(not value for value in values):
        raise ValueError(f"{name}: blank value")
    if name in sets:
        raise ValueError(f"{name}: defined by more than one source")

    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates and not math_series:
        raise ValueError(f"{name}: duplicate values {duplicates}")

    dropped = {}
    if duplicates:
        seen, kept = set(), []
        for index, value in enumerate(values):
            if value in seen:
                dropped[str(index)] = value
                continue
            seen.add(value)
            kept.append(value)
        values = kept

    entry = {
        "family": FAMILY_BY_SET.get(name),
        "value_kind": kind,
        "n_values": len(values),
        "expected_topology": expected,
        "source": source,
        "values": values,
    }
    if dropped:
        entry["dropped_duplicate_indices"] = dropped
    if notes:
        entry["notes"] = notes
    sets[name] = entry


def build():
    sets = {}

    for family in FAMILIES:
        _add(
            sets,
            family["name"],
            family["values"],
            kind="numeric" if family["name"] in {"years", "quantities"} else "text",
            source="scripts/concept_families.py",
            expected=family["expected"],
        )

    for family in PROPOSED:
        FAMILY_BY_SET.setdefault(family["name"], family["family"])
        _add(
            sets,
            family["name"],
            family["values"],
            kind="text",
            source="scripts/proposed_families.py",
            expected=family["expected"],
            notes={"review": "proposed, not yet reviewed"},
        )

    for name, prompt_set in PROMPT_SETS.items():
        _add(
            sets,
            name,
            prompt_set.values,
            kind="text",
            source="prompt_templates.PROMPT_SETS",
            expected=EXPECTED_BY_SET.get(name),
        )

    calendar = (REPO / "prompt_values" / "calendar_dates.txt").read_text().split("\n")
    _add(
        sets,
        "calendar_dates",
        [line for line in calendar if line.strip()],
        kind="text",
        source="prompt_values/calendar_dates.txt",
    )

    series = json.loads(SERIES_FILE.read_text())
    for name, payload in series.items():
        notes = {"indexing": payload["indexing"]}
        if "precision_note" in payload:
            notes["precision"] = payload["precision_note"]
        _add(
            sets,
            name,
            payload["values"],
            kind="numeric",
            source=SERIES_FILE.relative_to(REPO).as_posix(),
            notes=notes,
            math_series=True,
        )

    return dict(sorted(sets.items()))


def main():
    sets = build()
    document = {
        "schema_version": 1,
        "n_sets": len(sets),
        "sets": sets,
    }
    OUT.write_text(json.dumps(document, indent=2) + "\n")

    unresolved = [name for name, entry in sets.items() if entry["family"] is None]
    deduped = {
        name: entry["dropped_duplicate_indices"]
        for name, entry in sets.items()
        if "dropped_duplicate_indices" in entry
    }
    widest = max(len(name) for name in sets)
    print(f"wrote {len(sets)} concept sets to {OUT.relative_to(REPO)}")
    for name, entry in sets.items():
        family = entry["family"] or "UNASSIGNED"
        print(f"  {name:<{widest}}  {entry['n_values']:>3} values  {family}")
    if unresolved:
        print(f"\n{len(unresolved)} set(s) need a family label before use:")
        for name in unresolved:
            print(f"  {name}")
    if deduped:
        print(f"\n{len(deduped)} series repeated values; later occurrences dropped")
        print("so no two points coincide (the prompt carries no index to tell them")
        print("apart).  Stated indexing in the source file no longer applies:")
        for name, dropped in deduped.items():
            pairs = ", ".join(f"[{i}]={v}" for i, v in dropped.items())
            print(f"  {name:<{widest}}  dropped {pairs}")


if __name__ == "__main__":
    main()
