"""Emit ``expectations_v2.json``: the Betti-number answer key for every set.

Counts are derived from each set's family rather than written per set, so a
family's topology is stated once and cannot drift between the sets that share
it.  Ranges rather than points: a manifold recovered from finitely many noisy
activations may miss a feature the ideal shape has, and the ranges say how much
of that the scoring stage should tolerate.

The reasoning behind each family is recorded in ``FAMILY_TOPOLOGY`` and copied
into the output, because a wrong entry here makes correct topology code look
broken and the reader needs to audit the claim, not just the number.
"""

# ruff: noqa: E402

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "expectations_v2.json"
CONCEPT_SETS = REPO / "concept_sets.json"

FAMILY_TOPOLOGY = {
    "line": {
        "counts": {"H0": [1, 1], "H1": [0, 0], "H2": [0, 0]},
        "shape": "interval",
        "why": (
            "An ordered scale is a single arc: connected, contractible, no "
            "loop and no cavity."
        ),
    },
    "circle": {
        "counts": {"H0": [1, 1], "H1": [1, 1], "H2": [0, 0]},
        "shape": "S1",
        "why": (
            "A cyclic order closes back on itself, giving exactly one independent loop."
        ),
    },
    "plane": {
        "counts": {"H0": [1, 1], "H1": [0, 0], "H2": [0, 0]},
        "shape": "disc",
        "why": (
            "A bounded 2D patch is contractible; it is distinguished from a "
            "line by intrinsic dimension, not by Betti numbers."
        ),
    },
    "grid": {
        "counts": {"H0": [1, 1], "H1": [0, 0], "H2": [0, 0]},
        "shape": "disc",
        "why": (
            "A rectangular lattice is a 2D patch with no wraparound, so it is "
            "contractible like any other plane."
        ),
    },
    "sphere": {
        "counts": {"H0": [1, 1], "H1": [0, 0], "H2": [1, 1]},
        "shape": "S2",
        "why": (
            "Points spread over a globe enclose one cavity and, unlike a "
            "torus, admit no independent loop."
        ),
    },
    "tree": {
        "counts": {"H0": [1, 1], "H1": [0, 0], "H2": [0, 0]},
        "shape": "tree",
        "why": (
            "A hierarchy branches without ever rejoining, so it is connected "
            "and loop-free; branching shows up in geometry, not homology."
        ),
    },
    "spiral": {
        "counts": {"H0": [1, 1], "H1": [0, 0], "H2": [0, 0]},
        "shape": "arc",
        "why": (
            "A spiral is a single curve that never closes.  Periodicity may "
            "produce a near-loop with low persistence, which is why H1 is "
            "expected to be zero rather than one."
        ),
    },
    "torus": {
        "counts": {"H0": [1, 1], "H1": [2, 2], "H2": [1, 1]},
        "shape": "T2",
        "why": "Two independent cyclic coordinates give two loops and one cavity.",
    },
}

# Sets whose recovered topology is expected to be unreliable, with the reason.
# They stay in the file so a run can report them rather than silently omit
# them, but scoring should treat them separately.
LOW_CONFIDENCE = {
    "seasons": "n=4 cannot support a loop; a cyclic fit is saturated by construction",
    "chess_pieces": "n=6 is too few points to separate a loop from noise",
    "directions_3d": "n=6 samples three axis pairs, too sparse for a sphere",
    "days": "n=7 is marginal for a circle",
    "weekday_abbreviations": "n=7 is marginal for a circle",
    "quantities": "n=7 is marginal, though a line needs fewer points than a loop",
}

MIN_POINTS_FOR_LOOP = 8


def build() -> dict:
    document = json.loads(CONCEPT_SETS.read_text())
    sets = document["sets"]

    expectations = {}
    for name, entry in sorted(sets.items()):
        family = entry["family"]
        if family is None:
            raise ValueError(f"{name}: no family label; cannot state an expectation")
        if family not in FAMILY_TOPOLOGY:
            raise ValueError(f"{name}: unknown family {family!r}")
        topology = FAMILY_TOPOLOGY[family]

        record = {
            "family": family,
            "expected_counts": topology["counts"],
            "model_shape": topology["shape"],
            "rationale": topology["why"],
            "n_values": entry["n_values"],
            "expected_topology": entry["expected_topology"],
        }
        if entry.get("notes", {}).get("review"):
            record["review_status"] = "proposed"
        reasons = []
        if name in LOW_CONFIDENCE:
            reasons.append(LOW_CONFIDENCE[name])
        if topology["counts"]["H1"][1] > 0 and entry["n_values"] < MIN_POINTS_FOR_LOOP:
            reasons.append(
                f"fewer than {MIN_POINTS_FOR_LOOP} points for an expected loop"
            )
        if reasons:
            record["low_confidence"] = reasons
        expectations[name] = record

    return {
        "schema_version": 2,
        "n_sets": len(expectations),
        "family_topology": FAMILY_TOPOLOGY,
        "expectations": expectations,
    }


def main() -> int:
    document = build()
    OUT.write_text(json.dumps(document, indent=2) + "\n")
    expectations = document["expectations"]

    counts: dict[str, int] = {}
    for record in expectations.values():
        counts[record["family"]] = counts.get(record["family"], 0) + 1
    print(f"wrote {len(expectations)} expectations to {OUT.relative_to(REPO)}")
    for family, total in sorted(counts.items()):
        betti = FAMILY_TOPOLOGY[family]["counts"]
        signature = " ".join(f"{k}={v[0]}-{v[1]}" for k, v in betti.items())
        print(f"  {family:<8} {total:>3} sets   {signature}")

    flagged = [n for n, r in expectations.items() if "low_confidence" in r]
    proposed = [n for n, r in expectations.items() if r.get("review_status")]
    print(f"\n{len(proposed)} set(s) marked proposed, pending review")
    print(f"{len(flagged)} set(s) flagged low-confidence:")
    for name in flagged:
        print(f"  {name:<24} {expectations[name]['low_confidence'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
