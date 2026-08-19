"""Sanity-check extracted manifolds before they reach the topology pipeline.

Every failure mode here has already occurred once during development: repeated
concept values collapsing to one point, and two templates sharing a prefix so a
causal model returns identical vectors.  Both look like ordinary data
downstream, so they are checked here rather than diagnosed later from a strange
persistence diagram.
"""

# ruff: noqa: E402

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from templates import TEMPLATE_ORDER


def check(path: Path) -> list[str]:
    """Return one message per problem found in a single manifold archive."""
    problems: list[str] = []
    archive = np.load(path, allow_pickle=True)
    cloud = archive["activations"]
    labels = archive["labels"]
    # Name templates from the archive, not the current module: a stale archive
    # must report the template it actually stored.
    names = [str(name) for name in archive["templates"]]

    if cloud.ndim != 3:
        return [f"expected a 3-D (values, templates, hidden) array; got {cloud.shape}"]
    n_values, n_templates, _ = cloud.shape

    if n_values != len(labels):
        problems.append(f"{n_values} rows but {len(labels)} labels")
    if names != list(TEMPLATE_ORDER):
        problems.append("template axis does not match templates.TEMPLATE_ORDER")
    if not np.isfinite(cloud).all():
        problems.append("contains non-finite values")

    flat = cloud.reshape(-1, cloud.shape[-1])
    if len(np.unique(flat, axis=0)) != len(flat):
        problems.append(f"{len(flat) - len(np.unique(flat, axis=0))} coincident points")

    # A template axis that carries no variance means the framing never reached
    # the concept, which makes per-template offset removal meaningless.
    for first in range(n_templates):
        for second in range(first + 1, n_templates):
            if np.allclose(cloud[:, first], cloud[:, second]):
                problems.append(
                    f"templates {names[first]!r} and {names[second]!r} are "
                    "identical for every value"
                )

    # A cloud whose points are all near-parallel carries no usable geometry.
    centred = flat - flat.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    if float(norms.min()) <= 0.0:
        problems.append("a point sits exactly at the centroid")
    else:
        spread = float(norms.max() / norms.min())
        if spread > 1e4:
            problems.append(f"extreme norm spread ({spread:.1e}) suggests an outlier")

    return problems


def check_coverage(paths: list[Path]) -> list[str]:
    """Confirm the concept sets, the expectations, and the manifolds agree.

    These three files are written by separate steps, so a set added to one and
    not the others would otherwise surface as a KeyError deep in scoring.
    """
    problems: list[str] = []
    concepts = set(json.loads((REPO / "concept_sets.json").read_text())["sets"])
    expectations = set(
        json.loads((REPO / "expectations_v2.json").read_text())["expectations"]
    )
    extracted = {path.stem for path in paths}

    for label, missing in (
        ("no expectation", concepts - expectations),
        ("no concept set", expectations - concepts),
        ("not extracted", concepts - extracted),
        ("extracted but unknown", extracted - concepts),
    ):
        if missing:
            problems.append(f"{label}: {', '.join(sorted(missing))}")
    return problems


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "manifolds"
    paths = sorted(root.glob("*.npz"))
    if not paths:
        print(f"no manifolds found in {root}")
        return 1

    failures = 0
    widest = max(len(path.stem) for path in paths)
    for path in paths:
        problems = check(path)
        cloud = np.load(path, allow_pickle=True)["activations"]
        status = "ok" if not problems else "FAIL"
        print(f"  {path.stem:<{widest}}  {str(cloud.shape):<18} {status}")
        for problem in problems:
            print(f"      - {problem}")
        failures += bool(problems)

    coverage = check_coverage(paths) if root == REPO / "manifolds" else []
    for problem in coverage:
        print(f"  COVERAGE - {problem}")

    sidecars = sum((path.with_suffix(".json")).exists() for path in paths)
    print(f"\n{len(paths)} manifolds, {sidecars} sidecars, {failures} with problems")
    if paths:
        meta = json.loads(paths[0].with_suffix(".json").read_text())
        print(
            f"model={meta['model']} revision={meta['model_revision'][:12]} "
            f"layer={meta['layer']}/{meta['n_layers']} hidden={meta['hidden_size']}"
        )
    return 1 if failures or coverage else 0


if __name__ == "__main__":
    raise SystemExit(main())
