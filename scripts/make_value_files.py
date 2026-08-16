"""Write curated concept value files into the repo under prompt_values/."""

# ruff: noqa: E402  (sys.path setup must precede sibling imports)

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concept_families import FAMILIES

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "prompt_values"
OUT.mkdir(parents=True, exist_ok=True)

index = {}
for fam in FAMILIES:
    values = fam["values"]
    assert len(values) == len(set(values)), f"duplicate in {fam['name']}"
    assert all(v.strip() == v and v for v in values), f"bad value in {fam['name']}"
    (OUT / f"{fam['name']}.txt").write_text("\n".join(values) + "\n")
    index[fam["name"]] = {
        "template": fam["template"],
        "expected_topology": fam["expected"],
        "n_values": len(values),
    }

(OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
print(f"wrote {len(FAMILIES)} value files + index.json to {OUT}")
for name, meta in index.items():
    print(f"  {name:<18} {meta['n_values']:>3} values  template={meta['template']}")
