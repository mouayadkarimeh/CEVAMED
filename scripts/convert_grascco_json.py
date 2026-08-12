"""Convert GRASCCO PHI-annotation JSON files to plain text files.

Each JSON file is a UIMA CAS JSON export.  The plain document text lives in
the ``sofaString`` field of the Sofa feature-structure.  A companion
``<name>.meta.json`` file is written alongside each ``.txt`` to record the
PHI annotations (begin/end offsets + kind) so the evaluator can build
gold-standard answer spans without re-reading the original JSON.

Usage::

    python scripts/convert_grascco_json.py \
        --input  data/data-json/grascco-json/grascco_phi_annotation_json \
        --output data/grascco_texts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract(json_path: Path) -> tuple[str, list[dict]]:
    """Return (sofaString, list_of_phi_annotations) from a CAS JSON file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    feature_structures = data.get("%FEATURE_STRUCTURES", [])

    sofa_text: str = ""
    phi_annotations: list[dict] = []

    for fs in feature_structures:
        fs_type = fs.get("%TYPE", "")
        if fs_type == "uima.cas.Sofa":
            sofa_text = fs.get("sofaString", "")
        elif "PHI" in fs_type:
            phi_annotations.append(
                {
                    "begin": fs.get("begin", 0),
                    "end": fs.get("end", 0),
                    "kind": fs.get("kind", ""),
                }
            )

    return sofa_text, phi_annotations


def convert(input_dir: Path, output_dir: Path) -> int:
    """Convert all JSON files and return the number of files written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}", file=sys.stderr)
        return 0

    written = 0
    for json_path in json_files:
        # Derive a clean stem: strip trailing "_phi.json" or ".json"
        stem = json_path.stem  # e.g. "Colon_Fake_D.txt_phi"
        if stem.endswith("_phi"):
            stem = stem[:-4]  # → "Colon_Fake_D.txt"
        if stem.endswith(".txt"):
            stem = stem[:-4]  # → "Colon_Fake_D"

        sofa_text, phi_annotations = _extract(json_path)
        if not sofa_text:
            print(f"  SKIP {json_path.name}: no sofaString found", file=sys.stderr)
            continue

        txt_path = output_dir / f"{stem}.txt"
        txt_path.write_text(sofa_text, encoding="utf-8")

        meta_path = output_dir / f"{stem}.meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "doc_id": stem,
                    "source_file": json_path.name,
                    "phi_annotations": phi_annotations,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"  {json_path.name} → {txt_path.name} ({len(sofa_text)} chars, "
              f"{len(phi_annotations)} PHI annotations)")
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/data-json/grascco-json/grascco_phi_annotation_json",
        help="Directory containing GRASCCO PHI JSON files",
    )
    parser.add_argument(
        "--output",
        default="data/grascco_texts",
        help="Directory to write plain-text and meta files",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input) if Path(args.input).is_absolute() else repo_root / args.input
    output_dir = Path(args.output) if Path(args.output).is_absolute() else repo_root / args.output

    print(f"Converting JSON files from: {input_dir}")
    print(f"Writing text files to:      {output_dir}")
    n = convert(input_dir, output_dir)
    print(f"\nDone – {n} document(s) written.")


if __name__ == "__main__":
    main()
