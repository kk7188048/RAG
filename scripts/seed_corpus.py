"""Downloads MedQuAD (NIH) and normalizes it into per-source JSON files.

MedQuAD ships as one XML file per Document, grouped into 12 source folders
(CancerGov, GARD, GHR, MPlus, NIDDK, NINDS, SeniorHealth, NHLBI, CDC, ...).
This script shallow-clones the dataset, parses every <Document>/<QAPair>,
and writes normalized records to ingest/corpus/<source>.json:

    {"question": ..., "answer": ..., "focus_area": ..., "source": ...}

Usage: python scripts/seed_corpus.py
"""

import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

MEDQUAD_REPO = "https://github.com/abachaa/MedQuAD.git"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "ingest" / "corpus"


def parse_document(xml_path: Path, source: str) -> list[dict]:
    records = []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return records

    focus_el = root.find("Focus")
    focus_area = focus_el.text.strip() if focus_el is not None and focus_el.text else "Unknown"

    for qa_pair in root.findall(".//QAPair"):
        question_el = qa_pair.find("Question")
        answer_el = qa_pair.find("Answer")
        question = (question_el.text or "").strip() if question_el is not None else ""
        answer = (answer_el.text or "").strip() if answer_el is not None else ""
        if not question or not answer:
            continue
        records.append(
            {
                "question": question,
                "answer": answer,
                "focus_area": focus_area,
                "source": source,
            }
        )
    return records


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print(f"Cloning MedQuAD into {tmp_path} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", MEDQUAD_REPO, str(tmp_path)],
            check=True,
            capture_output=True,
        )

        source_folders = sorted(
            p for p in tmp_path.iterdir() if p.is_dir() and "QA" in p.name
        )

        total_records = 0
        for folder in source_folders:
            source_name = folder.name
            records: list[dict] = []
            for xml_file in folder.glob("*.xml"):
                records.extend(parse_document(xml_file, source_name))

            if not records:
                continue

            out_path = CORPUS_DIR / f"{source_name}.json"
            out_path.write_text(json.dumps(records, indent=2))
            total_records += len(records)
            print(f"  {source_name}: {len(records)} Q&A pairs -> {out_path.name}")

        print(f"Done. {total_records} total Q&A pairs across {len(source_folders)} sources.")
        print(f"Corpus written to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
