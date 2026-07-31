"""
Safety layer: medication extraction + static drug-drug interaction check.

This is what makes the assistant "healthcare-aware" rather than generic
RAG: independent of whatever the LLM says, we deterministically parse
medication lines out of the note text and cross-check every pair against a
curated interaction table, so a flag is never dependent on the model
noticing the interaction itself.

Data note: the interaction table (data/drug_interactions.csv) is a small,
hand-curated set of well-known, high-severity interactions -- NOT a
DrugBank export. DrugBank's structured dataset requires a commercial/
academic license for anything beyond very limited open data, so for a demo
project a hand-picked CSV is the honest, fastest option. Swap in a licensed
DrugBank extract (or RxNorm/DDInter, which are free) for production use --
the checker only needs a csv with drug_a,drug_b,severity,note columns.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

INTERACTIONS_CSV = Path(__file__).parent / "data" / "drug_interactions.csv"

MEDICATION_SECTION_NAMES = {"medications on admission", "discharge medications"}

# Strip common dose/route/frequency tokens off a medication line to recover
# just the drug name. Real systems would use a proper med-NER model
# (e.g. MedCAT, scispaCy) -- this regex approach is a deliberately simple
# stand-in that works for the structured "- Drugname 40 mg PO daily" format
# MIMIC discharge summaries actually use.
_DOSE_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|units?|mmol)\b.*$", re.IGNORECASE
)


def _clean_drug_name(line: str) -> str | None:
    line = line.strip().lstrip("-•").strip()
    if not line:
        return None
    name = _DOSE_PATTERN.sub("", line).strip()
    # Drop trailing parenthetical annotations like "(NEW)" or "(reduced dose)"
    name = re.sub(r"\(.*?\)", "", name).strip()
    if not name:
        return None
    return name.lower()


@dataclass
class InteractionFlag:
    drug_a: str
    drug_b: str
    severity: str
    note: str


def load_interaction_table() -> dict[frozenset, InteractionFlag]:
    table = {}
    with open(INTERACTIONS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = frozenset({row["drug_a"].lower(), row["drug_b"].lower()})
            table[key] = InteractionFlag(
                drug_a=row["drug_a"],
                drug_b=row["drug_b"],
                severity=row["severity"],
                note=row["note"],
            )
    return table


_INTERACTION_TABLE = load_interaction_table()


def extract_medications(note_text: str) -> list[str]:
    """Pull medication names out of medication-related sections of a note."""
    meds = set()
    current_section = None
    for raw_line in note_text.splitlines():
        stripped = raw_line.strip()
        header_match = re.match(r"^([A-Za-z /]+):\s*$", stripped)
        if header_match:
            current_section = header_match.group(1).strip().lower()
            continue
        if current_section in MEDICATION_SECTION_NAMES and stripped.startswith(("-", "•")):
            name = _clean_drug_name(stripped)
            if name:
                meds.add(name)
    return sorted(meds)


def check_interactions(medications: list[str]) -> list[InteractionFlag]:
    """Cross-reference every pair of extracted medications against the
    interaction table. O(n^2) over a medication list, which is always small
    (<20), so this is cheap."""
    flags = []
    meds_lower = [m.lower() for m in medications]
    for i in range(len(meds_lower)):
        for j in range(i + 1, len(meds_lower)):
            key = frozenset({meds_lower[i], meds_lower[j]})
            if key in _INTERACTION_TABLE:
                flags.append(_INTERACTION_TABLE[key])
    return flags


def check_note(note_text: str) -> dict:
    meds = extract_medications(note_text)
    flags = check_interactions(meds)
    return {"medications": meds, "flags": flags}


if __name__ == "__main__":
    from data.synthetic_notes import load_notes

    for note in load_notes():
        result = check_note(note.text)
        print(f"\n{note.note_id} (patient {note.patient_id}, admission {note.admission_id})")
        print("  meds:", result["medications"])
        if result["flags"]:
            for f in result["flags"]:
                print(f"  \u26a0\ufe0f  {f.severity.upper()}: {f.drug_a} + {f.drug_b} -- {f.note}")
        else:
            print("  no known interactions flagged")
