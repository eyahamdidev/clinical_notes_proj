"""
Resolves which note source to use.

Default: real MIMIC-IV Clinical Database Demo data (data/mimic_loader.py),
if the raw CSVs are present under data/mimic-iv-clinical-database-demo-2.2/.
Falls back to the fully-synthetic hand-written notes (data/synthetic_notes.py)
if the real data isn't present -- e.g. if you clone this repo without
downloading/keeping the MIMIC data, the app still runs.

Override explicitly with the DATA_SOURCE env var: "mimic" or "synthetic".
"""
import os

from data.mimic_loader import MIMIC_ROOT


def get_loader():
    forced = os.environ.get("DATA_SOURCE", "").lower().strip()
    if forced == "synthetic":
        from data.synthetic_notes import load_notes

        return load_notes, "synthetic"
    if forced == "mimic":
        from data.mimic_loader import load_notes

        return load_notes, "mimic"

    # auto-detect
    if (MIMIC_ROOT / "hosp" / "admissions.csv.gz").exists():
        from data.mimic_loader import load_notes

        return load_notes, "mimic"

    from data.synthetic_notes import load_notes

    return load_notes, "synthetic"


def load_notes():
    loader, source = get_loader()
    return loader()
