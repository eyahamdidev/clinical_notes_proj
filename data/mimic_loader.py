"""
Loads the real MIMIC-IV Clinical Database Demo (PhysioNet) and builds
discharge-summary-style Note objects from it.

IMPORTANT, read before trusting this data as "real clinical notes":
The MIMIC-IV Demo explicitly EXCLUDES free-text clinical notes -- see
data/mimic-iv-clinical-database-demo-2.2/README.txt: "The dataset includes
similar content to MIMIC-IV, but excludes free-text clinical notes."
Free-text notes live in the separate MIMIC-IV-Note project, which requires
its own PhysioNet credentialing (higher bar than the open demo, because
free text carries more re-identification risk).

So what this loader actually does: it pulls REAL, de-identified structured
fields for each hospital admission -- diagnoses (ICD-coded), prescriptions
(drug name/dose/route), abnormal labs, vitals (from omr), admission
metadata -- and renders them into a discharge-summary-shaped narrative
using a template. The underlying FACTS (which drugs, which diagnoses,
which lab values, in which combination) are real MIMIC-IV data, not
invented. The PROSE connecting them is templated, not a real physician's
dictation. Be precise about this distinction if you present this project:
it's "real structured EHR data, template-rendered as notes," not "real
discharge summaries."

Tables used (all under data/mimic-iv-clinical-database-demo-2.2/):
    hosp/admissions.csv.gz, hosp/patients.csv.gz
    hosp/diagnoses_icd.csv.gz, hosp/d_icd_diagnoses.csv.gz
    hosp/prescriptions.csv.gz
    hosp/labevents.csv.gz, hosp/d_labitems.csv.gz
    hosp/omr.csv.gz  (outpatient measurements: height/weight/BP)

Output note sections match the exact headers ingest.py's chunker expects:
Chief Complaint, History of Present Illness, Past Medical History,
Medications on Admission, Physical Exam, Assessment and Plan,
Discharge Medications, Discharge Instructions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

MIMIC_ROOT = Path(__file__).parent / "mimic-iv-clinical-database-demo-2.2"

# Admissions with more than this many distinct diagnoses/prescriptions get
# truncated in the note text, purely to keep chunks readable -- real
# discharge summaries don't list 40 ICD codes verbatim either.
MAX_DIAGNOSES_LISTED = 8
MAX_ABNORMAL_LABS_LISTED = 6


@dataclass
class Note:
    note_id: str
    patient_id: str
    admission_id: str
    text: str
    meta: dict = field(default_factory=dict)


def _fmt_dose(row) -> str:
    parts = [str(row["drug"]).strip()]
    if pd.notna(row.get("dose_val_rx")) and str(row["dose_val_rx"]).strip() not in ("", "0"):
        dose = f"{row['dose_val_rx']}"
        if pd.notna(row.get("dose_unit_rx")):
            dose += f" {row['dose_unit_rx']}"
        parts.append(dose)
    if pd.notna(row.get("route")) and str(row["route"]).strip():
        parts.append(str(row["route"]))
    return " ".join(parts)


def _age_at_admission(anchor_age: int, anchor_year: int, admittime) -> int:
    return int(anchor_age) + (admittime.year - int(anchor_year))


class MimicLoader:
    def __init__(self, root: Path = MIMIC_ROOT):
        self.root = root
        if not (root / "hosp" / "admissions.csv.gz").exists():
            raise FileNotFoundError(
                f"MIMIC-IV demo data not found at {root}. Download the demo from "
                "https://physionet.org/content/mimic-iv-demo/ and extract it there, "
                "or see README.md."
            )
        self._load_tables()

    def _load_tables(self):
        h = self.root / "hosp"
        self.admissions = pd.read_csv(h / "admissions.csv.gz", parse_dates=["admittime", "dischtime"])
        self.patients = pd.read_csv(h / "patients.csv.gz")
        self.diagnoses = pd.read_csv(h / "diagnoses_icd.csv.gz")
        self.d_diagnoses = pd.read_csv(h / "d_icd_diagnoses.csv.gz")
        self.prescriptions = pd.read_csv(
            h / "prescriptions.csv.gz",
            parse_dates=["starttime", "stoptime"],
            low_memory=False,
        )
        self.labevents = pd.read_csv(h / "labevents.csv.gz", parse_dates=["charttime"], low_memory=False)
        self.d_labitems = pd.read_csv(h / "d_labitems.csv.gz")
        self.omr = pd.read_csv(h / "omr.csv.gz", parse_dates=["chartdate"])

        self.diagnoses = self.diagnoses.merge(
            self.d_diagnoses, on=["icd_code", "icd_version"], how="left"
        ).sort_values(["hadm_id", "seq_num"])

    def _diagnoses_for(self, hadm_id: int) -> list[str]:
        rows = self.diagnoses[self.diagnoses.hadm_id == hadm_id]
        titles = [t for t in rows.long_title.tolist() if isinstance(t, str)]
        return titles

    def _prescriptions_for(self, hadm_id: int, admittime, dischtime) -> tuple[list[str], list[str]]:
        rows = self.prescriptions[self.prescriptions.hadm_id == hadm_id].dropna(subset=["drug"])
        if rows.empty:
            return [], []

        admission_window_end = admittime + pd.Timedelta(hours=24)
        discharge_window_start = dischtime - pd.Timedelta(hours=48)

        on_admission = rows[rows.starttime <= admission_window_end]
        near_discharge = rows[rows.starttime >= discharge_window_start]

        def dedup_format(df: pd.DataFrame) -> list[str]:
            seen = {}
            for _, r in df.iterrows():
                name = str(r["drug"]).strip().lower()
                if name not in seen:
                    seen[name] = _fmt_dose(r)
            return sorted(seen.values())

        return dedup_format(on_admission), dedup_format(near_discharge)

    def _abnormal_labs_for(self, hadm_id: int) -> list[str]:
        rows = self.labevents[(self.labevents.hadm_id == hadm_id) & (self.labevents.flag == "abnormal")]
        if rows.empty:
            return []
        rows = rows.merge(self.d_labitems[["itemid", "label"]], on="itemid", how="left")
        rows = rows.dropna(subset=["label"]).drop_duplicates(subset=["label"])
        out = []
        for _, r in rows.head(MAX_ABNORMAL_LABS_LISTED).iterrows():
            unit = f" {r['valueuom']}" if pd.notna(r.get("valueuom")) else ""
            out.append(f"{r['label']}: {r['value']}{unit} (abnormal)")
        return out

    def _vitals_for(self, subject_id: int, admittime, dischtime) -> list[str]:
        rows = self.omr[
            (self.omr.subject_id == subject_id)
            & (self.omr.chartdate >= admittime.normalize())
            & (self.omr.chartdate <= dischtime.normalize() + pd.Timedelta(days=1))
        ]
        if rows.empty:
            return []
        out = []
        for name in ("Blood Pressure", "Weight (Lbs)", "Height (Inches)", "BMI (kg/m2)"):
            match = rows[rows.result_name == name]
            if not match.empty:
                out.append(f"{name}: {match.iloc[-1]['result_value']}")
        return out

    def build_note_text(self, adm_row, patient_row) -> str:
        hadm_id = int(adm_row["hadm_id"])
        subject_id = int(adm_row["subject_id"])
        admittime, dischtime = adm_row["admittime"], adm_row["dischtime"]

        diagnoses = self._diagnoses_for(hadm_id)
        chief = diagnoses[0] if diagnoses else "Unspecified condition"
        other_dx = diagnoses[1:MAX_DIAGNOSES_LISTED]

        age = _age_at_admission(patient_row["anchor_age"], patient_row["anchor_year"], admittime)
        gender = "male" if patient_row["gender"] == "M" else "female"

        meds_admission, meds_discharge = self._prescriptions_for(hadm_id, admittime, dischtime)
        abnormal_labs = self._abnormal_labs_for(hadm_id)
        vitals = self._vitals_for(subject_id, admittime, dischtime)

        los_days = (dischtime - admittime).total_seconds() / 86400

        lines = []
        lines.append("Chief Complaint:")
        lines.append(f"{chief}.")
        lines.append("")

        lines.append("History of Present Illness:")
        hpi = (
            f"The patient is a {age}-year-old {gender} admitted via "
            f"{str(adm_row.get('admission_location', 'the emergency department')).lower()} "
            f"with {chief.lower()}. Admission type: {adm_row.get('admission_type', 'unspecified')}. "
            f"Length of stay: {los_days:.1f} days."
        )
        lines.append(hpi)
        lines.append("")

        lines.append("Past Medical History:")
        if other_dx:
            for i, dx in enumerate(other_dx, 1):
                lines.append(f"{i}. {dx}")
        else:
            lines.append("No additional coded diagnoses on file for this admission.")
        lines.append("")

        lines.append("Medications on Admission:")
        if meds_admission:
            for m in meds_admission:
                lines.append(f"- {m}")
        else:
            lines.append("No medications recorded near admission time.")
        lines.append("")

        lines.append("Physical Exam:")
        if vitals or abnormal_labs:
            for v in vitals:
                lines.append(v)
            for lab in abnormal_labs:
                lines.append(lab)
        else:
            lines.append("No vitals or abnormal labs on file for this admission window.")
        lines.append("")

        lines.append("Assessment and Plan:")
        dx_list = ", ".join(diagnoses[:MAX_DIAGNOSES_LISTED]) if diagnoses else "unspecified condition"
        ap = (
            f"{age}-year-old {gender} admitted with {chief.lower()}. "
            f"Coded diagnoses this admission: {dx_list}. "
            f"Discharged to: {adm_row.get('discharge_location', 'unspecified')}."
        )
        lines.append(ap)
        lines.append("")

        lines.append("Discharge Medications:")
        if meds_discharge:
            for m in meds_discharge:
                lines.append(f"- {m}")
        else:
            lines.append("No medications recorded near discharge time.")
        lines.append("")

        lines.append("Discharge Instructions:")
        lines.append(
            f"Follow up as directed. Discharge disposition: "
            f"{adm_row.get('discharge_location', 'unspecified')}."
        )

        return "\n".join(lines)

    def load_notes(self, limit: int | None = None) -> list[Note]:
        notes = []
        admissions = self.admissions.sort_values(["subject_id", "admittime"])
        if limit:
            admissions = admissions.head(limit)

        patients_idx = self.patients.set_index("subject_id")

        for _, adm_row in admissions.iterrows():
            subject_id = int(adm_row["subject_id"])
            if subject_id not in patients_idx.index:
                continue
            patient_row = patients_idx.loc[subject_id]
            adm_row = adm_row.copy()
            for col in ("discharge_location", "admission_location", "admission_type"):
                if pd.isna(adm_row.get(col)):
                    adm_row[col] = "not documented"
            text = self.build_note_text(adm_row, patient_row)
            notes.append(
                Note(
                    note_id=f"N{int(adm_row['hadm_id'])}",
                    patient_id=str(subject_id),
                    admission_id=str(int(adm_row["hadm_id"])),
                    text=text,
                    meta={
                        "admittime": str(adm_row["admittime"]),
                        "dischtime": str(adm_row["dischtime"]),
                        "age": _age_at_admission(
                            patient_row["anchor_age"], patient_row["anchor_year"], adm_row["admittime"]
                        ),
                        "gender": "M" if patient_row["gender"] == "M" else "F",
                        "admission_type": str(adm_row.get("admission_type", "unknown")),
                        "discharge_location": str(adm_row.get("discharge_location", "unknown")),
                    },
                )
            )
        return notes


def load_notes(limit: int | None = None) -> list[Note]:
    """Drop-in replacement for data.synthetic_notes.load_notes()."""
    return MimicLoader().load_notes(limit=limit)


if __name__ == "__main__":
    notes = load_notes(limit=5)
    for n in notes:
        print("=" * 70)
        print(n.note_id, n.patient_id, n.admission_id)
        print(n.text)
