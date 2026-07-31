"""
Generates synthetic, MIMIC-IV-discharge-summary-style clinical notes.

Why synthetic here: this sandbox has no network access to PhysioNet or
HuggingFace, so real MIMIC-IV-Demo data can't be pulled in this environment.
The notes below mimic real MIMIC-IV discharge summary structure (same
section headers, same terse clinical prose style) so the ingestion/chunking
code is a drop-in fit once you swap in real PhysioNet files -- see README.

To use real data instead: download the MIMIC-IV Clinical Database Demo
from https://physionet.org/content/mimic-iv-demo/ (free PhysioNet account +
short data use agreement, usually approved same-day), and point
`load_notes()` at the discharge.csv / discharge notes table instead.
"""
from dataclasses import dataclass, field


@dataclass
class Note:
    note_id: str
    patient_id: str
    admission_id: str
    text: str
    meta: dict = field(default_factory=dict)


_NOTES_RAW = [
    dict(
        patient_id="10001",
        admission_id="A1001",
        text="""
Chief Complaint:
Shortness of breath and lower extremity swelling for 5 days.

History of Present Illness:
The patient is a 72-year-old male with a history of congestive heart failure
(EF 30%), hypertension, and type 2 diabetes mellitus who presented with
progressive dyspnea on exertion and bilateral lower extremity edema over the
past 5 days. He reports orthopnea (3-pillow) and a weight gain of
approximately 8 lbs over the past week. He denies chest pain, fever, or
cough. He admits to dietary indiscretion (high sodium intake) over the
holiday weekend. No recent medication changes reported prior to admission.

Past Medical History:
1. Congestive heart failure, HFrEF, EF 30% (last echo 6 months ago)
2. Hypertension
3. Type 2 diabetes mellitus
4. Chronic kidney disease stage 3

Medications on Admission:
- Furosemide 40 mg PO daily
- Lisinopril 10 mg PO daily
- Metformin 500 mg PO BID
- Metoprolol succinate 50 mg PO daily
- Aspirin 81 mg PO daily

Physical Exam:
Vitals: T 98.2F, HR 92, BP 148/88, RR 20, SpO2 94% on room air.
Bilateral crackles at lung bases. 2+ pitting edema to the knees bilaterally.
JVP elevated to 10 cm.

Assessment and Plan:
72M with known HFrEF presenting with acute decompensated heart failure,
likely triggered by dietary sodium indiscretion. Diuresed with IV furosemide
80 mg BID with good urine output and symptomatic improvement. Transitioned
to oral furosemide 60 mg daily prior to discharge. Continued home
lisinopril and metoprolol. Diabetes managed on home metformin, blood
glucose well controlled during admission. Renal function stable at
baseline (creatinine 1.4).

Discharge Medications:
- Furosemide 60 mg PO daily (increased from 40 mg)
- Lisinopril 10 mg PO daily
- Metoprolol succinate 50 mg PO daily
- Metformin 500 mg PO BID
- Aspirin 81 mg PO daily
- Ibuprofen 400 mg PO PRN for knee pain (NEW)

Discharge Instructions:
Weigh yourself daily. Call your doctor if you gain more than 3 lbs in one
day or 5 lbs in one week. Follow a low-sodium diet (less than 2g/day).
Follow up with cardiology in 1 week and with primary care in 2 weeks.
""",
    ),
    dict(
        patient_id="10001",
        admission_id="A1002",
        text="""
Chief Complaint:
Follow-up admission for worsening renal function and leg swelling.

History of Present Illness:
This is the same 72-year-old male with HFrEF, CKD stage 3, and diabetes,
seen 3 months after his prior admission for decompensated heart failure,
now presenting with worsening bilateral leg swelling and decreased urine
output over the past week. He reports he has been taking ibuprofen
almost daily for knee pain since his last discharge. He also reports
missing several doses of furosemide due to cost concerns.

Past Medical History:
1. Congestive heart failure, HFrEF, EF 30%
2. Hypertension
3. Type 2 diabetes mellitus
4. Chronic kidney disease stage 3

Medications on Admission:
- Furosemide 60 mg PO daily (inconsistent adherence)
- Lisinopril 10 mg PO daily
- Metoprolol succinate 50 mg PO daily
- Metformin 500 mg PO BID
- Aspirin 81 mg PO daily
- Ibuprofen 400 mg PO daily (patient-reported, not prescribed at this dose)

Physical Exam:
Vitals: T 98.6F, HR 88, BP 156/92, RR 18, SpO2 96% on room air.
3+ pitting edema to the mid-shins. Labs notable for creatinine 2.1 (up
from baseline 1.4), potassium 5.4.

Assessment and Plan:
72M with HFrEF and CKD stage 3 presenting with acute kidney injury on
chronic kidney disease, most consistent with NSAID-induced nephrotoxicity
compounded by ACE-inhibitor use, in the setting of inconsistent diuretic
adherence. Ibuprofen discontinued. Held lisinopril temporarily given
hyperkalemia and rising creatinine. Furosemide dose held then restarted
at lower dose once renal function trended down. Renal function improved
to creatinine 1.6 by discharge. Counseled extensively on avoiding NSAIDs
given his CKD and concurrent ACE-inhibitor and diuretic use (triple
whammy nephrotoxicity).

Discharge Medications:
- Furosemide 40 mg PO daily
- Lisinopril 5 mg PO daily (reduced dose)
- Metoprolol succinate 50 mg PO daily
- Metformin 500 mg PO BID
- Aspirin 81 mg PO daily
- Acetaminophen 500 mg PO PRN for knee pain (replaces ibuprofen)

Discharge Instructions:
Avoid all NSAIDs (ibuprofen, naproxen, aspirin at higher doses) going
forward given kidney disease. Use acetaminophen for pain instead. Repeat
basic metabolic panel in 1 week with primary care. Follow up nephrology
in 2 weeks.
""",
    ),
    dict(
        patient_id="10002",
        admission_id="A2001",
        text="""
Chief Complaint:
Atrial fibrillation with rapid ventricular response.

History of Present Illness:
The patient is a 65-year-old female with a history of paroxysmal atrial
fibrillation, hypertension, and hyperlipidemia who presented to the
emergency department with palpitations and lightheadedness. She was found
to be in atrial fibrillation with rapid ventricular response, heart rate
140s. She denies chest pain or syncope. She takes warfarin for
anticoagulation, last INR checked 2 weeks ago was 2.3 (therapeutic).

Past Medical History:
1. Paroxysmal atrial fibrillation, on chronic anticoagulation
2. Hypertension
3. Hyperlipidemia
4. Osteoarthritis

Medications on Admission:
- Warfarin 5 mg PO daily
- Atorvastatin 40 mg PO daily
- Amlodipine 5 mg PO daily
- Naproxen 500 mg PO BID PRN for osteoarthritis pain

Physical Exam:
Vitals: T 98.4F, HR 138 irregularly irregular, BP 132/84, RR 18, SpO2 98%.
Irregularly irregular rhythm on exam, no murmurs appreciated.

Assessment and Plan:
65F with known paroxysmal AF presenting with rapid ventricular response,
successfully rate-controlled with IV diltiazem then transitioned to oral
metoprolol. Continued therapeutic anticoagulation with warfarin, INR
checked on admission was 2.1, within range. Cardiology was consulted;
no acute intervention needed, outpatient ablation discussion deferred to
clinic follow-up. Patient counseled on bleeding risk given concurrent
warfarin and chronic naproxen use for osteoarthritis, which significantly
increases GI bleeding risk; recommended transition to acetaminophen.

Discharge Medications:
- Warfarin 5 mg PO daily
- Metoprolol tartrate 25 mg PO BID (NEW, for rate control)
- Atorvastatin 40 mg PO daily
- Amlodipine 5 mg PO daily
- Acetaminophen 650 mg PO PRN for osteoarthritis pain (replaces naproxen)

Discharge Instructions:
Stop naproxen given bleeding risk with warfarin; use acetaminophen instead
for joint pain. Continue warfarin as directed, INR check in 1 week.
Follow up with cardiology in 2 weeks to discuss possible ablation.
""",
    ),
    dict(
        patient_id="10003",
        admission_id="A3001",
        text="""
Chief Complaint:
Community-acquired pneumonia.

History of Present Illness:
The patient is a 58-year-old male with a history of COPD (on home oxygen
2L nasal cannula) and type 2 diabetes who presented with 4 days of
productive cough, fever, and worsening dyspnea beyond his baseline. Chest
X-ray showed a right lower lobe infiltrate consistent with pneumonia.

Past Medical History:
1. COPD, GOLD stage III, on home oxygen
2. Type 2 diabetes mellitus
3. Former tobacco use (quit 5 years ago, 40 pack-year history)

Medications on Admission:
- Tiotropium inhaler daily
- Albuterol inhaler PRN
- Metformin 1000 mg PO BID
- Insulin glargine 20 units at bedtime

Physical Exam:
Vitals: T 101.4F, HR 104, BP 128/76, RR 24, SpO2 89% on 2L NC.
Decreased breath sounds and crackles at the right lower lobe.

Assessment and Plan:
58M with COPD presenting with community-acquired pneumonia, right lower
lobe. Started on IV ceftriaxone and azithromycin per CAP guidelines.
Sputum culture grew Streptococcus pneumoniae, sensitive to the treatment
regimen. Oxygen requirement increased transiently to 4L then weaned back
to baseline 2L by discharge. Blood glucose ran higher than home baseline
during illness and steroid-sparing management, adjusted insulin glargine
dose upward temporarily.

Discharge Medications:
- Amoxicillin-clavulanate 875 mg PO BID x 5 more days (to complete 7-day
  course)
- Tiotropium inhaler daily
- Albuterol inhaler PRN
- Metformin 1000 mg PO BID
- Insulin glargine 24 units at bedtime (increased from 20 units)

Discharge Instructions:
Complete the full course of antibiotics even if feeling better. Continue
home oxygen at 2L. Monitor blood sugars closely while insulin dose is
adjusted; follow up with primary care in 3-5 days to reassess glargine
dose. Return to ED if fever recurs or breathing worsens.
""",
    ),
]


def load_notes() -> list[Note]:
    notes = []
    for i, raw in enumerate(_NOTES_RAW):
        notes.append(
            Note(
                note_id=f"N{i+1:04d}",
                patient_id=raw["patient_id"],
                admission_id=raw["admission_id"],
                text=raw["text"].strip(),
            )
        )
    return notes


if __name__ == "__main__":
    for n in load_notes():
        print(n.note_id, n.patient_id, n.admission_id, len(n.text), "chars")
