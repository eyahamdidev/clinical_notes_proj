# Clinical Note RAG Assistant — Evaluation Results

Generator backend: `extractive-fallback`
QA set size: 20 (hand-labeled, 15 answerable / 5 intentionally unanswerable)

- **Answer accuracy:** 16/20 (80.0%)
- **Citation-section accuracy:** 12/15 (80.0%)
- **Correct-refusal rate:** 5/5 (100.0%)

| ID | Type | Query | Correct | Citation OK | Answer (truncated) |
|---|---|---|---|---|---|
| qa1 | answerable | What is the chief complaint for this admission? | yes | yes | Other specified retention of urine. [Chief Complaint] |
| qa2 | answerable | Does the patient have a history of asthma? | yes | yes | 1. Other anaphylactic reaction 2. Prolapse of vaginal vault after hysterectomy 3. Asthma, unspecified type, unspecified  |
| qa3 | answerable | What NSAID pain medication was given during this admission? | no | no | Not found in record. |
| qa4 | refusal | Is the patient on chemotherapy? | yes | n/a | Not found in record. |
| qa5 | answerable | What is the chief complaint for this admission? | yes | yes | Non-ST elevation (NSTEMI) myocardial infarction. [Chief Complaint] |
| qa6 | answerable | What anticoagulant was the patient discharged on? | no | yes | Follow up as directed. Discharge disposition: SKILLED NURSING FACILITY. [Discharge Instructions] |
| qa7 | answerable | Did the patient have a Clostridium difficile infection? | yes | yes | 1. Cardiogenic shock 2. Heart failure, unspecified 3. Other shock 4. Enterocolitis due to Clostridium difficile 5. Acute |
| qa8 | refusal | Is the patient on dialysis? | yes | n/a | Not found in record. |
| qa9 | answerable | What is the chief complaint for this admission? | yes | yes | Malignant neoplasm of lower third of esophagus. [Chief Complaint] |
| qa10 | answerable | Does the patient have Barrett's esophagus? | yes | yes | 1. Acidosis 2. Barrett's esophagus 3. Esophageal reflux [Past Medical History] |
| qa11 | answerable | Where was the patient discharged to? | yes | yes | Follow up as directed. Discharge disposition: HOME HEALTH CARE. [Discharge Instructions] |
| qa12 | refusal | Does the patient have a pacemaker? | yes | n/a | Not found in record. |
| qa13 | answerable | What is the chief complaint for this admission? | yes | yes | Coronary atherosclerosis of native coronary artery. [Chief Complaint] |
| qa14 | answerable | What statin is the patient on? | no | no | Not found in record. |
| qa15 | answerable | Does the patient have hyperlipidemia? | yes | yes | 1. Intermediate coronary syndrome 2. Hemorrhage complicating a procedure 3. Cardiac complications, not elsewhere classif |
| qa16 | refusal | Is the patient on insulin? | yes | n/a | Not found in record. |
| qa17 | answerable | What is the chief complaint for this admission? | yes | yes | Other encephalopathy. [Chief Complaint] |
| qa18 | answerable | Did the patient have ventilator associated pneumonia? | yes | yes | 1. Cerebral artery occlusion, unspecified with cerebral infarction 2. Pneumonitis due to inhalation of food or vomitus 3 |
| qa19 | answerable | What sedative infusion was used during this admission? | no | no | Not found in record. |
| qa20 | refusal | Does the patient have a history of diabetes? | yes | n/a | Not found in record. |