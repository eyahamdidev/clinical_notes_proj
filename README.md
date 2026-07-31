# Clinical Note RAG Assistant

A retrieval-augmented chatbot over real MIMIC-IV clinical data that cites
its sources, refuses to answer when the record doesn't say, and
independently flags dangerous drug combinations regardless of what the LLM
says.

**Not for clinical use   demo only.**

---

## What's in this repo

- Real, de-identified patient data from the **MIMIC-IV Clinical Database
  Demo** (100 patients, 275 hospital admissions), included in
  `data/mimic-iv-clinical-database-demo-2.2/`
- A loader that turns that structured data into discharge-summary-shaped
  notes (`data/mimic_loader.py`)   **read the "About the data" section
  below before calling these "real clinical notes,"** the distinction
  matters
- Section-aware chunking, TF-IDF retrieval scoped per-patient, a
  citation-forcing generation layer with a real "not found" refusal
  mechanism, and a deterministic drug-interaction safety layer
  independent of the LLM
- A FastAPI backend + a small Streamlit chat UI (intentionally simple for
  now   see "Roadmap" at the bottom)
- A hand-labeled 20-question eval set run against the real data:
  **80% answer accuracy, 100% correct-refusal rate** (never hallucinated)
    see `eval/eval_results.md`

---

## About the data   read this before presenting the project

The **MIMIC-IV Clinical Database Demo** (what's bundled in this repo)
explicitly **excludes free-text clinical notes**. Straight from
`data/mimic-iv-clinical-database-demo-2.2/README.txt`:

> "The dataset includes similar content to MIMIC-IV, but excludes
> free-text clinical notes."

Free-text discharge summaries live in a separate PhysioNet project
(MIMIC-IV-Note), which requires its own credentialing beyond what the open
demo needs, because free text carries more re-identification risk than
structured/coded fields.

**So what `data/mimic_loader.py` actually does:** it pulls real,
de-identified structured fields per hospital admission   ICD-coded
diagnoses, prescriptions (drug/dose/route), abnormal labs, vitals   and
renders them into a discharge-summary-shaped narrative using a template.

- The **facts** (which drugs, which diagnoses, which lab values, which
  combinations) are real MIMIC-IV data, not invented.
- The **prose** connecting them is templated, not a real physician's
  dictation.

Be precise about this distinction if you present this project: it's "real
structured EHR data, template-rendered as notes," not "real discharge
summaries." This is honestly still a strong basis for a RAG demo   the
retrieval, citation, and safety-flagging problems are exactly the same
either way, and the drug interactions the safety layer catches are real
combinations real patients were actually prescribed.

**If you get access to real free-text notes later** (MIMIC-IV-Note, via
additional PhysioNet credentialing), point `data_source.py` at a new
loader for it   nothing else in the pipeline needs to change, since
everything downstream only depends on the `Note` dataclass shape
(`note_id`, `patient_id`, `admission_id`, `text`).

### Data license

The MIMIC-IV Demo is distributed under the **ODC Open Database License
(ODbL)**   see `data/mimic-iv-clinical-database-demo-2.2/LICENSE.txt`.
Keep that license file with the data if you redistribute this repo, and
cite:

- Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark,
  R. (2023). MIMIC-IV (version 2.2). PhysioNet.
  https://doi.org/10.13026/6mm1-ek67
- Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely
  accessible electronic health record dataset. *Sci Data* 10, 1 (2023).
- Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
  *Circulation* 101(23), e215–e220.

---

## Architecture

```
data/mimic-iv-clinical-database-demo-2.2/   Real MIMIC-IV demo CSVs (ODbL)
data/mimic_loader.py                        Structured data -> Note objects
data/synthetic_notes.py                     Fallback: fully synthetic notes
                                             (used automatically if the real
                                             data folder is ever removed)
data_source.py                              Picks mimic vs synthetic loader
        │
        ▼
ingest.py                 Section-aware chunking (header-split, then
                           token-budget sub-chunking only for long sections)
        │
        ▼
embed_store.py             TF-IDF embedding (+ stemming tokenizer) +
                           numpy cosine vector store, per-patient scoped
                           retrieval. Pluggable: swap in bge-small-en or
                           PubMedBERT sentence embeddings, see file docstring.
        │
        ▼
generate.py                Answer generation:
                             - ExtractiveGenerator: fully local, zero
                               dependency, zero API key, grounded by
                               construction (can only say what's in the
                               retrieved chunks)
        │
        ▼
safety.py                  Deterministic medication extraction (regex) +
                           drug interaction lookup (data/drug_interactions.csv),
                           runs independently of the generator
        │
        ▼
api.py                     FastAPI: /patients, /patients/{id}/notes, /chat
        │
        ├──▶ frontend/app.py        Streamlit chat UI, calls FastAPI over HTTP
        │                           (two processes, local dev)
        │
        └──▶ frontend-react/        Three-pane clinical dashboard (React +
                                     Tailwind), calls FastAPI over HTTP.
                                     Patient context / query log / document
                                     reference panes, drug-interaction alert
                                     cards, live eval-metrics panel.

streamlit_app.py           Standalone Streamlit app: calls the pipeline
                           in-process (no separate FastAPI backend needed).
                           Use this for free single-service deployment
                           (Streamlit Community Cloud) -- see "Deployment".
```

---

## Frontends: which one to use

There are three UIs in this repo, all hitting the same backend logic:

| File | What it is | When to use it |
|---|---|---|
| `frontend-react/` | Three-pane clinical dashboard (patient context / query log / document reference), drug-interaction alert cards, eval-metrics panel | The polished option   what you'd show someone. Needs the FastAPI backend running separately. |
| `frontend/app.py` | Simple Streamlit chat UI, calls FastAPI over HTTP | Quick local testing without touching Node/npm. |
| `streamlit_app.py` | Same UI as above, but calls the pipeline directly in-process (no FastAPI needed) | Free single-service deployment (see below) or a minimal local demo with one process instead of two. |

## Running it

**Backend (required for `frontend/app.py` and `frontend-react/`):**
```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

**React dashboard** (recommended UI):
```bash
cd frontend-react
npm install
npm run dev
```
Open the URL Vite prints (usually `http://localhost:5173`). It talks to
the backend at `http://localhost:8000` by default   override with
`VITE_API_BASE` (see `.env.example`) if you're running the backend
elsewhere.

**Streamlit chat UI (two-process version)**:
```bash
streamlit run frontend/app.py
```

**Streamlit chat UI (standalone, one process, no FastAPI needed)**:
```bash
streamlit run streamlit_app.py
```

Pick a patient (MIMIC `subject_id`, e.g. `10001725`), ask a question.

By default the app auto-detects and uses the real MIMIC data bundled in
this repo. To force the synthetic fallback instead (e.g. to test without
the real data present): `DATA_SOURCE=synthetic uvicorn api:app --reload`.

**What's actually been tested, concretely:** every backend layer
(chunking, retrieval, safety, FastAPI endpoints) via `TestClient`. The
React dashboard was verified against the *live* running backend, not
mocked data   health check, patient list (100 real patients), per-patient
metadata fetch, asking a question end-to-end through `/chat`, and the
citation-pill-click → source-card-highlight interaction were all exercised
programmatically against a real `uvicorn` process, confirming the full
fetch contract works, not just that the code compiles. `frontend/app.py`
(the simpler Streamlit version) was checked for contract correctness but
not click-tested in an actual browser   worth a quick smoke test before
you demo it live.

---




## Evaluation

`eval/qa_set.json` has 20 hand-labeled QA pairs against 5 real admissions
(15 answerable, 5 deliberately unanswerable   e.g. asking about
chemotherapy or dialysis when the record never mentions it). Patients with
only a single admission in the demo dataset were chosen deliberately, so
"what was the chief complaint" style questions aren't ambiguous across
multiple visits (see "Known limitations" below   that ambiguity is real
and worth knowing about even though the eval set avoids it).

```bash
python3 eval/run_eval.py
```

### Current results (TF-IDF + extractive-fallback backend, on real MIMIC data)

| Metric | Score |
|---|---|
| Answer accuracy | 16/20 (80.0%) |
| Citation-section accuracy | 12/15 (80.0%) |
| **Correct-refusal rate (never hallucinated)** | **5/5 (100%)** |

Full per-question breakdown: `eval/eval_results.md`.

Misses are concentrated exactly where you'd expect from a TF-IDF-only
baseline: vocabulary mismatch on drug-class questions ("What statin..."
not matching "Simvastatin" by name, "What NSAID..." not matching
"Ibuprofen"/"Ketorolac"). A dense embedding model would close most of this
gap since it captures semantic relationships (drug → drug class) that
bag-of-words retrieval structurally cannot.

---

## Known limitations (worth stating out loud)

- **Notes are template-rendered from structured data, not real free-text
  dictation**   see "About the data" above. This is the single most
  important caveat for this project.
- **Retrieval is scoped per-patient, not per-admission.** A patient with
  multiple hospital stays in the dataset will have all their admissions'
  chunks searched together, so an admission-specific question ("what was
  *this* discharge's chief complaint") can retrieve the wrong visit. Fix:
  add an `admission_id` filter to the UI/API alongside `patient_id`.
- **TF-IDF retrieval has real vocabulary-mismatch failures**, especially
  on drug-class terms (statin/NSAID/anticoagulant vs. specific drug
  names)   see eval results above. Biggest lever for improving accuracy:
  swap in a dense embedder (see `embed_store.py` docstring).
- **Medication extraction is regex-based**, tuned to the
  `"- Drugname 40 mg PO daily"` bullet format. A real system would use a
  clinical NER model (e.g. MedCAT, scispaCy).
- **The interaction table is a small, hand-curated CSV** (~18 pairs), not
  a DrugBank export   RxNorm and DDInter are free, more complete
  alternatives worth integrating for anything beyond a demo.
- **The generator returns one chunk verbatim** rather than synthesizing
  across chunks   plugging in a real local model (e.g. Llama via Ollama)
  would fix this; see the swap point noted in `generate.py`.

---

## Roadmap (intentionally not done yet)

This is v2: real data wired in, backend solid, a real three-pane clinical
dashboard shipped. Next, roughly in priority order:
1. Admission-level filtering in the UI (patient dropdown → admission
   dropdown), not just patient-level   the dashboard currently surfaces
   this ambiguity with a warning when a patient has multiple admissions,
   but doesn't let you disambiguate yet
2. Dense embeddings (`bge-small-en` or a biomedical sentence embedder)
3. Real LLM generation by default instead of the extractive fallback
4. A live `/eval` endpoint so the eval-metrics panel reflects the current
   pipeline config instead of a snapshot you have to re-run manually
5. Bigger, DrugBank/RxNorm-backed interaction table
6. Chat history persistence across page reloads in the React dashboard
