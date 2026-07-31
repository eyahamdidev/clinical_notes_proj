"""
FastAPI backend for the Clinical Note RAG Assistant.

Run:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /patients                 -> list patient IDs available in the index
    GET  /patients/{pid}/notes     -> list note/admission summaries for a patient
    POST /chat                     -> ask a question, scoped to one patient
    GET  /health                   -> basic liveness check

NOT FOR CLINICAL USE. This is a demo system on synthetic data.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_source import get_loader
from embed_store import build_default_store
from generate import get_default_generator
from ingest import SECTION_HEADERS, chunk_all, chunk_note
from safety import check_note

DISCLAIMER = "Not for clinical use \u2014 demo only."

app = FastAPI(title="Clinical Note RAG Assistant", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Build the index once at startup -----------------------------------
_load_notes, _data_source_name = get_loader()
_notes = _load_notes()
_chunks = chunk_all(_notes)
_store = build_default_store(_chunks)
_generator = get_default_generator()


def _available_sections(note) -> list[str]:
    """Which of the canonical sections actually have real content in this
    note, vs. a "No X recorded" placeholder (mimic_loader emits these when
    the underlying structured data had nothing for that section). Used by
    the UI's per-patient section checklist."""
    present = {c.section for c in chunk_note(note)}
    available = []
    for header in SECTION_HEADERS:
        if header not in present:
            continue
        body = next((c.text for c in chunk_note(note) if c.section == header), "")
        if body.strip().lower().startswith("no "):
            continue
        available.append(header)
    return available


class ChatRequest(BaseModel):
    patient_id: str
    query: str
    top_k: int = 5


class Citation(BaseModel):
    section: str
    note_id: str
    chunk_id: str
    score: float
    excerpt: str


class SafetyFlag(BaseModel):
    drug_a: str
    drug_b: str
    severity: str
    note: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    backend: str
    safety_flags: list[SafetyFlag]
    disclaimer: str = DISCLAIMER


@app.get("/health")
def health():
    return {
        "status": "ok",
        "generator_backend": _generator.backend_name,
        "data_source": _data_source_name,
        "num_patients": len({n.patient_id for n in _notes}),
        "num_admissions": len(_notes),
        "disclaimer": DISCLAIMER,
    }


@app.get("/patients")
def list_patients():
    patient_ids = sorted({n.patient_id for n in _notes})
    return {"patients": patient_ids, "disclaimer": DISCLAIMER}


@app.get("/patients/{patient_id}/notes")
def patient_notes(patient_id: str):
    notes = [n for n in _notes if n.patient_id == patient_id]
    if not notes:
        raise HTTPException(status_code=404, detail=f"No notes found for patient {patient_id}")
    return {
        "patient_id": patient_id,
        "notes": [
            {
                "note_id": n.note_id,
                "admission_id": n.admission_id,
                "char_count": len(n.text),
                "age": n.meta.get("age"),
                "gender": n.meta.get("gender"),
                "admission_type": n.meta.get("admission_type"),
                "discharge_location": n.meta.get("discharge_location"),
                "available_sections": _available_sections(n),
            }
            for n in notes
        ],
        "disclaimer": DISCLAIMER,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    patient_notes_list = [n for n in _notes if n.patient_id == req.patient_id]
    if not patient_notes_list:
        raise HTTPException(status_code=404, detail=f"Unknown patient_id {req.patient_id!r}")

    # --- retrieval, scoped strictly to this patient ---
    retrieved = _store.query(req.query, top_k=req.top_k, patient_id=req.patient_id)
    answer = _generator.generate(req.query, retrieved)

    # --- safety layer: independent of the LLM's answer, deterministic ---
    all_flags = []
    seen = set()
    for note in patient_notes_list:
        result = check_note(note.text)
        for flag in result["flags"]:
            key = (flag.drug_a, flag.drug_b)
            if key not in seen:
                seen.add(key)
                all_flags.append(
                    SafetyFlag(drug_a=flag.drug_a, drug_b=flag.drug_b, severity=flag.severity, note=flag.note)
                )

    return ChatResponse(
        answer=answer.text,
        citations=[Citation(**c) for c in answer.citations],
        backend=answer.backend,
        safety_flags=all_flags,
    )


# --- Serve the built React frontend (if present) on the same origin -----
# Mounted LAST so it only catches requests that don't match an API route
# above (e.g. "/", "/assets/*"), never overriding /health, /patients, /chat.
from pathlib import Path

from fastapi.staticfiles import StaticFiles

_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
