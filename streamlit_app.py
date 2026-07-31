"""
Standalone deployable version of the Clinical Note RAG Assistant.

Unlike frontend/app.py (which calls a separately-running FastAPI backend
over HTTP), this file calls the retrieval/generation/safety pipeline
directly, in-process. That makes it deployable as a single free service
(e.g. Streamlit Community Cloud) with zero backend to manage separately.

Run locally:
    streamlit run streamlit_app.py

Deploy free: push this repo to GitHub, then on https://share.streamlit.io
point a new app at this file (streamlit_app.py) in your repo. No other
setup needed -- requirements.txt at the repo root is picked up
automatically.

If you'd rather keep the FastAPI backend (e.g. because you want a real
HTTP API other clients can hit, not just this UI), use frontend/app.py +
api.py instead and deploy them separately -- see README.md "Deployment".
"""
import streamlit as st

from data_source import get_loader
from embed_store import build_default_store
from generate import get_default_generator
from ingest import chunk_all
from safety import check_note

st.set_page_config(page_title="Clinical Note RAG Assistant", page_icon="\U0001FA7A", layout="centered")


@st.cache_resource(show_spinner="Loading patient data and building index...")
def load_pipeline():
    load_notes, data_source_name = get_loader()
    notes = load_notes()
    chunks = chunk_all(notes)
    store = build_default_store(chunks)
    generator = get_default_generator()
    notes_by_patient = {}
    for n in notes:
        notes_by_patient.setdefault(n.patient_id, []).append(n)
    return {
        "notes": notes,
        "notes_by_patient": notes_by_patient,
        "store": store,
        "generator": generator,
        "data_source": data_source_name,
    }


pipeline = load_pipeline()

st.warning("\u26a0\ufe0f **Not for clinical use \u2014 demo only.**", icon="\u26a0\ufe0f")
st.title("\U0001FA7A Clinical Note RAG Assistant")
st.caption(
    f"Data source: `{pipeline['data_source']}` \u00b7 "
    f"{len(pipeline['notes_by_patient'])} patients \u00b7 "
    f"{len(pipeline['notes'])} admissions"
)

if pipeline["data_source"] == "mimic":
    st.caption(
        "Notes are template-rendered from real, de-identified MIMIC-IV Demo structured "
        "data (diagnoses, prescriptions, labs) -- not real free-text dictation. See README."
    )

patient_ids = sorted(pipeline["notes_by_patient"].keys())
patient_id = st.selectbox("Patient", patient_ids, index=0)

patient_notes = pipeline["notes_by_patient"][patient_id]
with st.expander(f"Admissions on file for patient {patient_id}"):
    for n in patient_notes:
        st.write(f"- Note `{n.note_id}`, admission `{n.admission_id}` ({len(n.text)} chars)")
    if len(patient_notes) > 1:
        st.caption(
            "\u26a0\ufe0f This patient has multiple admissions. Retrieval is scoped to the "
            "patient, not a single admission, so admission-specific answers may pull from "
            "the wrong visit. See README known limitations."
        )

if "history" not in st.session_state:
    st.session_state.history = {}
if patient_id not in st.session_state.history:
    st.session_state.history[patient_id] = []

for msg in st.session_state.history[patient_id]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for flag in msg.get("safety_flags", []):
            st.error(f"**{flag['severity'].upper()} interaction:** {flag['drug_a']} + {flag['drug_b']} \u2014 {flag['note']}")
        if msg.get("citations"):
            with st.expander("Sources"):
                for c in msg["citations"]:
                    st.markdown(f"**[{c['section']}]** (note `{c['note_id']}`, score {c['score']})")
                    st.text(c["excerpt"])

query = st.chat_input("Ask about this patient's notes...")
if query:
    st.session_state.history[patient_id].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            retrieved = pipeline["store"].query(query, top_k=5, patient_id=patient_id)
            answer = pipeline["generator"].generate(query, retrieved)

            safety_flags = []
            seen = set()
            for note in patient_notes:
                result = check_note(note.text)
                for flag in result["flags"]:
                    key = (flag.drug_a, flag.drug_b)
                    if key not in seen:
                        seen.add(key)
                        safety_flags.append(
                            {"drug_a": flag.drug_a, "drug_b": flag.drug_b, "severity": flag.severity, "note": flag.note}
                        )

        st.markdown(answer.text)
        st.caption(f"backend: `{answer.backend}`")
        for flag in safety_flags:
            st.error(f"**{flag['severity'].upper()} interaction:** {flag['drug_a']} + {flag['drug_b']} \u2014 {flag['note']}")
        if answer.citations:
            with st.expander("Sources"):
                for c in answer.citations:
                    st.markdown(f"**[{c['section']}]** (note `{c['note_id']}`, score {c['score']})")
                    st.text(c["excerpt"])

    st.session_state.history[patient_id].append(
        {
            "role": "assistant",
            "content": answer.text,
            "citations": answer.citations,
            "safety_flags": safety_flags,
        }
    )
