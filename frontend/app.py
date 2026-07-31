"""
Streamlit chat UI for the Clinical Note RAG Assistant.

Run (with the FastAPI backend already running on :8000):
    streamlit run frontend/app.py
"""
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Clinical Note RAG Assistant", page_icon="\U0001FA7A", layout="centered")

st.warning("\u26a0\ufe0f **Not for clinical use \u2014 demo only.** Synthetic patient data.", icon="\u26a0\ufe0f")
st.title("\U0001FA7A Clinical Note RAG Assistant")
st.caption("Ask questions about a patient's discharge notes. Every answer is cited to a note section.")


@st.cache_data(ttl=30)
def get_patients():
    try:
        resp = requests.get(f"{API_BASE}/patients", timeout=5)
        resp.raise_for_status()
        return resp.json()["patients"]
    except requests.RequestException:
        return None


patients = get_patients()

if patients is None:
    st.error(
        "Can't reach the backend API. Start it first with:\n\n"
        "`uvicorn api:app --reload --port 8000`"
    )
    st.stop()

patient_id = st.selectbox("Patient", patients, index=0)

with st.expander("Notes on file for this patient"):
    try:
        notes_resp = requests.get(f"{API_BASE}/patients/{patient_id}/notes", timeout=5).json()
        for n in notes_resp["notes"]:
            st.write(f"- Note `{n['note_id']}`, admission `{n['admission_id']}` ({n['char_count']} chars)")
    except requests.RequestException:
        st.write("Could not load notes.")

if "history" not in st.session_state:
    st.session_state.history = {}
if patient_id not in st.session_state.history:
    st.session_state.history[patient_id] = []

for msg in st.session_state.history[patient_id]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("safety_flags"):
            for flag in msg["safety_flags"]:
                st.error(
                    f"**{flag['severity'].upper()} interaction:** "
                    f"{flag['drug_a']} + {flag['drug_b']} \u2014 {flag['note']}"
                )
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
            try:
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={"patient_id": patient_id, "query": query, "top_k": 5},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                st.stop()

        st.markdown(data["answer"])
        st.caption(f"backend: `{data['backend']}`")

        for flag in data["safety_flags"]:
            st.error(
                f"**{flag['severity'].upper()} interaction:** "
                f"{flag['drug_a']} + {flag['drug_b']} \u2014 {flag['note']}"
            )

        if data["citations"]:
            with st.expander("Sources"):
                for c in data["citations"]:
                    st.markdown(f"**[{c['section']}]** (note `{c['note_id']}`, score {c['score']})")
                    st.text(c["excerpt"])

    st.session_state.history[patient_id].append(
        {
            "role": "assistant",
            "content": data["answer"],
            "citations": data["citations"],
            "safety_flags": data["safety_flags"],
        }
    )
