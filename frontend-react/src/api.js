// In dev, set VITE_API_BASE=http://localhost:8000 in a local .env file.
// In production (built and served by FastAPI itself), leave it unset so
// requests go to the same origin the page was loaded from.
const API_BASE = import.meta.env.VITE_API_BASE || "";

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

export function getHealth() {
  return request("/health");
}

export function listPatients() {
  return request("/patients");
}

export function getPatientNotes(patientId) {
  return request(`/patients/${encodeURIComponent(patientId)}/notes`);
}

export function askQuestion(patientId, query, topK = 5) {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify({ patient_id: patientId, query, top_k: topK }),
  });
}

export { API_BASE };
