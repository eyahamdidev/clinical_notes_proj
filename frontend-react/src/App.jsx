import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  ChevronDown,
  AlertTriangle,
  CheckCircle2,
  Circle,
  Send,
  Activity,
  FileText,
  X,
  Loader2,
  WifiOff,
} from "lucide-react";
import { getHealth, listPatients, getPatientNotes, askQuestion, API_BASE } from "./api.js";

const SECTION_ORDER = [
  "Chief Complaint",
  "History of Present Illness",
  "Past Medical History",
  "Medications on Admission",
  "Physical Exam",
  "Assessment and Plan",
  "Discharge Medications",
  "Discharge Instructions",
];

// Static snapshot from eval/run_eval.py -- there's no live /eval endpoint
// (the QA set is meant to be run offline against a fixed pipeline
// configuration), so this panel shows the last recorded run rather than
// a real-time metric. Re-run `python3 eval/run_eval.py` and update these
// numbers after any retrieval/generation change.
const EVAL_METRICS = {
  answerAccuracy: 80.0,
  citationValidity: 80.0,
  refusalRate: 100.0,
  qaSetSize: 20,
  backend: "extractive-fallback",
};

function SeverityIcon({ severity }) {
  const color = severity === "major" ? "text-red-600" : "text-amber-600";
  return <AlertTriangle className={`h-4 w-4 ${color} flex-shrink-0`} />;
}

function InteractionAlert({ flag }) {
  const isMajor = flag.severity === "major";
  return (
    <div
      className={`border-l-4 ${
        isMajor ? "border-red-600 bg-red-50" : "border-amber-500 bg-amber-50"
      } border-y border-r ${
        isMajor ? "border-y-red-200 border-r-red-200" : "border-y-amber-200 border-r-amber-200"
      } rounded px-3 py-2 mb-2`}
    >
      <div className="flex items-start gap-2">
        <SeverityIcon severity={flag.severity} />
        <div className="min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className={`text-xs font-semibold tracking-wide uppercase ${
                isMajor ? "text-red-700" : "text-amber-700"
              }`}
            >
              {flag.severity} interaction
            </span>
            <span className="text-xs font-mono text-slate-500">
              {flag.drug_a} + {flag.drug_b}
            </span>
          </div>
          <p className="text-sm text-slate-700 mt-0.5 leading-snug">{flag.note}</p>
        </div>
      </div>
    </div>
  );
}

function CitationPill({ citation, onClick, active }) {
  return (
    <button
      onClick={onClick}
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors ${
        active
          ? "border-teal-600 bg-teal-50 text-teal-700"
          : "border-slate-300 bg-white text-slate-600 hover:border-teal-500 hover:text-teal-700"
      }`}
    >
      <FileText className="h-3 w-3" />
      Source: {citation.section}
    </button>
  );
}

function SectionChecklistItem({ label, available }) {
  return (
    <div className="flex items-center gap-2 py-1">
      {available ? (
        <CheckCircle2 className="h-3.5 w-3.5 text-teal-600 flex-shrink-0" />
      ) : (
        <Circle className="h-3.5 w-3.5 text-slate-300 flex-shrink-0" />
      )}
      <span className={`text-xs ${available ? "text-slate-700" : "text-slate-400"}`}>{label}</span>
    </div>
  );
}

function MetricRow({ label, value, tone }) {
  const toneClass = tone === "emerald" ? "text-emerald-700 bg-emerald-50" : "text-teal-700 bg-teal-50";
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-slate-600">{label}</span>
      <span className={`text-xs font-mono font-semibold rounded px-1.5 py-0.5 ${toneClass}`}>{value}</span>
    </div>
  );
}

function EvalMetricsPanel({ onClose }) {
  return (
    <div className="absolute right-4 top-11 z-20 w-72 rounded-md border border-slate-300 bg-white shadow-lg">
      <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-600">
          Evaluation metrics
        </span>
        <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="px-3 py-3 space-y-2.5">
        <MetricRow label="Answer accuracy" value={`${EVAL_METRICS.answerAccuracy.toFixed(1)}%`} tone="teal" />
        <MetricRow label="Citation validity" value={`${EVAL_METRICS.citationValidity.toFixed(1)}%`} tone="teal" />
        <MetricRow label="Correct-refusal rate" value={`${EVAL_METRICS.refusalRate.toFixed(1)}%`} tone="emerald" />
        <div className="pt-1.5 border-t border-slate-100 text-xs text-slate-400 font-mono">
          n = {EVAL_METRICS.qaSetSize} \u00b7 backend: {EVAL_METRICS.backend}
        </div>
        <p className="text-xs text-slate-400 pt-1">
          Snapshot from the last <code className="font-mono">eval/run_eval.py</code> run, not live.
        </p>
      </div>
    </div>
  );
}

function ConnectionBanner({ status, apiBase }) {
  if (status === "ok") return null;
  return (
    <div className="flex items-center gap-1.5 bg-red-50 border-t border-red-200 px-4 py-1">
      <WifiOff className="h-3 w-3 text-red-600 flex-shrink-0" />
      <span className="text-xs text-red-800 font-medium">
        {status === "checking"
          ? "Connecting to backend\u2026"
          : `Can't reach the API at ${apiBase}. Start it with: uvicorn api:app --reload --port 8000`}
      </span>
    </div>
  );
}

export default function App() {
  const [connStatus, setConnStatus] = useState("checking"); // checking | ok | error
  const [dataSource, setDataSource] = useState(null);

  const [patientIds, setPatientIds] = useState([]);
  const [patientId, setPatientId] = useState(null);
  const [patientLoading, setPatientLoading] = useState(false);
  const [patientMeta, setPatientMeta] = useState(null); // { age, gender, admission_type, discharge_location, available_sections, admission_id }

  const [entries, setEntries] = useState([]);
  const [query, setQuery] = useState("");
  const [asking, setAsking] = useState(false);
  const [activeEntryIdx, setActiveEntryIdx] = useState(null);
  const [activeCitationIdx, setActiveCitationIdx] = useState(null);
  const [showEval, setShowEval] = useState(false);

  const scrollRef = useRef(null);
  const docRefs = useRef({});

  // --- initial health check + patient list ---
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const health = await getHealth();
        if (cancelled) return;
        setDataSource(health.data_source);
        setConnStatus("ok");
        const { patients } = await listPatients();
        if (cancelled) return;
        setPatientIds(patients);
        if (patients.length > 0) setPatientId(patients[0]);
      } catch (e) {
        if (!cancelled) setConnStatus("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // --- load per-patient metadata whenever the selected patient changes ---
  useEffect(() => {
    if (!patientId) return;
    let cancelled = false;
    setPatientLoading(true);
    setEntries([]);
    setActiveEntryIdx(null);
    setActiveCitationIdx(null);
    (async () => {
      try {
        const data = await getPatientNotes(patientId);
        if (cancelled) return;
        // Multiple admissions -> show the most recent one's demographics/
        // sections in the left pane, but note the ambiguity risk explicitly.
        const primary = data.notes[data.notes.length - 1];
        setPatientMeta({ ...primary, admissionCount: data.notes.length });
      } catch (e) {
        if (!cancelled) setPatientMeta(null);
      } finally {
        if (!cancelled) setPatientLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [patientId]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  const handleAsk = useCallback(async () => {
    if (!query.trim() || !patientId || asking) return;
    const q = query;
    setQuery("");
    setAsking(true);
    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    try {
      const res = await askQuestion(patientId, q);
      const entryIdx = entries.length;
      setEntries((prev) => [
        ...prev,
        {
          query: q,
          answer: res.answer,
          citations: res.citations,
          flags: res.safety_flags,
          backend: res.backend,
          timestamp,
        },
      ]);
      setActiveEntryIdx(entryIdx);
      setActiveCitationIdx(res.citations.length > 0 ? 0 : null);
    } catch (e) {
      const entryIdx = entries.length;
      setEntries((prev) => [
        ...prev,
        {
          query: q,
          answer: `Request failed: ${e.message}`,
          citations: [],
          flags: [],
          backend: "error",
          timestamp,
          isError: true,
        },
      ]);
      setActiveEntryIdx(entryIdx);
    } finally {
      setAsking(false);
    }
  }, [query, patientId, asking, entries.length]);

  function jumpToCitation(entryIdx, citationIdx) {
    setActiveEntryIdx(entryIdx);
    setActiveCitationIdx(citationIdx);
    const key = `${entryIdx}-${citationIdx}`;
    requestAnimationFrame(() => {
      const el = docRefs.current[key];
      if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  const activeEntry = activeEntryIdx !== null ? entries[activeEntryIdx] : null;
  const availableSections = patientMeta?.available_sections || [];

  return (
    <div className="h-screen w-full flex flex-col bg-slate-50 text-slate-800 font-sans">
      {/* Top bar */}
      <header className="flex-shrink-0 border-b border-slate-300 bg-white">
        <div className="flex items-center justify-between px-4 py-2 relative">
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded bg-slate-800 text-white text-xs font-bold font-mono">
              CN
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-900 leading-none">
                Clinical Note RAG Assistant
              </h1>
              <p className="text-xs text-slate-400 leading-none mt-0.5">
                {dataSource ? `Data source: ${dataSource}` : "Retrieval-grounded chart review"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowEval((v) => !v)}
              className="flex items-center gap-1.5 rounded border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-600 hover:border-slate-400 hover:text-slate-800"
            >
              <Activity className="h-3.5 w-3.5" />
              Eval metrics
              <ChevronDown className="h-3 w-3" />
            </button>
          </div>
          {showEval && <EvalMetricsPanel onClose={() => setShowEval(false)} />}
        </div>
        <div className="flex items-center gap-1.5 bg-amber-50 border-t border-amber-200 px-4 py-1">
          <AlertTriangle className="h-3 w-3 text-amber-600 flex-shrink-0" />
          <span className="text-xs text-amber-800 font-medium">
            Not for clinical use \u2014 demo only. De-identified data.
          </span>
        </div>
        <ConnectionBanner status={connStatus} apiBase={API_BASE} />
      </header>

      {connStatus !== "ok" ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            {connStatus === "checking" ? (
              <Loader2 className="h-6 w-6 text-slate-400 animate-spin mx-auto mb-2" />
            ) : (
              <WifiOff className="h-6 w-6 text-red-500 mx-auto mb-2" />
            )}
            <p className="text-sm text-slate-500">
              {connStatus === "checking" ? "Connecting to the backend\u2026" : "Backend unreachable."}
            </p>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex min-h-0">
          {/* LEFT: Patient context */}
          <aside className="w-64 flex-shrink-0 border-r border-slate-300 bg-white overflow-y-auto">
            <div className="p-3">
              <label className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Patient
              </label>
              <div className="relative mt-1">
                <select
                  value={patientId || ""}
                  onChange={(e) => setPatientId(e.target.value)}
                  className="w-full appearance-none rounded border border-slate-300 bg-white px-2.5 py-1.5 pr-7 text-sm font-mono text-slate-800 focus:border-teal-600 focus:outline-none focus:ring-1 focus:ring-teal-600"
                >
                  {patientIds.map((id) => (
                    <option key={id} value={id}>
                      {id}
                    </option>
                  ))}
                </select>
                <ChevronDown className="pointer-events-none absolute right-2 top-2 h-3.5 w-3.5 text-slate-400" />
              </div>
            </div>

            {patientLoading || !patientMeta ? (
              <div className="mx-3 rounded border border-slate-200 bg-slate-50 px-3 py-4 text-center">
                <Loader2 className="h-4 w-4 text-slate-400 animate-spin mx-auto" />
              </div>
            ) : (
              <>
                <div className="mx-3 rounded border border-slate-200 bg-slate-50 px-3 py-2.5">
                  <div className="grid grid-cols-2 gap-y-1.5 text-xs">
                    <span className="text-slate-400">Patient ID</span>
                    <span className="text-right font-mono text-slate-700">{patientId}</span>
                    <span className="text-slate-400">Age</span>
                    <span className="text-right font-mono text-slate-700">{patientMeta.age ?? "\u2014"}</span>
                    <span className="text-slate-400">Gender</span>
                    <span className="text-right font-mono text-slate-700">{patientMeta.gender ?? "\u2014"}</span>
                    <span className="text-slate-400">Admission</span>
                    <span className="text-right font-mono text-slate-700">{patientMeta.admission_id}</span>
                    <span className="text-slate-400">Disposition</span>
                    <span className="text-right font-mono text-slate-700 text-[11px] break-words">
                      {patientMeta.discharge_location ?? "\u2014"}
                    </span>
                  </div>
                  {patientMeta.admissionCount > 1 && (
                    <p className="mt-2 pt-2 border-t border-slate-200 text-xs text-amber-700 leading-snug">
                      \u26a0 This patient has {patientMeta.admissionCount} admissions. Retrieval is
                      scoped to the patient, not one visit \u2014 answers may pull from a different
                      admission than the one shown here.
                    </p>
                  )}
                </div>

                <div className="p-3 mt-1">
                  <label className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Available sections
                  </label>
                  <div className="mt-1.5">
                    {SECTION_ORDER.map((s) => (
                      <SectionChecklistItem key={s} label={s} available={availableSections.includes(s)} />
                    ))}
                  </div>
                </div>
              </>
            )}
          </aside>

          {/* MIDDLE: Query / documentation log */}
          <main className="flex-1 flex flex-col min-w-0 bg-slate-50">
            <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
              {entries.length === 0 && (
                <div className="flex h-full items-center justify-center">
                  <p className="text-sm text-slate-400">
                    {patientId
                      ? `Ask a question about patient ${patientId}'s record to begin.`
                      : "Select a patient to begin."}
                  </p>
                </div>
              )}
              {entries.map((entry, entryIdx) => (
                <div key={entryIdx} className="rounded border border-slate-300 bg-white">
                  <div className="flex items-center justify-between border-b border-slate-100 px-3 py-1.5">
                    <span className="text-xs font-mono text-slate-400">{entry.timestamp}</span>
                    <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Query response
                    </span>
                  </div>
                  <div className="px-3 py-2 border-b border-slate-100">
                    <p className="text-sm text-slate-500 italic">"{entry.query}"</p>
                  </div>
                  <div className="px-3 py-2.5">
                    {entry.flags?.map((flag, i) => (
                      <InteractionAlert key={i} flag={flag} />
                    ))}
                    <p
                      className={`text-sm leading-relaxed whitespace-pre-line ${
                        entry.isError ? "text-red-600" : "text-slate-800"
                      }`}
                    >
                      {entry.answer}
                    </p>
                    {entry.citations?.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mt-2.5">
                        {entry.citations.map((c, citationIdx) => (
                          <CitationPill
                            key={citationIdx}
                            citation={c}
                            active={activeEntryIdx === entryIdx && activeCitationIdx === citationIdx}
                            onClick={() => jumpToCitation(entryIdx, citationIdx)}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {asking && (
                <div className="flex items-center gap-2 text-xs text-slate-400 px-1">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Retrieving and answering\u2026
                </div>
              )}
            </div>

            <div className="flex-shrink-0 border-t border-slate-300 bg-white px-4 py-3">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                  disabled={!patientId || asking}
                  placeholder="Ask about this patient's record..."
                  className="flex-1 rounded border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-teal-600 focus:outline-none focus:ring-1 focus:ring-teal-600 disabled:bg-slate-50"
                />
                <button
                  onClick={handleAsk}
                  disabled={!patientId || asking || !query.trim()}
                  className="flex items-center gap-1.5 rounded bg-slate-800 px-3.5 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {asking ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
                  Ask
                </button>
              </div>
              <p className="mt-1.5 text-xs text-slate-400">
                Answers are grounded to retrieved note excerpts only. Unsupported questions return
                "Not found in record."
              </p>
            </div>
          </main>

          {/* RIGHT: Document reference / source citations */}
          <aside className="w-80 flex-shrink-0 border-l border-slate-300 bg-white overflow-y-auto">
            <div className="border-b border-slate-200 px-3 py-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Document reference
              </span>
              <p className="text-xs text-slate-400 mt-0.5">
                Raw excerpts backing the active response
              </p>
            </div>
            <div className="p-3 space-y-2.5">
              {!activeEntry || activeEntry.citations.length === 0 ? (
                <p className="text-xs text-slate-400">No sources cited yet.</p>
              ) : (
                activeEntry.citations.map((doc, citationIdx) => {
                  const isActive = activeCitationIdx === citationIdx;
                  return (
                    <div
                      key={citationIdx}
                      ref={(el) => (docRefs.current[`${activeEntryIdx}-${citationIdx}`] = el)}
                      className={`rounded border transition-all px-3 py-2.5 ${
                        isActive ? "border-teal-600 ring-1 ring-teal-600 bg-teal-50" : "border-slate-200 bg-white"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold text-slate-700">{doc.section}</span>
                        <span className="text-xs font-mono text-slate-400">score {doc.score.toFixed(2)}</span>
                      </div>
                      <p className="text-xs font-mono text-slate-500 mb-1">
                        {doc.note_id} \u00b7 {doc.chunk_id}
                      </p>
                      <p className="text-xs text-slate-700 leading-snug whitespace-pre-line border-l-2 border-slate-200 pl-2">
                        {doc.excerpt}
                      </p>
                    </div>
                  );
                })
              )}
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
