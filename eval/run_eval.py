"""
Evaluation harness: runs the hand-labeled QA set through the live pipeline
and scores:
  - Answer accuracy: for answerable questions, does the answer contain the
    expected keyword(s)? For unanswerable questions, did the system
    correctly say "Not found in record" instead of hallucinating?
  - Citation correctness: for answerable questions, does at least one
    citation point to the expected section?

Run:
    python3 eval/run_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_source import get_loader  # noqa: E402
from embed_store import build_default_store  # noqa: E402
from generate import NOT_FOUND_PHRASE, get_default_generator  # noqa: E402
from ingest import chunk_all  # noqa: E402

QA_SET_PATH = Path(__file__).parent / "qa_set.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.md"


def load_qa_set():
    with open(QA_SET_PATH) as f:
        return json.load(f)


def score_answer(item: dict, answer_text: str) -> bool:
    lower = answer_text.lower()
    if not item["answerable"]:
        return NOT_FOUND_PHRASE.lower() in lower
    return any(kw.lower() in lower for kw in item["expected_keywords"])


def score_citation(item: dict, citations: list[dict]) -> bool | None:
    if not item["answerable"]:
        return None  # not applicable for refusal cases
    if not citations:
        return False
    return any(c["section"] == item["expected_section"] for c in citations)


def run():
    load_notes, data_source_name = get_loader()
    chunks = chunk_all(load_notes())
    store = build_default_store(chunks)
    generator = get_default_generator()
    print(f"Data source: {data_source_name}")

    qa_set = load_qa_set()
    rows = []
    for item in qa_set:
        retrieved = store.query(item["query"], top_k=5, patient_id=item["patient_id"])
        answer = generator.generate(item["query"], retrieved)
        citations = answer.citations
        answer_correct = score_answer(item, answer.text)
        citation_correct = score_citation(item, citations)
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "answerable": item["answerable"],
                "answer": answer.text.replace("\n", " ")[:120],
                "answer_correct": answer_correct,
                "citation_correct": citation_correct,
            }
        )

    n = len(rows)
    n_answer_correct = sum(r["answer_correct"] for r in rows)
    citation_applicable = [r for r in rows if r["citation_correct"] is not None]
    n_citation_correct = sum(1 for r in citation_applicable if r["citation_correct"])

    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable_rows = [r for r in rows if not r["answerable"]]
    n_correct_refusals = sum(r["answer_correct"] for r in unanswerable_rows)

    answer_accuracy = n_answer_correct / n * 100
    citation_accuracy = (n_citation_correct / len(citation_applicable) * 100) if citation_applicable else 0.0
    refusal_accuracy = (n_correct_refusals / len(unanswerable_rows) * 100) if unanswerable_rows else 0.0

    print(f"Generator backend: {generator.backend_name}")
    print(f"Overall answer accuracy: {n_answer_correct}/{n} ({answer_accuracy:.1f}%)")
    print(
        f"Citation-section accuracy (answerable only): "
        f"{n_citation_correct}/{len(citation_applicable)} ({citation_accuracy:.1f}%)"
    )
    print(
        f"Correct-refusal rate (unanswerable questions correctly flagged "
        f"'not found'): {n_correct_refusals}/{len(unanswerable_rows)} ({refusal_accuracy:.1f}%)"
    )
    print()
    for r in rows:
        mark = "\u2713" if r["answer_correct"] else "\u2717"
        cite_mark = "" if r["citation_correct"] is None else (" | cite \u2713" if r["citation_correct"] else " | cite \u2717")
        print(f"[{mark}] {r['id']} ({'answerable' if r['answerable'] else 'refusal'}){cite_mark}: {r['query']}")
        print(f"      -> {r['answer']}")

    # Write a markdown report too, useful to attach to a CV/portfolio.
    lines = [
        "# Clinical Note RAG Assistant \u2014 Evaluation Results",
        "",
        f"Generator backend: `{generator.backend_name}`",
        f"QA set size: {n} (hand-labeled, {len(answerable_rows)} answerable / "
        f"{len(unanswerable_rows)} intentionally unanswerable)",
        "",
        f"- **Answer accuracy:** {n_answer_correct}/{n} ({answer_accuracy:.1f}%)",
        f"- **Citation-section accuracy:** {n_citation_correct}/{len(citation_applicable)} "
        f"({citation_accuracy:.1f}%)",
        f"- **Correct-refusal rate:** {n_correct_refusals}/{len(unanswerable_rows)} "
        f"({refusal_accuracy:.1f}%)",
        "",
        "| ID | Type | Query | Correct | Citation OK | Answer (truncated) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        cite_str = "n/a" if r["citation_correct"] is None else ("yes" if r["citation_correct"] else "no")
        lines.append(
            f"| {r['id']} | {'answerable' if r['answerable'] else 'refusal'} | {r['query']} | "
            f"{'yes' if r['answer_correct'] else 'no'} | {cite_str} | {r['answer']} |"
        )
    RESULTS_PATH.write_text("\n".join(lines))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    run()
