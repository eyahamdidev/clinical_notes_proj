"""
Answer generation with forced citations and "not found" refusal.

Backend: ExtractiveGenerator - a fully local, zero-API-key, zero-network
generator that composes an answer directly from the retrieved chunks. It
can't paraphrase or reason the way an LLM can, but it is grounded by
construction (it literally cannot say anything not in the chunks).

Swap point: to plug in a real model later, add a class with a
`generate(query, retrieved) -> Answer` method (matching ExtractiveGenerator's
signature) that calls whatever local/self-hosted model you want (e.g. Llama
via Ollama or vLLM), and return it from get_default_generator() instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

NOT_FOUND_PHRASE = "Not found in record."


@dataclass
class Answer:
    text: str
    citations: list[dict]
    backend: str


_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "does", "do", "did", "have",
    "has", "had", "of", "on", "in", "to", "and", "or", "patient", "history",
    "what", "when", "where", "why", "how", "with", "for", "at", "be", "any",
}


def _stem(word: str) -> str:
    # Crude suffix-stripping so "discharged"/"discharge"/"discharges" and
    # similar inflections are treated as the same content word. Not a real
    # stemmer, just enough to avoid false "not found" refusals over
    # surface-form mismatches.
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def _content_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
    return {_stem(w) for w in words if w not in _STOPWORDS and len(w) > 2}


class ExtractiveGenerator:
    """Zero-API-key fallback: ranks retrieved chunks, returns the most
    relevant chunk's text as the answer, always cited, always grounded.

    Cosine similarity alone isn't a strong enough guardrail with TF-IDF: a
    query like "history of cancer" scores non-trivially against ANY "Past
    Medical History" chunk just because they share generic structural
    words, even if cancer is never mentioned. So on top of the similarity
    threshold, we require at least one specific content word from the
    query to literally appear in the top chunk before treating it as a
    real match -- otherwise we refuse rather than return a topically
    similar but non-answering chunk.
    """

    backend_name = "extractive-fallback"

    def generate(self, query: str, retrieved: list[tuple]) -> Answer:
        if not retrieved or retrieved[0][1] <= 0.0:
            return Answer(text=NOT_FOUND_PHRASE, citations=[], backend=self.backend_name)

        # TF-IDF ranking can put the truly-answering chunk at rank 2 or 3
        # instead of rank 1 (e.g. "Medications on Admission" outscoring
        # "Discharge Medications" on a "what was the patient discharged
        # on" query, since both sections are full of drug-name overlap).
        # Rather than trusting rank-1 blindly, scan top-k in order and pick
        # the first chunk that both clears the similarity floor AND
        # actually contains a query content-word -- a cheap stand-in for a
        # real cross-encoder reranker.
        query_terms = _content_terms(query)
        top_chunk = None
        for chunk, score in retrieved:
            if score < 0.03:
                continue
            chunk_terms = _content_terms(chunk.text) | _content_terms(chunk.section)
            if not query_terms or (query_terms & chunk_terms):
                top_chunk = chunk
                break

        if top_chunk is None:
            return Answer(text=NOT_FOUND_PHRASE, citations=[], backend=self.backend_name)

        text = f"{top_chunk.text.strip()} [{top_chunk.section}]"
        citations = [
            {
                "section": c.section,
                "note_id": c.note_id,
                "chunk_id": c.chunk_id,
                "score": round(s, 3),
                "excerpt": c.text[:200],
            }
            for c, s in retrieved
        ]
        return Answer(text=text, citations=citations, backend=self.backend_name)


def get_default_generator():
    """Always returns the local, fully offline extractive generator.
    No API keys, no network calls, nothing external."""
    return ExtractiveGenerator()


if __name__ == "__main__":
    from data.synthetic_notes import load_notes
    from ingest import chunk_all
    from embed_store import build_default_store

    chunks = chunk_all(load_notes())
    store = build_default_store(chunks)
    gen = get_default_generator()
    print("Using backend:", gen.backend_name)

    for q, pid in [
        ("What was the patient discharged on?", "10001"),
        ("Does the patient have a history of cancer?", "10001"),
    ]:
        retrieved = store.query(q, top_k=3, patient_id=pid)
        ans = gen.generate(q, retrieved)
        print(f"\nQ: {q} (patient {pid})\nA: {ans.text}")
