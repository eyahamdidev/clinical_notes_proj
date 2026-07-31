"""
Embedding + vector store.

Default embedder: TF-IDF (scikit-learn). This runs fully offline with zero
model downloads, which matters because this dev sandbox has no route to
HuggingFace or model hubs. It's a legitimate baseline, just not
domain-tuned.

To upgrade to a real dense embedding model once you're running this outside
the sandbox (recommended for production quality):

    pip install sentence-transformers
    # generic, strong general-purpose retrieval model:
    #   BAAI/bge-small-en-v1.5
    # biomedical-tuned sentence embedder (better for clinical text):
    #   pritamdeka/S-PubMedBert-MS-MARCO
    #   NeuML/pubmedbert-base-embeddings

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    vectors = model.encode(texts, normalize_embeddings=True)

Then set embedder=DenseEmbedder(model) below instead of TfidfEmbedder().
The VectorStore class is embedding-agnostic (just needs fit/embed methods),
so no other code changes are required.

Vector store: a small numpy cosine-similarity index. For >>10k chunks, swap
in FAISS (`faiss-cpu`) or Chroma with the same .add/.query interface -- the
data volumes in a per-patient clinical RAG system rarely justify the extra
infra for a demo like this one.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]+")


def _stem(word: str) -> str:
    """Crude suffix stripping (no NLTK/network dependency) so surface-form
    variants like discharge/discharged/discharges share a vocabulary token.
    Plain TF-IDF has zero notion of this -- without it, a query for "what
    was the patient discharged on" gets literally 0.0 similarity with the
    "Discharge Medications" section, since "discharged" and "Discharge"
    are different tokens to a bag-of-words model. This is exactly the kind
    of vocabulary-mismatch problem dense embedding models (bge-small-en,
    PubMedBERT) solve properly -- see module docstring for how to swap one
    in."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "be", "been", "of", "on", "in",
    "to", "and", "or", "for", "at", "with", "this", "that", "it", "as",
    "by", "from", "he", "she", "his", "her", "who", "which",
}


def _stemming_tokenizer(text: str) -> list[str]:
    return [_stem(w.lower()) for w in _TOKEN_RE.findall(text) if w.lower() not in _STOPWORDS]


class TfidfEmbedder:
    """Default, dependency-light embedder. No network access required."""

    name = "tfidf"

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=_stemming_tokenizer,
            lowercase=False,  # tokenizer already lowercases
            ngram_range=(1, 2),
            max_features=20000,
            token_pattern=None,  # required by sklearn when tokenizer is set
        )
        self._fitted = False

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() before embed().")
        vecs = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class DenseEmbedder:
    """Wraps a sentence-transformers model. See module docstring for setup."""

    name = "dense"

    def __init__(self, model):
        self.model = model

    def fit(self, texts: list[str]):
        return self  # dense models are pretrained, nothing to fit

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks = []
        self.vectors = None

    @staticmethod
    def _embedding_text(chunk) -> str:
        # Prepend the section header to the embedded text (but not to what's
        # shown/cited) so queries like "discharged on" match the
        # "Discharge Medications" section even when the word "discharge"
        # doesn't appear in the body text itself.
        return f"{chunk.section}. {chunk.text}"

    def build(self, chunks: list):
        self.chunks = chunks
        texts = [self._embedding_text(c) for c in chunks]
        self.embedder.fit(texts)
        self.vectors = self.embedder.embed(texts)
        return self

    def query(self, query_text: str, top_k: int = 5, patient_id: str | None = None):
        """Return top_k (chunk, score) pairs, optionally scoped to a single
        patient_id. Patient scoping happens BEFORE ranking, not as a
        post-filter, so top_k always returns the k best in-scope results
        rather than potentially fewer after filtering."""
        if patient_id is not None:
            candidate_idx = [i for i, c in enumerate(self.chunks) if c.patient_id == patient_id]
        else:
            candidate_idx = list(range(len(self.chunks)))

        if not candidate_idx:
            return []

        q_vec = self.embedder.embed([query_text])[0]
        cand_vectors = self.vectors[candidate_idx]
        scores = cand_vectors @ q_vec  # cosine sim (vectors are pre-normalized)

        order = np.argsort(-scores)[:top_k]
        results = []
        for o in order:
            idx = candidate_idx[o]
            results.append((self.chunks[idx], float(scores[o])))
        return results


def build_default_store(chunks: list) -> VectorStore:
    return VectorStore(TfidfEmbedder()).build(chunks)


if __name__ == "__main__":
    from data.synthetic_notes import load_notes
    from ingest import chunk_all

    chunks = chunk_all(load_notes())
    store = build_default_store(chunks)

    for q, pid in [
        ("what medications was the patient discharged on", "10001"),
        ("was the patient on anticoagulation", "10002"),
        ("oxygen requirement", "10003"),
    ]:
        print(f"\nQuery: {q!r} (patient {pid})")
        for chunk, score in store.query(q, top_k=3, patient_id=pid):
            print(f"  score={score:.3f} [{chunk.section}] {chunk.text[:70]!r}")
