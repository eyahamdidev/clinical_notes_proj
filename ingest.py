"""
Section-aware chunking for clinical discharge summaries.

MIMIC-IV discharge notes are semi-structured: they use a predictable set of
capitalized headers (e.g. "Chief Complaint:", "History of Present Illness:",
"Medications on Admission:"). We split on those headers first, then apply
token-budget chunking *within* a section only if that section is unusually
long, so we never merge unrelated clinical content into one chunk and never
split a coherent thought (e.g. an Assessment and Plan) across chunks
unnecessarily.
"""
import re
from dataclasses import dataclass

# Canonical section headers we expect in a MIMIC-style discharge summary.
# Matching is case-insensitive and tolerant of minor header variants.
SECTION_HEADERS = [
    "Chief Complaint",
    "History of Present Illness",
    "Past Medical History",
    "Medications on Admission",
    "Physical Exam",
    "Assessment and Plan",
    "Discharge Medications",
    "Discharge Instructions",
]

_HEADER_PATTERN = re.compile(
    r"^(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r"):\s*$",
    re.IGNORECASE | re.MULTILINE,
)

CHUNK_TARGET_TOKENS = 400  # approx tokens (we use whitespace-word count as a cheap proxy)
CHUNK_OVERLAP_TOKENS = 60


@dataclass
class Chunk:
    chunk_id: str
    patient_id: str
    admission_id: str
    note_id: str
    section: str
    text: str

    def metadata(self):
        return {
            "chunk_id": self.chunk_id,
            "patient_id": self.patient_id,
            "admission_id": self.admission_id,
            "note_id": self.note_id,
            "section": self.section,
        }


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (section_name, section_body) in document order."""
    matches = list(_HEADER_PATTERN.finditer(text))
    if not matches:
        return [("Full Note", text.strip())]

    sections = []
    for i, m in enumerate(matches):
        header = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((header, body))
    return sections


def _word_chunks(text: str, target: int, overlap: int) -> list[str]:
    """Fallback token-budget chunking for a long section, word-based with overlap."""
    words = text.split()
    if len(words) <= target:
        return [text]
    chunks = []
    step = target - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + target]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + target >= len(words):
            break
    return chunks


def chunk_note(note) -> list[Chunk]:
    """Section-aware chunking: one chunk per section, further split only if
    the section exceeds the token budget."""
    chunks = []
    sections = _split_into_sections(note.text)
    for sec_idx, (section_name, body) in enumerate(sections):
        pieces = _word_chunks(body, CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS)
        for part_idx, piece in enumerate(pieces):
            suffix = f"-{part_idx}" if len(pieces) > 1 else ""
            chunk_id = f"{note.note_id}-{sec_idx}{suffix}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    patient_id=note.patient_id,
                    admission_id=note.admission_id,
                    note_id=note.note_id,
                    section=section_name,
                    text=piece,
                )
            )
    return chunks


def chunk_all(notes) -> list[Chunk]:
    all_chunks = []
    for note in notes:
        all_chunks.extend(chunk_note(note))
    return all_chunks


if __name__ == "__main__":
    from data.synthetic_notes import load_notes

    notes = load_notes()
    chunks = chunk_all(notes)
    print(f"{len(notes)} notes -> {len(chunks)} chunks\n")
    for c in chunks[:6]:
        print(f"[{c.chunk_id}] patient={c.patient_id} section={c.section!r}")
        print("  ", c.text[:90].replace("\n", " "), "...")
