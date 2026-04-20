"""
Enhanced Chunking for CA Study Materials

Design decisions for CA content:
- CA answers require 400-600 words minimum for full marks
- Chunk size targets ~500 words = ~2500-3000 characters
- Overlap ensures concepts spanning boundaries are not lost
- Heading-aware: new heading = new chunk (preserves section context)
- Tables get their own chunks (already handled in enhanced_upload_service)

CHAR vs WORD note:
  Average English word = 5.5 chars
  2800 chars ≈ 500 words  ← optimal for CA exam answers
  3500 chars ≈ 636 words  ← ceiling before LLM context gets diluted
"""

from typing import List, Dict, Any, Optional
from .table_processor import table_to_text

# ── Chunk size constants ──────────────────────────────────────────────────────
MAX_CHARS     = 3500    # Hard ceiling — never exceed this
OPTIMAL_CHARS = 2800    # Target size (≈ 500 words)
MIN_CHARS     = 300     # Minimum to be worth indexing
CHUNK_OVERLAP = 400     # Overlap between consecutive chunks (≈ 70 words)


# ── Helper: split one large text block with sentence-boundary awareness ───────

def split_text_with_overlap(
    text: str,
    chunk_size: int = OPTIMAL_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a large text block into overlapping chunks.
    Tries to break at sentence boundaries to preserve readability.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks   = []
    start    = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # Try to break at sentence boundary within last 300 chars
            search_start    = max(end - 300, start)
            sentence_markers = [". ", ".\n", "? ", "! ", ":\n", ";\n"]
            best_break = -1
            for marker in sentence_markers:
                pos = text.rfind(marker, search_start, end)
                if pos > best_break:
                    best_break = pos + len(marker)

            if best_break > start:
                end = best_break
            else:
                # Word boundary fallback
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= text_len:
            break
        start = end - overlap
        if start < 0:
            start = 0

    return chunks


# ── Helper: merge consecutive short chunks within the same section ────────────

def merge_short_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge back-to-back short chunks that belong to the same heading section.
    Prevents tiny fragments from polluting Pinecone with low-quality vectors.
    """
    if not chunks:
        return []

    merged        = []
    current_chunk = chunks[0].copy()

    for next_chunk in chunks[1:]:
        current_len = len(current_chunk["text"])
        next_len    = len(next_chunk["text"])

        should_merge = (
            (current_len < MIN_CHARS or next_len < MIN_CHARS)
            and (current_len + next_len + 2) <= MAX_CHARS
            and current_chunk.get("heading") == next_chunk.get("heading")
        )

        if should_merge:
            current_chunk["text"]   += "\n\n" + next_chunk["text"]
            current_chunk["length"]  = len(current_chunk["text"])
            # Extend page range
            current_chunk["page_end"] = next_chunk.get("page_end") or next_chunk.get("page_start")
        else:
            merged.append(current_chunk)
            current_chunk = next_chunk.copy()

    merged.append(current_chunk)
    return merged


# ── Main chunking function ────────────────────────────────────────────────────

def create_chunks_enhanced(
    elements: List[Dict[str, Any]],
    images:   Optional[List[Dict[str, Any]]] = None,   # kept for API compat, unused
    tables:   Optional[List[Dict[str, Any]]] = None,   # kept for API compat, unused
) -> List[Dict[str, Any]]:
    """
    Convert a flat list of parsed document elements into RAG-ready chunks.

    Rules:
    1. A new heading element always flushes the current chunk and starts a new one.
       This keeps each section's content together under its heading metadata.
    2. Text paragraphs are accumulated until OPTIMAL_CHARS is reached.
    3. When MAX_CHARS would be exceeded, the current chunk is saved and a new
       one starts with CHUNK_OVERLAP characters of context from the previous chunk.
    4. Very large single paragraphs are split with split_text_with_overlap().
    5. After all elements are processed, short adjacent chunks are merged.

    Args:
        elements: Output from FastDoclingParser — each element has:
                  { "type": "heading"|"paragraph"|"table",
                    "text": str,
                    "page": int (1-indexed) }
        images:   Ignored — CA PDFs contain no meaningful image content.
        tables:   Ignored — table chunks are built separately in upload_service.

    Returns:
        List of chunk dicts suitable for embedding and Pinecone upsert.
    """
    chunks          = []
    current_text    = ""
    current_heading = ""
    current_page    = None
    page_start      = None

    for el in elements:
        el_type = el.get("type", "paragraph")
        el_page = el.get("page")
        el_text = el.get("text", "").strip()

        if not el_text:
            continue

        # Track page numbers
        if el_page:
            current_page = el_page
            if page_start is None:
                page_start = el_page

        # ── HEADING: flush current chunk, start a new section ─────────────────
        if el_type == "heading":
            if current_text and len(current_text) >= MIN_CHARS:
                chunks.append({
                    "text":       current_text.strip(),
                    "heading":    current_heading,
                    "length":     len(current_text),
                    "page_start": page_start or 1,
                    "page_end":   current_page or page_start or 1,
                })

            current_heading = el_text
            current_text    = el_text   # heading text is the start of new chunk
            page_start      = current_page

        # ── TABLE element from Docling (pdfplumber tables handled separately) ──
        elif el_type == "table":
            table_text = table_to_text(el_text)
            combined   = current_text + "\n\n[Table]\n" + table_text

            if len(combined) > MAX_CHARS:
                # Flush current chunk before table
                if len(current_text) >= MIN_CHARS:
                    chunks.append({
                        "text":       current_text.strip(),
                        "heading":    current_heading,
                        "length":     len(current_text),
                        "page_start": page_start or 1,
                        "page_end":   current_page or page_start or 1,
                    })
                current_text = f"[Table]\n{table_text}"
                page_start   = current_page
            else:
                current_text = combined

        # ── REGULAR TEXT paragraph ─────────────────────────────────────────────
        else:
            # Split oversized paragraphs before accumulating
            if len(el_text) > MAX_CHARS:
                parts = split_text_with_overlap(el_text, OPTIMAL_CHARS, CHUNK_OVERLAP)
            else:
                parts = [el_text]

            for part in parts:
                if len(current_text) + len(part) + 2 > MAX_CHARS:
                    # Save current chunk
                    if len(current_text) >= MIN_CHARS:
                        chunks.append({
                            "text":       current_text.strip(),
                            "heading":    current_heading,
                            "length":     len(current_text),
                            "page_start": page_start or 1,
                            "page_end":   current_page or page_start or 1,
                        })

                    # Start new chunk with overlap context
                    overlap_text = current_text[-CHUNK_OVERLAP:] if len(current_text) > CHUNK_OVERLAP else current_text
                    current_text = (overlap_text + "\n\n" + part) if overlap_text.strip() else part
                    page_start   = current_page
                else:
                    sep          = "\n\n" if current_text else ""
                    current_text += sep + part

    # ── Final chunk ────────────────────────────────────────────────────────────
    if current_text and len(current_text) >= MIN_CHARS // 2:
        chunks.append({
            "text":       current_text.strip(),
            "heading":    current_heading,
            "length":     len(current_text),
            "page_start": page_start or 1,
            "page_end":   current_page or page_start or 1,
        })

    # Merge tiny adjacent chunks within the same section
    chunks = merge_short_chunks(chunks)

    return chunks


# ── Backward-compat wrapper ───────────────────────────────────────────────────

def create_chunks(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Backward compatible wrapper (used by older routes)."""
    return create_chunks_enhanced(elements)