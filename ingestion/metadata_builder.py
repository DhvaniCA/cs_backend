from datetime import datetime
from pathlib import Path


def extract_clean_filename(file_name: str) -> str:
    """
    Extract a human-readable name from a filename.

    Examples:
        "Chapter_1_Introduction_to_Accounting.pdf"  → "Chapter 1 Introduction to Accounting"
        "CA_Foundation_Tax_Law.pdf"                  → "CA Foundation Tax Law"
    """
    name = Path(file_name).stem
    name = name.replace("_", " ")
    name = " ".join(name.split())
    return name


def build_metadata(chunk: dict, file_name: str, extra_meta: dict) -> dict:
    """
    Build Pinecone vector metadata from a chunk and admin-supplied metadata.

    Naming logic
    ─────────────
    • If admin did NOT supply a chapter:
        chapter = clean filename   (e.g. "Chapter 1 Preliminary")
        unit    = extra_meta unit or ""

    • If admin DID supply a chapter:
        chapter = admin chapter    (e.g. "Chapter 5: Depreciation")
        unit    = clean filename   (e.g. "Depreciation Methods")

    This always gives a meaningful chapter name for RAG filtering.

    Text truncation
    ───────────────
    Stored text is truncated to 3000 chars to stay within Pinecone
    metadata limits while matching chunk sizes of up to ~2800 chars.
    (Old limit was 2000 which silently cut off chunk content.)
    """
    text           = chunk.get("text", "")
    clean_filename = extract_clean_filename(file_name)
    admin_chapter  = (extra_meta.get("chapter") or "").strip()

    if not admin_chapter:
        # Case 1: no chapter provided → filename is the chapter
        chapter_name = clean_filename
        unit_name    = (extra_meta.get("unit") or "").strip()
    else:
        # Case 2: chapter provided → filename becomes the unit
        chapter_name = admin_chapter
        unit_name    = clean_filename

    metadata = {
        # ── Content (PRIMARY — what the LLM reads) ──────────────────────────
        "text":           text[:3000],   # Raised from 2000 → matches chunk size

        # ── Document identity ────────────────────────────────────────────────
        "source":         file_name,
        "doc_title":      (extra_meta.get("title") or clean_filename),

        # ── CA hierarchy (used for subject-filter in RAG queries) ────────────
        "course":         (extra_meta.get("course")  or ""),
        "level":          (extra_meta.get("level")   or ""),
        "subject":        (extra_meta.get("subject") or ""),

        # ── Chapter / unit (intelligent naming, see above) ───────────────────
        "chapter":        chapter_name,
        "unit":           unit_name,

        # ── Optional extra fields ────────────────────────────────────────────
        "section":        (extra_meta.get("section")        or ""),
        "module":         (extra_meta.get("module")         or ""),
        "custom_heading": (extra_meta.get("custom_heading") or ""),

        # ── Chunk location ───────────────────────────────────────────────────
        "topic":          (chunk.get("heading") or ""),
        "page_start":     (chunk.get("page_start") or 1),
        "page_end":       (chunk.get("page_end")   or chunk.get("page_start") or 1),

        # ── System ──────────────────────────────────────────────────────────
        "type":           "text",
        "uploaded_by":    (extra_meta.get("uploaded_by") or ""),
        "uploaded_at":    datetime.utcnow().isoformat(),
    }

    # Safety pass: Pinecone rejects None values — replace with empty string
    return {k: ("" if v is None else v) for k, v in metadata.items()}