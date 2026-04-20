"""
LIGHTWEIGHT Fast Parser for CA Study Material PDFs

Optimized for:
- Text-heavy CA PDFs (law, accounting, tax, audit)
- Accurate page number tracking per element
- Better heading detection for section-level metadata
- Table extraction via pdfplumber

Images: intentionally disabled — CA PDFs are text/table documents.
Processing time: 5-15 seconds per PDF.
"""

from docling.document_converter import DocumentConverter
import pdfplumber
from typing import List, Dict, Any
from pathlib import Path

print("⚡ FAST Parser loaded (CA-optimized, page-aware)")


class FastDoclingParser:
    """
    Fast CA-optimized parser.

    Produces:
    - ✅ Text elements with page numbers
    - ✅ Accurate heading detection (numbered sections, ALLCAPS headings)
    - ✅ Table extraction via pdfplumber
    - ❌ Images — intentionally skipped (CA PDFs are text/table only)
    """

    def __init__(self):
        self.converter = DocumentConverter()

    # ── Table extraction ──────────────────────────────────────────────────────

    def extract_tables_fast(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables with page numbers via pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    for table_index, table in enumerate(page_tables):
                        if not table or len(table) < 2:
                            continue
                        cleaned = []
                        for row in table:
                            cleaned_row = [(cell.strip() if cell else "") for cell in row]
                            if any(cleaned_row):
                                cleaned.append(cleaned_row)
                        if len(cleaned) >= 2:
                            tables.append({
                                "page":        page_num,
                                "table_index": table_index,
                                "headers":     cleaned[0],
                                "rows":        cleaned[1:],
                                "num_rows":    len(cleaned) - 1,
                                "num_columns": len(cleaned[0]),
                                "raw_data":    cleaned,
                            })
        except Exception as e:
            print(f"  ⚠️  Table extraction error: {e}")
        return tables

    # ── Heading detection ─────────────────────────────────────────────────────

    @staticmethod
    def _is_heading(text: str) -> bool:
        """
        Detect headings in CA legal/accounting documents.

        Matches:
          - Markdown headings: ## Introduction
          - Numbered sections: 1. Introduction / 2.3 Application
          - ALL-CAPS short lines: DEFINITIONS, PRELIMINARY
          - Clause-style: Section 2(1) / Clause (a)
        """
        t = text.strip()
        if not t or len(t) > 120:
            return False
        if t.startswith("#"):
            return True
        import re
        # Numbered section headings: "1.", "2.3", "Section 1", "Clause 5"
        if re.match(r"^(\d+\.)+\s+\w", t):
            return True
        if re.match(r"^(section|clause|rule|schedule|chapter|part)\s+\d", t, re.IGNORECASE):
            return True
        # ALL CAPS heading (at least 3 words or one meaningful word)
        words = t.split()
        if len(words) <= 8 and t == t.upper() and any(c.isalpha() for c in t):
            return True
        return False

    # ── Main parse ────────────────────────────────────────────────────────────

    def parse_pdf_fast(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF: extract text elements with page numbers + tables.

        Strategy:
        1. Use Docling to get structured text (handles multi-column, headers)
        2. Use pdfplumber page-text to assign page numbers to each element
        3. Use pdfplumber for table extraction
        """
        print(f"  ⚡ Parsing: {Path(file_path).name}")

        # ── Step 1: Docling text extraction ───────────────────────────────────
        result       = self.converter.convert(file_path)
        raw_elements = self._extract_elements_from_docling(result)

        # ── Step 2: Assign page numbers via pdfplumber ────────────────────────
        page_texts = self._extract_page_texts(file_path)
        elements   = self._assign_pages(raw_elements, page_texts)

        # ── Step 3: Table extraction ──────────────────────────────────────────
        tables = self.extract_tables_fast(file_path)

        print(f"  ✅ Parsed: {len(elements)} elements, {len(tables)} tables, {len(page_texts)} pages")

        return {
            "elements": elements,
            "tables":   tables,
            "images":   [],   # CA PDFs: no image processing needed
            "metadata": {
                "total_elements": len(elements),
                "total_tables":   len(tables),
                "total_images":   0,
                "total_pages":    len(page_texts),
                "filename":       Path(file_path).name,
                "fast_mode":      True,
            },
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_elements_from_docling(self, result) -> List[Dict[str, Any]]:
        """Convert Docling output to a flat list of {type, text} elements."""
        elements = []
        try:
            if hasattr(result.document, "export_to_markdown"):
                md   = result.document.export_to_markdown()
                for para in md.split("\n\n"):
                    para = para.strip()
                    if not para:
                        continue
                    # Strip markdown heading markers but preserve text
                    clean = para.lstrip("#").strip()
                    if not clean:
                        continue
                    typ = "heading" if (para.startswith("#") or self._is_heading(clean)) else "paragraph"
                    elements.append({"type": typ, "text": clean})

            elif hasattr(result.document, "export_to_dict"):
                doc_dict = result.document.export_to_dict()
                for item in doc_dict.get("texts", []):
                    text = item.get("text", "").strip()
                    if text:
                        typ = "heading" if self._is_heading(text) else "paragraph"
                        elements.append({"type": typ, "text": text})

            else:
                # Last resort: stringify
                text = str(result.document)
                for para in text.split("\n\n"):
                    if para.strip():
                        elements.append({"type": "paragraph", "text": para.strip()})

        except Exception as e:
            print(f"  ⚠️  Docling parse error: {e}")
            elements = [{"type": "paragraph", "text": "Could not extract text from document"}]

        return elements

    @staticmethod
    def _extract_page_texts(file_path: str) -> List[str]:
        """Return list of raw text per page (index 0 = page 1)."""
        pages = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    pages.append((page.extract_text() or "").lower())
        except Exception as e:
            print(f"  ⚠️  Page text extraction error: {e}")
        return pages

    @staticmethod
    def _assign_pages(
        elements: List[Dict[str, Any]],
        page_texts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Assign a page number to each element by substring matching.

        For each element we search which page contains the first ~80 chars
        of its text.  We walk pages forward from the last matched page to
        keep assignment monotonically increasing.
        """
        if not page_texts:
            # No page info available — assign page 1 to everything
            for el in elements:
                el["page"] = 1
            return elements

        current_page = 1   # 1-indexed

        for el in elements:
            snippet = el["text"][:80].lower().strip()
            if not snippet:
                el["page"] = current_page
                continue

            # Search from current page onwards (documents are sequential)
            found = False
            for pg_idx in range(current_page - 1, len(page_texts)):
                if snippet[:40] in page_texts[pg_idx]:
                    current_page = pg_idx + 1   # convert to 1-indexed
                    found = True
                    break

            if not found:
                # Fallback: try all pages (handles Docling reordering)
                for pg_idx, pg_text in enumerate(page_texts):
                    if snippet[:40] in pg_text:
                        current_page = pg_idx + 1
                        found = True
                        break

            el["page"] = current_page

        return elements


# ── Module-level convenience function ────────────────────────────────────────

def parse_pdf_fast(file_path: str) -> Dict[str, Any]:
    """Convenience wrapper for direct use."""
    parser = FastDoclingParser()
    return parser.parse_pdf_fast(file_path)