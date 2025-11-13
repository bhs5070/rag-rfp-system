from pathlib import Path
from typing import Dict, Any, List
import fitz  # PyMuPDF

def parse_pdf(path: Path) -> Dict[str, Any]:
    doc = fitz.open(path)
    pages: List[Dict[str, Any]] = []
    for i, page in enumerate(doc):
        text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
        pages.append({"page": i + 1, "text": text})
    return {
        "file_path": str(path),
        "n_pages": len(doc),
        "pages": pages,
    }
