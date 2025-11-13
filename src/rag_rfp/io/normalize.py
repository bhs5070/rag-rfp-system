import re
from typing import Dict, Any

def normalize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    pages = []
    for p in doc["pages"]:
        text = p["text"]
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append({**p, "text": text.strip()})
    return {**doc, "pages": pages}
