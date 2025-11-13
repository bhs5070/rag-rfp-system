from typing import List, Dict

def sliding_window(text: str, max_tokens: int = 512, stride: int = 128) -> List[str]:
    # 간단한 문자 단위 슬라이딩 윈도우 청킹
    max_chars = max_tokens * 4
    step = max(1, max_chars - stride * 4)
    return [text[i:i + max_chars] for i in range(0, len(text), step)]

def chunk_pages(pages: List[Dict], **kw) -> List[Dict]:
    chunks = []
    for p in pages:
        for ch in sliding_window(p["text"], **kw):
            if len(ch) < kw.get("min_chars", 200):
                continue
            chunks.append({"page": p["page"], "text": ch})
    return chunks
