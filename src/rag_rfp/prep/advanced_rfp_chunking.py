from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class AdvancedRFPChunker:
    """Notebook에서 분리한 RFP 특화 청킹 모듈."""

    def __init__(self) -> None:
        self.rfp_section_patterns = {
            "사업개요": [r"사업\s*개요", r"추진\s*배경", r"목적"],
            "요구사항": [r"요구\s*사항", r"기능\s*명세", r"기술\s*규격"],
            "예산": [r"예산", r"사업비", r"계약금액", r"소요예산"],
            "일정": [r"일정", r"기간", r"납기", r"완료시기"],
            "제출사항": [r"제출", r"접수", r"서류", r"양식"],
            "평가기준": [r"평가", r"심사", r"선정", r"기준"],
            "계약조건": [r"계약", r"조건", r"이행보증"],
        }

    def process_extracted_texts(self, jsonl_file: str | Path) -> Dict[str, List[Dict[str, Any]]]:
        """페이지 단위 추출 텍스트 jsonl을 읽어 여러 청킹 전략을 생성한다."""
        documents = self._load_and_preprocess(Path(jsonl_file))
        return {
            "parent_child": self._parent_child_chunking(documents),
            "structure_aware": self._structure_aware_chunking(documents),
            "multi_aspect": self._multi_aspect_chunking(documents),
            "sliding_window": self._sliding_window_chunking(documents),
            "paragraph": self._paragraph_chunking(documents),
            "page": self._page_chunking(documents),
            "semantic": self._semantic_chunking(documents),
        }

    def save_chunking_results(
        self,
        chunking_results: Dict[str, List[Dict[str, Any]]],
        output_dir: str | Path,
    ) -> Dict[str, Dict[str, float]]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for strategy, chunks in chunking_results.items():
            chunk_file = output_path / f"chunks_{strategy}.jsonl"
            with chunk_file.open("w", encoding="utf-8") as handle:
                for chunk in chunks:
                    handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        stats = {
            strategy: {
                "chunk_count": len(chunks),
                "avg_size": float(np.mean([c["size"] for c in chunks])) if chunks else 0.0,
                "total_characters": int(sum(c["size"] for c in chunks)),
            }
            for strategy, chunks in chunking_results.items()
        }

        with (output_path / "chunking_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)

        return stats

    def _load_and_preprocess(self, jsonl_file: Path) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []

        with jsonl_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                data = json.loads(line)
                text = str(data.get("text", "")).strip()
                if len(text) <= 50:
                    continue

                documents.append(
                    {
                        "doc_id": self._extract_doc_id(str(data.get("filename", ""))),
                        "filename": data.get("filename", ""),
                        "page": data.get("page", 0),
                        "text": self._preprocess_text_enhanced(text),
                        "status": data.get("status"),
                        "method": data.get("method"),
                    }
                )

        return self._group_by_document(documents)

    def _preprocess_text_enhanced(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([가-힣])(○|—|∼)([가-힣])", r"\1 \2 \3", text)
        text = re.sub(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})", r"\1. \2. \3.", text)
        text = re.sub(r"￦", "원 ", text)
        text = re.sub(r"([.])([가-힣○])", r"\1\n\2", text)
        text = re.sub(r"(○|—)", r"\n\1", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_doc_id(self, filename: str) -> str:
        return filename.replace("pdf_files/", "").replace(".pdf", "")

    def _group_by_document(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        doc_groups: Dict[str, Dict[str, Any]] = {}

        for doc in documents:
            doc_id = doc["doc_id"]
            if doc_id not in doc_groups:
                doc_groups[doc_id] = {
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "pages": [],
                    "full_text": "",
                }

            doc_groups[doc_id]["pages"].append(
                {
                    "page": doc["page"],
                    "text": doc["text"],
                    "status": doc["status"],
                }
            )

        grouped_docs: List[Dict[str, Any]] = []
        for doc_data in doc_groups.values():
            doc_data["pages"].sort(key=lambda item: item["page"])
            doc_data["full_text"] = "\n\n".join(page["text"] for page in doc_data["pages"])
            doc_data["page_count"] = len(doc_data["pages"])
            grouped_docs.append(doc_data)

        return grouped_docs

    def _parent_child_chunking(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for doc in documents:
            parent_chunks = self._split_text(doc["full_text"], chunk_size=1500, overlap=300)
            for parent_index, parent_text in enumerate(parent_chunks):
                parent_id = f"{doc['doc_id']}_parent_{parent_index}"
                child_chunks = self._split_text(parent_text, chunk_size=500, overlap=100)

                for child_index, child_text in enumerate(child_chunks):
                    chunks.append(
                        {
                            "chunk_id": f"{parent_id}_child_{child_index}",
                            "parent_id": parent_id,
                            "doc_id": doc["doc_id"],
                            "filename": doc["filename"],
                            "text": child_text,
                            "parent_text": parent_text,
                            "chunk_type": "parent_child",
                            "size": len(child_text),
                        }
                    )

        return chunks

    def _structure_aware_chunking(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for doc in documents:
            sections = self._identify_rfp_sections(doc["full_text"])
            for section_name, section_text in sections.items():
                if len(section_text.strip()) < 100:
                    continue

                if section_name in {"예산", "일정"}:
                    section_chunks = self._split_text(section_text, chunk_size=300, overlap=50)
                elif section_name in {"요구사항", "기술규격"}:
                    section_chunks = self._split_text(section_text, chunk_size=800, overlap=150)
                else:
                    section_chunks = self._split_text(section_text, chunk_size=600, overlap=100)

                for chunk_index, chunk_text in enumerate(section_chunks):
                    chunks.append(
                        {
                            "chunk_id": f"{doc['doc_id']}_{section_name}_{chunk_index}",
                            "doc_id": doc["doc_id"],
                            "filename": doc["filename"],
                            "text": chunk_text,
                            "section": section_name,
                            "chunk_type": "structure_aware",
                            "size": len(chunk_text),
                        }
                    )

        return chunks

    def _identify_rfp_sections(self, text: str) -> Dict[str, str]:
        sections = {"기타": ""}
        current_section = "기타"

        for line in text.split("\n"):
            detected_section = None
            for section_name, patterns in self.rfp_section_patterns.items():
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
                    detected_section = section_name
                    break

            if detected_section:
                current_section = detected_section
                sections.setdefault(current_section, "")

            sections[current_section] += line + "\n"

        return {name: value.strip() for name, value in sections.items() if value.strip()}

    def _multi_aspect_chunking(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for doc in documents:
            base_chunks = self._split_text(doc["full_text"], chunk_size=600, overlap=100)
            for chunk_index, chunk_text in enumerate(base_chunks):
                base_chunk_id = f"{doc['doc_id']}_multi_{chunk_index}"

                chunks.append(
                    {
                        "chunk_id": f"{base_chunk_id}_original",
                        "doc_id": doc["doc_id"],
                        "filename": doc["filename"],
                        "text": chunk_text,
                        "aspect": "original",
                        "chunk_type": "multi_aspect",
                        "size": len(chunk_text),
                    }
                )

                keywords = self._extract_key_terms(chunk_text)
                if keywords:
                    keyword_text = " ".join(keywords)
                    chunks.append(
                        {
                            "chunk_id": f"{base_chunk_id}_keywords",
                            "doc_id": doc["doc_id"],
                            "filename": doc["filename"],
                            "text": keyword_text,
                            "original_text": chunk_text,
                            "aspect": "keywords",
                            "chunk_type": "multi_aspect",
                            "size": len(keyword_text),
                        }
                    )

                summary = self._create_simple_summary(chunk_text)
                if summary:
                    chunks.append(
                        {
                            "chunk_id": f"{base_chunk_id}_summary",
                            "doc_id": doc["doc_id"],
                            "filename": doc["filename"],
                            "text": summary,
                            "original_text": chunk_text,
                            "aspect": "summary",
                            "chunk_type": "multi_aspect",
                            "size": len(summary),
                        }
                    )

        return chunks

    def _sliding_window_chunking(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for doc in documents:
            doc_chunks = self._split_text(doc["full_text"], chunk_size=500, overlap=100)
            for chunk_index, chunk_text in enumerate(doc_chunks):
                chunks.append(
                    {
                        "chunk_id": f"{doc['doc_id']}_sliding_{chunk_index}",
                        "doc_id": doc["doc_id"],
                        "filename": doc["filename"],
                        "text": chunk_text,
                        "chunk_type": "sliding_window",
                        "size": len(chunk_text),
                    }
                )

        return chunks

    def _paragraph_chunking(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for doc in documents:
            paragraphs = self._extract_paragraphs(doc["full_text"])
            paragraph_chunks = self._combine_units(paragraphs, target_size=700, overlap_units=1)

            for chunk_index, chunk_text in enumerate(paragraph_chunks):
                chunks.append(
                    {
                        "chunk_id": f"{doc['doc_id']}_paragraph_{chunk_index}",
                        "doc_id": doc["doc_id"],
                        "filename": doc["filename"],
                        "text": chunk_text,
                        "chunk_type": "paragraph",
                        "size": len(chunk_text),
                    }
                )

        return chunks

    def _page_chunking(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for doc in documents:
            for page_item in doc["pages"]:
                page_text = page_item["text"].strip()
                if len(page_text) < 50:
                    continue

                chunks.append(
                    {
                        "chunk_id": f"{doc['doc_id']}_page_{page_item['page']}",
                        "doc_id": doc["doc_id"],
                        "filename": doc["filename"],
                        "page": page_item["page"],
                        "text": page_text,
                        "chunk_type": "page",
                        "size": len(page_text),
                    }
                )

        return chunks

    def _semantic_chunking(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for doc in documents:
            units = self._extract_semantic_units(doc["full_text"])
            semantic_chunks = self._combine_semantic_units(units, target_size=700, max_size=1100)

            for chunk_index, chunk_text in enumerate(semantic_chunks):
                chunks.append(
                    {
                        "chunk_id": f"{doc['doc_id']}_semantic_{chunk_index}",
                        "doc_id": doc["doc_id"],
                        "filename": doc["filename"],
                        "text": chunk_text,
                        "chunk_type": "semantic",
                        "size": len(chunk_text),
                    }
                )

        return chunks

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        if len(text) <= chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        step = max(chunk_size - overlap, 1)

        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                next_period = text.find(".", end)
                next_newline = text.find("\n", end)
                boundaries = [boundary for boundary in (next_period, next_newline) if boundary != -1]
                if boundaries:
                    end = min(boundaries) + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break
            start = max(end - overlap, start + step)

        return chunks

    def _extract_key_terms(self, text: str) -> List[str]:
        korean_terms = re.findall(r"[가-힣]{2,}", text)
        term_counts = Counter(korean_terms)
        return [term for term, _count in term_counts.most_common(5)]

    def _create_simple_summary(self, text: str) -> str:
        sentences = [sentence.strip() for sentence in re.split(r"[.!?]\s+", text) if sentence.strip()]
        if not sentences:
            return ""

        first_sentence = sentences[0]
        longest_sentence = max(sentences, key=len)
        if first_sentence == longest_sentence:
            return first_sentence[:200]
        return f"{first_sentence}. {longest_sentence}"[:300]

    def _extract_paragraphs(self, text: str) -> List[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
        if paragraphs:
            return paragraphs
        return [line.strip() for line in text.split("\n") if line.strip()]

    def _combine_units(self, units: List[str], target_size: int, overlap_units: int = 0) -> List[str]:
        if not units:
            return []

        chunks: List[str] = []
        current_units: List[str] = []
        current_size = 0

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue

            if current_units and current_size + len(unit) > target_size:
                chunks.append("\n\n".join(current_units).strip())
                current_units = current_units[-overlap_units:] if overlap_units > 0 else []
                current_size = sum(len(item) for item in current_units)

            current_units.append(unit)
            current_size += len(unit)

        if current_units:
            chunks.append("\n\n".join(current_units).strip())

        return [chunk for chunk in chunks if chunk]

    def _extract_semantic_units(self, text: str) -> List[str]:
        units: List[str] = []
        paragraphs = self._extract_paragraphs(text)

        for paragraph in paragraphs:
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+|\n", paragraph)
                if sentence.strip()
            ]
            if not sentences:
                continue

            current = sentences[0]
            current_terms = self._semantic_terms(current)

            for sentence in sentences[1:]:
                sentence_terms = self._semantic_terms(sentence)
                should_split = False

                if self._looks_like_heading(sentence):
                    should_split = True
                elif len(current) >= 500:
                    should_split = True
                elif current_terms and sentence_terms:
                    overlap = len(current_terms & sentence_terms) / max(len(current_terms | sentence_terms), 1)
                    if overlap < 0.08 and len(current) >= 250:
                        should_split = True

                if should_split:
                    units.append(current.strip())
                    current = sentence
                    current_terms = sentence_terms
                else:
                    current = f"{current} {sentence}".strip()
                    current_terms |= sentence_terms

            if current.strip():
                units.append(current.strip())

        return units

    def _combine_semantic_units(self, units: List[str], target_size: int, max_size: int) -> List[str]:
        if not units:
            return []

        chunks: List[str] = []
        current_units: List[str] = []
        current_terms: set[str] = set()
        current_size = 0

        for unit in units:
            unit_terms = self._semantic_terms(unit)
            overlap = len(current_terms & unit_terms) / max(len(current_terms | unit_terms), 1) if current_terms else 1.0

            should_flush = False
            if current_units and current_size >= target_size and overlap < 0.08:
                should_flush = True
            if current_units and current_size + len(unit) > max_size:
                should_flush = True

            if should_flush:
                chunks.append("\n\n".join(current_units).strip())
                current_units = []
                current_terms = set()
                current_size = 0

            current_units.append(unit)
            current_terms |= unit_terms
            current_size += len(unit)

        if current_units:
            chunks.append("\n\n".join(current_units).strip())

        return [chunk for chunk in chunks if chunk]

    def _semantic_terms(self, text: str) -> set[str]:
        korean_terms = set(re.findall(r"[가-힣]{2,}", text))
        numeric_terms = set(re.findall(r"\d+(?:[./-]\d+)*", text))
        english_terms = {term.lower() for term in re.findall(r"[A-Za-z]{2,}", text)}
        return korean_terms | numeric_terms | english_terms

    def _looks_like_heading(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if len(stripped) <= 40 and re.search(r"(제\s*\d+\s*장|^\d+[.)]|^[가-힣A-Za-z0-9\s]+:?$)", stripped):
            return True
        return False


def main() -> None:
    input_path = os.getenv("RFP_EXTRACTED_JSONL", "all_extracted_texts.jsonl")
    output_dir = os.getenv("RFP_CHUNK_OUTPUT_DIR", "advanced_chunks")

    chunker = AdvancedRFPChunker()
    results = chunker.process_extracted_texts(input_path)
    stats = chunker.save_chunking_results(results, output_dir)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
