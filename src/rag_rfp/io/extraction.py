import os
import re
import json
import fitz
import pdfplumber
from tqdm import tqdm
import easyocr


# ============================================
# 0) OCR 초기화
# ============================================
ocr_reader = easyocr.Reader(['ko', 'en'])


# ============================================
# 1) 줄바꿈 정리
# ============================================
def normalize_linebreaks(text: str) -> str:
    if not text:
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines)


# ============================================
# 2) 깨짐 검출
# ============================================
def is_broken_text(text: str) -> bool:
    if not text or len(text.strip()) == 0:
        return True
    if re.search(r'(.)\1{7,}', text):
        return True
    unique_ratio = len(set(text)) / max(len(text), 1)
    if unique_ratio < 0.15:
        return True
    if not re.search(r'[가-힣A-Za-z]', text):
        return True
    return False


# ============================================
# 3) 목차 후처리
# ============================================
def postprocess_toc_like(text: str) -> str:
    """목차 페이지를 라인 단위로 완벽하게 재구성하는 고급 후처리."""

    # 목차인지 판단
    if "Contents" not in text and "목차" not in text:
        return text

    t = text

    # 0) 노이즈 제거
    t = re.sub(r"[\"']", "", t)          # 따옴표 제거
    t = re.sub(r"[＊*●■□◆◇…]+", "", t)  # 점선·특수문자류 제거
    t = re.sub(r"\b[gG]{2,}\b", "", t)   # gg 제거

    # 1) 번호 패턴 정규화
    #  "2 운영 현황" → "2. 운영 현황"
    t = re.sub(r"\b(\d+)\s+(?=[가-힣A-Za-z])", r"\1. ", t)

    # 2) 번호. 제목 앞에 줄바꿈 강제
    #   예: "1. 사업 개요" → "\n1. 사업 개요"
    t = re.sub(r"\s*(\d+)\.\s*", r"\n\1. ", t)

    # 3) 제목 + 페이지번호 패턴 정리
    #   "사업 개요 ....... 1" 형태로 정규화
    #   숫자가 끝에 단독으로 오는 경우 페이지 번호로 판단
    t = re.sub(r"([가-힣A-Za-z\)])\s+(\d{1,4})(?=\D|$)", r"\1 ....... \2", t)

    # 4) 중간에 붙은 노이즈 다시 정리
    t = re.sub(r"[ ]{2,}", " ", t)

    # 5) 라인 단위 정리
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]

    # 6) Contents를 맨 위로 강제 배치
    if not lines[0].startswith("Contents"):
        lines.insert(0, "Contents")

    # 최종 라인 조합
    return "\n".join(lines)




# ============================================
# 4) easyocr bounding box 기반 레이아웃 OCR
# ============================================
def ocr_page_layout_easyocr(path: str, page_idx: int) -> str:
    """easyocr 박스 좌표 기반으로 줄 단위 재조합"""

    # A) PDF 페이지 → 이미지
    doc = fitz.open(path)
    page = doc.load_page(page_idx)
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("png")
    doc.close()

    tmp_img = "/tmp/ocr_page_easy.png"
    with open(tmp_img, "wb") as f:
        f.write(img_bytes)

    # B) OCR 실행 (박스 + 텍스트)
    result = ocr_reader.readtext(tmp_img, detail=1)
    if not result:
        return ""

    # C) 박스 중심값 계산해서 라인 그룹핑
    items = []
    for box, text, score in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x_center = sum(xs) / 4.0
        y_center = sum(ys) / 4.0

        # 특수문자만 있는 노이즈 제거
        if len(text.strip()) == 0:
            continue
        if re.fullmatch(r"[^가-힣A-Za-z0-9]+", text.strip()):
            continue

        items.append((y_center, x_center, text.strip()))

    if not items:
        return ""

    # y 기준 정렬 (위→아래)
    items.sort(key=lambda x: x[0])

    # D) y 중심값으로 줄 그룹
    lines = []
    current_line = []
    current_y = None
    y_threshold = 15  # 한 줄로 묶을 최대 y 오차

    for y, x, text in items:
        if current_y is None:
            current_y = y
            current_line = [(x, text)]
        else:
            if abs(y - current_y) <= y_threshold:
                current_line.append((x, text))
            else:
                current_line.sort(key=lambda t: t[0])
                line_text = " ".join([t[1] for t in current_line])
                lines.append(line_text)

                current_y = y
                current_line = [(x, text)]

    # 마지막 줄
    if current_line:
        current_line.sort(key=lambda t: t[0])
        line_text = " ".join([t[1] for t in current_line])
        lines.append(line_text)

    return "\n".join(lines)


# ============================================
# 5) PDF 페이지 텍스트 추출
# ============================================
def extract_page_text(path: str, page_idx: int):

    # 1) fitz 먼저 시도
    try:
        doc = fitz.open(path)
        page = doc.load_page(page_idx)
        txt = page.get_text() or ""
        doc.close()
        txt = normalize_linebreaks(txt)

        if txt.strip() and not is_broken_text(txt):
            return {"text": txt, "status": "ok", "method": "fitz"}
    except:
        pass

    # 2) pdfplumber fallback
    try:
        with pdfplumber.open(path) as pdf:
            page = pdf.pages[page_idx]
            txt = page.extract_text() or ""
        txt = normalize_linebreaks(txt)

        if txt.strip() and not is_broken_text(txt):
            return {"text": txt, "status": "ok", "method": "pdfplumber"}
    except:
        pass

    # 3) easyocr layout-aware fallback
    try:
        txt = ocr_page_layout_easyocr(path, page_idx)
        txt = normalize_linebreaks(txt)
        txt = postprocess_toc_like(txt)

        if txt.strip():
            return {"text": txt, "status": "ocr", "method": "easyocr_layout"}
    except Exception as e:
        pass

    # 4) 완전 실패
    return {"text": "", "status": "broken", "method": "none"}


# ============================================
# 6) PDF 전체 처리
# ============================================
def extract_pdf(path: str, output_jsonl: str):

    doc = fitz.open(path)
    n_pages = len(doc)
    doc.close()

    results = []

    for p in tqdm(range(n_pages), desc=f"Extracting {os.path.basename(path)}"):
        info = extract_page_text(path, p)
        results.append({
            "page": p,
            "text": info["text"],
            "status": info["status"],
            "method": info["method"],
        })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "pages": n_pages,
        "ok_pages": sum(r["status"] == "ok" for r in results),
        "ocr_pages": sum(r["status"] == "ocr" for r in results),
        "broken_pages": sum(r["status"] == "broken" for r in results),
        "output": output_jsonl
    }


# ============================================
# 7) 실행 예시
# ============================================
if __name__ == "__main__":
    pdf_path = "/home/bhs1581/rag-rfp-system/data/pdf_files_2/한국철도공사 (용역)_모바일오피스 시스템 고도화 용역(총체 및 1차).pdf"
    output_path = "/home/bhs1581/rag-rfp-system/chunking/extraction/extracted_text4.jsonl"

    summary = extract_pdf(pdf_path, output_path)
    print(summary)
