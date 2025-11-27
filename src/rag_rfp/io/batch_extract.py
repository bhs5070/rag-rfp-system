import os
from extract_pdf import extract_pdf

PDF_DIR = "/home/bhs1581/rag-rfp-system/data/pdf_files_2"
OUTPUT_DIR = "/home/bhs1581/rag-rfp-system/chunking/extraction"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])

print(f"총 {len(pdf_files)}개 PDF 감지됨")

for i, pdf in enumerate(pdf_files):
    pdf_path = os.path.join(PDF_DIR, pdf)
    output_jsonl = os.path.join(OUTPUT_DIR, f"{os.path.splitext(pdf)[0]}.jsonl")

    print(f"\n[{i+1}/{len(pdf_files)}] Processing: {pdf}")
    summary = extract_pdf(pdf_path, output_jsonl)
    print(summary)
