from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from src.rag_rfp.io.parse_pdf import parse_pdf
from src.rag_rfp.prep.advanced_rfp_chunking import AdvancedRFPChunker

app = typer.Typer()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def extract_folder_to_jsonl(input_dir: Path, output_jsonl: Path) -> int:
    pdf_files = sorted(input_dir.glob("*.pdf"))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for pdf_path in pdf_files:
            logger.info("Extracting %s", pdf_path.name)
            parsed = parse_pdf(pdf_path)

            for page in parsed["pages"]:
                record = {
                    "filename": pdf_path.name,
                    "page": page["page"],
                    "text": page["text"],
                    "status": "ok",
                    "method": "pymupdf",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                row_count += 1

    return row_count


@app.command()
def main(
    input_dir: str = typer.Option("pdf_files 3", help="PDF folder path"),
    extracted_jsonl: str = typer.Option(
        "data/interim/all_extracted_texts.jsonl",
        help="Extracted page-level jsonl output path",
    ),
    chunks_dir: str = typer.Option(
        "data/processed/advanced_chunks",
        help="Chunk output directory",
    ),
) -> None:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise typer.BadParameter(f"Input folder not found: {input_path}")

    pdf_count = len(list(input_path.glob("*.pdf")))
    if pdf_count == 0:
        raise typer.BadParameter(f"No PDF files found in: {input_path}")

    extracted_path = Path(extracted_jsonl)
    chunk_output_dir = Path(chunks_dir)

    row_count = extract_folder_to_jsonl(input_path, extracted_path)
    logger.info("Extracted %s PDFs into %s page rows", pdf_count, row_count)

    chunker = AdvancedRFPChunker()
    chunking_results = chunker.process_extracted_texts(extracted_path)
    stats = chunker.save_chunking_results(chunking_results, chunk_output_dir)

    logger.info("Saved chunk outputs to %s", chunk_output_dir)
    logger.info("Chunk stats: %s", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    app()
