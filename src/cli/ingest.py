import csv, json
from pathlib import Path
import typer
from loguru import logger
from pydantic import BaseModel
from src.rag_rfp.io.parse_pdf import parse_pdf
from src.rag_rfp.io.parse_hwp import parse_hwp
from src.rag_rfp.io.normalize import normalize_doc
from src.rag_rfp.prep.chunk import chunk_pages

app = typer.Typer()

class Cfg(BaseModel):
    data_meta: str
    raw_dir: str
    processed_dir: str
    chunk_max_tokens: int = 512
    chunk_stride: int = 128
    min_chars: int = 200

def load_cfg(path: Path) -> Cfg:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    d = y["data"]
    p = y["preprocess"]["chunk"]
    return Cfg(
        data_meta=d["meta_csv"],
        raw_dir=d["raw_dir"],
        processed_dir=d["processed_dir"],
        chunk_max_tokens=p["max_tokens"],
        chunk_stride=p["stride"],
        min_chars=p["min_chars"],
    )

@app.command()
def main(config: str = typer.Option(..., help="path to config yaml")):
    cfg = load_cfg(Path(config))
    raw_dir = Path(cfg.raw_dir)
    out_path = Path(cfg.processed_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(cfg.data_meta, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    all_chunks = []
    for r in rows:
        fpath = raw_dir / r["path"]
        logger.info(f"Parsing {fpath}")
        if fpath.suffix.lower() == ".pdf":
            doc = parse_pdf(fpath)
        elif fpath.suffix.lower() == ".hwp":
            doc = parse_hwp(fpath)
        else:
            logger.warning(f"Skip unsupported: {fpath}")
            continue

        doc = normalize_doc(doc)
        chunks = chunk_pages(doc["pages"], max_tokens=cfg.chunk_max_tokens,
                             stride=cfg.chunk_stride, min_chars=cfg.min_chars)
        for c in chunks:
            all_chunks.append({
                "doc_id": r["doc_id"],
                "title": r.get("title", ""),
                "page": c["page"],
                "text": c["text"],
                "meta": {k: r[k] for k in r.keys() if k not in ("path",)}
            })

    with open(out_path / "chunks.jsonl", "w", encoding="utf-8") as w:
        for c in all_chunks:
            w.write(json.dumps(c, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    app()
