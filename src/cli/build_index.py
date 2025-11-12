import json
from pathlib import Path
import typer
from loguru import logger
from src.rag_rfp.index.embed import embed_texts
from src.rag_rfp.index.vectordb import FaissIndex

app = typer.Typer()

@app.command()
def main(config: str = typer.Option(...)):
    import yaml
    with open(config, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    idx_dir = Path(y["index"]["path"])
    dim = int(y["index"]["dim"])

    chunks_path = Path(y["data"]["processed_dir"]) / "chunks.jsonl"
    lines = [json.loads(l) for l in open(chunks_path, "r", encoding="utf-8")]
    texts = [l["text"] for l in lines]
    logger.info(f"Embedding {len(texts)} chunks...")
    vecs = embed_texts(texts)

    metas = [{"text": t, **{k: v for k, v in l.items() if k != "text"}} for t, l in zip(texts, lines)]
    index = FaissIndex(dim=dim, path=str(idx_dir))
    index.add(vecs, metas)
    index.save()
    logger.info(f"Saved index to {idx_dir}")

if __name__ == "__main__":
    app()
