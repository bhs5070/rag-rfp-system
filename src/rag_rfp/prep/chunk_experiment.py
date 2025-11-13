# src/rag_rfp/prep/chunk_experiment.py
from pathlib import Path
import pandas as pd

# langchain 텍스트 스플리터 (패키지: langchain-text-splitters)
from langchain_text_splitters import RecursiveCharacterTextSplitter

def run_chunk_experiments(
    input_dir: str,
    output_path: str,
    sizes: list[int],
    overlaps: list[int],
) -> pd.DataFrame:
    """
    input_dir 안의 *.txt들을 읽어 sizes x overlaps 조합으로 청킹 후,
    청크 개수/평균 길이를 집계하여 CSV로 저장하고 DataFrame을 반환.
    """
    input_dir = Path(input_dir)
    texts = []
    for fp in input_dir.glob("*.txt"):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())

    results = []
    for size in sizes:
        for overlap in overlaps:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap
            )
            all_chunks = []
            for t in texts:
                all_chunks.extend(splitter.split_text(t))

            chunk_count = len(all_chunks)
            avg_len = (sum(len(c) for c in all_chunks) / chunk_count) if chunk_count else 0

            results.append({
                "chunk_size": size,
                "overlap": overlap,
                "chunk_count": chunk_count,
                "avg_len": avg_len,
            })

    df = pd.DataFrame(results).sort_values(["chunk_size", "overlap"]).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df