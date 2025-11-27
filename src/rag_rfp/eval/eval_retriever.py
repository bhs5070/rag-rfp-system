import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

from rag_rfp.retrieve.retriever import ChunkRetriever
 # 네 프로젝트에 맞게 import

BASE_DIR = Path(__file__).resolve().parents[3]  # rag-rfp-system/
EVAL_PATH = BASE_DIR / "data" / "eval" / "rag_eval.jsonl"
OUT_PATH = BASE_DIR / "outputs" / "eval" / "retriever_eval.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_eval_dataset(path: Path):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            samples.append(obj)
    return samples


def evaluate_retriever(k_values=(1, 3, 5)):
    # retriever 초기화 (네 retriever 초기화 방식에 맞게 수정해도 됨)
    retriever = ChunkRetriever()

    samples = load_eval_dataset(EVAL_PATH)
    print(f"Loaded {len(samples)} eval samples from {EVAL_PATH}")

    # k별로 hits/total 계산
    hits_at_k = defaultdict(int)
    total = len(samples)

    for sample in samples:
        question = sample["question"]
        gt_ids = set(sample["relevant_chunk_ids"])

        # 🔴 여기서 retriever 메서드 이름만 네 코드에 맞게 바꾸면 됨
        # 예: retriever.search, retriever.retrieve, retriever.get_top_k 등
        results = retriever.search(question, top_k=max(k_values))

        # results가 [{"id": "...", "score": ...}, ...] 형태라고 가정
        retrieved_ids = [r["doc_id"] for r in results]

        for k in k_values:
            top_k_ids = set(retrieved_ids[:k])
            if gt_ids & top_k_ids:
                hits_at_k[k] += 1

    # 계산 결과
    metrics = {
        f"Recall@{k}": hits_at_k[k] / total if total else 0.0
        for k in k_values
    }
    
    #출력
    print("\nRetriever Evaluation Metrics:")
    for k in sorted(k_values):
        recall = hits_at_k[k] / total if total > 0 else 0.0
        print(f"Recall@{k}: {recall:.4f}")
        
    # 파일로 저장
    save_obj = {
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen3-Embedding-0.6B",
        "eval_file": str(EVAL_PATH),
        "metrics": metrics,
    }

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(save_obj, f, indent=2, ensure_ascii=False)

    print(f"\nSaved evaluation → {OUT_PATH}")


if __name__ == "__main__":
    evaluate_retriever(k_values=(1, 3, 5))

    