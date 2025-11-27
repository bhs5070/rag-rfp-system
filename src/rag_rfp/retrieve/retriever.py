import faiss
import numpy as np
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
from collections import defaultdict

# ==========================================
# CONFIG
# ==========================================
CHUNK_FILE_PATH = "/home/bhs1581/rag-rfp-system/chunking/chunks/chunks_multi_aspect (1).jsonl"
FAISS_INDEX_PATH = "/home/bhs1581/rag-rfp-system/scenario_a/hnsw_index.faiss"
EVAL_JSONL_PATH = "/home/bhs1581/rag-rfp-system/chunking/eval/our_clean_eval_style.jsonl"

TOP_K_LIST = [1, 3, 5, 10]

# Hybrid config
MAX_K_DENSE = 20
MAX_K_SPARSE = 20
HYBRID_TOP_K = 20
RRF_K = 60

OKT = Okt()
ALL_CHUNK_TEXTS = []
BM25_MODEL = None


# ==========================================
# 1. Load chunks + BM25 build
# ==========================================
def load_chunk_mapping(chunk_path):
    global ALL_CHUNK_TEXTS, BM25_MODEL

    mapping = {}
    chunks = []

    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            chunks.append(item)

    for idx, c in enumerate(chunks):
        mapping[idx] = c["doc_id"]
        ALL_CHUNK_TEXTS.append(c["text"])

    print(f"Loaded {len(mapping)} chunks.")

    # ----- BM25 초기화 -----
    tokenized_corpus = []
    print("BM25 Tokenizing (OKT nouns)...")
    for txt in tqdm(ALL_CHUNK_TEXTS):
        tokens = OKT.nouns(txt)
        tokenized_corpus.append(tokens)

    BM25_MODEL = BM25Okapi(tokenized_corpus)
    print("BM25 initialization complete.")

    return mapping, chunks


# ==========================================
# 2. FAISS index load
# ==========================================
def load_faiss_index(path: str):
    print("Loading FAISS index...")
    index = faiss.read_index(path)
    print(f"FAISS index loaded: {index.ntotal} vectors")
    return index


# ==========================================
# 3. Dense Search (FAISS)
# ==========================================
def dense_search(model, index, query: str, top_k: int):
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    ranks = {}
    for rank, idx in enumerate(I[0]):
        if idx >= 0:
            ranks[int(idx)] = rank + 1
    return ranks


# ==========================================
# 4. Sparse Search (BM25)
# ==========================================
def sparse_search(query: str, top_k: int):
    tokens = OKT.nouns(query)
    scores = BM25_MODEL.get_scores(tokens)
    ranked = np.argsort(scores)[::-1]

    ranks = {}
    for r, idx in enumerate(ranked[:top_k]):
        ranks[int(idx)] = r + 1

    return ranks


# ==========================================
# 5. Reciprocal Rank Fusion (RRF)
# ==========================================
def reciprocal_rank_fusion(rank_dicts):
    fused_scores = defaultdict(float)
    for ranks in rank_dicts:
        for idx, rank in ranks.items():
            fused_scores[idx] += 1.0 / (RRF_K + rank)
    sorted_idx = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return sorted_idx[:HYBRID_TOP_K]


# ==========================================
# 6. Hybrid Search (Dense + Sparse + RRF)
# ==========================================
def hybrid_search(model, index, query: str):
    dense_ranks = dense_search(model, index, query, MAX_K_DENSE)
    sparse_ranks = sparse_search(query, MAX_K_SPARSE)
    fused = reciprocal_rank_fusion([dense_ranks, sparse_ranks])
    return fused


# ==========================================
# 7. R@k Evaluation (doc_id match)
# ==========================================
def evaluate_recall(search_fn, model, index, eval_data, chunk_mapping):
    results = {f"R@{k}": 0 for k in TOP_K_LIST}
    total = len(eval_data)

    for item in tqdm(eval_data, desc="Evaluating Recall"):
        query = item["question"]
        gt_doc = item["gt_doc_id"]

        retrieved_indices = search_fn(model, index, query)
        retrieved_docs = [chunk_mapping[idx] for idx in retrieved_indices]

        for k in TOP_K_LIST:
            if gt_doc in retrieved_docs[:k]:
                results[f"R@{k}"] += 1

    for k in results:
        results[k] /= total

    return results


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("Loading bge-m3 model...")
    model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
    model = model.to("cuda")

    chunk_mapping, chunks = load_chunk_mapping(CHUNK_FILE_PATH)
    index = load_faiss_index(FAISS_INDEX_PATH)

    with open(EVAL_JSONL_PATH, "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f]

    print(f"Evaluation dataset size: {len(eval_data)}")

    # -------- Baseline Evaluation --------
    print("\n===== Baseline (Dense Only) =====")
    baseline_results = evaluate_recall(
        search_fn=lambda m, idx, q: dense_search(m, idx, q, top_k=HYBRID_TOP_K),
        model=model,
        index=index,
        eval_data=eval_data,
        chunk_mapping=chunk_mapping
    )
    for k, v in baseline_results.items():
        print(f"{k}: {v:.3f}")

    # -------- Hybrid Evaluation --------
    print("\n===== Hybrid (Dense + Sparse + RRF) =====")
    hybrid_results = evaluate_recall(
        search_fn=hybrid_search,
        model=model,
        index=index,
        eval_data=eval_data,
        chunk_mapping=chunk_mapping
    )
    for k, v in hybrid_results.items():
        print(f"{k}: {v:.3f}")

    print("===========================================================")
