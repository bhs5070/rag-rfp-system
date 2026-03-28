import faiss
import numpy as np
import json
from openai import OpenAI
from tqdm import tqdm
from typing import List, Dict, Tuple
import os
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import defaultdict
import time # LLM 호출 지연 방지용

# --- 설정값: 실제 파일 경로로 수정하세요 ---
FAISS_INDEX_PATH = "/home/bhs1581/rag-rfp-system/chunking/vector_db/vectordb_multi_aspect_index.faiss" 
EVAL_JSONL_PATH = "/home/bhs1581/rag-rfp-system/chunking/eval/our_clean_eval_style.jsonl" 
CHUNK_FILE_PATH = "/home/bhs1581/rag-rfp-system/chunking/chunks/chunks_multi_aspect (1).jsonl"
# 🚨 쿼리 변환 결과를 저장할 캐시 파일 경로
QUERY_CACHE_PATH = "./rewritten_queries_cache.json" 

# --- Reranker 설정 ---
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"  
RERANK_TOP_K = 10 

# --- 성능 측정 상수 ---
TARGET_DIMENSION = 1536 
MAX_K_DENSE = 20        
MAX_K_SPARSE = 20       
HYBRID_K = 20           
RRF_K = 60              # 최적 성능 K=60 고정
TEST_PREFIX = "RFP의 필수 정보를 찾고 있습니다: " 

# OpenAI 클라이언트 초기화
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) 
except Exception as e:
    print(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
    exit()

# --- 전역 변수 및 모델 초기화 ---
ALL_CHUNK_TEXTS: List[str] = []
BM25_MODEL = None
OKT = Okt() 
QUERY_CACHE: Dict[str, List[str]] = {} # 쿼리 변환 결과를 저장할 메모리 캐시

try:
    print(f"BGE Reranker Base 모델({RERANKER_MODEL_NAME}) 로드를 시작합니다...")
    RERANKER_TOKENIZER = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
    RERANKER_MODEL = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RERANKER_MODEL.to(DEVICE)
    print(f"✅ Reranker 모델 로드 성공: {RERANKER_MODEL_NAME} (Device: {DEVICE})")
except Exception as e:
    print(f"❌ Reranker 모델 로드 실패: {RERANKER_MODEL_NAME} 모델 로드 중 오류 발생.")
    print(f"상세 오류: {e}")
    exit()

# --- 1. 데이터 로드 및 BM25 객체 생성 (기존 코드와 동일) ---
def load_chunk_data(chunk_file_path: str) -> Dict[int, str]:
    global ALL_CHUNK_TEXTS, BM25_MODEL
    mapping = {}
    with open(chunk_file_path, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]
    for index, chunk in enumerate(chunks):
        mapping[index] = chunk['doc_id'] 
        ALL_CHUNK_TEXTS.append(chunk['text'])
    print(f"청크-DocID 매핑 테이블 생성 완료. 총 {len(mapping)}개 항목.")
    tokenized_corpus = []
    for doc in tqdm(ALL_CHUNK_TEXTS, desc="BM25 Corpus Tokenizing (Okt - Nouns)"):
        tokens = OKT.nouns(doc)
        tokenized_corpus.append(tokens)
    BM25_MODEL = BM25Okapi(tokenized_corpus)
    print("✅ BM25 모델 (Okt 명사 추출 적용) 초기화 완료.")
    return mapping

# --- 2. 임베딩 및 FAISS 로드 (기존 코드와 동일) ---
def load_faiss_index(path: str):
    try:
        index = faiss.read_index(path)
        if index.d != TARGET_DIMENSION:
            print(f"❌ 경고: 로드된 인덱스 차원({index.d})이 목표 차원({TARGET_DIMENSION})과 다릅니다.")
        print(f"FAISS 인덱스 로드 성공: {path} (총 {index.ntotal}개 벡터)")
        return index
    except Exception as e:
        print(f"FAISS 인덱스 로드 실패: {e}")
        return None

def get_query_embeddings(queries: List[str]) -> np.ndarray:
    OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    response = client.embeddings.create(input=queries, model=OPENAI_EMBEDDING_MODEL, dimensions=TARGET_DIMENSION)
    embeddings = [data.embedding for data in response.data]
    return np.array(embeddings, dtype=np.float32)

# --- 신규 캐시 관리 함수 ---
def load_query_cache(path: str, eval_data_questions: List[str]) -> Dict[str, List[str]]:
    """ 캐시 파일을 로드합니다. 파일이 없거나 불완전하면 빈 딕셔너리를 반환합니다. """
    cache = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cache_list = json.load(f)
            # 리스트를 딕셔너리로 변환 (원본 쿼리: 변환 쿼리 리스트)
            for item in cache_list:
                cache[item['original_query']] = item['rewritten_queries']
        
        # 캐시의 완전성 검사
        if len(cache) == len(eval_data_questions):
            print(f"✅ Query Rewriting Cache 로드 성공: {len(cache)}개 쿼리 (완전)")
        else:
            print(f"⚠️ Query Rewriting Cache가 불완전합니다. 재구성을 시도합니다. (요청 {len(eval_data_questions)}개, 캐시 {len(cache)}개)")
            # 불완전한 캐시를 사용하면 변동성을 유발하므로 다시 LLM 호출을 유도
            return {} 
        return cache
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"⚠️ Query Rewriting Cache 파일을 찾을 수 없거나 형식이 잘못되었습니다: {path}. LLM 호출을 진행합니다.")
        return {}

def save_query_cache(path: str, cache: Dict[str, List[str]]):
    """ 현재 메모리 캐시를 파일로 저장합니다. """
    cache_list = [{'original_query': q, 'rewritten_queries': r} for q, r in cache.items()]
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache_list, f, ensure_ascii=False, indent=4)
        print(f"💾 Query Rewriting Cache 파일 저장 성공: {path}")
    except Exception as e:
        print(f"❌ Query Rewriting Cache 파일 저장 실패: {e}")

# --- 3. Multi-step Query Transformation (캐시 적용) ---
def transform_query(query: str) -> List[str]:
    """
    LLM을 사용하여 원본 쿼리를 여러 개의 명확한 검색 쿼리로 변환합니다.
    캐시가 있다면 캐시를 사용하고, 없다면 LLM을 호출하여 캐시를 채웁니다.
    """
    global QUERY_CACHE
    
    # 1. 캐시 히트 체크
    if query in QUERY_CACHE:
        return QUERY_CACHE[query]

    # 2. 캐시 미스 시 LLM 호출
    print(f"   [LLM 호출] {query} -> Rewriting...")
    prompt = f"""
    당신은 RFP 문서를 검색하기 위한 전문 검색어 생성기입니다.
    주어진 사용자 쿼리를 분석하여, RFP 문서에서 관련 정보를 찾을 수 있는 3가지의 독립적이고 명확한 검색 쿼리를 생성하세요.
    결과는 오직 쿼리 3개만, 각 줄에 하나씩 나열되어야 합니다. 다른 설명이나 문장은 포함하지 마세요.
    
    사용자 쿼리: "{query}"
    """
    try:
        # Rate Limit 회피를 위해 잠시 대기
        time.sleep(0.5) 
        
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        transformed_queries = response.choices[0].message.content.strip().split('\n')
        
        valid_queries = [q.strip() for q in transformed_queries if q.strip()]
        if not valid_queries:
            valid_queries = [query]
            
        # LLM 호출 결과를 캐시에 저장 (메모리)
        QUERY_CACHE[query] = valid_queries 
        return valid_queries
    
    except Exception as e:
        print(f"❌ 쿼리 변환 실패 (LLM 호출): {e}")
        return [query]

# --- 4. Hybrid Search 핵심 로직 (기존 코드와 동일) ---
def dense_search(index: faiss.Index, query_vector: np.ndarray, top_k: int) -> Dict[int, float]:
    faiss.normalize_L2(query_vector.reshape(1, -1))
    D, I = index.search(query_vector.reshape(1, -1), top_k)
    results = {}
    for rank, idx in enumerate(I[0]):
        if idx >= 0:
            results[int(idx)] = rank + 1
    return results

def sparse_search(query: str, top_k: int) -> Dict[int, float]:
    tokenized_query = OKT.nouns(query) 
    scores = BM25_MODEL.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1]
    results = {}
    for rank, idx in enumerate(ranked_indices[:top_k]):
        results[int(idx)] = rank + 1
    return results

def reciprocal_rank_fusion(all_ranks: List[Dict[int, float]], k: int = RRF_K) -> List[int]:
    fused_scores = defaultdict(float)
    for ranks in all_ranks:
        for index, rank in ranks.items():
            score = 1.0 / (k + rank)
            fused_scores[index] += score
            
    sorted_indices = sorted(fused_scores.keys(), key=lambda idx: fused_scores[idx], reverse=True)
    return sorted_indices[:HYBRID_K]

def multi_step_hybrid_search(index: faiss.Index, original_query: str) -> List[int]:
    
    # 1. 쿼리 변환 (캐시 사용)
    transformed_queries = transform_query(original_query)
    
    all_ranks: List[Dict[int, float]] = []

    for query in transformed_queries:
        dense_query = f"{TEST_PREFIX}{query}"
        query_vector = get_query_embeddings([dense_query])[0]
        dense_ranks = dense_search(index, query_vector, MAX_K_DENSE)
        sparse_ranks = sparse_search(query, MAX_K_SPARSE)
        
        all_ranks.append(dense_ranks)
        all_ranks.append(sparse_ranks)

    # 5. RRF 통합
    fused_indices = reciprocal_rank_fusion(all_ranks)
    return fused_indices

# --- 5. Reranker 로직 (기존 코드와 동일) ---
def rerank_results(query: str, retrieved_indices: List[int], chunk_mapping: Dict[int, str], top_k: int) -> List[int]:
    pairs = []
    for idx in retrieved_indices:
        chunk_text = ALL_CHUNK_TEXTS[idx]
        pairs.append([query, chunk_text])
        
    if not pairs: return []

    inputs = RERANKER_TOKENIZER(pairs, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        RERANKER_MODEL.eval()
        outputs = RERANKER_MODEL(**inputs)
        scores = outputs.logits.squeeze(dim=1).cpu().numpy()
    
    indexed_scores = list(zip(retrieved_indices, scores))
    indexed_scores.sort(key=lambda item: item[1], reverse=True)
    reranked_indices = [idx for idx, score in indexed_scores]
    
    return reranked_indices[:top_k]

# --- 6. Multi-step Reranked Recall 계산 함수 (R@10 포함) ---
def evaluate_reranked_recall(
    index: faiss.Index, eval_data: List[Dict], chunk_mapping: Dict[int, str], max_k: int = RERANK_TOP_K
) -> Dict[str, float]:
    
    k_list = sorted([k for k in [1, 3, 5, 10] if k <= max_k]) 
    results = {f"R@{k}": 0 for k in k_list}
    total_queries = len(eval_data)
    
    gt_doc_ids = [item['gt_doc_id'] for item in eval_data]
    original_queries = [item['question'] for item in eval_data] 

    for q_idx in tqdm(range(total_queries), desc="Multi-step Reranked Recall 평가 진행 중"):
        query_text = original_queries[q_idx] 
        ground_truth_doc = gt_doc_ids[q_idx]
        
        retrieved_indices_hybrid = multi_step_hybrid_search(index, query_text)
        reranked_indices = rerank_results(query_text, retrieved_indices_hybrid, chunk_mapping, top_k=RERANK_TOP_K)
        
        retrieved_docs = [chunk_mapping.get(idx) for idx in reranked_indices if idx != -1] 
        
        for k in k_list:
            if ground_truth_doc in retrieved_docs[:k]:
                results[f"R@{k}"] += 1
                
    for k in results.keys():
        results[k] /= total_queries
        
    return results

# --- 7. 메인 실행 함수 (캐시 관리 로직 추가) ---

if __name__ == "__main__":
    
    print("\n===================================================")
    print(f"🚀 최종 시스템 평가 (Multi-step Hybrid + Reranker Base) - RRF K={RRF_K}")
    print("===================================================")
    
    index = load_faiss_index(FAISS_INDEX_PATH)
    if index is None: exit()
    
    try:
        with open(EVAL_JSONL_PATH, 'r', encoding='utf-8') as f:
            eval_data = [json.loads(line) for line in f]
        print(f"평가 데이터셋 로드 성공: 총 {len(eval_data)}개 쿼리")
    except Exception as e:
        print(f"❌ 평가 데이터셋 로드 실패: {EVAL_JSONL_PATH} 파일 확인 필요. {e}")
        exit()
        
    try:
        chunk_mapping = load_chunk_data(CHUNK_FILE_PATH)
    except Exception as e:
        print(f"❌ 청크 파일 로드 및 BM25 초기화 실패: {CHUNK_FILE_PATH} 파일 경로/형식 확인 필요. {e}")
        exit()

    # 🚨 쿼리 캐시 로드 및 LLM 호출/캐시 생성 로직 
    eval_questions = [item['question'] for item in eval_data]
    QUERY_CACHE = load_query_cache(QUERY_CACHE_PATH, eval_questions)

    print("\n---------------------------------------------------")
    print(f"   - 1차 검색: Multi-step Hybrid Search (RRF K={RRF_K}, Top {HYBRID_K})")
    print(f"   - 2차 순위 조정: {RERANKER_MODEL_NAME} (Top {RERANK_TOP_K})")
    print("---------------------------------------------------")

    final_results = evaluate_reranked_recall(
        index, 
        eval_data, 
        chunk_mapping, 
        max_k=RERANK_TOP_K
    )
    
    # 🚨 LLM 호출이 이루어졌고, 캐시 파일이 생성되지 않은 경우 (최초 실행) 저장
    if not os.path.exists(QUERY_CACHE_PATH) and len(QUERY_CACHE) == len(eval_questions):
        print("\n최초 실행 완료. 성능 변동을 막기 위해 쿼리 변환 결과를 캐싱합니다.")
        save_query_cache(QUERY_CACHE_PATH, QUERY_CACHE)
        
    print("\n---------------------------------------------------")
    print("🌟 Multi-step Hybrid + Reranker Base 최종 성능:")
    for k, score in final_results.items():
        print(f"{k}: {score:.3f}")
    print("===================================================")
