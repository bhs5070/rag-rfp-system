import numpy as np
import faiss
import time
from tqdm import tqdm
# 메타데이터 로드 추가
import pickle

with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"✅ 메타데이터 로드: {len(metadata)}개")
# 기존 임베딩 로드
embeddings = np.load("embeddings.npy")
dim = embeddings.shape[1]
n_vectors = embeddings.shape[0]

def create_multiple_indexes(embeddings):
    """다양한 FAISS 인덱스 생성"""
    indexes = {}
    
    print(f"Creating indexes for {n_vectors:,} vectors, {dim}D")
    
    # 1. IndexFlatIP (현재 사용중 - 베이스라인)
    print("1. IndexFlatIP (Exact Inner Product)")
    start_time = time.time()
    index_flat_ip = faiss.IndexFlatIP(dim)
    index_flat_ip.add(embeddings)
    indexes['FlatIP'] = {
        'index': index_flat_ip,
        'build_time': time.time() - start_time,
        'type': 'exact'
    }
    
    # 2. IndexFlatL2 (정확한 L2 거리)
    print("2. IndexFlatL2 (Exact L2 Distance)")
    start_time = time.time()
    index_flat_l2 = faiss.IndexFlatL2(dim)
    index_flat_l2.add(embeddings)
    indexes['FlatL2'] = {
        'index': index_flat_l2,
        'build_time': time.time() - start_time,
        'type': 'exact'
    }
    
    # 3. IndexIVFFlat (근사 검색)
    print("3. IndexIVFFlat (Approximate)")
    nlist = min(4 * int(np.sqrt(n_vectors)), n_vectors // 39)  # 클러스터 수
    quantizer = faiss.IndexFlatIP(dim)
    start_time = time.time()
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index_ivf.train(embeddings)
    index_ivf.add(embeddings)
    index_ivf.nprobe = min(nlist // 4, 50)  # 검색할 클러스터 수
    indexes['IVFFlat'] = {
        'index': index_ivf,
        'build_time': time.time() - start_time,
        'type': 'approximate',
        'nlist': nlist,
        'nprobe': index_ivf.nprobe
    }
    
    # 4. IndexIVFPQ (압축 + 근사)
    print("4. IndexIVFPQ (Compressed + Approximate)")
    m = 64  # PQ 서브벡터 수
    nbits = 8  # 비트 수
    start_time = time.time()
    index_ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    index_ivfpq.train(embeddings)
    index_ivfpq.add(embeddings)
    index_ivfpq.nprobe = min(nlist // 4, 50)
    indexes['IVFPQ'] = {
        'index': index_ivfpq,
        'build_time': time.time() - start_time,
        'type': 'compressed',
        'nlist': nlist,
        'nprobe': index_ivfpq.nprobe,
        'm': m,
        'nbits': nbits
    }
    
    # 5. IndexHNSWFlat (고속 근사)
    print("5. IndexHNSWFlat (Fast Approximate)")
    M = 32  # 연결 수
    start_time = time.time()
    index_hnsw = faiss.IndexHNSWFlat(dim, M)
    index_hnsw.hnsw.efConstruction = 200
    index_hnsw.add(embeddings)
    index_hnsw.hnsw.efSearch = 128
    indexes['HNSW'] = {
        'index': index_hnsw,
        'build_time': time.time() - start_time,
        'type': 'graph',
        'M': M,
        'efConstruction': 200,
        'efSearch': 128
    }
    
    return indexes

def benchmark_indexes(indexes, query_embeddings, k=5):
    """인덱스별 성능 측정"""
    results = {}
    
    for name, index_info in indexes.items():
        print(f"\n🔍 Testing {name}...")
        
        index = index_info['index']
        search_times = []
        
        # 검색 시간 측정
        for query_emb in tqdm(query_embeddings, desc=f"{name} search"):
            start_time = time.time()
            D, I = index.search(query_emb.reshape(1, -1), k)
            search_times.append(time.time() - start_time)
        
        avg_search_time = np.mean(search_times)
        qps = len(query_embeddings) / sum(search_times)  # Queries Per Second
        
        # 메모리 사용량 추정 (수정된 부분)
        try:
            # FAISS 인덱스별 메모리 사용량 계산
            if name == 'FlatIP' or name == 'FlatL2':
                # Flat 인덱스: 전체 벡터 저장
                memory_mb = (n_vectors * dim * 4) / (1024**2)  # float32
            elif 'IVF' in name:
                # IVF 인덱스: 클러스터 + 벡터
                nlist = index_info.get('nlist', 100)
                if 'PQ' in name:
                    # PQ 압축된 경우
                    m = index_info.get('m', 64)
                    memory_mb = (n_vectors * m + nlist * dim * 4) / (1024**2)
                else:
                    # IVFFlat
                    memory_mb = (n_vectors * dim * 4 + nlist * dim * 4) / (1024**2)
            elif name == 'HNSW':
                # HNSW: 벡터 + 그래프 구조
                M = index_info.get('M', 32)
                memory_mb = (n_vectors * (dim * 4 + M * 4)) / (1024**2)
            else:
                # 기본값
                memory_mb = (n_vectors * dim * 4) / (1024**2)
                
        except Exception as e:
            print(f"   ⚠️ 메모리 계산 오류: {e}")
            memory_mb = (n_vectors * dim * 4) / (1024**2)  # 기본값
        
        results[name] = {
            'avg_search_time_ms': avg_search_time * 1000,
            'qps': qps,
            'build_time_sec': index_info['build_time'],
            'memory_mb': memory_mb,
            'type': index_info['type']
        }
        
        print(f"   ⏱️  평균 검색시간: {avg_search_time*1000:.2f}ms")
        print(f"   🚀 QPS: {qps:.1f}")
        print(f"   🏗️  빌드 시간: {index_info['build_time']:.2f}초")
        print(f"   💾 메모리: {memory_mb:.1f}MB")
    
    return results


def measure_recall_accuracy(indexes, evaluation_queries, metadata, model, k_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """인덱스별 Recall 정확도 측정 (R@1~10)"""
    recall_results = {}
    
    for index_name, index_info in indexes.items():
        print(f"\n📊 Measuring recall for {index_name}...")
        
        index = index_info['index']
        recalls = {f'R@{k}': 0 for k in k_values}
        
        for query_data in tqdm(evaluation_queries, desc=f"{index_name} recall"):
            query_text = query_data['question']
            gt_doc_id = query_data['gt_doc_id']
            
            # 쿼리 임베딩 (BGE-M3 사용)
            q_emb = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
            
            # 최대 k값으로 검색 (10으로 변경)
            max_k = 10
            D, I = index.search(q_emb, max_k)
            
            # 검색된 문서들의 doc_id 추출
            retrieved_doc_ids = []
            for idx in I[0]:
                if idx < len(metadata):
                    retrieved_doc_ids.append(metadata[idx]['doc_id'])
            
            # 각 k값에 대해 Recall 계산
            for k in k_values:
                if gt_doc_id in retrieved_doc_ids[:k]:
                    recalls[f'R@{k}'] += 1
        
        # 평균 Recall 계산
        for k in k_values:
            recalls[f'R@{k}'] /= len(evaluation_queries)
        
        recall_results[index_name] = recalls
        
        # 결과 출력 - 한 줄에 표시
        recall_str = " | ".join([f"R@{k}: {recalls[f'R@{k}']:.3f}" for k in k_values])
        print(f"   {recall_str}")
    
    return recall_results


def find_best_index(performance_results, recall_results, weights={'recall_5': 0.5, 'qps': 0.3, 'memory': 0.2}):
    """종합 점수로 최적 인덱스 선택"""
    
    print("\n🏆 종합 성능 분석")
    print("="*60)
    
    # 정규화를 위한 최대/최소값
    max_recall = max([r['R@5'] for r in recall_results.values()])
    max_qps = max([p['qps'] for p in performance_results.values()])
    min_memory = min([p['memory_mb'] for p in performance_results.values()])
    max_memory = max([p['memory_mb'] for p in performance_results.values()])
    
    scores = {}
    
    for name in performance_results.keys():
        recall_5 = recall_results[name]['R@5']
        qps = performance_results[name]['qps']
        memory = performance_results[name]['memory_mb']
        
        # 정규화 (0-1 범위)
        norm_recall = recall_5 / max_recall if max_recall > 0 else 0
        norm_qps = qps / max_qps if max_qps > 0 else 0
        norm_memory = (max_memory - memory) / (max_memory - min_memory) if max_memory > min_memory else 1
        
        # 가중 점수 계산
        total_score = (
            weights['recall_5'] * norm_recall +
            weights['qps'] * norm_qps +
            weights['memory'] * norm_memory
        )
        
        scores[name] = {
            'total_score': total_score,
            'recall_5': recall_5,
            'qps': qps,
            'memory_mb': memory,
            'type': performance_results[name]['type']
        }
    
    # 결과 출력
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    
    print(f"{'Rank':<4} {'Index':<10} {'Score':<6} {'R@5':<6} {'QPS':<8} {'Memory':<10} {'Type'}")
    print("-"*60)
    
    for i, (name, score_info) in enumerate(sorted_scores, 1):
        print(f"{i:<4} {name:<10} {score_info['total_score']:.3f}  "
              f"{score_info['recall_5']:.3f}  {score_info['qps']:<8.1f} "
              f"{score_info['memory_mb']:<10.1f} {score_info['type']}")
    
    best_index = sorted_scores[0][0]
    print(f"\n🎯 최적 인덱스: {best_index}")
    
    return best_index, scores


def print_detailed_recall_comparison(recall_results):
    """상세한 Recall 비교 테이블 출력"""
    print("\n📊 상세 Recall@K 비교 결과")
    print("=" * 100)
    
    # 헤더 출력
    header = "Index    "
    for k in range(1, 11):
        header += f" R@{k:<2}"
    print(header)
    print("-" * 100)
    
    # 각 인덱스별 결과 출력
    for index_name, recalls in recall_results.items():
        row = f"{index_name:<8} "
        for k in range(1, 11):
            recall_value = recalls[f'R@{k}']
            row += f" {recall_value:<4.3f}"
        print(row)
    
    # 최고 성능 표시
    print("\n🏆 각 K값별 최고 성능:")
    for k in range(1, 11):
        best_recall = max([recalls[f'R@{k}'] for recalls in recall_results.values()])
        best_indexes = [name for name, recalls in recall_results.items() 
                       if recalls[f'R@{k}'] == best_recall]
        print(f"   R@{k}: {best_recall:.3f} ({', '.join(best_indexes)})")


def find_best_index_detailed(performance_results, recall_results, 
                           weights={'recall_1': 0.1, 'recall_3': 0.2, 'recall_5': 0.3, 
                                   'recall_10': 0.2, 'qps': 0.1, 'memory': 0.1}):
    """다양한 Recall@K를 고려한 최적 인덱스 선택"""
    
    print("\n🏆 종합 성능 분석 (R@1~10 고려)")
    print("="*80)
    
    # 정규화를 위한 최대/최소값
    max_values = {}
    for k in [1, 3, 5, 10]:
        max_values[f'recall_{k}'] = max([r[f'R@{k}'] for r in recall_results.values()])
    
    max_qps = max([p['qps'] for p in performance_results.values()])
    min_memory = min([p['memory_mb'] for p in performance_results.values()])
    max_memory = max([p['memory_mb'] for p in performance_results.values()])
    
    scores = {}
    
    for name in performance_results.keys():
        # 각 Recall 값들
        r1 = recall_results[name]['R@1']
        r3 = recall_results[name]['R@3']
        r5 = recall_results[name]['R@5']
        r10 = recall_results[name]['R@10']
        
        qps = performance_results[name]['qps']
        memory = performance_results[name]['memory_mb']
        
        # 정규화 (0-1 범위)
        norm_r1 = r1 / max_values['recall_1'] if max_values['recall_1'] > 0 else 0
        norm_r3 = r3 / max_values['recall_3'] if max_values['recall_3'] > 0 else 0
        norm_r5 = r5 / max_values['recall_5'] if max_values['recall_5'] > 0 else 0
        norm_r10 = r10 / max_values['recall_10'] if max_values['recall_10'] > 0 else 0
        norm_qps = qps / max_qps if max_qps > 0 else 0
        norm_memory = (max_memory - memory) / (max_memory - min_memory) if max_memory > min_memory else 1
        
        # 가중 점수 계산
        total_score = (
            weights['recall_1'] * norm_r1 +
            weights['recall_3'] * norm_r3 +
            weights['recall_5'] * norm_r5 +
            weights['recall_10'] * norm_r10 +
            weights['qps'] * norm_qps +
            weights['memory'] * norm_memory
        )
        
        scores[name] = {
            'total_score': total_score,
            'r1': r1, 'r3': r3, 'r5': r5, 'r10': r10,
            'qps': qps, 'memory_mb': memory,
            'type': performance_results[name]['type']
        }
    
    # 결과 출력
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    
    print(f"{'Rank':<4} {'Index':<10} {'Score':<6} {'R@1':<5} {'R@3':<5} {'R@5':<5} {'R@10':<5} {'QPS':<7} {'Memory':<8} {'Type'}")
    print("-"*80)
    
    for i, (name, score_info) in enumerate(sorted_scores, 1):
        print(f"{i:<4} {name:<10} {score_info['total_score']:.3f}  "
              f"{score_info['r1']:<5.3f} {score_info['r3']:<5.3f} "
              f"{score_info['r5']:<5.3f} {score_info['r10']:<5.3f} "
              f"{score_info['qps']:<7.1f} {score_info['memory_mb']:<8.1f} "
              f"{score_info['type']}")
    
    best_index = sorted_scores[0][0]
    print(f"\n🎯 최적 인덱스: {best_index}")
    
    return best_index, scores





# 필요한 데이터 로드
embeddings = np.load("embeddings.npy")
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
with open("our_clean_eval_style.jsonl", "r") as f:
    eval_queries = [json.loads(line) for line in f]

# BGE-M3 모델 로드
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)

dim = embeddings.shape[1]
n_vectors = embeddings.shape[0]

# 테스트용 쿼리 임베딩
np.random.seed(42)
test_query_embeddings = np.random.random((50, dim)).astype('float32')
faiss.normalize_L2(test_query_embeddings)

# 실행 (k_values를 1~10으로 확장)
print("🚀 FAISS 인덱스 성능 비교 시작! (R@1~10)")

# 1. 인덱스 생성
indexes = create_multiple_indexes(embeddings)

# 2. 속도 벤치마크
performance_results = benchmark_indexes(indexes, test_query_embeddings)

# 3. Recall 정확도 측정 (R@1~10)
recall_results = measure_recall_accuracy(indexes, eval_queries, metadata, model, 
                                       k_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 4. 상세 결과 출력
print_detailed_recall_comparison(recall_results)

# 5. 최적 인덱스 선택 (다양한 Recall 고려)
best_index, all_scores = find_best_index_detailed(performance_results, recall_results)

# 6. 최적 인덱스 저장
print(f"\n💾 최적 인덱스 ({best_index}) 저장 중...")
faiss.write_index(indexes[best_index]['index'], f"best_{best_index.lower()}.index")
print(f"✅ 저장 완료: best_{best_index.lower()}.index")

for name, index_info in indexes.items():
    filename = f"{name.lower()}_index.faiss"
    faiss.write_index(index_info['index'], filename)
    print(f"✅ {name} 저장: {filename}")