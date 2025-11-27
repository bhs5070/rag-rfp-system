"""
FAISS Vector Database Indexing Module
=====================================
임베딩 벡터를 FAISS 인덱스로 구축하여 고속 검색을 지원하는 모듈

Features:
- 다양한 FAISS 인덱스 타입 지원
- 메모리 효율적인 인덱싱
- 검색 성능 최적화
- 인덱스 저장/로드 기능

Author: 원후 (Bidding Mate RAG Team)
"""

import json
import numpy as np
import faiss
import os
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pickle
import time


class FAISSIndexer:
    """FAISS 벡터 데이터베이스 인덱싱 클래스"""
    
    def __init__(self, embedding_dim: int = 1536):
        """
        Args:
            embedding_dim: 임베딩 벡터 차원
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
        self.index_type = None
        
        print(f"FAISS 인덱서 초기화 완료")
        print(f"임베딩 차원: {self.embedding_dim}")
    
    def load_embeddings_and_chunks(self, embedding_file: str, chunks_file: str) -> Tuple[np.ndarray, List[Dict]]:
        """임베딩과 청크 데이터 로드"""
        print(f"데이터 로드 중...")
        
        # 임베딩 로드
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"임베딩 파일이 없습니다: {embedding_file}")
        
        embeddings = np.load(embedding_file)
        print(f"임베딩 로드: {embeddings.shape}")
        
        # 청크 데이터 로드
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"청크 파일이 없습니다: {chunks_file}")
        
        chunks = []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                chunks.append(chunk)
        
        print(f"청크 데이터 로드: {len(chunks):,}개")
        
        # 데이터 일관성 검증
        if len(embeddings) != len(chunks):
            raise ValueError(f"데이터 불일치: 임베딩({len(embeddings)}) vs 청크({len(chunks)})")
        
        return embeddings, chunks
    
    def create_flat_index(self, embeddings: np.ndarray, metric: str = "ip") -> faiss.Index:
        """Flat 인덱스 생성 (정확한 검색)"""
        print(f"Flat 인덱스 생성 중... (metric: {metric})")
        
        if metric == "ip":  # Inner Product (cosine similarity)
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif metric == "l2":  # L2 distance
            index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"지원하지 않는 metric: {metric}")
        
        # 임베딩 정규화 (cosine similarity를 위해)
        if metric == "ip":
            normalized_embeddings = embeddings.copy()
            faiss.normalize_L2(normalized_embeddings)
            index.add(normalized_embeddings.astype(np.float32))
        else:
            index.add(embeddings.astype(np.float32))
        
        print(f"Flat 인덱스 생성 완료: {index.ntotal:,}개 벡터")
        return index
    
    def create_ivf_index(self, embeddings: np.ndarray, nlist: int = None, metric: str = "ip") -> faiss.Index:
        """IVF 인덱스 생성 (근사 검색, 고속)"""
        print(f"IVF 인덱스 생성 중... (metric: {metric})")
        
        # nlist 자동 설정
        if nlist is None:
            nlist = min(4 * int(np.sqrt(len(embeddings))), len(embeddings) // 39)
            nlist = max(nlist, 1)
        
        print(f"클러스터 수: {nlist}")
        
        if metric == "ip":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        elif metric == "l2":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)
        else:
            raise ValueError(f"지원하지 않는 metric: {metric}")
        
        # 임베딩 정규화 (cosine similarity를 위해)
        if metric == "ip":
            normalized_embeddings = embeddings.copy()
            faiss.normalize_L2(normalized_embeddings)
            embeddings_to_train = normalized_embeddings.astype(np.float32)
        else:
            embeddings_to_train = embeddings.astype(np.float32)
        
        # 훈련 및 추가
        print("인덱스 훈련 중...")
        index.train(embeddings_to_train)
        print("벡터 추가 중...")
        index.add(embeddings_to_train)
        
        print(f"IVF 인덱스 생성 완료: {index.ntotal:,}개 벡터")
        return index
    
    def create_hnsw_index(self, embeddings: np.ndarray, M: int = 16, metric: str = "ip") -> faiss.Index:
        """HNSW 인덱스 생성 (고성능 근사 검색)"""
        print(f"HNSW 인덱스 생성 중... (M: {M}, metric: {metric})")
        
        if metric == "ip":
            index = faiss.IndexHNSWFlat(self.embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
        elif metric == "l2":
            index = faiss.IndexHNSWFlat(self.embedding_dim, M, faiss.METRIC_L2)
        else:
            raise ValueError(f"지원하지 않는 metric: {metric}")
        
        # 임베딩 정규화 (cosine similarity를 위해)
        if metric == "ip":
            normalized_embeddings = embeddings.copy()
            faiss.normalize_L2(normalized_embeddings)
            index.add(normalized_embeddings.astype(np.float32))
        else:
            index.add(embeddings.astype(np.float32))
        
        print(f"HNSW 인덱스 생성 완료: {index.ntotal:,}개 벡터")
        return index
    
    def build_index(self, embeddings: np.ndarray, chunks: List[Dict], 
                   index_type: str = "flat", **kwargs) -> None:
        """지정된 타입의 인덱스 구축"""
        print(f"{index_type.upper()} 인덱스 구축 시작")
        print(f"데이터: {len(embeddings):,}개 벡터 × {embeddings.shape[1]}차원")
        
        start_time = time.time()
        
        if index_type.lower() == "flat":
            self.index = self.create_flat_index(embeddings, kwargs.get('metric', 'ip'))
        elif index_type.lower() == "ivf":
            self.index = self.create_ivf_index(embeddings, kwargs.get('nlist'), kwargs.get('metric', 'ip'))
        elif index_type.lower() == "hnsw":
            self.index = self.create_hnsw_index(embeddings, kwargs.get('M', 16), kwargs.get('metric', 'ip'))
        else:
            raise ValueError(f"지원하지 않는 인덱스 타입: {index_type}")
        
        self.chunks = chunks
        self.index_type = index_type
        
        build_time = time.time() - start_time
        
        print(f"인덱스 구축 완료!")
        print(f"구축 시간: {build_time:.2f}초")
        print(f"인덱스 크기: {self.get_index_size():.1f}MB")
    
    def get_index_size(self) -> float:
        """인덱스 메모리 사용량 추정 (MB)"""
        if self.index is None:
            return 0.0
        
        # 대략적인 메모리 사용량 계산
        base_size = self.index.ntotal * self.embedding_dim * 4  # float32
        if self.index_type == "ivf":
            # IVF는 클러스터 오버헤드 추가
            base_size *= 1.2
        elif self.index_type == "hnsw":
            # HNSW는 그래프 오버헤드 추가  
            base_size *= 1.5
        
        return base_size / (1024 ** 2)
    
    def search(self, query_embedding: np.ndarray, k: int = 5, nprobe: int = None) -> Tuple[List[float], List[int]]:
        """벡터 검색 수행"""
        if self.index is None:
            raise ValueError("인덱스가 구축되지 않았습니다.")
        
        # 쿼리 벡터 정규화 (cosine similarity를 위해)
        query_norm = query_embedding.copy().reshape(1, -1)
        faiss.normalize_L2(query_norm)
        
        # IVF 인덱스 검색 파라미터 설정
        if self.index_type == "ivf" and nprobe:
            self.index.nprobe = nprobe
        
        # 검색 수행
        scores, indices = self.index.search(query_norm.astype(np.float32), k)
        
        return scores[0].tolist(), indices[0].tolist()
    
    def save_index(self, index_file: str, chunks_file: str = None) -> None:
        """인덱스와 청크 데이터 저장"""
        if self.index is None:
            raise ValueError("저장할 인덱스가 없습니다.")
        
        print(f"인덱스 저장 중: {index_file}")
        
        # FAISS 인덱스 저장
        faiss.write_index(self.index, index_file)
        
        # 메타데이터 저장
        metadata = {
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'total_vectors': self.index.ntotal,
            'created_at': time.time()
        }
        
        metadata_file = index_file.replace('.index', '_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"인덱스 저장 완료")
        print(f"인덱스: {index_file}")
        print(f"메타데이터: {metadata_file}")
        
        # 청크 데이터 저장 (옵션)
        if chunks_file and self.chunks:
            print(f"청크 데이터 저장: {chunks_file}")
            with open(chunks_file, 'w', encoding='utf-8') as f:
                for chunk in self.chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            print(f"청크 데이터 저장 완료: {len(self.chunks):,}개")
    
    def load_index(self, index_file: str, chunks_file: str = None) -> None:
        """저장된 인덱스와 청크 데이터 로드"""
        print(f"인덱스 로드 중: {index_file}")
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"인덱스 파일이 없습니다: {index_file}")
        
        # FAISS 인덱스 로드
        self.index = faiss.read_index(index_file)
        
        # 메타데이터 로드
        metadata_file = index_file.replace('.index', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.index_type = metadata.get('index_type', 'unknown')
            print(f"메타데이터 로드: {self.index_type} 인덱스")
        
        # 청크 데이터 로드 (옵션)
        if chunks_file and os.path.exists(chunks_file):
            print(f"청크 데이터 로드: {chunks_file}")
            chunks = []
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk = json.loads(line)
                    chunks.append(chunk)
            self.chunks = chunks
            print(f"청크 데이터 로드: {len(chunks):,}개")
        
        print(f"인덱스 로드 완료: {self.index.ntotal:,}개 벡터")
    
    def benchmark_search(self, test_queries: List[np.ndarray], k: int = 5) -> Dict:
        """검색 성능 벤치마크"""
        if self.index is None:
            raise ValueError("인덱스가 구축되지 않았습니다.")
        
        print(f"검색 성능 벤치마크 시작: {len(test_queries)}개 쿼리")
        
        total_time = 0
        for query in tqdm(test_queries, desc="검색 벤치마크"):
            start_time = time.time()
            self.search(query, k)
            total_time += time.time() - start_time
        
        avg_time = total_time / len(test_queries)
        qps = len(test_queries) / total_time
        
        benchmark_result = {
            'total_queries': len(test_queries),
            'total_time': total_time,
            'avg_time_per_query': avg_time,
            'queries_per_second': qps,
            'index_type': self.index_type,
            'k': k
        }
        
        print(f"벤치마크 완료!")
        print(f" 평균 응답시간: {avg_time*1000:.2f}ms")
        print(f" 초당 쿼리 수: {qps:.1f} QPS")
        
        return benchmark_result


def main():
    """메인 실행 함수"""
    print("FAISS 인덱싱 시스템 실행")
    print("="*50)
    
    # 파일 경로 설정
    embedding_file = "multi_aspect_embeddings.npy"
    chunks_file = "multi_aspect_chunks_processed.jsonl"
    index_file = "multi_aspect_faiss.index"
    index_chunks_file = "multi_aspect_index_chunks.jsonl"
    
    # FAISS 인덱서 초기화
    indexer = FAISSIndexer(embedding_dim=1536)
    
    # 데이터 로드
    try:
        embeddings, chunks = indexer.load_embeddings_and_chunks(embedding_file, chunks_file)
    except FileNotFoundError as e:
        print(f"{e}")
        print("   임베딩 생성을 먼저 실행해주세요.")
        return
    
    # 인덱스 구축 - 여러 타입 비교
    index_configs = [
        {"index_type": "flat", "metric": "ip"},
        {"index_type": "ivf", "metric": "ip", "nlist": 100},
        {"index_type": "hnsw", "metric": "ip", "M": 16}
    ]
    
    best_config = None
    best_qps = 0
    
    # 테스트 쿼리 생성 (랜덤 샘플)
    test_queries = [embeddings[i] for i in np.random.choice(len(embeddings), 10, replace=False)]
    
    for config in index_configs:
        print(f"\n{config['index_type'].upper()} 인덱스 테스트")
        print("-" * 30)
        
        # 인덱스 구축
        indexer.build_index(embeddings, chunks, **config)
        
        # 성능 벤치마크
        benchmark = indexer.benchmark_search(test_queries, k=5)
        
        # 최고 성능 추적
        if benchmark['queries_per_second'] > best_qps:
            best_qps = benchmark['queries_per_second']
            best_config = config.copy()
    
    # 최고 성능 인덱스로 최종 구축
    print(f"\n최고 성능 인덱스로 최종 구축: {best_config['index_type'].upper()}")
    print(f"성능: {best_qps:.1f} QPS")
    
    indexer.build_index(embeddings, chunks, **best_config)
    
    # 인덱스 저장
    indexer.save_index(index_file, index_chunks_file)
    
    # 최종 통계
    print("\nFAISS 인덱싱 완료 통계")
    print("-" * 30)
    print(f"인덱스 타입: {indexer.index_type.upper()}")
    print(f"총 벡터 수: {indexer.index.ntotal:,}개")
    print(f"임베딩 차원: {indexer.embedding_dim}")
    print(f"인덱스 크기: {indexer.get_index_size():.1f}MB")
    print(f"검색 성능: {best_qps:.1f} QPS")
    print(f"출력 파일:")
    print(f"  - 인덱스: {index_file}")
    print(f"  - 청크: {index_chunks_file}")
    
    print(f"\nFAISS 인덱싱 완료!")


if __name__ == "__main__":
    main()