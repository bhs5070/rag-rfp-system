"""
Embedding Generation Module
===========================
청킹된 텍스트를 OpenAI 임베딩으로 변환하는 모듈

Features:
- OpenAI text-embedding-3-small 모델 사용
- 배치 처리로 API 효율성 최적화
- 임베딩 결과 저장 및 로드
- 진행상황 시각화

Author: 원후 (Bidding Mate RAG Team)
"""

import json
import numpy as np
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
from openai import OpenAI
import pickle


class EmbeddingGenerator:
    """OpenAI 임베딩 생성 클래스"""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        """
        Args:
            api_key: OpenAI API 키 (환경변수에서 자동 로드)
            model: 사용할 임베딩 모델
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.embedding_dim = 1536  # text-embedding-3-small 차원
        self.batch_size = 100  # API 배치 크기
        
        print(f"임베딩 생성기 초기화 완료")
        print(f" 모델: {self.model}")
        print(f" 차원: {self.embedding_dim}")
        print(f" 배치 크기: {self.batch_size}")
    
    def load_chunks(self, jsonl_file: str) -> List[Dict]:
        """청킹된 데이터 로드"""
        print(f"청크 데이터 로드: {jsonl_file}")
        
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"청크 파일이 없습니다: {jsonl_file}")
        
        chunks = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Line {line_no} JSON 파싱 오류: {e}")
                    continue
        
        print(f"{len(chunks):,}개 청크 로드 완료")
        return chunks
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 단위로 임베딩 생성"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            embeddings = [embedding.embedding for embedding in response.data]
            return embeddings
            
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            # 개별적으로 재시도
            embeddings = []
            for text in texts:
                try:
                    response = self.client.embeddings.create(
                        input=[text],
                        model=self.model
                    )
                    embeddings.append(response.data[0].embedding)
                except Exception as e2:
                    print(f"개별 텍스트 임베딩 실패: {e2}")
                    # 제로 벡터로 대체
                    embeddings.append([0.0] * self.embedding_dim)
            
            return embeddings
    
    def generate_embeddings(self, chunks: List[Dict], cache_file: str = None) -> Tuple[np.ndarray, List[Dict]]:
        """전체 청크에 대해 임베딩 생성"""
        print(f"임베딩 생성 시작 - {len(chunks):,}개 청크")
        
        # 캐시 파일 확인
        if cache_file and os.path.exists(cache_file):
            print(f"캐시 파일 발견: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print("캐시에서 임베딩 로드 완료")
                return cached_data['embeddings'], cached_data['chunks']
            except Exception as e:
                print(f"캐시 로드 실패: {e}")
        
        embeddings_list = []
        processed_chunks = []
        
        # 배치 단위로 처리
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(0, len(chunks), self.batch_size), 
                             desc="임베딩 생성", total=total_batches):
            
            batch_chunks = chunks[batch_idx:batch_idx + self.batch_size]
            batch_texts = [chunk['text'] for chunk in batch_chunks]
            
            # 임베딩 생성
            batch_embeddings = self.create_embeddings_batch(batch_texts)
            
            # 결과 저장
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                embeddings_list.append(embedding)
                processed_chunks.append(chunk)
            
            # API 요청 간격 조절 (rate limit 방지)
            if batch_idx > 0 and batch_idx % (self.batch_size * 5) == 0:
                time.sleep(1)
        
        # numpy 배열로 변환
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        
        print(f"임베딩 생성 완료!")
        print(f" 임베딩 형태: {embeddings_array.shape}")
        print(f" 메모리 사용량: {embeddings_array.nbytes / (1024**2):.1f}MB")
        
        # 캐시 저장
        if cache_file:
            print(f"임베딩 캐시 저장: {cache_file}")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'embeddings': embeddings_array,
                        'chunks': processed_chunks,
                        'model': self.model,
                        'created_at': time.time()
                    }, f)
                print("캐시 저장 완료")
            except Exception as e:
                print(f"캐시 저장 실패: {e}")
        
        return embeddings_array, processed_chunks
    
    def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict], 
                       embedding_file: str, chunks_file: str = None) -> None:
        """임베딩과 청크 데이터 저장"""
        print(f"임베딩 저장: {embedding_file}")
        
        # 임베딩 저장 (numpy 형식)
        np.save(embedding_file, embeddings)
        print(f"임베딩 저장 완료: {embeddings.shape}")
        
        # 청크 데이터 저장 (옵션)
        if chunks_file:
            print(f"청크 데이터 저장: {chunks_file}")
            with open(chunks_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            print(f"청크 데이터 저장 완료: {len(chunks):,}개")
    
    def load_embeddings(self, embedding_file: str, chunks_file: str = None) -> Tuple[np.ndarray, List[Dict]]:
        """저장된 임베딩과 청크 데이터 로드"""
        print(f"임베딩 로드: {embedding_file}")
        
        # 임베딩 로드
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"임베딩 파일이 없습니다: {embedding_file}")
        
        embeddings = np.load(embedding_file)
        print(f"임베딩 로드 완료: {embeddings.shape}")
        
        # 청크 데이터 로드 (옵션)
        chunks = []
        if chunks_file and os.path.exists(chunks_file):
            print(f"청크 데이터 로드: {chunks_file}")
            chunks = self.load_chunks(chunks_file)
        
        return embeddings, chunks
    
    def validate_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]) -> bool:
        """임베딩 데이터 검증"""
        print("임베딩 데이터 검증 중...")
        
        # 기본 검증
        if len(embeddings) != len(chunks):
            print(f"길이 불일치: 임베딩({len(embeddings)}) vs 청크({len(chunks)})")
            return False
        
        if embeddings.shape[1] != self.embedding_dim:
            print(f"차원 불일치: {embeddings.shape[1]} vs {self.embedding_dim}")
            return False
        
        # NaN, Inf 검사
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            print("NaN 또는 Inf 값 발견")
            return False
        
        # 제로 벡터 검사
        zero_count = np.sum(np.all(embeddings == 0, axis=1))
        if zero_count > 0:
            print(f"제로 벡터 발견: {zero_count}개 ({zero_count/len(embeddings)*100:.1f}%)")
        
        # 유사도 범위 검사
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"벡터 노름 범위: {norms.min():.3f} ~ {norms.max():.3f}")
        
        print("임베딩 데이터 검증 완료")
        return True


def main():
    """메인 실행 함수"""
    print("임베딩 생성 시스템 실행")
    print("="*50)
    
    # 설정
    chunk_file = "multi_aspect_chunks.jsonl"  # 입력: 청킹된 파일
    embedding_file = "multi_aspect_embeddings.npy"  # 출력: 임베딩 파일
    chunks_output_file = "multi_aspect_chunks_processed.jsonl"  # 출력: 처리된 청크 파일
    cache_file = "embedding_cache.pkl"  # 캐시 파일
    
    # OpenAI API 키 확인
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   export OPENAI_API_KEY='your-api-key'")
        return
    
    # 임베딩 생성기 초기화
    generator = EmbeddingGenerator(api_key=api_key)
    
    # 청크 데이터 로드
    if not os.path.exists(chunk_file):
        print(f"청크 파일이 없습니다: {chunk_file}")
        print("   청킹 모듈을 먼저 실행해주세요.")
        return
    
    chunks = generator.load_chunks(chunk_file)
    
    # 임베딩 생성
    embeddings, processed_chunks = generator.generate_embeddings(
        chunks, cache_file=cache_file
    )
    
    # 데이터 검증
    if not generator.validate_embeddings(embeddings, processed_chunks):
        print("임베딩 검증 실패")
        return
    
    # 결과 저장
    generator.save_embeddings(
        embeddings, processed_chunks, 
        embedding_file, chunks_output_file
    )
    
    # 최종 통계
    print("\n임베딩 생성 완료 통계")
    print("-" * 30)
    print(f"총 임베딩 수: {len(embeddings):,}개")
    print(f"임베딩 차원: {embeddings.shape[1]}")
    print(f"메모리 사용량: {embeddings.nbytes / (1024**2):.1f}MB")
    print(f"출력 파일:")
    print(f"  - 임베딩: {embedding_file}")
    print(f"  - 청크: {chunks_output_file}")
    
    print(f"\n임베딩 생성 완료!")


if __name__ == "__main__":
    main()