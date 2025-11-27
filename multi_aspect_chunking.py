"""
Multi-Aspect Chunking Module
============================
RFP 문서를 다각도로 표현하는 청킹 전략 구현

하나의 텍스트를 3가지 방식으로 표현:
- 원문 (original): 원본 텍스트 그대로
- 키워드 (keywords): 핵심 키워드 추출
- 요약 (summary): 내용 요약

Author: 원후 (Bidding Mate RAG Team)
"""

import json
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class MultiAspectChunker:
    """Multi-Aspect 청킹 클래스 - 하나의 텍스트를 다각도로 표현"""
    
    def __init__(self, chunk_size: int = 600, overlap: int = 150):
        """
        Args:
            chunk_size: 기본 청크 크기
            overlap: 청크 간 겹침 크기
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # RFP 도메인 중요 키워드들
        self.rfp_keywords = {
            '기술': ['시스템', '플랫폼', '소프트웨어', 'API', '데이터베이스', '서버', '클라우드'],
            '사업': ['사업', '프로젝트', '구축', '개발', '운영', '유지보수', '납품'],
            '요구사항': ['요구사항', '기능', '성능', '보안', '표준', '규격', '품질'],
            '예산': ['예산', '비용', '금액', '계약', '사업비', '총액', '단가'],
            '일정': ['기간', '일정', '완료', '납기', '단계', '마일스톤'],
            '평가': ['평가', '심사', '선정', '기준', '배점', '가점', '점수']
        }
        
        print("✅ Multi-Aspect 청킹 시스템 초기화 완료")
    
    def split_text_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """겹침이 있는 텍스트 분할"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # 문장 경계에서 자르기 시도
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size * 0.7:
                    end = sentence_end + 1
                else:
                    # 줄바꿈에서 자르기 시도
                    line_end = text.rfind('\n', start, end)
                    if line_end > start + chunk_size * 0.7:
                        end = line_end + 1
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:
                chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - overlap
        
        return chunks
    
    def extract_keywords(self, text: str, top_k: int = 8) -> str:
        """TF-IDF 기반 키워드 추출"""
        try:
            # 간단한 키워드 추출
            words = re.findall(r'\b[가-힣]{2,}\b', text)
            
            # RFP 도메인 키워드 우선 추출
            domain_keywords = []
            for category, keywords in self.rfp_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        domain_keywords.append(keyword)
            
            # 빈도 기반 키워드 추출
            word_freq = {}
            for word in words:
                if len(word) > 1:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # 상위 키워드 선택
            frequent_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # 도메인 키워드 + 빈도 키워드 조합
            all_keywords = domain_keywords[:3]
            for word, _ in frequent_keywords:
                if word not in all_keywords and len(all_keywords) < top_k:
                    all_keywords.append(word)
            
            return ', '.join(all_keywords[:top_k]) if all_keywords else text[:100]
            
        except Exception as e:
            print(f"키워드 추출 오류: {e}")
            return text[:100]
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """텍스트 요약 (간단한 추출 요약)"""
        try:
            sentences = re.split(r'[.!?]\s+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) <= 2:
                return text[:max_length]
            
            # 중요 문장 선택 (길이, 키워드 포함 기준)
            scored_sentences = []
            for sentence in sentences:
                score = len(sentence)  # 기본 점수는 길이
                
                # RFP 도메인 키워드 포함 시 가점
                for category, keywords in self.rfp_keywords.items():
                    for keyword in keywords:
                        if keyword in sentence:
                            score += 50
                
                scored_sentences.append((sentence, score))
            
            # 상위 문장들 선택
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            summary_sentences = []
            current_length = 0
            
            for sentence, _ in scored_sentences:
                if current_length + len(sentence) <= max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            if not summary_sentences:
                return text[:max_length]
            
            return '. '.join(summary_sentences) + '.'
            
        except Exception as e:
            print(f"요약 오류: {e}")
            return text[:max_length]
    
    def create_multi_aspect_chunks(self, documents: List[Dict]) -> List[Dict]:
        """Multi-Aspect 청킹 실행"""
        print(f"Multi-Aspect 청킹 시작 - {len(documents)}개 문서")
        
        all_chunks = []
        
        for doc in tqdm(documents, desc="Multi-Aspect 청킹"):
            doc_id = doc['filename'].replace('pdf_files/', '').replace('.pdf', '')
            text = doc['text'].strip()
            
            if len(text) < 100:
                continue
            
            # 기본 청크 분할
            base_chunks = self.split_text_with_overlap(text, self.chunk_size, self.overlap)
            
            # 각 청크를 3가지 방식으로 표현
            for chunk_idx, chunk_text in enumerate(base_chunks):
                # 1. 원문 (Original)
                original_chunk = {
                    "chunk_id": f"{doc_id}_multi_original_{chunk_idx}",
                    "doc_id": doc_id,
                    "aspect": "original",
                    "text": f"원문: {chunk_text}",
                    "metadata": {
                        "filename": doc['filename'],
                        "chunk_type": "multi_aspect",
                        "aspect_type": "original",
                        "chunk_index": chunk_idx,
                        "original_length": len(chunk_text)
                    }
                }
                all_chunks.append(original_chunk)
                
                # 2. 키워드 (Keywords)
                keywords = self.extract_keywords(chunk_text)
                keyword_chunk = {
                    "chunk_id": f"{doc_id}_multi_keywords_{chunk_idx}",
                    "doc_id": doc_id,
                    "aspect": "keywords", 
                    "text": f"키워드: {keywords}",
                    "metadata": {
                        "filename": doc['filename'],
                        "chunk_type": "multi_aspect",
                        "aspect_type": "keywords",
                        "chunk_index": chunk_idx,
                        "original_text": chunk_text
                    }
                }
                all_chunks.append(keyword_chunk)
                
                # 3. 요약 (Summary)
                summary = self.summarize_text(chunk_text)
                summary_chunk = {
                    "chunk_id": f"{doc_id}_multi_summary_{chunk_idx}",
                    "doc_id": doc_id,
                    "aspect": "summary",
                    "text": f"요약: {summary}",
                    "metadata": {
                        "filename": doc['filename'],
                        "chunk_type": "multi_aspect", 
                        "aspect_type": "summary",
                        "chunk_index": chunk_idx,
                        "original_text": chunk_text
                    }
                }
                all_chunks.append(summary_chunk)
        
        print(f"Multi-Aspect 청킹 완료: {len(all_chunks)}개 청크 생성")
        print(f"청크 구성: 원문({len(all_chunks)//3}개), 키워드({len(all_chunks)//3}개), 요약({len(all_chunks)//3}개)")
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict], output_file: str) -> None:
        """청크 결과 저장"""
        print(f"청크 저장: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        print(f"{len(chunks)}개 청크 저장 완료")
    
    def load_extracted_texts(self, jsonl_file: str) -> List[Dict]:
        """추출된 텍스트 로드"""
        print(f"텍스트 파일 로드: {jsonl_file}")
        
        documents = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                documents.append(data)
        
        print(f"{len(documents)}개 문서 로드 완료")
        return documents


def main():
    """메인 실행 함수"""
    print("🚀 Multi-Aspect 청킹 시스템 실행")
    print("="*50)
    
    # 청킹 시스템 초기화
    chunker = MultiAspectChunker(chunk_size=600, overlap=150)
    
    # 텍스트 데이터 로드
    input_file = "all_extracted_texts.jsonl"  # 추출된 텍스트 파일 경로
    
    if not os.path.exists(input_file):
        print(f"❌ 입력 파일이 없습니다: {input_file}")
        print("   텍스트 추출을 먼저 실행해주세요.")
        return
    
    documents = chunker.load_extracted_texts(input_file)
    
    # Multi-Aspect 청킹 실행
    chunks = chunker.create_multi_aspect_chunks(documents)
    
    # 결과 저장
    output_file = "multi_aspect_chunks.jsonl"
    chunker.save_chunks(chunks, output_file)
    
    # 통계 출력
    print("\nMulti-Aspect 청킹 결과 통계")
    print("-" * 30)
    print(f"총 청크 수: {len(chunks):,}개")
    
    # 측면별 통계
    aspect_counts = {}
    for chunk in chunks:
        aspect = chunk['aspect']
        aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
    
    for aspect, count in aspect_counts.items():
        print(f"{aspect}: {count:,}개")
    
    print(f"\nMulti-Aspect 청킹 완료!")
    print(f"   출력 파일: {output_file}")


if __name__ == "__main__":
    main()