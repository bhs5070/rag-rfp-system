"""
Clean Eval Style Dataset Creation Module
=======================================
우리 RFP 문서 기반으로 Clean Eval 스타일 평가 데이터셋을 생성하는 모듈

Features:
- Clean Eval 데이터셋과 동일한 구조 (id, question, answer, gt_doc_id)
- OpenAI GPT-4o-mini를 사용한 자연스러운 질문 생성
- 고품질 청크 선별 및 문서별 균등 분배
- RFP 도메인 특화 질문 패턴

Author: 원후 (Bidding Mate RAG Team)
"""

import json
import os
import random
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from openai import OpenAI


class CleanEvalDatasetGenerator:
    """Clean Eval 스타일 평가 데이터셋 생성 클래스"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: OpenAI API 키 (환경변수에서 자동 로드)
        """
        self.client = OpenAI(api_key=api_key)
        self.chunk_files = [
            "efficient_chunks_structure_aware.jsonl",
            # 필요하면 다른 청킹 파일들도 추가 가능
        ]
        
        # 고품질 청크 선별 기준
        self.min_text_length = 300
        self.required_keywords = ['사업', '시스템', '예산', '기간', '요구사항', '구축', '개발']
        self.skip_keywords = ['[표]', '[그림]', '빈 페이지', '[이미지]']
        self.chunks_per_doc = 3  # 문서당 선택할 청크 수
        
        print("✅ Clean Eval 데이터셋 생성기 초기화 완료")
        print(f"   🤖 OpenAI 클라이언트 설정됨")
        print(f"   📦 청크 소스: {len(self.chunk_files)}개 파일")
    
    def load_chunks(self) -> List[Dict]:
        """청킹된 데이터 로드"""
        print("📂 청킹 데이터 로드 중...")
        
        chunks = []
        for chunk_file in self.chunk_files:
            if os.path.exists(chunk_file):
                print(f"   📄 로딩: {chunk_file}")
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        try:
                            chunk = json.loads(line)
                            chunks.append(chunk)
                        except json.JSONDecodeError as e:
                            print(f"   ⚠️ JSON 오류 (라인 {line_no}): {e}")
                            continue
            else:
                print(f"   ❌ 파일 없음: {chunk_file}")
        
        print(f"✅ 총 {len(chunks):,}개 청크 로드 완료")
        return chunks
    
    def filter_high_quality_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """고품질 청크 선별"""
        print("🔍 고품질 청크 선별 중...")
        
        selected_chunks = []
        
        for chunk in chunks:
            text = chunk.get('text', '').strip()
            
            # 길이 조건
            if len(text) < self.min_text_length:
                continue
            
            # 키워드 조건 (하나 이상 포함)
            if not any(keyword in text for keyword in self.required_keywords):
                continue
            
            # 제외 키워드 체크
            if any(skip_word in text for skip_word in self.skip_keywords):
                continue
            
            # 의미있는 내용인지 추가 검증
            if self._is_meaningful_content(text):
                selected_chunks.append(chunk)
        
        print(f"✅ {len(selected_chunks):,}개 고품질 청크 선별 완료")
        return selected_chunks
    
    def _is_meaningful_content(self, text: str) -> bool:
        """의미있는 내용인지 판단"""
        # 너무 반복적인 내용 제외
        words = text.split()
        if len(set(words)) / len(words) < 0.3:  # 고유 단어 비율이 30% 미만
            return False
        
        # 숫자나 특수문자만 있는 경우 제외
        if len([c for c in text if c.isalpha()]) < len(text) * 0.5:
            return False
        
        return True
    
    def balance_chunks_by_document(self, chunks: List[Dict]) -> List[Dict]:
        """문서별로 균등하게 청크 선택"""
        print("⚖️ 문서별 균등 분배 중...")
        
        # 문서별 그룹화
        doc_chunks = {}
        for chunk in chunks:
            doc_id = chunk.get('doc_id', 'unknown')
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(chunk)
        
        print(f"   📊 총 {len(doc_chunks)}개 문서")
        
        # 각 문서에서 균등하게 선택
        final_chunks = []
        for doc_id, doc_chunk_list in doc_chunks.items():
            # 품질 순으로 정렬 (길이 기준)
            doc_chunk_list.sort(key=lambda x: len(x.get('text', '')), reverse=True)
            
            # 상위 N개 선택
            selected_count = min(self.chunks_per_doc, len(doc_chunk_list))
            selected_from_doc = doc_chunk_list[:selected_count]
            
            final_chunks.extend(selected_from_doc)
            print(f"   📄 {doc_id[:30]}...: {selected_count}개 선택")
        
        print(f"✅ 총 {len(final_chunks)}개 청크 최종 선택")
        return final_chunks
    
    def generate_question_and_answer(self, chunk_text: str, doc_id: str) -> Tuple[Optional[str], Optional[str]]:
        """청크 기반으로 Clean Eval 스타일 질문과 답변 생성"""
        
        question_prompt = f"""다음 RFP 문서 내용을 보고, Clean Eval 데이터셋 스타일의 질문을 생성해주세요.

문서: {doc_id}
내용: {chunk_text[:600]}

Clean Eval 스타일 질문 특징:
- 구체적이고 실무적인 정보를 묻는 질문
- "무엇인가?", "어떻게 되나요?", "얼마인가?" 등 자연스러운 표현
- 복합 질문도 가능 (예: 사업명과 예산을 함께 묻는 등)
- 실제 업무에서 필요한 핵심 정보 중심

예시 질문 패턴:
- "이 사업의 공식 명칭은 무엇인가?"
- "본 프로젝트의 사업기간과 예산 규모는 어떻게 되나요?"
- "시스템 구축에 필요한 주요 기술 요구사항은 무엇인가?"

문서 내용에 맞는 자연스러운 질문 1개만 생성:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question_prompt}],
                max_tokens=120,
                temperature=0.7
            )
            
            question = response.choices[0].message.content.strip()
            
            # 질문 형태 검증
            if not self._validate_question(question):
                print(f"⚠️ 질문 형태 부적절: {question[:50]}...")
                return None, None
            
            # 답변은 원본 청크 텍스트 사용 (Clean Eval 스타일)
            answer = chunk_text.strip()
            
            return question, answer
            
        except Exception as e:
            print(f"❌ LLM 생성 오류: {e}")
            return None, None
    
    def _validate_question(self, question: str) -> bool:
        """생성된 질문이 적절한지 검증"""
        if not question:
            return False
        
        # 질문 형태 체크
        if not ("?" in question or "인가" in question or question.endswith("요?")):
            return False
        
        # 최소 길이 체크
        if len(question.strip()) < 10:
            return False
        
        # 적절한 키워드 포함 체크
        question_keywords = ["무엇", "어떻게", "얼마", "언제", "어디서", "누가", "왜"]
        if not any(keyword in question for keyword in question_keywords):
            return False
        
        return True
    
    def create_evaluation_dataset(self, output_file: str = "our_clean_eval_style.jsonl") -> List[Dict]:
        """Clean Eval 스타일 평가 데이터셋 생성"""
        print("🚀 Clean Eval 스타일 평가 데이터셋 생성 시작")
        print("=" * 60)
        
        # 1. 청킹 데이터 로드
        chunks = self.load_chunks()
        if not chunks:
            print("❌ 청킹 데이터가 없습니다.")
            return []
        
        # 2. 고품질 청크 선별
        high_quality_chunks = self.filter_high_quality_chunks(chunks)
        if not high_quality_chunks:
            print("❌ 적합한 청크가 없습니다.")
            return []
        
        # 3. 문서별 균등 분배
        final_chunks = self.balance_chunks_by_document(high_quality_chunks)
        
        # 4. 질문-답변 쌍 생성
        print("🧠 질문-답변 쌍 생성 중...")
        eval_dataset = []
        failed_count = 0
        
        for idx, chunk in enumerate(tqdm(final_chunks, desc="평가 데이터 생성"), 1):
            question, answer = self.generate_question_and_answer(
                chunk.get('text', ''), 
                chunk.get('doc_id', 'unknown')
            )
            
            if question and answer:
                eval_item = {
                    "id": f"eval_{idx:03d}",
                    "question": question,
                    "answer": answer,
                    "gt_doc_id": chunk.get('doc_id', 'unknown')
                }
                eval_dataset.append(eval_item)
            else:
                failed_count += 1
        
        print(f"✅ 질문-답변 생성 완료: {len(eval_dataset)}개 성공, {failed_count}개 실패")
        
        # 5. 데이터셋 저장
        self.save_dataset(eval_dataset, output_file)
        
        # 6. 통계 출력
        self.print_dataset_statistics(eval_dataset)
        
        return eval_dataset
    
    def save_dataset(self, eval_dataset: List[Dict], output_file: str) -> None:
        """데이터셋 저장"""
        print(f"💾 데이터셋 저장: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in eval_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ {len(eval_dataset)}개 평가 데이터 저장 완료")
    
    def print_dataset_statistics(self, eval_dataset: List[Dict]) -> None:
        """데이터셋 통계 출력"""
        if not eval_dataset:
            return
        
        print("\n📊 데이터셋 통계")
        print("-" * 40)
        print(f"총 QA 쌍: {len(eval_dataset)}개")
        
        # 문서별 분포
        doc_distribution = {}
        for item in eval_dataset:
            doc_id = item['gt_doc_id']
            doc_distribution[doc_id] = doc_distribution.get(doc_id, 0) + 1
        
        print(f"고유 문서: {len(doc_distribution)}개")
        print(f"문서당 평균 질문: {len(eval_dataset)/len(doc_distribution):.1f}개")
        
        print(f"\n📄 문서별 질문 분포:")
        for doc_id, count in sorted(doc_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"   {doc_id[:35]}...: {count}개")
        
        # 질문 길이 통계
        question_lengths = [len(item['question']) for item in eval_dataset]
        print(f"\n📝 질문 길이 통계:")
        print(f"   평균: {sum(question_lengths)/len(question_lengths):.1f}자")
        print(f"   최대: {max(question_lengths)}자")
        print(f"   최소: {min(question_lengths)}자")
        
        # 답변 길이 통계
        answer_lengths = [len(item['answer']) for item in eval_dataset]
        print(f"\n💬 답변 길이 통계:")
        print(f"   평균: {sum(answer_lengths)/len(answer_lengths):.0f}자")
        print(f"   최대: {max(answer_lengths):,}자")
        print(f"   최소: {min(answer_lengths)}자")
        
        # 샘플 질문 출력
        print(f"\n🔍 샘플 질문 (처음 3개):")
        for i, item in enumerate(eval_dataset[:3], 1):
            print(f"   {i}. {item['question']}")


def main():
    """메인 실행 함수"""
    print("🚀 Clean Eval 스타일 데이터셋 생성 시스템 실행")
    print("=" * 60)
    
    # OpenAI API 키 확인
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   export OPENAI_API_KEY='your-api-key'")
        return
    
    # 데이터셋 생성기 초기화
    generator = CleanEvalDatasetGenerator(api_key=api_key)
    
    # 평가 데이터셋 생성
    dataset = generator.create_evaluation_dataset()
    
    if dataset:
        print(f"\n🎉 Clean Eval 스타일 데이터셋 생성 완료!")
        print(f"   📄 파일: our_clean_eval_style.jsonl")
        print(f"   📊 총 평가 쌍: {len(dataset)}개")
        print(f"\n✅ 이제 4가지 청킹 전략 성능 평가를 진행할 수 있습니다!")
    else:
        print("❌ 데이터셋 생성에 실패했습니다.")


if __name__ == "__main__":
    main()
