# 📘 문서 구조 분석 종합 리포트 (Final Summary)

![Project](https://img.shields.io/badge/RAG_RFP-문서구조_분석-blue)
![Status](https://img.shields.io/badge/상태-RAG_실험_준비중-orange)
![Last_Updated](https://img.shields.io/badge/최종_업데이트-2025--11--12-success)
![Author](https://img.shields.io/badge/작성자-문서구조담당-informational)

> 분석 범위: RFP PDF 100건  
> 분석 단계: PDF 품질 → 문서 구조 통계 → OCR/레이아웃 비교 → 청킹 품질 → 임베딩 유사도 검증  

---

## 1️⃣ 개요
이번 보고서는 RAG 구축 전 단계에서 **문서 품질과 구조를 정량적으로 점검**하고,  
**청킹·전처리·임베딩 전략을 표준화하기 위한 기초 데이터 분석 결과**를 정리한 것이다.  

- **대상 문서:** 100개 RFP PDF  
- **분석 목적:** 구조적 일관성 확인, OCR 불필요성 검증, 청킹 품질 확인, Retrieval 유사도 점검  
- **출력 경로:** `outputs/structure_check/`  
- **관련 노트북:** `notebooks/06_rag_simulation.ipynb`

---

## 2️⃣ 주요 결과 요약

### 📁 PDF 품질 점검 (Task 1)
- 총 문서 **100개**, 중복 없음, 열람 오류 없음, 스캔형 PDF 없음  
- 일부 파일명에 **한글 특수문자·공백 포함** → 경로 인코딩 필요  

**활용 포인트:**  
- OCR 전처리 불필요, 바로 텍스트 추출 가능  
- `unicodedata.normalize()`로 파일명 정규화 권장  

---

### 🧱 문서 구조 분석 (Task 2)
- 평균 **heading: 약 179개**, **bullet: 약 511개**, **텍스트 길이: 약 72,000자**  
- 복잡도 높은 문서: 고려대, 을지대, 사회서비스원 등 (`layout_complexity > 16`)  

**인사이트:**  
- 대부분 문서가 **계층적 구조(heading–bullet–paragraph)** 유지  
- heading 기반 청킹 경계 자동화 가능  
- 복잡도가 높은 문서는 **가변 청킹 길이(dynamic window)** 적용 필요  

**활용 포인트:**  
- heading/bullet 분포를 이용해 semantic boundary 규칙 설계  
- 평균 문단 길이를 기준으로 tokenizer 세그먼트 길이 설정  

---

### 🧩 OCR 및 레이아웃 비교 (Task 3)
- **OCR Gain Ratio 평균 < 0.05**, 최대 0.064 → 거의 모든 PDF가 디지털 텍스트 기반  
- 표(Table) 검출 상위 문서  
  - 고려대(420), GKL(367), 평택시 BIS(353)  

**인사이트:**  
- OCR 효과가 미미 → OCR 불필요  
- 표 데이터가 풍부 → 추출 과정에서 **표 구조 보존 필요**  

**활용 포인트:**  
- OCR 대신 **LayoutParser** 또는 **Camelot** 활용  
- 표 데이터를 별도 텍스트 변환 파이프라인으로 처리  

---

### 🔀 청킹 품질 분석 (Task 4)
- 평균 **chunk 수: 107개**, **chunk 길이: 약 675 tokens**, **의미 밀도: 0.52 (안정적)**  
- 의미 밀도 상위 문서:  
  - 한국대학스포츠협의회(0.66), 농어촌공사 네팔사업(0.56) 등  

**인사이트:**  
- chunk 간 길이·의미밀도 분포가 균질 → **청킹 기준 적정**  
- semantic density 상위 문서는 **양질의 학습 샘플**로 활용 가능  

**활용 포인트:**  
- Python 청킹 기준(약 675 tokens) 유지  
- 의미 밀도 기반으로 청킹 샘플 선정 (RAG 학습용 후보)

---

### 🔍 Retrieval 품질 검증 (Task 5)
- **임베딩 모델:** `sentence-transformers/distiluse-base-multilingual-cased-v2`  
- **임베딩 차원:** 512  
- **intra-doc similarity:** 0.442  
- **inter-doc similarity:** 0.377  

**인사이트:**  
- 문서 간 구분력이 약함 (0.06의 차이) → retrieval precision 낮을 가능성  
- 문서 구조가 유사한 RFP 특성상 **도메인 맞춤 임베딩 필요**  

**활용 포인트:**  
- 임베딩 교체 또는 파인튜닝 권장  
  - 예: `intfloat/multilingual-e5-base`  
- 문서 간 유사도 정규화(normalization)로 retrieval 정확도 개선  

---

## 3️⃣ 향후 활용 전략

| 단계 | 개선 포인트 | 기대 효과 |
|------|--------------|------------|
| 파일 정제 | 파일명 인코딩 및 공백 정리 | 인덱싱 경로 오류 방지 |
| 텍스트 추출 | OCR 생략, LayoutParser/Camelot 활용 | 전처리 시간 절감 |
| 구조 기반 청킹 | heading + bullet 경계 기반 | 의미 단위 유지 |
| 청킹 검증 | semantic_density=0.52 유지 | 균질한 chunk 확보 |
| 임베딩 개선 | 도메인 특화 모델로 교체 | retrieval 정확도 향상 |

---

## 4️⃣ 프로젝트 폴더 구조 예시
outputs/
structure_check/
pdf_inventory.md
document_structure.md
ocr_comparison.md
chunking_quality.md
retrieval_quality.md
summary_final.md ← 이 파일
notebooks/
01_data_structure.ipynb

---


---

## 5️⃣ 다음 단계 제안
1. **임베딩 모델 교체** 또는 RFP 데이터로 도메인 파인튜닝 수행  
2. **표(table)** 중심 문서의 별도 처리 경로 구축  
3. 현재 청킹 기준 유지, Retrieval hit-rate 실험으로 검증  

---

## 💡 결론
✅ 데이터 품질 및 구조 안정성 우수 (OCR 불필요)  
✅ 청킹 품질 균질, RAG 전처리 단계 완료 수준  
⚠️ 문서 간 유사도 높음 → Retrieval 구분력 개선 필요  

> **지금 상태에서 RAG 실험(요약·질의·검색) 바로 가능**  
> 다음 핵심 과제: **Retriever fine-tuning 또는 임베딩 모델 교체**
