# 🔍 Retrieval Quality Report

- 임베딩 모델: `sentence-transformers/distiluse-base-multilingual-cased-v2`
- 총 청크 수: 475
- 임베딩 차원: 512

## 유사도 통계

|   embedding_dim |   similarity_mean |   similarity_std |   intra_doc_sim |   inter_doc_sim |   semantic_density_corr |   avg_chunk_len_corr |
|----------------:|------------------:|-----------------:|----------------:|----------------:|------------------------:|---------------------:|
|             512 |          0.378909 |         0.208729 |        0.441697 |        0.377053 |                     nan |                  nan |

## 시각화

![Retrieval Similarity](retrieval_similarity_hist.png)
