# Results — Simple RAG Baseline

## CovidQA Domain (246 examples)

| Metric | Our Score | Paper GPT-3.5 |
|--------|-----------|---------------|
| Context Relevance RMSE | 0.2410 | 0.18 |
| Context Utilization RMSE | 0.1627 | 0.11 |
| Completeness RMSE | 0.4500 | N/A |
| Adherence AUCROC | 0.6446 | 0.57 |

**Setup:**
- Dataset: RAGBench CovidQA test split
- Embedding: all-MiniLM-L6-v2
- Retriever: FAISS top-3
- Generator: llama-3.1-8b-instant
- Judge LLM: llama-3.1-8b-instant (RAGBench paper prompt)
- Date: June 2026
