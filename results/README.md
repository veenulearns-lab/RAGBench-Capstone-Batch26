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


## General Knowledge (hotpotqa) — 390 examples

| Metric | Our Score | Paper GPT-3.5 |
|--------|-----------|---------------|
| Context Relevance RMSE | 0.1662 | 0.18 |
| Context Utilization RMSE | 0.1618 | 0.11 |
| Completeness RMSE | 0.7714 | N/A |
| Adherence AUCROC | 0.6390 | 0.57 |

- Dataset: RAGBench hotpotqa test split (full)
- Same pipeline as Biomedical baseline

## Finance (finqa) — 195 examples

| Metric | Our Score |
|--------|-----------|
| Context Relevance RMSE | 0.4212 |
| Context Utilization RMSE | 0.1670 |
| Completeness RMSE | 0.8314 |
| Adherence AUCROC | 0.5929 |

- Dataset: RAGBench finqa test split (subset)
- Same pipeline as Biomedical baseline

## Legal (cuad) — 195 examples

| Metric | Our Score |
|--------|-----------|
| Context Relevance RMSE | 0.7353 |
| Context Utilization RMSE | 0.5399 |
| Completeness RMSE | 0.8130 |
| Adherence AUCROC | 0.5220 |

- Dataset: RAGBench cuad test split (subset)
- Note: documents truncated to 2000 chars to avoid Groq API 413 token limit errors
  (cuad legal contracts are significantly longer than other domains)
- This is a key CP-2 finding: Legal domain needs proper clause-level chunking,
  not truncation, to preserve full document context
