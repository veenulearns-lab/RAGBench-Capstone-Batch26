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

## Finance (finqa) — Final Results (corrected TRACe metrics)

**Note:** an earlier version of this section used a TRACe metrics function with a bug —
judge-hallucinated sentence keys weren't filtered against the real document key set,
which could push ratios (and RMSE) past their valid 0-1 range. Fixed by clipping judge
output to the real key set before computing ratios. All numbers below use the fix.

Full per-experiment scores (mean values, not RMSE) are in
`results/finance_all_experiments_summary_table.csv`. Summary:

| Run | LLM | Embedding | Retrieval | Chunking | Rel RMSE | Util RMSE | Comp RMSE | Adh AUCROC | Samples |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (200) | llama-3.1-8b | MiniLM | Dense | Whole doc | 0.3526 | 0.1387 | 0.6361 | 0.5212 | 197 |
| Best Combo (200) | llama-3.3-70b | MiniLM | Dense | Whole doc | 0.3769 | 0.1406 | 0.6224 | **0.5802** | 197 |

**Winning change: generator LLM (llama-3.1-8b → llama-3.3-70b-versatile).** At full scale
(200 samples), this gives a modest, real improvement in Adherence (0.52→0.58) and
Completeness, at a small cost to Relevance, with Utilization essentially unchanged.

**Important calibration note:** the 25-sample Phase 2/3 runs showed a much larger apparent
Adherence gain (0.5379→0.75) than the 200-sample validation confirmed (0.5212→0.5802).
This is expected — Adherence AUCROC is unstable on small samples with few negative-class
examples (see Exp-004's anomalous 0.19 AUCROC, also on a 3-negative-example split). The
200-sample result is the one to trust; the 25-sample result correctly identified the
*direction* of improvement but overstated its *size*.

**8 experiments tested per capstone requirements:** k (3 vs 5), prompt style (full vs
minimal), chunking (whole-doc vs metadata-aware), embedding (MiniLM vs BGE-large vs
FinBERT — domain-specific per capstone doc), retrieval (dense vs hybrid BM25+RRF), and
generator LLM (8B vs 70B vs qwen3-32b). Only the generator-LLM swap showed a reliable,
reproducible improvement at scale; all other single-factor changes were within noise.

**Observation for error analysis:** across all experiments, model-predicted Relevance/
Utilization ran well above the finqa ground-truth reference values (~0.08-0.09), while
predicted Completeness ran well below the reference (~0.83-0.89) — indicating the
pipeline's retrieval is more liberal about marking sentences relevant/utilized than
the dataset's human annotators, and consequently captures less of what they considered
essential context. Worth deeper investigation in the manual error analysis deliverable.

### Update: Domain-recommended combo (EXP-013) — new best result

Testing the domain-matrix-recommended stack together (metadata-aware chunking + FinBERT
+ hybrid retrieval + llama-3.3-70b generator), at 200 samples:

| Run | Rel RMSE | Util RMSE | Comp RMSE | Adh AUCROC |
|---|---|---|---|---|
| Baseline (8B) | 0.3526 | 0.1387 | 0.6361 | 0.5212 |
| Generator-only combo (70B) | 0.3769 | 0.1406 | 0.6224 | 0.5802 |
| **Domain-recommended combo (70B+FinBERT+Hybrid+Metadata)** | 0.3561 | 0.1583 | **0.5964** | **0.5905** |

This is the best Completeness and Adherence of any Finance run, despite FinBERT, hybrid
retrieval, and metadata-aware chunking each underperforming individually in isolated
testing (see EXP-004, 006, 007) — the components combine better than they perform alone,
at a small cost to Relevance/Utilization. **This is now the recommended Finance pipeline.**

### Update: whole_doc_finbert_dense_sides — VALIDATED at 200 samples

| Config | Rel RMSE↓ | Util RMSE↓ | Comp RMSE↓ | Adh AUCROC↑ | Samples |
|---|---|---|---|---|---|
| whole_doc + FinBERT + dense + sides order | 0.3534 | **0.1067** | 0.6035 | **0.6196** | 200 |
| EXP-013 (previous best) | 0.3561 | 0.1583 | 0.5964 | 0.5905 | 197 |
| Baseline | 0.3526 | 0.1387 | 0.6361 | 0.5212 | 197 |

This candidate beats EXP-013 on Relevance, Utilization, and Adherence at full scale
(only Completeness is marginally worse). As expected from the documented 25-vs-200
sample instability, its 25-sample Adherence (0.7045) shrank at 200 samples (0.6196) —
still real, still the best Adherence of any 200-sample config tested so far.

Second candidate (`largesem_bge_dense_fwd`) still pending 200-sample validation —
this section will be updated once that result is in. **Do not treat either candidate
as final until both are validated and compared.**



### New candidate combos (25 samples, PRELIMINARY — not yet validated at 200)

Testing gaps in the chunking/embedding/retrieval/rerank/context-order grid not
covered by EXP-001–013. Two candidates beat EXP-013's 25-sample-equivalent
Adherence and are queued for 200-sample validation:

| Combo | Rel RMSE | Util RMSE | Comp RMSE | Adh AUCROC |
|---|---|---|---|---|
| whole_doc + FinBERT + dense + sides order | 0.4486 | 0.1780 | 0.6390 | **0.7045** |
| large_semantic + BGE-large + dense + forward | 0.3736 | **0.0717** | 0.6216 | 0.6429 |

Reranking (cross-encoder) tested on top of EXP-013's exact setup did NOT improve
Adherence (0.4286 vs. EXP-013's 0.5905) — not adopted.

**Caution:** given the documented 25-vs-200-sample Adherence instability (see EXP-010
vs EXP-012), neither candidate above should be treated as a new best pipeline until
validated at 200 samples. Full new-combo table: `results/finance_new_combos_25sample_preliminary.csv`.


Full per-run scores: `results/finance_all_experiments_summary_table.csv` (13 experiments).



- Dataset: RAGBench finqa test split
- Uses the verbatim Friel et al. (2024) Appendix 7.4 judge prompt
- Judge fixed at llama-3.1-8b-instant throughout; only the generator model varies across LLM experiments

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

## Customer Support (techqa) — 192 examples

| Metric | Our Score |
|--------|-----------|
| Context Relevance RMSE | 0.3938 |
| Context Utilization RMSE | 0.5420 |
| Completeness RMSE | 0.6523 |
| Adherence AUCROC | 0.5254 |

- Dataset: RAGBench techqa test split (subset)
- Documents truncated to 2000 chars (some technical manuals exceed token limit)

---

# CP-1 FINAL SUMMARY — All 5 Domains Complete

| Domain | Examples | Relevance RMSE↓ | Utilization RMSE↓ | Completeness RMSE↓ | Adherence AUCROC↑ |
|--------|----------|------------------|---------------------|----------------------|----------------------|
| Biomedical (covidqa) | 246 | 0.2410 | 0.1627 | 0.4500 | 0.6446 |
| General Knowledge (hotpotqa) | 390 | 0.1662 | 0.1618 | 0.7714 | 0.6390 |
| Finance (finqa) | 195 | 0.4212 | 0.1670 | 0.8314 | 0.5929 |
| Legal (cuad) | 195 | 0.7353 | 0.5399 | 0.8130 | 0.5220 |
| Customer Support (techqa) | 192 | 0.3938 | 0.5420 | 0.6523 | 0.5254 |

**Total examples evaluated: 1,218 across 5 domains**

Paper reference (GPT-3.5 judge): Adherence AUCROC 0.57. We beat this on 2/5 domains
(Biomedical, General Knowledge) using Llama 3.1 8B as Judge LLM.

Key finding: Legal and Customer Support show higher RMSE due to document truncation
(2000 char limit) needed to avoid Groq API 413 errors. This is a primary target for
CP-2 — proper domain-specific chunking instead of truncation should significantly
improve these scores.
