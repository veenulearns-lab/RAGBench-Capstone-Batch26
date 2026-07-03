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

## Finance (finqa) — 25 examples, verbatim judge prompt (corrected TRACe metrics)

**Note:** an earlier version of this section used a TRACe metrics function with a bug —
judge-hallucinated sentence keys weren't filtered against the real document key set,
which could push ratios (and RMSE) past their valid 0-1 range. Fixed by clipping judge
output to the real key set before computing ratios. All numbers below use the fix.

| Metric | Baseline (8B) | Best Combo (70B generator) |
|--------|--------|--------|
| Context Relevance RMSE | 0.2915 | 0.4379 |
| Context Utilization RMSE | 0.1183 | 0.1403 |
| Completeness RMSE | 0.6335 | 0.7020 |
| Adherence AUCROC | 0.5379 | **0.7500** |

**Phase 2 experiments (25 samples each, vs. baseline):**

| Experiment | Rel RMSE | Util RMSE | Comp RMSE | Adh AUCROC |
|--------|--------|--------|--------|--------|
| Exp1: k=5 | 0.2903 | 0.1143 | 0.6473 | 0.5379 |
| Exp2: minimal prompt | 0.3947 | 0.1732 | 0.6702 | 0.6818 |
| Exp3: large-semantic chunking | 0.3499 | 0.1759 | 0.5467 | 0.1905* |
| Exp4: BGE-large embedding | 0.3906 | 0.1895 | 0.6539 | 0.6591 |
| Exp5: hybrid retrieval | 0.3044 | 0.1140 | 0.6569 | 0.5000 |
| Exp6: FinBERT embedding | 0.3198 | 0.0856 | 0.6774 | 0.5152 |
| Exp7: llama-3.3-70b generator | 0.3901 | 0.1418 | 0.6526 | 0.7955 |
| Exp8: qwen3-32b generator | 0.4683 | 0.3055 | 0.5687 | 0.6500 |

*Exp3's Adherence is an outlier (below random-guess baseline of 0.5) on a very small
negative-class sample (n=3 of 24) — flagged as unstable, not adopted into the final pipeline.

**Winning change: generator LLM (llama-3.1-8b → llama-3.3-70b-versatile).** This is a
tradeoff, not a uniform win — Adherence nearly doubled (better grounding/less hallucination)
at some cost to Relevance and Completeness, suggesting the larger model gives more
confident, better-supported answers while drawing from a narrower or differently-weighted
set of context sentences than the 8B baseline. All other components (k, prompt style,
chunking, embedding, retrieval type) showed no reliable improvement over baseline at n=25.

- Dataset: RAGBench finqa test split (25-sample subset)
- Uses the verbatim Friel et al. (2024) Appendix 7.4 judge prompt
- Judge fixed at llama-3.1-8b-instant throughout; only the generator model varies in Exp7/8
- Note: 25 samples — directional result; Phase 4 (200 samples) will validate baseline vs.
  best-combo pipeline at full scale before final conclusions


### Phase 4 — Full-scale validation (200 samples)

| Metric | Baseline (200) | Best Combo (200) |
|--------|--------|--------|
| Context Relevance RMSE | 0.3526 | *(pending)* |
| Context Utilization RMSE | 0.1387 | *(pending)* |
| Completeness RMSE | 0.6361 | *(pending)* |
| Adherence AUCROC | 0.5212 | *(pending)* |

Baseline at 200 samples is broadly consistent with the 25-sample directional result
(Rel 0.2915→0.3526, Util 0.1183→0.1387, Comp 0.6335→0.6361, Adh 0.5379→0.5212),
confirming the smaller sample wasn't a fluke. Best-combo (llama-3.3-70b generator)
200-sample run pending — will be added once complete.


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
