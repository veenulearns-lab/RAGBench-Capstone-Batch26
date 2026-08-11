# RAGBench Capstone — Batch 26

**Real-World RAG: Building, Testing and Tuning Domain-Specific Retrieval-Augmented Generation Pipelines**

AIML PG-Level Capstone, IIIT Hyderabad. Veenu Dandekar · Sridhar Vedula · Prerna Saratkar · Rinku Rajput
Supervisor: Dr. Manish Shrivastava · Mentors: Gopichand, Lokesh

---

## What this project asks

1. **Attribution** — which pipeline stage actually carries the variation in answer quality?
2. **Generality** — is the best configuration the same in every domain?
3. **Validity** — are our own measurements trustworthy?

The third turned out to be the hardest and produced the most transferable findings.

## Locked configurations

All five domains at N=200, generator `llama-3.3-70b-versatile` (temperature 0),
judge `openai/gpt-oss-120b`. Chunking `sliding_5o2` = sliding window, 5 sentences, 2 overlap.

| Domain | Dataset | Exp ID | Chunking | Embedder | Retrieval | Rerank | Rel RMSE ↓ | Util RMSE ↓ | Comp RMSE ↓ | Adherence ↑ | AUROC ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Customer Support | TechQA | `CS-064` | sliding_5o2 | all-MiniLM-L6-v2 | hybrid | cross_encoder | 0.2732 | 0.1762 | 0.6179 | 0.6996 | 0.5282 |
| Finance | FinQA | `FIN-031` | fixed_150w | FinBERT | hybrid | none | 0.3275 | 0.2815 | 0.4496 | 0.8250 | 0.4480 |
| General Knowledge | HotpotQA | `GK-031` | sliding_5o2 | e5-large-v2 | hybrid | none | 0.1085 | 0.0832 | 0.3782 | 0.8773 | 0.6221 |
| Legal | CUAD | `LEG-025` | sliding_5o2 | legal-bert-base-uncased | dense | none | 0.2651 | 0.1346 | 0.6911 | 0.8825 | 0.6524 |
| Biomedical | CovidQA | `BIO-004` | whole_doc | S-PubMedBert-MS-MARCO | dense | cross_encoder | 0.2398 | 0.1932 | 0.4605 | 0.8010 | 0.6277 |

All five use forward context ordering. Three of five use a sliding window; three of five use
hybrid retrieval. The embedding model is what really varies, and it varies in the direction
each domain suggests.

## Two campaigns

**Greedy (stage-wise ladder)** — 93 experiments. Vary one component per stage, keep the winner.
Visits the sum of levels rather than the product, so every locked choice traces to one named
experiment. Run over roughly two months.

**Factorial (full grid)** — 96 cells per domain: 2 embedders × 4 chunking strategies ×
2 retrieval methods × 2 reranking settings × 3 context orderings. Reports main effects with
other factors marginalised out.

**The locked configurations above come from the factorial campaign.** The greedy ladder varied
the generator and judge as part of its search, and in places used a judge the same size as or
smaller than the generator — which is not an independent measurement. The factorial grid held
the generator fixed at llama-3.3-70b and judged everything with the larger gpt-oss-120b. Only
one of the two campaigns measured with a valid instrument. Greedy results are retained in full
because they document the ablation method and because the agreements and disagreements between
the two searches are themselves a finding.

Report-grade cells after the ≥0.80 judge-coverage quarantine: General Knowledge 96/96,
Finance 95/96, Customer Support 77/96, Legal 68/96.

## Protocol

- SEED=42 · k=5 final chunks · N=25 for screening, N=200 for every reported figure
- Judge must be **stronger than the generator**. On an identical pipeline an 8b judge scored
  0.639 where a 70b judge scored 0.875 — the pipeline had not changed, the instrument had.
- Judge prompt pasted verbatim from RAGBench Appendix 7.4; documents sent as raw nested
  `[[key, sentence]]` lists; documents renumbered in retrieval order
- Judge labels that do not match a real retrieved sentence are discarded before scoring
- Judge coverage below 0.80 excluded from reporting; excluded rows published in
  `report_<domain>_EXCLUDED.csv` rather than dropped silently
- FAISS flat index throughout (exact search, no approximation error). BM25 is added alongside
  FAISS for hybrid retrieval via RRF, never in place of it
- OpenRouter primary transport with Groq key rotation as fallback
- Triple-save checkpointing every 5 examples: local, Drive, GitHub

## TRACe metrics

| Metric | Question | Formula | Scored by |
|---|---|---|---|
| Context Relevance | Was the retrieved material about the question? | \|relevant\| ÷ \|all context sentences\| | RMSE ↓ |
| Context Utilization | Did the answer use that material? | \|utilized\| ÷ \|all context sentences\| | RMSE ↓ |
| Completeness | Did it use all of the relevant material? | \|relevant ∩ utilized\| ÷ \|relevant\| | RMSE ↓ |
| Adherence | Is every claim supported by the documents? | 1 if all answer sentences supported, else 0 | AUROC ↑ |

AUROC sits near chance in some domains. The reference labels describe whether the benchmark's
own answer was supported by the context its annotators saw; our pipeline retrieves different
context and writes a different answer, so the label stops describing the thing being measured.
Absolute adherence is reported next to AUROC everywhere rather than choosing one.

## Layout

```
RAGBench/
  src/                    five domain pipeline scripts + finance backfill utility
  notebooks/              executed notebooks
  results/
    factorial/<domain>/   96-cell grid: master CSV, report-grade rows, excluded rows, registry
    greedy/<domain>/      stage-wise ladder experiments
    judge_swap/           5 domains x 2 judges diagnostic, n=100
RGB/
  results/v2/             five models x four abilities
  notebooks/
archive/                  CP1 baselines, CP2 ablation tracks, RGB v1 (superseded, retained)
```

## RGB — behaviour when retrieval fails

TRACe measures a good answer; RGB measures what the model does when the evidence is wrong,
missing or scattered. Five models across noise robustness, negative rejection, information
integration and counterfactual robustness.

The v1 campaign was discarded: the counterfactual test read the dataset's clean-document field
instead of the corrupted one, so the test designed to plant factual errors contained none.
v2 fixed that plus two further defects and re-ran everything. Copying the RGB prompt verbatim
rather than paraphrasing it was worth 44.7 points; text normalisation (`cannot` ≡ `can not`)
was hiding a further 29.7.

## References

- RAGBench — Friel et al. 2024, https://arxiv.org/abs/2407.11005
- RGB — Chen et al. 2023, https://arxiv.org/abs/2309.01431
- Dataset — https://huggingface.co/datasets/rungalileo/ragbench
