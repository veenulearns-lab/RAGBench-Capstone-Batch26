# RAGBench — answer quality across five professional domains

~100,000 examples across five domains. Each example ships its own documents plus reference
TRACe scores, so our scoring can be validated against the benchmark's own labels rather than
merely reported.

## Two campaigns, and why both

**Greedy — stage-wise ladder, 93 experiments.** Vary one component per stage, keep the winner,
move on. Visits the sum of the levels (13) rather than the product (96), so it is cheap and
every choice traces to one named experiment. Run over roughly two months.

**Factorial — full grid, 96 cells per domain.** 2 embedders x 4 chunking strategies x
2 retrieval methods x 2 reranking settings x 3 context orderings. Reports main effects with
the other factors marginalised out.

**The locked configurations come from the factorial campaign.** The ladder varied the generator
and judge as part of its search and in places used a judge no larger than the generator, which
is not an independent measurement. The grid held the generator fixed at llama-3.3-70b and
judged everything with the larger gpt-oss-120b. Greedy results are retained in full because
they document the ablation method, and because where the two searches agree and disagree is
itself a finding — they agree on every negative result about domain encoders and diverge only
where components interact.

Report-grade cells after the >=0.80 judge-coverage quarantine:
General Knowledge 96/96 · Finance 95/96 · Customer Support 77/96 · Legal 68/96.

## Which component actually mattered

Marginal spread — the gap between the best and worst average for each component, measured at
N=25 on runs with at least 80% judge coverage.

| Domain | Chunking | Context order | Embedding | Retrieval | Reranking |
|---|---|---|---|---|---|
| Legal | **0.2686** | 0.0120 | 0.0739 | 0.0175 | 0.0175 |
| Customer Support | **0.1662** | 0.0862 | 0.0492 | 0.0488 | 0.0285 |
| Biomedical | **0.1269** | 0.0284 | 0.0276 | 0.0011 | 0.0027 |
| General Knowledge | **0.0903** | 0.0016 | 0.0056 | 0.0078 | 0.0221 |
| Finance | 0.0658 | **0.1131** | 0.0312 | 0.0012 | 0.0075 |

Chunking leads in four of five domains and in Legal dwarfs everything else combined. Context
ordering leads only Finance, where a number and its qualifying sentence must stay adjacent.
Retrieval and reranking move least almost everywhere — 0.001 and 0.003 in Biomedical. That is
the opposite of what we expected, and it is the clearest single lesson of the project.

## What each domain taught us

**Customer Support (TechQA)** — hybrid retrieval plus reranking won here, a combination we had
tested and rejected earlier under the old judge. Fixing the judge reversed the decision.

**Finance (FinQA)** — the only domain where a specialist embedding model earned its place.
Context ordering was the biggest lever, not chunking. A fact is a number plus its qualifying
sentence; split them and it is destroyed.

**General Knowledge (HotpotQA)** — our strongest continuous scores anywhere, relevance error
0.109. AUROC is near-blind here: the gold adherence positive rate is 1.000, so there is no
negative class to separate.

**Legal (CUAD)** — our strongest adherence at 0.8825 against a reference of 0.905, and the best
AUROC of any domain. Chunking dominated everything (spread 0.269) — yet the clause-aware
chunker we built specifically for contracts did **not** win. A plain sliding sentence window did.

**Biomedical (CovidQA)** — judge coverage 1.00 on every configuration, the cleanest measurement
conditions of any domain. BIO-004 screened *below chance* at N=25 and won at N=200, while the
screening leader fell from 0.773 to 0.590.

## Screen at 25, report only at 200

At 25 examples the group of unsupported answers is tiny, so a few lucky calls move the score a
long way — and the error runs in both directions. Customer Support CS-064 went from AUROC 0.767
at N=25 to 0.528 at N=200. Biomedical BIO-004 went from 0.409 to 0.628. Both are large enough
to reverse a ranking decision, which is why no 25-example score is ever reported.

## Choosing when the metrics disagreed

Stated in advance rather than chosen to fit the answer:

1. A difference below ~0.02 is not interpretable at this sample size.
2. When AUROC disagrees with the direct measures, prefer the direct measures — AUROC is
   weakened by the label-transfer effect.
3. Where the direct measures still disagree, adherence carries most weight; groundedness is
   what the project set out to improve.

Applied: FIN-031 locked over FIN-075, whose AUROC lead of 0.0012 is a twentieth of the noise
floor against an adherence deficit of 0.0405. GK-031 locked over GK-040 on the same reasoning.
Customer Support was decided on judge coverage instead — CS-037 leads adherence but its score
rests on 82% of examples against CS-064's 94.5%. Both runner-ups are published in full.

## Reading the results files

| File | What it is |
|---|---|
| `results_<domain>.csv` | append-only master, every run, deduplicated on exp_id + n_examples |
| `report_<domain>.csv` | report-grade rows only, judge coverage >= 0.80 |
| `report_<domain>_EXCLUDED.csv` | rows below the coverage threshold, published rather than dropped |
| `experiment_registry.json` | exp_id to configuration mapping |

Finance additionally carries `results_finance_canonical*.csv` and `audit_finance_judgeonly*.csv`
from the 37-column schema backfill, which reconstructed per-example metrics from checkpoint
JSON without re-running anything.

## Layout

```
src/                    five domain pipeline scripts + finance backfill utility
notebooks/              executed notebooks
results/factorial/      96-cell grid per domain
results/greedy/         stage-wise ladder per domain
results/judge_swap/     5 domains x 2 judges diagnostic, n=100
```

`results/judge_swap/` is the evidence for the judge-stronger-than-generator rule: the same
pipeline scored 0.639 under an 8b judge and 0.875 under a 70b one.

## Reference

Friel et al. 2024, *RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems*,
https://arxiv.org/abs/2407.11005
