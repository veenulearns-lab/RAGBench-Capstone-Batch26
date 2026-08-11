# RGB — Retrieval-Augmented Generation Benchmark (CP-4)

TRACe measures whether an answer is good given the retrieval it got. RGB measures what the
model does when retrieval **fails** — when the evidence is noisy, absent, scattered across
documents, or factually wrong. In production, retrieval failing is the normal case.

English set only; the Chinese set was not used. Five models, four abilities.

## The four abilities

| Ability | What it tests | Formula | n |
|---|---|---|---|
| Noise robustness | Finding the answer with distractors mixed in | correct / total, at each noise ratio | 300 |
| Negative rejection | Refusing when the evidence genuinely is not there | refusals / unanswerable questions | 300 |
| Information integration | Combining evidence from several documents | correct / multi-source questions | 100 |
| Counterfactual robustness | Noticing that a retrieved document is factually wrong | errors identified / corrupted docs | 100 |

## Results

### Noise robustness — accuracy % by noise ratio

| Model | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|---|
| llama-3.3-70b-versatile | 100.00 | 100.00 | 99.33 | 99.33 | 99.00 |
| qwen2.5-72b-instruct | 99.67 | 99.67 | 99.00 | 99.00 | 96.00 |
| llama-3.1-8b-instruct | 99.67 | 99.33 | 98.33 | 96.67 | 86.00 |
| gpt-oss-120b | 98.00 | 98.00 | 98.00 | 98.33 | 94.67 |
| gpt-oss-20b | 97.33 | 98.00 | 97.00 | 97.33 | 95.00 |

Only llama-3.1-8b breaks, and only at 0.8 — a 10.7-point cliff where every other model loses
under 5 points across the whole range. Noise is not where these models fail.

### Negative rejection — rejection rate %

| Model | Rejection |
|---|---|
| llama-3.1-8b-instruct | 73.00 |
| qwen2.5-72b-instruct | 73.00 |
| gpt-oss-120b | 71.67 |
| gpt-oss-20b | 70.33 |
| llama-3.3-70b-versatile | 67.33 |

Roughly a third of the time, models answer anyway when the evidence is not there. Note the
inversion: the 8b model rejects most and detects corrupted evidence least — it is not being
discriminating, it is answering less confidently overall.

### Information integration — accuracy % by noise ratio

| Model | 0.0 | 0.2 | 0.4 |
|---|---|---|---|
| llama-3.3-70b-versatile | 80.0 | 79.0 | 74.0 |
| qwen2.5-72b-instruct | 78.0 | 76.0 | 71.0 |
| gpt-oss-120b | 75.0 | 72.0 | 66.0 |
| gpt-oss-20b | 70.0 | 66.0 | 63.0 |
| llama-3.1-8b-instruct | 68.0 | 64.0 | 55.0 |

Scored all-parts, not any-part: an answer counts only if every required fact is present.

### Counterfactual robustness — %

| Model | Acc | Acc_doc | Error detection | Correction (of detected) | Correction (overall) |
|---|---|---|---|---|---|
| qwen2.5-72b-instruct | 92.0 | 71.0 | 60.0 | 90.00 | 54.0 |
| llama-3.3-70b-versatile | 93.0 | 61.0 | 54.0 | 92.59 | 50.0 |
| gpt-oss-120b | 91.0 | 51.0 | 60.0 | 83.33 | 50.0 |
| gpt-oss-20b | 71.0 | 26.0 | 40.0 | 65.00 | 26.0 |
| llama-3.1-8b-instruct | 82.0 | 12.0 | 24.0 | 41.67 | 10.0 |

This is the real weakness. Accuracy without documents stays high (71–93) while Acc_doc
collapses — models answer correctly *and* fail to notice the planted error. Robustness here
tracks model scale monotonically, but even the best model misses 29% of corrupted documents.
RAG fails safest when retrieval is noisy and least safe when retrieval is confidently wrong.

## Why this is version 2

The first campaign was discarded. Three defects, none of which crashed:

1. **Counterfactual read the wrong field** — the harness used the dataset's clean-document
   field instead of the corrupted one, so the test designed to plant factual errors contained
   none. Every model passed it, which we briefly took as a good sign.
2. **Information integration scored any-part instead of all-parts** — partial answers counted
   as correct.
3. **The Figure 3 prompt was paraphrased rather than copied** — worth up to 44.7 points in
   llama-3.3-70b's noise-0.8 accuracy on its own.

A unified rescoring pass was then applied idempotently: NFKC normalisation, diacritic folding,
and contraction equivalence. Treating `cannot` and `can not` as different answers had been
scoring correct refusals as failures and was hiding 29.7 points.

The lesson: inspect a handful of real examples from a new test set **before** running the
batch, not after.

## Running it

- 9-key rotation with paid OpenRouter credits as overflow
- Halt-guard stops the run if `[ERROR` replies begin, so a provider failure cannot silently
  fill results with blanks
- Triple-location checkpointing: Colab, Drive, GitHub

## Contents

```
results/v2/   cp4_rgb_summary_<timestamp>.csv  — the scored results
              cp4_rgb_results_<timestamp>.png  — plots
              cp4v2_rgb_<model>_<ability>_progress.csv — per-run state
notebooks/    RGB_Evaluation_v2_CP4_Batch26.ipynb
```

The superseded v1 harness and its results are in `archive/RGB_v1/`, retained rather than
deleted so the defects above are inspectable.

## Reference

Chen et al. 2023, *Benchmarking Large Language Models in Retrieval-Augmented Generation*,
https://arxiv.org/abs/2309.01431
