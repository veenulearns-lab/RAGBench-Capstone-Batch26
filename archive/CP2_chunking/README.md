# CP-2 Chunking Strategy Experiments

## Final Winning Strategy Per Domain

| Domain | Winning Strategy | Rel RMSE | Util RMSE | Comp RMSE | Adh AUCROC | Key Insight |
|--------|------------------|----------|-----------|-----------|------------|-------------|
| Biomedical | CP-1 baseline (whole doc) | 0.2410 | 0.1627 | 0.4500 | 0.6446 | No truncation needed in CP-1 — chunking added no value, only hurt |
| Legal | Clause-level (~800 chars) | 0.3168 | 0.1672 | 0.6645 | 0.6077 | Fixes CP-1's destructive truncation — biggest win in project |
| Customer Support | Short fixed (~300 chars) | 0.0843 | 0.0619 | 0.5564 | 0.5357 | Massive win — short queries need short precise chunks |
| Finance | Larger chunks (~1100 chars) | 0.2620 | 0.2307 | 0.5000 | 0.6016 | Keeps numeric figures with labels intact |
| General Knowledge | Round 1 paragraph (~500 chars) | 0.1849 | 0.2279 | 0.5254 | 0.6141 | Best completeness gain; CP-1 still wins raw relevance |

## Full Round-by-Round Results

### Biomedical (covidqa)
| Version | Strategy | Examples | Rel RMSE | Util RMSE | Comp RMSE | Adh AUCROC |
|---------|----------|----------|----------|-----------|-----------|------------|
| CP-1 baseline | Whole document | 246 | 0.2410 | 0.1627 | 0.4500 | 0.6446 |
| Round 1 | Sentence-level | 150 | 0.2537 | 0.2323 | 0.5664 | 0.5708 |
| Round 2 | Small2Big (window=1) | 145 | 0.2714 | 0.1605 | 0.5337 | 0.5170 |

### Legal (cuad)
| Version | Strategy | Examples | Rel RMSE | Util RMSE | Comp RMSE | Adh AUCROC |
|---------|----------|----------|----------|-----------|-----------|------------|
| CP-1 baseline | Truncation (2000 chars) | 195 | 0.7353 | 0.5399 | 0.8130 | 0.5220 |
| Round 1 | Clause-level (~800 chars) | 70 | 0.3168 | 0.1672 | 0.6645 | 0.6077 |

### Customer Support (techqa)
| Version | Strategy | Examples | Rel RMSE | Util RMSE | Comp RMSE | Adh AUCROC |
|---------|----------|----------|----------|-----------|-----------|------------|
| CP-1 baseline | Truncation (2000 chars) | 192 | 0.3938 | 0.5420 | 0.6523 | 0.5254 |
| Round 1 | Short fixed (~300 chars) | 60 | 0.0843 | 0.0619 | 0.5564 | 0.5357 |

### Finance (finqa)
| Version | Strategy | Examples | Rel RMSE | Util RMSE | Comp RMSE | Adh AUCROC |
|---------|----------|----------|----------|-----------|-----------|------------|
| CP-1 baseline | Whole document (200 subset) | 195 | 0.4212 | 0.1670 | 0.8314 | 0.5929 |
| Round 1 | Semantic (~600 chars) | 150 | 0.3071 | 0.2688 | 0.5223 | 0.5680 |
| Round 2 | Larger semantic (~1100 chars) | 150 | 0.2620 | 0.2307 | 0.5000 | 0.6016 |

### General Knowledge (hotpotqa)
| Version | Strategy | Examples | Rel RMSE | Util RMSE | Comp RMSE | Adh AUCROC |
|---------|----------|----------|----------|-----------|-----------|------------|
| CP-1 baseline | Whole document (390 full) | 390 | 0.1662 | 0.1618 | 0.7714 | 0.6390 |
| Round 1 | Paragraph-level (~500 chars) | 150 | 0.1849 | 0.2279 | 0.5254 | 0.6141 |
| Round 2 | Smaller paragraph (~350 chars) | 145 | 0.1876 | 0.2100 | 0.5377 | 0.6373 |

## Key Finding

Chunking strategy effectiveness is highly domain-dependent. Domains where CP-1's
whole-document approach hit API token limits and required truncation (Legal,
Customer Support) saw dramatic improvements from proper chunking — confirming
truncation was the actual bottleneck, not the architecture. Finance benefited
from larger chunks that preserve numerical context. Biomedical and General
Knowledge, where CP-1 succeeded without data loss, showed that chunking
introduces a precision/completeness trade-off rather than a universal
improvement. This demonstrates the core CP-2 finding: chunking strategy must
be selected based on each domain's specific failure mode in the baseline,
not applied uniformly.
