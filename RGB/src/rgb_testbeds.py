"""Testbed builders for the four RGB abilities.
Extracted from RGB_Evaluation_v2_CP4_Batch26.ipynb, cell 5.
"""

# CELL 5: Data Loader + Testbed Builder — v2 FIXES:
#   (1) counterfactual now uses positive_wrong (docs WITH planted errors) — v1 bug used clean docs
#   (2) info integration: en_int "positive" is a LIST OF LISTS (one doc-group per sub-answer);
#       official RGB code samples one doc per group — v1 fed stringified nested lists
#   (3) adds a no-documents counterfactual condition for the Acc column (mentor Table 7 format)
import json, random

def load_jsonl(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_noise_testbed(records, noise_ratio, max_examples=None, seed=42):
    random.seed(seed)
    testbed = []
    recs = records[:max_examples] if max_examples else records
    for rec in recs:
        positives = rec.get("positive", [])
        negatives = rec.get("negative", [])
        if not positives:
            continue
        n_docs = 5
        n_noise = round(n_docs * noise_ratio)
        n_pos = n_docs - n_noise
        pos_sample = random.sample(positives, min(n_pos, len(positives)))
        neg_sample = random.sample(negatives, min(n_noise, len(negatives))) if negatives else []
        while len(pos_sample) < n_pos and positives:
            pos_sample.append(random.choice(positives))
        docs = pos_sample + neg_sample
        random.shuffle(docs)
        testbed.append({"id": rec["id"], "query": rec["query"], "answer": rec["answer"],
                        "docs": docs, "noise_ratio": noise_ratio, "ability": "noise_robustness"})
    return testbed


def build_negative_rejection_testbed(records, max_examples=None, seed=42):
    random.seed(seed)
    testbed = []
    recs = records[:max_examples] if max_examples else records
    for rec in recs:
        negatives = rec.get("negative", [])
        if not negatives:
            continue
        n_docs = 5
        neg_sample = random.sample(negatives, min(n_docs, len(negatives)))
        while len(neg_sample) < n_docs and negatives:
            neg_sample.append(random.choice(negatives))
        testbed.append({"id": rec["id"], "query": rec["query"], "answer": rec["answer"],
                        "docs": neg_sample, "ability": "negative_rejection"})
    return testbed


def build_info_integration_testbed(records, noise_ratio=0.0, max_examples=None, seed=42):
    # Official RGB sampling: "positive" is a list of doc-groups, one group per sub-answer.
    # One doc per group first (integration is impossible without every sub-answer represented),
    # then fill remaining positive slots by depth, then negatives per noise ratio (ceil(5*r)).
    import math
    random.seed(seed)
    recs = records[:max_examples] if max_examples else records
    testbed = []
    for rec in recs:
        groups = [list(g) for g in rec.get("positive", [])]
        for g in groups:
            random.shuffle(g)
        n_neg = math.ceil(5 * noise_ratio)
        n_pos = 5 - n_neg
        docs = [g[0] for g in groups if g]
        depth = 1
        max_depth = max((len(g) for g in groups), default=0)
        while len(docs) < n_pos and depth < max_depth:
            for g in groups:
                if len(g) > depth and len(docs) < n_pos:
                    docs.append(g[depth])
            depth += 1
        negatives = list(rec.get("negative", []))
        random.shuffle(negatives)
        docs += negatives[:5 - len(docs)]
        random.shuffle(docs)
        testbed.append({"id": rec["id"], "query": rec["query"], "answer": rec["answer"],
                        "docs": docs, "ability": "info_integration"})
    return testbed


def build_counterfactual_testbed(records, max_examples=None):
    # FIX: use positive_wrong — the documents WITH planted factual errors.
    recs = records[:max_examples] if max_examples else records
    testbed = []
    for rec in recs:
        docs = rec.get("positive_wrong", [])[:5]
        if not docs:
            continue
        testbed.append({"id": rec["id"], "query": rec["query"], "answer": rec["answer"],
                        "fakeanswer": rec.get("fakeanswer", ""),
                        "docs": docs, "ability": "counterfactual_robustness"})
    return testbed


def build_counterfactual_nodocs_testbed(records, max_examples=None):
    # Acc column of paper Table 7: model answers WITHOUT documents (own knowledge).
    recs = records[:max_examples] if max_examples else records
    return [{"id": rec["id"], "query": rec["query"], "answer": rec["answer"],
             "docs": [], "ability": "counterfactual_nodocs"} for rec in recs]


en_refine = load_jsonl(os.path.join(DATA_DIR, "en_refine.json"))
en_int    = load_jsonl(os.path.join(DATA_DIR, "en_int.json"))
en_fact   = load_jsonl(os.path.join(DATA_DIR, "en_fact.json"))

print(f"Loaded: en_refine={len(en_refine)}, en_int={len(en_int)}, en_fact={len(en_fact)}")
print(f"en_fact keys (expect positive_wrong + fakeanswer): {list(en_fact[0].keys())}")
print(f"en_int positive is nested: {isinstance(en_int[0]['positive'][0], list)}")
