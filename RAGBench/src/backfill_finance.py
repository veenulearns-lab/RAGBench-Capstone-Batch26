# ===== Finance backfill, ALL N groups — Colab. No API calls. =====
# Scans every checkpoint, groups by (fingerprint, n_examples), and writes one
# canonical CSV per group. No experiment IDs or N values are hardcoded.
import os, json, glob, math, random, hashlib, ast, shutil
from collections import defaultdict
import numpy as np, pandas as pd
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, mean_squared_error

BASE     = "/content/drive/MyDrive/RAGBench_Results/NEW_FIN_Aug/finance"
CKPT_DIR = f"{BASE}/ckpt"          # <- point at ckpt_backup_25 to rebuild the full N=25 table
DSCFG, SEED = "finqa", 42
MIN_VALID_FRAC = 0.80
K_RETRIEVE, K_FINAL = 12, 5
METRICS = ["context_relevance", "context_utilization", "completeness", "adherence"]
TARGET = ["exp_id","timestamp","domain","status","embedder","chunk_config","gen_model",
    "judge_model","retrieval","rerank","context_order","k_retrieve","k_final","seed",
    "n_examples","context_relevance","context_utilization","completeness","adherence",
    "ref_context_relevance","ref_context_utilization","ref_completeness","ref_adherence",
    "context_relevance_rmse","context_utilization_rmse","completeness_rmse","adherence_rmse",
    "context_relevance_mae","context_utilization_mae","completeness_mae","adherence_mae",
    "adherence_auroc","adherence_accuracy","adherence_pos_rate","n_gold_adherence",
    "judge_used","runtime_sec"]

def gold_metrics(row):
    def g(*names):
        for n in names:
            if n in row and row[n] is not None:
                try: return float(row[n])
                except Exception: pass
        return None
    return {"context_relevance":   g("relevance_score","gpt3_context_relevance","context_relevance"),
            "context_utilization": g("utilization_score","gpt3_utilization","context_utilization"),
            "completeness":        g("completeness_score","adherence_completeness","completeness"),
            "adherence":           g("adherence_score","gpt3_adherence","adherence")}

def build_sample(n):
    ds  = load_dataset("rungalileo/ragbench", DSCFG, split="test")
    idx = sorted(random.Random(SEED).sample(range(len(ds)), min(n, len(ds))))
    rows = [{"question": r["question"], "documents": r["documents"], "gold": gold_metrics(r)}
            for r in ds.select(idx)]
    h = hashlib.sha1(); h.update(f"finance|{DSCFG}|{SEED}|{len(rows)}".encode())
    for ex in rows:
        docs = ex["documents"]
        if isinstance(docs, str):
            try: docs = ast.literal_eval(docs)
            except Exception: docs = [docs]
        h.update(str(ex["question"]).strip().encode("utf-8","ignore"))
        h.update(f"|{len(docs)}|".encode())
    return [r["gold"] for r in rows], h.hexdigest()[:16]

def stats(preds, golds):
    o = {}
    pairs = [(p, g) for p, g in zip(preds, golds) if p is not None]
    if not pairs: return o
    P = [p for p, _ in pairs]
    for m in METRICS:
        o[m] = round(float(np.mean([p[m] for p in P])), 4)
        gp = [(p[m], g[m]) for p, g in pairs if g.get(m) is not None]
        if gp:
            pv, gv = [a for a,_ in gp], [b for _,b in gp]
            o[f"ref_{m}"]  = round(float(np.mean(gv)), 4)
            o[f"{m}_rmse"] = round(math.sqrt(mean_squared_error(gv, pv)), 4)
            o[f"{m}_mae"]  = round(float(np.mean(np.abs(np.array(pv)-np.array(gv)))), 4)
    adh = [(p["adherence"], g["adherence"]) for p, g in pairs if g.get("adherence") is not None]
    if adh:
        lab = [int(round(b)) for _, b in adh]; sc = [a for a,_ in adh]
        o["n_gold_adherence"]   = len(adh)
        o["adherence_pos_rate"] = round(float(np.mean(lab)), 4)
        o["adherence_accuracy"] = round(float(np.mean([int(round(s))==l for s,l in zip(sc,lab)])), 4)
        if len(set(lab)) > 1: o["adherence_auroc"] = round(roc_auc_score(lab, sc), 4)
    o["judge_used"] = round(float(np.mean([p.get("_judge_used",0.0) for p in P])), 4)
    return o

# ---- snapshot before anything (ckpt files are keyed on exp_id only) ----
BACKUP = f"{BASE}/ckpt_snapshot_latest"
if os.path.abspath(CKPT_DIR) == os.path.abspath(f"{BASE}/ckpt"):
    if os.path.exists(BACKUP): shutil.rmtree(BACKUP)
    shutil.copytree(CKPT_DIR, BACKUP)
    print(f"snapshot: {len(os.listdir(BACKUP))} files -> {BACKUP}")

# ---- group checkpoints ----
files = sorted(glob.glob(f"{CKPT_DIR}/*.json"))
if not files: raise SystemExit(f"no checkpoints in {CKPT_DIR}")
groups = defaultdict(list)
for f in files:
    c = json.load(open(f))
    groups[(c.get("fingerprint"), c.get("n_examples"))].append((f, c))
print(f"{len(files)} checkpoints in {len(groups)} group(s):")
for (fpv, n), items in sorted(groups.items(), key=lambda x: -len(x[1])):
    print(f"  N={n:<5} fp={fpv}  {len(items)} exp: "
          f"{', '.join(sorted(c['exp_id'] for _, c in items)[:6])}"
          f"{' ...' if len(items) > 6 else ''}")

master = pd.read_csv(f"{BASE}/results_finance.csv")
master["timestamp"] = pd.to_datetime(master["timestamp"], errors="coerce", utc=True)

# ---- backfill each group ----
for (fpv, N), items in sorted(groups.items(), key=lambda x: -len(x[1])):
    print(f"\n=== N={N} ({len(items)} experiments) ===")
    golds, rebuilt = build_sample(N)
    print(f"rebuilt fp={rebuilt}  ckpt fp={fpv}  -> "
          f"{'MATCH' if rebuilt == fpv else '*** MISMATCH — group skipped ***'}")
    if rebuilt != fpv:
        continue

    rows, audit = [], []
    for _, c in items:
        eid, spec = c["exp_id"], c["spec"]
        preds = c["preds"][:N] + [None] * max(0, N - len(c["preds"]))
        a = stats(preds, golds)
        j = stats([p if (p and float(p.get("_judge_used",0.0)) >= 1.0) else None
                   for p in preds], golds)
        sel = master[(master.exp_id == eid) & (master.n_examples == N)].sort_values("timestamp")
        m = sel.tail(1)
        r = {k: "" for k in TARGET}
        r.update({"exp_id": eid, "domain": "finance",
                  "timestamp": m.timestamp.iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ") if len(m) else "",
                  "embedder": spec["embedder"].split("/")[-1],
                  "chunk_config": spec["chunk_config"], "gen_model": spec["gen_model"],
                  "judge_model": spec["judge_model"], "retrieval": spec["retrieval"],
                  "rerank": spec["rerank"], "context_order": spec["context_order"],
                  "k_retrieve": K_RETRIEVE, "k_final": K_FINAL, "seed": SEED, "n_examples": N,
                  "runtime_sec": m.elapsed_s.iloc[0] if len(m) else ""})
        for k, v in a.items():
            if k in r: r[k] = v
        r["status"] = ("report-grade" if a.get("judge_used", 0) >= MIN_VALID_FRAC
                       else "LOW_JUDGE_COVERAGE")
        rows.append(r)
        audit.append({"exp_id": eid, "n_examples": N, "judge_used": a.get("judge_used"),
                      "n_fallback": sum(1 for p in preds if p and p.get("_judge_used",0.0) < 1.0),
                      "adherence": a.get("adherence"), "adherence_judgeonly": j.get("adherence"),
                      "auroc": a.get("adherence_auroc"), "auroc_judgeonly": j.get("adherence_auroc")})

    out = pd.DataFrame(rows)[TARGET].sort_values("exp_id").reset_index(drop=True)
    out.to_csv(f"{BASE}/results_finance_canonical_N{N}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(f"{BASE}/audit_finance_judgeonly_N{N}.csv",
                               index=False, encoding="utf-8-sig")
    blank = [c for c in TARGET if out[c].astype(str).str.strip().eq("").all()]
    print(f"wrote results_finance_canonical_N{N}.csv  {out.shape[0]} rows")
    print(f"  blank columns: {blank or 'none'}")
    if len(out):
        npos = int(out.adherence_pos_rate.iloc[0] * out.n_gold_adherence.iloc[0])
        ntot = int(out.n_gold_adherence.iloc[0])
        print(f"  gold balance: {npos} pos / {ntot-npos} neg of {ntot}")
    if len(out) <= 5:
        print(out[["exp_id","embedder","chunk_config","retrieval","rerank","context_order",
                   "adherence","adherence_auroc","adherence_accuracy","judge_used"]]
              .to_string(index=False))
