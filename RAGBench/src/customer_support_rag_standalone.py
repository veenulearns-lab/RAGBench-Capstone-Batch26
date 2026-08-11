"""
customer_support_rag_standalone.py — Customer Support (RAGBench TechQA) standalone pipeline.

Self-contained RAG + TRACe evaluation for the Customer Support domain.
Judge prompt: verbatim RAGBench Appendix 7.4 (from Final Working notebooks).
All logic embedded — no shared engine dependency.

Run:  python customer_support_rag_standalone.py           (real data; needs GROQ_API_KEY + deps)
      USE_SYNTHETIC=1 python customer_support_rag_standalone.py   (offline smoke test)
"""
from __future__ import annotations
import os
import re
import csv
import json
import math
import time
import random
import pickle
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any

import numpy as np

REJECTION = "i cannot answer the question because of insufficient information in documents"

# --------------------------------------------------------------------------- #
# Device: use the GPU only if a real embedding op succeeds on it, else CPU.
# --------------------------------------------------------------------------- #
def select_device() -> str:
    force = os.environ.get("ST_DEVICE")
    if force in ("cpu", "cuda"):
        return force
    try:
        import torch
        if torch.cuda.is_available():
            emb = torch.nn.Embedding(4, 4).to("cuda")
            _ = emb(torch.zeros(2, dtype=torch.long, device="cuda")).sum().item()
            return "cuda"
    except Exception:
        pass
    return "cpu"

DEVICE = select_device()
SAVE_DIR = Path(os.environ.get("SAVE_PATH", "."))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# PERSISTENCE LAYER — experiment IDs, Drive/local checkpoints, git sync, resume
# --------------------------------------------------------------------------- #
# Cadence:
#   * checkpoint (local + Drive) every CHECKPOINT_EVERY examples   -> default 5
#   * git push every GIT_PUSH_EVERY_N_CKPT checkpoints             -> default 5
#     (= every 25 examples) plus one forced push when an experiment finishes.
# Resume:
#   * finished experiments are skipped entirely (CS-001 done -> next run starts CS-002)
#   * a half-finished experiment resumes at the exact example it died on
# --------------------------------------------------------------------------- #
EXP_PREFIX  = os.environ.get("EXP_PREFIX", "CS")
DOMAIN_TAG  = os.environ.get("DOMAIN_TAG", "customer_support")
LOCAL_ROOT  = Path(os.environ.get("LOCAL_ROOT", str(SAVE_DIR)))
DRIVE_ROOT  = Path(os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/RAGBench_Results"))
REPO_ROOT   = Path(os.environ.get("REPO_ROOT", "/content/RAGBench-Capstone-Batch26"))
REPO_SUBDIR = os.environ.get("REPO_SUBDIR", "results")
GIT_BRANCH  = os.environ.get("GIT_BRANCH", "main")
GIT_USER    = os.environ.get("GIT_USER", "veenulearns-lab")
GIT_REPO    = os.environ.get("GIT_REPO", "veenulearns-lab/RAGBench-Capstone-Batch26")

CHECKPOINT_EVERY      = int(os.environ.get("CHECKPOINT_EVERY", "5"))
GIT_PUSH_EVERY_N_CKPT = int(os.environ.get("GIT_PUSH_EVERY_N_CKPT", "5"))
ENABLE_GIT            = os.environ.get("ENABLE_GIT", "1") not in ("0", "false", "False")
GUARD_PAUSE           = float(os.environ.get("GUARD_PAUSE", "3"))   # sanity pause before each run

# ---- LLM call settings (all env-overridable) ----
MAX_RETRIES      = int(os.environ.get("MAX_RETRIES", "3"))          # attempts per API call
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "4096"))  # TRACe JSON is long
GEN_MAX_TOKENS   = int(os.environ.get("GEN_MAX_TOKENS", "300"))     # answer length
JUDGE_JSON_MODE  = os.environ.get("JUDGE_JSON_MODE", "1") not in ("0", "false", "False")
_no_json_mode: set = set()   # models observed to reject response_format

DRIVE_OK = DRIVE_ROOT.parent.exists()          # /content/drive/MyDrive present == mounted
if DRIVE_OK:
    try:
        (DRIVE_ROOT / DOMAIN_TAG).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[drive] mount looks present but write failed ({e}); local-only mode")
        DRIVE_OK = False
(LOCAL_ROOT / DOMAIN_TAG).mkdir(parents=True, exist_ok=True)
print(f"[persist] local={LOCAL_ROOT/DOMAIN_TAG}  drive={'ON: ' + str(DRIVE_ROOT/DOMAIN_TAG) if DRIVE_OK else 'OFF'}  "
      f"ckpt_every={CHECKPOINT_EVERY}  git_every={GIT_PUSH_EVERY_N_CKPT} ckpts")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _roots() -> List[Path]:
    return [LOCAL_ROOT] + ([DRIVE_ROOT] if DRIVE_OK else [])

def _rel(*parts) -> str:
    return str(Path(DOMAIN_TAG, *parts))

def _atomic_write(path: Path, obj) -> None:
    """tmp + os.replace so a Colab disconnect mid-write can never truncate a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def _write_all(relpath: str, obj) -> None:
    for root in _roots():
        try:
            _atomic_write(root / relpath, obj)
        except Exception as e:
            print(f"[persist] write failed at {root/relpath}: {type(e).__name__}: {e}")

def _read_any(relpath: str, prefer_more: Optional[str] = None):
    """Read local + Drive copies; if both exist keep the richer one (more work done)."""
    found = []
    for root in _roots():
        p = root / relpath
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    found.append(json.load(f))
        except Exception:
            pass
    if not found:
        return None
    if len(found) == 1 or not prefer_more:
        return found[0]
    return max(found, key=lambda o: len(o.get(prefer_more, []) or []))

def _ckey(*parts) -> str:
    """Stable string cache key (JSON-serializable, unlike the old tuple keys)."""
    blob = "||".join(json.dumps(p, sort_keys=True, default=str) for p in parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()

def _fingerprint(examples: List[dict]) -> str:
    probe = [str(e.get("question", ""))[:120] for e in examples[:3] + examples[-1:]]
    return _ckey(len(examples), probe)

# ----------------------------- experiment registry (stable CS-001, CS-002 ...) #
_REGISTRY_REL = _rel("experiment_registry.json")
_registry: Dict[str, str] = {}

def _load_registry() -> Dict[str, str]:
    global _registry
    obj = _read_any(_REGISTRY_REL)
    _registry = (obj or {}).get("signatures", {}) if isinstance(obj, dict) else {}
    return _registry

def registry_exp_id(signature: str) -> str:
    """
    Map a config signature -> CS-001 / CS-002 / ... IDs are assigned once and
    persisted, so adding an embedder later does NOT renumber existing rows.
    """
    if not _registry:
        _load_registry()
    if signature in _registry:
        return _registry[signature]
    used = {int(v.split("-")[-1]) for v in _registry.values() if v.split("-")[-1].isdigit()}
    nxt = 1
    while nxt in used:
        nxt += 1
    exp_id = f"{EXP_PREFIX}-{nxt:03d}"
    _registry[signature] = exp_id
    _write_all(_REGISTRY_REL, {"updated_at": _now(), "prefix": EXP_PREFIX,
                               "signatures": _registry})
    return exp_id

# ----------------------------------------------------------------- checkpoints #
def save_checkpoint(exp_id: str, records: List[dict], meta: Optional[dict] = None) -> None:
    payload = {"exp_id": exp_id, "domain": DOMAIN_TAG, "updated_at": _now(),
               "n_done": len(records),
               "done_indices": sorted(int(r["idx"]) for r in records),
               "meta": meta or {}, "records": records}
    _write_all(_rel(exp_id, "checkpoint.json"), payload)

def load_checkpoint(exp_id: str, fingerprint: Optional[str] = None):
    """Returns (records, done_indices_set, meta). Stale-data guard included."""
    obj = _read_any(_rel(exp_id, "checkpoint.json"), prefer_more="done_indices")
    if not obj:
        return [], set(), {}
    meta = obj.get("meta", {}) or {}
    if fingerprint and meta.get("fingerprint") and meta["fingerprint"] != fingerprint:
        print(f"[resume] {exp_id}: dataset fingerprint changed — discarding stale checkpoint")
        return [], set(), {}
    records = obj.get("records", []) or []
    done = {int(r["idx"]) for r in records}
    if done:
        print(f"[resume] {exp_id}: {len(done)} examples already done (last write {obj.get('updated_at')})")
    return records, done, meta

def save_progress(exp_id: str, n_done: int, total: int, status: str = "running",
                  extra: Optional[dict] = None, fingerprint: Optional[str] = None) -> None:
    _write_all(_rel(exp_id, "progress.json"),
               {"exp_id": exp_id, "domain": DOMAIN_TAG, "status": status,
                "n_done": int(n_done), "total": int(total),
                "pct": round(100.0 * n_done / total, 2) if total else 0.0,
                "fingerprint": fingerprint, "updated_at": _now(), "extra": extra or {}})

def load_progress(exp_id: str) -> dict:
    return _read_any(_rel(exp_id, "progress.json")) or {}

def experiment_completed(exp_id: str, expected_n: Optional[int] = None,
                         fingerprint: Optional[str] = None) -> bool:
    """
    True once the experiment finished FOR THIS DATA SLICE — this is what makes the
    next run start at CS-002.

    Guarded on both N and the dataset fingerprint, so a smoke run at N_EXAMPLES=1
    can never mark the sweep 'done' and cause the real N=200 pass to skip everything.
    """
    if os.environ.get("FORCE_RERUN", "").strip() in ("1", "true", "True"):
        return False
    prog = load_progress(exp_id)
    if not prog:
        if expected_n:
            _, done, meta = load_checkpoint(exp_id, fingerprint)
            return len(done) >= expected_n
        return False
    if fingerprint and prog.get("fingerprint") and prog["fingerprint"] != fingerprint:
        return False                      # different N / different examples -> must rerun
    if expected_n and prog.get("total") and int(prog["total"]) != int(expected_n):
        return False
    if prog.get("status") == "done":
        return True
    if expected_n and prog.get("n_done", 0) >= expected_n:
        return True
    if expected_n:
        _, done, _ = load_checkpoint(exp_id, fingerprint)
        return len(done) >= expected_n
    return False

# ---------------------------------------------------------------------- cache #
def save_cache() -> None:
    """Judge + generation caches survive a disconnect, so a resume costs no API calls."""
    _write_all(_rel("cache_gen.json"),
               {"updated_at": _now(), "n": len(_gen_cache), "entries": _gen_cache})
    _write_all(_rel("cache_judge.json"),
               {"updated_at": _now(), "n": len(_judge_cache), "entries": _judge_cache})

def load_cache() -> None:
    g = _read_any(_rel("cache_gen.json"))
    j = _read_any(_rel("cache_judge.json"))
    if g:
        _gen_cache.update(g.get("entries", {}) or {})
    if j:
        _judge_cache.update(j.get("entries", {}) or {})
    print(f"[cache] restored gen={len(_gen_cache)} judge={len(_judge_cache)} entries")

# ----------------------------------------------------------------- results csv #
def _csv_paths(csv_name: str) -> List[Path]:
    return [root / DOMAIN_TAG / csv_name for root in _roots()]

def csv_has_exp(csv_name: str, exp_id: str, n_examples: Optional[int] = None) -> bool:
    for p in _csv_paths(csv_name):
        try:
            if p.exists():
                with open(p, newline="", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        if r.get("exp_id") != exp_id:
                            continue
                        if n_examples is None or str(r.get("n_examples")) == str(n_examples):
                            return True
        except Exception:
            pass
    return False

def append_results_csv(row: dict, fields: List[str], csv_name: str) -> None:
    """Append-only (the old 'w' mode wiped previous runs on every restart)."""
    clean = {k: row.get(k, "") for k in fields}
    for p in _csv_paths(csv_name):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            new = (not p.exists()) or p.stat().st_size == 0
            with open(p, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if new:
                    w.writeheader()
                w.writerow(clean)
        except Exception as e:
            print(f"[csv] append failed at {p}: {e}")

def read_results_csv(csv_name: str) -> List[dict]:
    for p in _csv_paths(csv_name):
        try:
            if p.exists():
                with open(p, newline="", encoding="utf-8") as f:
                    return list(csv.DictReader(f))
        except Exception:
            pass
    return []

# ------------------------------------------------------------------------ git #
def _git_token() -> Optional[str]:
    tok = os.environ.get("GIT_PAT_V") or os.environ.get("GH_PAT_NEW")
    if tok:
        return tok
    try:
        from google.colab import userdata
        return userdata.get("GIT_PAT_V")
    except Exception:
        return None

def git_sync(message: str, exp_id: Optional[str] = None) -> bool:
    """Third leg of the triple-save. Always non-fatal — git must never kill a run."""
    if not ENABLE_GIT or not (REPO_ROOT / ".git").is_dir():
        return False
    try:
        src_root = LOCAL_ROOT / DOMAIN_TAG
        dst_root = REPO_ROOT / REPO_SUBDIR / DOMAIN_TAG
        dst_root.mkdir(parents=True, exist_ok=True)
        srcs = [src_root / exp_id] if exp_id else [src_root]
        for s in srcs:
            if not s.exists():
                continue
            for f in s.rglob("*"):
                if f.is_file() and f.suffix in (".json", ".csv") and not f.name.endswith(".tmp"):
                    rel = f.relative_to(src_root)
                    out = dst_root / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(f.read_bytes())
        for f in src_root.glob("*.csv"):
            (dst_root / f.name).write_bytes(f.read_bytes())

        tok = _git_token()
        if tok:
            subprocess.run(["git", "remote", "set-url", "origin",
                            f"https://{GIT_USER}:{tok}@github.com/{GIT_REPO}.git"],
                           cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
        for cmd in (["git", "add", "-A"],
                    ["git", "commit", "-m", message],
                    ["git", "push", "origin", GIT_BRANCH]):
            r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
            if r.returncode != 0:
                blob = (r.stdout + r.stderr).lower()
                if "nothing to commit" in blob:
                    return True
                print(f"[git] {cmd[1]} -> {(r.stderr or r.stdout).strip()[:180]}")
                return False
        print(f"[git] pushed: {message}")
        return True
    except Exception as e:
        print(f"[git] sync failed (non-fatal): {type(e).__name__}: {e}")
        return False

# --------------------------------------------------------------------------- #
# Text utils
# --------------------------------------------------------------------------- #
_STOP = set("the a an of to in for with on by is are was were be this that as from and or it "
            "how what when which does do you your".split())

def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if s.strip()]

def content_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if len(w) > 2 and w not in _STOP}

def coverage(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    return len(ta & tb) / len(ta) if ta else 0.0

# --------------------------------------------------------------------------- #
# Chunking — every strategy, per document (provenance preserved)
# --------------------------------------------------------------------------- #
def chunk_document(doc_id: str, text: str, cc: dict) -> List[Dict]:
    strat = cc["strategy"]
    sents = split_sentences(text)
    chunks: List[List[str]] = []
    if strat == "whole_doc":
        chunks = [sents] if sents else []
    elif strat == "sentence":
        step = max(1, cc["size"] - cc.get("overlap", 0))
        chunks = [sents[i:i + cc["size"]] for i in range(0, len(sents), step) if sents[i:i + cc["size"]]]
    elif strat == "sliding":
        size, step = cc["size"], max(1, cc["size"] - cc.get("overlap", 0))
        i = 0
        while i < len(sents):
            chunks.append(sents[i:i + size])
            if i + size >= len(sents):
                break
            i += step
    elif strat == "fixed":
        words = text.split()
        size, step = cc["size"], max(1, cc["size"] - cc.get("overlap", 0))
        for i in range(0, len(words), step):
            piece = " ".join(words[i:i + size])
            if piece:
                chunks.append([piece])
            if i + size >= len(words):
                break
    elif strat == "large_semantic":
        max_chars = cc.get("max_chars", 1100)
        cur, cur_len = [], 0
        for s in sents:
            cur.append(s); cur_len += len(s) + 1
            if cur_len >= max_chars:
                chunks.append(cur); cur, cur_len = [], 0
        if cur:
            chunks.append(cur)
    else:
        raise ValueError(f"Unknown chunk strategy: {strat}")
    return [{"chunk_id": f"{doc_id}::c{i}", "doc_id": doc_id, "sentences": c, "text": " ".join(c)}
            for i, c in enumerate(chunks) if c]

def build_corpus(documents: List[Dict], cc: dict) -> List[Dict]:
    out = []
    for d in documents:
        out.extend(chunk_document(d["doc_id"], d["text"], cc))
    return out

# --------------------------------------------------------------------------- #
# Embeddings (domain models via mean-pooling; cache) + normalized cosine
# --------------------------------------------------------------------------- #
_RAW_BERT = {
    "ProsusAI/finbert", "nlpaueb/legal-bert-base-uncased",
    "dmis-lab/biobert-base-cased-v1.1", "yiyanghkust/finbert-tone",
}
_embedder_cache: Dict[str, Any] = {}

class Embedder:
    def __init__(self, model_name: str, corpus_hint: Optional[List[str]] = None):
        self.model_name = model_name
        self.backend = "tfidf"
        try:
            if os.environ.get("USE_SYNTHETIC"):
                raise RuntimeError("synthetic mode -> TF-IDF")
            from sentence_transformers import SentenceTransformer, models
            if model_name in _RAW_BERT:
                word = models.Transformer(model_name)
                pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
                self.model = SentenceTransformer(modules=[word, pool], device=DEVICE)
            else:
                self.model = SentenceTransformer(model_name, device=DEVICE)
            self.backend = "st"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer()
            self.vec.fit(corpus_hint or ["placeholder"])

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend == "st":
            return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype="float32")
        v = self.vec.transform(texts).toarray().astype("float32")
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-8, None)

def get_embedder(model_name: str, corpus_hint: Optional[List[str]] = None) -> Embedder:
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = Embedder(model_name, corpus_hint)
    return _embedder_cache[model_name]

# --------------------------------------------------------------------------- #
# Retrieval — dense cosine + BM25 hybrid RRF
# --------------------------------------------------------------------------- #
def make_bm25(corpus_texts: List[str]):
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi([t.lower().split() for t in corpus_texts])
    except Exception:
        return None

def dense_rank(qvec: np.ndarray, mat: np.ndarray) -> List[int]:
    return list(np.argsort(-(mat @ qvec)))

def hybrid_rrf_rank(query: str, qvec: np.ndarray, mat: np.ndarray, bm25, kk: int = 60) -> List[int]:
    dense_order = dense_rank(qvec, mat)
    if bm25 is None:
        return dense_order
    bm_order = list(np.argsort(-np.asarray(bm25.get_scores(query.lower().split()))))
    rrf: Dict[int, float] = {}
    for rank, i in enumerate(dense_order):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (kk + rank + 1)
    for rank, i in enumerate(bm_order):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (kk + rank + 1)
    return sorted(rrf, key=lambda i: -rrf[i])

# --------------------------------------------------------------------------- #
# Reranking + context ordering
# --------------------------------------------------------------------------- #
_cross_encoder: Dict[str, Any] = {}
def cross_encoder_rerank(query: str, chunks: List[Dict], model_name: str, k: int) -> List[Dict]:
    try:
        if model_name not in _cross_encoder:
            from sentence_transformers import CrossEncoder
            _cross_encoder[model_name] = CrossEncoder(model_name, device=DEVICE)
        scores = _cross_encoder[model_name].predict([(query, c["text"]) for c in chunks])
        return [chunks[i] for i in np.argsort(-np.asarray(scores))[:k]]
    except Exception:
        return chunks[:k]

def order_context(chunks: List[Dict], strategy: str) -> List[Dict]:
    if strategy == "forward":
        return chunks
    if strategy == "reverse":
        return list(reversed(chunks))
    if strategy == "sides":
        left, right = [], []
        for i, c in enumerate(chunks):
            (left if i % 2 == 0 else right).append(c)
        return left + list(reversed(right))
    raise ValueError(f"Unknown context order: {strategy}")

# --------------------------------------------------------------------------- #
# Groq API Key Rotation (Avoids Rate Limits on 200 samples)
# --------------------------------------------------------------------------- #
# ------------------------------------------------------------------
# LLM Provider Configuration
# ------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

GROQ_KEYS = [
    k.strip()
    for k in os.environ.get("GROQ_API_KEYS", "").split(",")
    if k.strip()
]

_single = os.environ.get("GROQ_API_KEY", "").strip()

if _single and _single not in GROQ_KEYS:
    GROQ_KEYS.insert(0, _single)

USE_OPENROUTER = bool(OPENROUTER_API_KEY)

_key_idx = 0
_keys_exhausted_until = 0.0

def get_llm_client():

    if USE_OPENROUTER:

        from openai import OpenAI

        return OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

    from groq import Groq

    if not GROQ_KEYS:
        raise RuntimeError(
            "No OpenRouter or Groq API key configured."
        )

    key = GROQ_KEYS[_key_idx % len(GROQ_KEYS)]

    return Groq(api_key=key)

def rotate_key():
    global _key_idx
    if GROQ_KEYS:
        _key_idx = (_key_idx + 1) % len(GROQ_KEYS)

def _parse_retry_after(err):
    """Extract wait time from Groq rate-limit error message (e.g. 'try again in 2m45.024s')."""
    m = re.search(r'try again in\s+(?:(\d+)m)?([\d.]+)s', str(err), re.IGNORECASE)
    if m:
        minutes = int(m.group(1) or 0)
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    return None

def _is_tpd_limit(err):
    """Check if this is a daily token limit (TPD) vs per-minute/per-request limit."""
    m = str(err).lower()
    return 'tokens per day' in m or 'tpd' in m

def _groq_cooldown():
    """Sleep between API calls. Override with GROQ_COOLDOWN (seconds). Default 0.5."""
    try:
        delay = float(os.environ.get("GROQ_COOLDOWN", "0.5"))
    except ValueError:
        delay = 0.5
    if delay > 0:
        time.sleep(delay)
# ---------------------------------------------------------------------------
# Provider-specific model mapping
# ---------------------------------------------------------------------------
def resolve_model(model: str) -> str:

    if USE_OPENROUTER:

        mapping = {
            "llama-3.3-70b-versatile":
                "meta-llama/llama-3.3-70b-instruct",

            "openai/gpt-oss-120b":
                "openai/gpt-oss-120b",
        }

        return mapping.get(model, model)

    return model

# --------------------------------------------------------------------------- #
# Generation with rate-limit retry + caching
# --------------------------------------------------------------------------- #
_gen_cache: Dict[str, str] = {}   # hashed keys -> persistable across disconnects

def _is_transient(err: Exception) -> bool:
    m = str(err).lower()
    return any(t in m for t in ("429", "rate limit", "timeout", "timed out", "503", "502",
                                "overloaded", "temporarily unavailable", "connection reset",
                                "connection aborted"))

def make_generator(model: str) -> Callable[[str, List[str]], str]:
    if not os.environ.get("USE_SYNTHETIC"):
        def gen(question: str, contexts: List[str]) -> str:
            global _keys_exhausted_until
            ck = _ckey(model, question, list(contexts))
            if ck in _gen_cache:
                return _gen_cache[ck]
            # If we know all keys are TPD-exhausted, wait or skip
            if _keys_exhausted_until > time.time():
                wait = _keys_exhausted_until - time.time()
                if wait > 300:  # More than 5 min — skip gracefully
                    print(f"[gen] TPD exhausted, {wait:.0f}s remaining — returning fallback for this sample")
                    out = contexts[0] if contexts else REJECTION
                    _gen_cache[ck] = out
                    return out
                print(f"[gen] TPD cooldown: waiting {wait:.0f}s for quota reset...")
                time.sleep(wait + 2)
                _keys_exhausted_until = 0.0
            ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
            prompt = (f"Answer ONLY from the context. If insufficient, reply exactly:\n'{REJECTION}'.\n\n"
                      f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:")
            delay = 5.0
            max_attempts = MAX_RETRIES
            for attempt in range(max_attempts):
                try:
                    r = get_llm_client().chat.completions.create(
                        model=resolve_model(model),
                        temperature=0, max_tokens=GEN_MAX_TOKENS,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    out = (r.choices[0].message.content or "").strip()
                    _gen_cache[ck] = out
                    _groq_cooldown()
                    return out
                except Exception as e:
                    if attempt < max_attempts - 1 and _is_transient(e):
                        if _is_tpd_limit(e):
                            retry_after = _parse_retry_after(e) or 180.0
                            rotate_key()
                            if attempt >= len(GROQ_KEYS):
                                _keys_exhausted_until = time.time() + retry_after
                                print(f"[gen] All {len(GROQ_KEYS)} keys TPD-exhausted. "
                                      f"Waiting {retry_after:.0f}s for quota reset...")
                                time.sleep(min(retry_after + 2, 300))
                                continue
                            time.sleep(min(delay, 10)); continue
                        elif "429" in str(e).lower() or "rate limit" in str(e).lower():
                            rotate_key()
                            retry_after = _parse_retry_after(e)
                            wait = retry_after if retry_after else delay
                            time.sleep(wait); delay = min(delay * 2, 120.0); continue
                        time.sleep(delay)
                        delay = min(delay * 2, 120.0)
                        continue
                    # Final attempt failed — gracefully degrade instead of crashing
                    print(f"[gen] FAILED after {attempt + 1} attempts: {type(e).__name__}: {str(e)[:200]}")
                    out = contexts[0] if contexts else REJECTION
                    _gen_cache[ck] = out
                    return out
        return gen

    def gen_offline(question: str, contexts: List[str]) -> str:
        if not contexts:
            return REJECTION
        ck = _ckey("offline", question, list(contexts))
        if ck in _gen_cache:
            return _gen_cache[ck]
        out = sorted(contexts, key=lambda c: -coverage(question, c))[0]
        _gen_cache[ck] = out
        return out
    return gen_offline

# --------------------------------------------------------------------------- #
# Verbatim RAGBench Appendix 7.4 judge prompt (from Final Working notebooks)
# --------------------------------------------------------------------------- #
JUDGE_PROMPT = """I asked someone to answer a question based on one or more
documents. Your task is to review their response and assess whether or not each
sentence in that response is supported by text in the documents. And if so, which
sentences in the documents provide that support. You will also tell me which
of the documents contain useful information for answering the question, and
which of the documents the answer was sourced from.

Here are the documents, each of which is split into sentences. Alongside each
sentence is associated key, such as '0a.' or '0b.' that you can use to refer
to it:

```
{documents}
```

The question was:
```
{question}
```

Here is their response, split into sentences. Alongside each sentence is
associated key, such as 'a.' or 'b.' that you can use to refer to it. Note
that these keys are unique to the response, and are not related to the keys
in the documents:

```
{answer}
```

You must respond with a JSON object matching this schema:

{{
  "relevance_explanation": string,
  "all_relevant_sentence_keys": [string],
  "overall_supported_explanation": string,
  "overall_supported": boolean,
  "sentence_support_information": [
    {{
      "response_sentence_key": string,
      "explanation": string,
      "supporting_sentence_keys": [string],
      "fully_supported": boolean
    }}
  ],
  "all_utilized_sentence_keys": [string]
}}

The relevance_explanation field is a string explaining which documents
contain useful information for answering the question. Walk through the
information in the documents step by step and how it is useful for
answering the question.

The all_relevant_sentence_keys field is a list of all document sentence
keys (e.g. '0a') that are relevant to the question. Include every sentence
that is useful and relevant to the question, even if it was not used in the
response, or if only parts of the sentence are useful. Base this judgement
only on the documents and the question -- ignore the response entirely when
deciding relevance. Leave out sentences that could be removed from the
document without affecting someone's ability to answer the question.

The overall_supported_explanation field is a string explaining why the
response *as a whole* is or is not supported by the documents. Walk through
each claim in the response individually and assess its support (or lack of
support) in the documents one at a time, before drawing any conclusion about
the response as a whole.

The overall_supported field is a boolean reflecting the conclusion you
reached at the end of overall_supported_explanation: whether the response as
a whole is supported by the documents.

The sentence_support_information field is a list of objects, one for each sentence
in the response. Each object MUST have the following fields:
- response_sentence_key: a string identifying the sentence in the response. This
key is the same as the one used in the response above.- explanation: a string
explaining why the sentence is or is not supported by the documents.
- supporting_sentence_keys: keys (e.g. '0a') of sentences from the documents that
support the response sentence. If the sentence is not supported, this list MUST
be empty. If the sentence is supported, this list MUST contain one or more keys.
In special cases where the sentence is supported, but not by any specific sentence,
you can use the string "supported_without_sentence" to indicate that the sentence
is generally supported by the documents. Consider cases where the sentence is
expressing inability to answer the question due to lack of relevant information
in the provided contex as "supported_without_sentence". In cases
where the sentence is making a general statement (e.g. outlining the steps to produce
an answer, or summarizing previously stated sentences, or a transition sentence), use
the sting "general".In cases where the sentence is correctly stating a well-known fact,
like a mathematical formula, use the string "well_known_fact". In cases where the
sentence is performing numerical reasoning (e.g. addition, multiplication), use
the string "numerical_reasoning".
- fully_supported: a boolean indicating whether the sentence is fully supported by
the documents.
  - This value should reflect the conclusion  you drew at the end of your step-by-step
    breakdown in explanation.
  - If supporting_sentence_keys is an empty list, then fully_supported must be false.
  - Otherwise, use fully_supported to clarify whether everything in the response
  sentence is fully supported by the document text indicated in supporting_sentence_keys
  (fully_supported = true), or whether the sentence is only partially or incompletely
  supported by that document text (fully_supported = false).

The all_utilized_sentence_keys field is a list of all sentences keys (e.g. '0a') that
were used to construct the answer. Include every sentence that either directly supported
the answer, or was implicitly used to construct the answer, even if it was not used
in its entirety. Omit sentences that were not used, and could have been removed from
the documents without affecting the answer.

You must respond with a valid JSON string. Use escapes for quotes, e.g. '\\"', and
newlines, e.g. '\\n'. Do not write anything before or after the JSON string. Do not
wrap the JSON string in backticks like ''' or '''json.

As a reminder: your task is to review the response and assess which documents contain
useful information pertaining to the question, and how each sentence in the response
is supported by the text in the documents."""

# --------------------------------------------------------------------------- #
# Sentence-key helpers + judge call with retry
# --------------------------------------------------------------------------- #
def _skey(i: int) -> str:
    s = ""
    i += 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s

def sentence_keyed(chunks: List[Dict]) -> List[Tuple[str, str]]:
    keyed = []
    for d_i, ch in enumerate(chunks):
        for s_i, sent in enumerate(ch["sentences"]):
            keyed.append((f"{d_i}{_skey(s_i)}", sent))
    return keyed

_judge_cache: Dict[str, Any] = {}   # hashed keys -> persistable across disconnects
_JUDGE_MISS = object()

def _judge_completion(resolved_model: str, prompt: str) -> str:
    """
    One judge call at temperature 0 with response_format={"type":"json_object"}.

    JSON mode is what stops the judge wrapping its verdict in prose or fences —
    it constrains the OUTPUT only, the Appendix 7.4 prompt text is sent verbatim
    and untouched. If a provider/model rejects the parameter we remember that and
    fall back to plain completion for the rest of the run instead of failing.
    """
    kwargs = dict(model=resolved_model, temperature=0, max_tokens=JUDGE_MAX_TOKENS,
                  messages=[{"role": "user", "content": prompt}])
    use_json = JUDGE_JSON_MODE and resolved_model not in _no_json_mode
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        r = get_llm_client().chat.completions.create(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        if use_json and ("response_format" in msg or "json_object" in msg
                         or "json mode" in msg or "unsupported" in msg):
            print(f"[judge] {resolved_model} rejected JSON mode — falling back to plain completion")
            _no_json_mode.add(resolved_model)
            kwargs.pop("response_format", None)
            r = get_llm_client().chat.completions.create(**kwargs)
        else:
            raise
    return (r.choices[0].message.content or "").strip()


def trace_via_judge(question: str, keyed: List[Tuple[str, str]], answer: str, model: str,
                    max_retries: int = MAX_RETRIES) -> Optional[dict]:
    if os.environ.get("USE_SYNTHETIC"):
        return None
    global _keys_exhausted_until
    ck = _ckey(model, question, [k for k, _ in keyed], answer)
    hit = _judge_cache.get(ck, _JUDGE_MISS)
    if hit is not _JUDGE_MISS:
        return hit

    # Skip if TPD-exhausted and wait is too long
    if _keys_exhausted_until > time.time():
        wait = _keys_exhausted_until - time.time()
        if wait > 300:
            print(f"[judge] TPD exhausted ({wait:.0f}s left) — skipping judge for this sample")
            _judge_cache[ck] = None; return None
        print(f"[judge] TPD cooldown: waiting {wait:.0f}s...")
        time.sleep(wait + 2)
        _keys_exhausted_until = 0.0

    docs = "\n".join(f"{k}. {s}" for k, s in keyed)
    ans = "\n".join(f"{_skey(i)}. {s}" for i, s in enumerate(split_sentences(answer)))
    prompt = JUDGE_PROMPT.format(documents=docs, question=question, answer=ans)
    delay = 5.0
    for attempt in range(max_retries):
        try:
            raw = _judge_completion(resolve_model(model), prompt)
            if "```" in raw:
                raw = raw.split("```")[1]
                raw = raw[4:] if raw.startswith("json") else raw
            raw = raw[raw.find("{"): raw.rfind("}") + 1]
            try:
                out = json.loads(raw)
            except Exception:
                print("=" * 80)
                print("RAW JUDGE RESPONSE")
                print(raw)
                print("=" * 80)
                raise
            _judge_cache[ck] = out
            _groq_cooldown()
            return out
        except Exception as e:
            if attempt < max_retries - 1 and _is_transient(e):
                if _is_tpd_limit(e):
                    retry_after = _parse_retry_after(e) or 180.0
                    rotate_key()
                    if attempt >= len(GROQ_KEYS):
                        _keys_exhausted_until = time.time() + retry_after
                        print(f"[judge] All keys TPD-exhausted. Waiting {retry_after:.0f}s...")
                        time.sleep(min(retry_after + 2, 300)); continue
                    time.sleep(min(delay, 10)); continue
                elif '429' in str(e).lower() or 'rate limit' in str(e).lower():
                    rotate_key()
                    retry_after = _parse_retry_after(e)
                    wait = retry_after if retry_after else delay
                    time.sleep(wait); delay = min(delay * 2, 120.0); continue
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
                continue
            print(f"[judge] FAILED after {attempt + 1} attempt(s): {type(e).__name__}: {str(e)[:140]}")
            _judge_cache[ck] = None
            return None
    _judge_cache[ck] = None
    return None

def compute_trace(question: str, chunks: List[Dict], answer: str, reference: str, model: str) -> Dict[str, float]:
    keyed = sentence_keyed(chunks)
    all_keys = [k for k, _ in keyed]
    total = len(all_keys)
    if total == 0:
        return {"context_relevance": 0.0, "context_utilization": 0.0, "completeness": 0.0,
                "adherence": 0.0, "_judge_used": 0.0}
    j = trace_via_judge(question, keyed, answer, model)
    if j is not None:
        real = set(all_keys)
        relevant = set(j.get("all_relevant_sentence_keys", [])) & real
        utilized = set(j.get("all_utilized_sentence_keys", [])) & real
        supp = j.get("sentence_support_information", [])
        adherence = (sum(1 for s in supp if s.get("fully_supported")) / len(supp)) if supp \
            else (1.0 if j.get("overall_supported") else 0.0)
        return {"context_relevance": round(len(relevant) / total, 4),
                "context_utilization": round(len(utilized) / total, 4),
                "completeness": round(len(relevant & utilized) / len(relevant), 4) if relevant else 0.0,
                "adherence": round(float(adherence), 4),
                "_judge_used": 1.0}
    oracle = question + " " + (reference or "")
    relevant = {k for (k, s) in keyed if coverage(s, oracle) >= 0.34 or coverage(oracle, s) >= 0.5}
    # utilized: the answer is largely drawn from sentence s (coverage(answer, s)) OR the
    # short sentence is fully absorbed by the answer. Using only coverage(s, answer) biased
    # the metric against long context sentences.
    utilized = {k for (k, s) in keyed if coverage(answer, s) >= 0.5 or coverage(s, answer) >= 0.5}
    ans_sents = split_sentences(answer)
    grounded = sum(1 for a in ans_sents if any(coverage(a, s) >= 0.4 for _, s in keyed))
    return {"context_relevance": round(len(relevant) / total, 4),
            "context_utilization": round(len(utilized) / total, 4),
            "completeness": round(len(relevant & utilized) / len(relevant), 4) if relevant else 0.0,
            "adherence": round(grounded / len(ans_sents), 4) if ans_sents else (1.0 if REJECTION in answer.lower() else 0.0),
            "_judge_used": 0.0}

# --------------------------------------------------------------------------- #
# AUROC / RMSE against RAGBench gold labels
# --------------------------------------------------------------------------- #
def gold_metrics(row: dict) -> Dict[str, Optional[float]]:
    def g(*names):
        for n in names:
            if n in row and row[n] is not None:
                try: return float(row[n])
                except Exception: pass
        return None
    return {
        "context_relevance": g("relevance_score", "gpt3_context_relevance", "context_relevance"),
        "context_utilization": g("utilization_score", "gpt3_utilization", "context_utilization"),
        "completeness": g("completeness_score", "adherence_completeness", "completeness"),
        "adherence": g("adherence_score", "gpt3_adherence", "adherence"),
    }

def gold_eval(preds: List[dict], golds: List[dict]) -> Dict[str, float]:
    """
    Full TRACe scoring vs RAGBench reference labels.

      Relevance / Utilization / Completeness -> RMSE + MAE  (continuous, per paper)
      Adherence                              -> AUROC (primary) + RMSE + accuracy
      ref_*                                  -> mean of the reference labels themselves
                                                (the 'GT ref' column in the sheet)
    Everything is computed with numpy so a missing sklearn only costs AUROC.
    """
    out: Dict[str, float] = {}
    cont = ("context_relevance", "context_utilization", "completeness")

    for m in cont + ("adherence",):
        pairs = [(float(p[m]), float(g[m])) for p, g in zip(preds, golds)
                 if g.get(m) is not None and p.get(m) is not None]
        out[f"n_gold_{m}"] = len(pairs)
        if not pairs:
            continue
        pv = np.asarray([a for a, _ in pairs], dtype=float)
        gv = np.asarray([b for _, b in pairs], dtype=float)
        out[f"ref_{m}"] = round(float(gv.mean()), 4)
        out[f"{m}_rmse"] = round(float(np.sqrt(np.mean((pv - gv) ** 2))), 4)
        out[f"{m}_mae"] = round(float(np.mean(np.abs(pv - gv))), 4)

    adh = [(float(p["adherence"]), float(g["adherence"])) for p, g in zip(preds, golds)
           if g.get("adherence") is not None]
    if adh:
        scores = np.asarray([a for a, _ in adh], dtype=float)
        labels = np.asarray([int(round(b)) for _, b in adh], dtype=int)
        out["adherence_pos_rate"] = round(float(labels.mean()), 4)
        out["adherence_accuracy"] = round(float(((scores >= 0.5).astype(int) == labels).mean()), 4)
        if len(set(labels.tolist())) > 1:
            try:
                from sklearn.metrics import roc_auc_score
                out["adherence_auroc"] = round(float(roc_auc_score(labels, scores)), 4)
            except Exception:
                # rank-based fallback so the primary metric never silently vanishes
                order = np.argsort(scores, kind="mergesort")
                ranks = np.empty_like(order, dtype=float)
                ranks[order] = np.arange(1, len(scores) + 1)
                npos, nneg = labels.sum(), len(labels) - labels.sum()
                if npos and nneg:
                    out["adherence_auroc"] = round(
                        float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)), 4)
        else:
            out["adherence_auroc"] = ""   # single-class slice: AUROC undefined, not 0.5
    return out

# --------------------------------------------------------------------------- #
# Config + data loading
# --------------------------------------------------------------------------- #
@dataclass
class DomainConfig:
    domain: str
    dataset_config: str
    embedders: Tuple[str, ...]
    chunk_configs: Tuple[dict, ...]
    retrievals: Tuple[str, ...] = ("dense", "hybrid")
    rerank_options: Tuple[str, ...] = ("none", "cross_encoder")
    context_orders: Tuple[str, ...] = ("forward", "reverse", "sides")
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gen_model: str = "llama-3.3-70b-versatile"
    gen_models: Tuple[str, ...] = ()
    judge_model: str = "openai/gpt-oss-120b"
    k_retrieve: int = 12
    k_final: int = 5
    n_examples: int = 200
    seed: int = 42
    results_csv: str = "results.csv"

    def __post_init__(self):
        # RULE: the judge must be a DIFFERENT and STRONGER model than the generator.
        # No self-grading, and a weaker judge cannot reliably grade a stronger generator.
        _RANK = {"allam-2-7b": 1, "llama-3.1-8b-instant": 2, "openai/gpt-oss-20b": 3,
                 "qwen/qwen3.6-27b": 4, "llama-3.3-70b-versatile": 5, "openai/gpt-oss-120b": 6}
        for g in (self.gen_models or (self.gen_model,)):
            if self.judge_model == g:
                raise ValueError(f"[{self.domain}] judge_model must DIFFER from generator "
                                 f"'{g}' - a model cannot grade its own output.")
            gr, jr = _RANK.get(g), _RANK.get(self.judge_model)
            if gr and jr and jr <= gr:
                raise ValueError(f"[{self.domain}] judge '{self.judge_model}' (tier {jr}) must be "
                                 f"STRONGER than generator '{g}' (tier {gr}).")

def apply_runtime_overrides(cfg):
    if os.environ.get("N_EXAMPLES"):
        try: cfg.n_examples = max(1, int(os.environ["N_EXAMPLES"]))
        except ValueError: pass
    if os.environ.get("DENSE_ONLY", "").strip() in ("1", "true", "True", "yes"):
        cfg.retrievals = ("dense",)
        print(f"[{cfg.domain}] DENSE_ONLY=1 -> retrievals=('dense',)")
    if os.environ.get("QUICK_SWEEP", "").strip() in ("1", "true", "True", "yes"):
        cfg.embedders = cfg.embedders[:1]
        cfg.chunk_configs = cfg.chunk_configs[:1]
        cfg.retrievals = ("dense",)
        cfg.rerank_options = ("none",)
        cfg.context_orders = ("forward",)
        print(f"[{cfg.domain}] QUICK_SWEEP=1 -> emb={len(cfg.embedders)} chunk={len(cfg.chunk_configs)} "
              f"ret={cfg.retrievals} rr={cfg.rerank_options} order={cfg.context_orders}")
    return cfg

def load_examples(cfg, synthetic_fn):
    if not os.environ.get("USE_SYNTHETIC"):
        try:
            from datasets import load_dataset
            import ast
            ds = load_dataset("rungalileo/ragbench", cfg.dataset_config, split="test")
            n = min(cfg.n_examples, len(ds))
            print(f"[data] HuggingFace rungalileo/ragbench/{cfg.dataset_config} test={len(ds)} -> sampling {n}")
            idx = sorted(random.Random(cfg.seed).sample(range(len(ds)), n))
            out = []
            for row in ds.select(idx):
                docs = row["documents"]
                if isinstance(docs, str):
                    try: docs = ast.literal_eval(docs)
                    except Exception: docs = [docs]
                out.append({"question": row["question"],
                            "documents": [{"doc_id": f"d{i}", "text": t} for i, t in enumerate(docs)],
                            "reference": row.get("response", "") or "",
                            "gold": gold_metrics(row)})
            if out: return out
        except Exception as e:
            print(f"[data] RAGBench load failed ({type(e).__name__}: {e}); using synthetic set.")
    return synthetic_fn()

# --------------------------------------------------------------------------- #
# The sweep — one CS-### experiment per config, resumable at both levels
# --------------------------------------------------------------------------- #
METRIC_KEYS = ["context_relevance", "context_utilization", "completeness", "adherence"]

RESULT_FIELDS = [
    "exp_id", "timestamp", "domain", "status",
    # config
    "embedder", "chunk_config", "gen_model", "judge_model", "retrieval", "rerank",
    "context_order", "k_retrieve", "k_final", "seed", "n_examples",
    # predicted TRACe means
    "context_relevance", "context_utilization", "completeness", "adherence",
    # reference (gold) means
    "ref_context_relevance", "ref_context_utilization", "ref_completeness", "ref_adherence",
    # error vs reference
    "context_relevance_rmse", "context_utilization_rmse", "completeness_rmse", "adherence_rmse",
    "context_relevance_mae", "context_utilization_mae", "completeness_mae", "adherence_mae",
    # adherence classification
    "adherence_auroc", "adherence_accuracy", "adherence_pos_rate",
    # bookkeeping
    "n_gold_adherence", "judge_used", "runtime_sec",
]


def build_plan(cfg) -> List[dict]:
    """Enumerate the sweep and attach a stable CS-### id to every config."""
    gen_models = cfg.gen_models or (cfg.gen_model,)
    plan = []
    for emb in cfg.embedders:
        for cc in cfg.chunk_configs:
            for gen in gen_models:
                for ret in cfg.retrievals:
                    for rr in cfg.rerank_options:
                        for order in cfg.context_orders:
                            sig = "|".join([emb, cc["label"], gen, ret, rr, order,
                                            f"k{cfg.k_retrieve}/{cfg.k_final}",
                                            f"judge:{cfg.judge_model}", f"seed:{cfg.seed}"])
                            plan.append({"exp_id": registry_exp_id(sig), "signature": sig,
                                         "embedder": emb, "chunk": cc, "gen_model": gen,
                                         "retrieval": ret, "rerank": rr, "order": order})
    return plan


def _aggregate_row(cfg, spec, records, runtime_sec) -> dict:
    preds = [r["pred"] for r in records]
    golds = [r.get("gold", {}) or {} for r in records]
    row = {
        "exp_id": spec["exp_id"], "timestamp": _now(), "domain": cfg.domain, "status": "done",
        "embedder": spec["embedder"].split("/")[-1], "chunk_config": spec["chunk"]["label"],
        "gen_model": spec["gen_model"], "judge_model": cfg.judge_model,
        "retrieval": spec["retrieval"], "rerank": spec["rerank"], "context_order": spec["order"],
        "k_retrieve": cfg.k_retrieve, "k_final": cfg.k_final, "seed": cfg.seed,
        "n_examples": len(preds), "runtime_sec": round(runtime_sec, 1),
        "judge_used": round(float(np.mean([p.get("_judge_used", 0.0) for p in preds])), 3),
    }
    for m in METRIC_KEYS:
        row[m] = round(float(np.mean([p[m] for p in preds])), 4)
    row.update(gold_eval(preds, golds))
    return row


def run_one(cfg, spec, prepared, fingerprint) -> Optional[dict]:
    """Run (or finish) a single CS-### experiment with checkpoint + git cadence."""
    exp_id, total = spec["exp_id"], len(prepared)
    records, done, meta = load_checkpoint(exp_id, fingerprint)
    runtime = float(meta.get("runtime_sec", 0.0))
    ckpt_count = int(meta.get("ckpt_count", 0))
    meta_base = {"fingerprint": fingerprint, "signature": spec["signature"],
                 "config": {k: (v["label"] if k == "chunk" else v)
                            for k, v in spec.items() if k != "signature"}}

    print(f"\n=== {exp_id} | emb={spec['embedder'].split('/')[-1]} chunk={spec['chunk']['label']} "
          f"gen={spec['gen_model'].split('/')[-1]} ret={spec['retrieval']} rr={spec['rerank']} "
          f"order={spec['order']} ===")
    print(f"*** N_EXAMPLES={total} | resuming from {len(done)} ***")
    time.sleep(GUARD_PAUSE)   # guard pause: verify N before a long run (GUARD_PAUSE=0 to skip)

    generator = make_generator(spec["gen_model"])
    save_progress(exp_id, len(done), total, status="running", fingerprint=fingerprint)
    t0 = time.time()
    try:
        for i, (ex, corpus, mat, qvec, bm25) in enumerate(prepared):
            if i in done:
                continue
            ranked = (hybrid_rrf_rank(ex["question"], qvec, mat, bm25)
                      if spec["retrieval"] == "hybrid" else dense_rank(qvec, mat))
            got = [corpus[j] for j in ranked[:min(cfg.k_retrieve, len(corpus))]]
            got = (cross_encoder_rerank(ex["question"], got, cfg.reranker_model, cfg.k_final)
                   if spec["rerank"] == "cross_encoder" else got[:cfg.k_final])
            got = order_context(got, spec["order"])
            answer = generator(ex["question"], [c["text"] for c in got])
            pred = compute_trace(ex["question"], got, answer, ex["reference"], cfg.judge_model)

            records.append({"idx": i, "question": ex["question"], "answer": answer,
                            "pred": pred, "gold": ex.get("gold", {})})
            done.add(i)

            if len(done) % CHECKPOINT_EVERY == 0:
                ckpt_count += 1
                elapsed = runtime + (time.time() - t0)
                save_checkpoint(exp_id, records, {**meta_base, "runtime_sec": round(elapsed, 1),
                                                 "ckpt_count": ckpt_count})
                save_progress(exp_id, len(done), total, status="running", fingerprint=fingerprint)
                save_cache()
                print(f"  [ckpt {ckpt_count}] {exp_id} {len(done)}/{total} "
                      f"({100*len(done)/total:.0f}%) elapsed={elapsed:.0f}s")
                if ckpt_count % GIT_PUSH_EVERY_N_CKPT == 0:
                    git_sync(f"checkpoint {exp_id} {len(done)}/{total}", exp_id=exp_id)

    except KeyboardInterrupt:
        elapsed = runtime + (time.time() - t0)
        save_checkpoint(exp_id, records, {**meta_base, "runtime_sec": round(elapsed, 1),
                                          "ckpt_count": ckpt_count})
        save_progress(exp_id, len(done), total, status="paused", fingerprint=fingerprint)
        save_cache()
        git_sync(f"paused {exp_id} at {len(done)}/{total}", exp_id=exp_id)
        print(f"[paused] {exp_id} at {len(done)}/{total} — rerun resumes from here")
        raise
    except Exception as e:
        elapsed = runtime + (time.time() - t0)
        save_checkpoint(exp_id, records, {**meta_base, "runtime_sec": round(elapsed, 1),
                                          "ckpt_count": ckpt_count})
        save_progress(exp_id, len(done), total, status="failed", fingerprint=fingerprint,
                      extra={"error": f"{type(e).__name__}: {str(e)[:400]}"})
        save_cache()
        print(f"[FAIL] {exp_id} at {len(done)}/{total}: {type(e).__name__}: {str(e)[:200]}")
        return None

    runtime += time.time() - t0
    if not records:
        save_progress(exp_id, 0, total, status="failed", fingerprint=fingerprint,
                      extra={"error": "no records"})
        return None

    records.sort(key=lambda r: r["idx"])
    save_checkpoint(exp_id, records, {**meta_base, "runtime_sec": round(runtime, 1),
                                      "ckpt_count": ckpt_count})
    row = _aggregate_row(cfg, spec, records, runtime)
    _write_all(_rel(exp_id, "summary.json"), row)
    if not csv_has_exp(cfg.results_csv, exp_id, len(records)):
        append_results_csv(row, RESULT_FIELDS, cfg.results_csv)
    save_progress(exp_id, len(done), total, status="done", fingerprint=fingerprint,
                  extra={"summary": row})
    save_cache()
    git_sync(f"complete {exp_id} n={len(records)} adh_auroc={row.get('adherence_auroc','na')}",
             exp_id=exp_id)

    print(f"  -> {exp_id} rel={row['context_relevance']} util={row['context_utilization']} "
          f"comp={row['completeness']} adh={row['adherence']} | "
          f"RMSE rel={row.get('context_relevance_rmse','-')} util={row.get('context_utilization_rmse','-')} "
          f"comp={row.get('completeness_rmse','-')} | AUROC(adh)={row.get('adherence_auroc','-')} "
          f"| {runtime:.0f}s")
    return row


def run_experiment(cfg, synthetic_fn):
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    cfg = apply_runtime_overrides(cfg)
    examples = load_examples(cfg, synthetic_fn)
    n = len(examples)
    fingerprint = _fingerprint(examples)
    mode = "synthetic" if os.environ.get("USE_SYNTHETIC") else cfg.dataset_config
    have_gold = any(any(v is not None for v in ex.get("gold", {}).values()) for ex in examples)
    load_cache()

    plan = build_plan(cfg)
    only = {s.strip().upper() for s in os.environ.get("EXP_ONLY", "").split(",") if s.strip()}
    if only:
        plan = [s for s in plan if s["exp_id"] in only]
    todo, skipped = [], []
    for s in plan:
        (skipped if experiment_completed(s["exp_id"], n, fingerprint) else todo).append(s)

    print(f"[{cfg.domain}] {n} examples ({mode}) · device={DEVICE} · gold_labels={have_gold} "
          f"· judge={cfg.judge_model}")
    print(f"[{cfg.domain}] plan = {len(plan)} experiments "
          f"({plan[0]['exp_id'] if plan else '-'} .. {plan[-1]['exp_id'] if plan else '-'}) "
          f"| done={len(skipped)} | to run={len(todo)}")
    if skipped:
        print(f"[{cfg.domain}] skipping completed: {', '.join(s['exp_id'] for s in skipped)}")
    if not todo:
        print(f"[{cfg.domain}] nothing left to run (set FORCE_RERUN=1 to redo)")
        return read_results_csv(cfg.results_csv)

    need_bm25 = any(s["retrieval"] == "hybrid" for s in todo)
    groups: Dict[tuple, List[dict]] = {}
    for s in todo:                       # group so embeddings are built once per (emb, chunk)
        groups.setdefault((s["embedder"], s["chunk"]["label"]), []).append(s)

    rows = []
    for (emb_name, cc_label), specs in groups.items():
        cc = specs[0]["chunk"]
        print(f"\n[{cfg.domain}] building corpus+embeddings: {emb_name.split('/')[-1]} / {cc_label} "
              f"for {len(specs)} experiment(s)")
        prepared, embedder = [], None
        for ex in examples:
            corpus = build_corpus(ex["documents"], cc)
            if not corpus:
                continue
            texts = [c["text"] for c in corpus]
            if embedder is None:
                embedder = get_embedder(emb_name, corpus_hint=texts)
                if embedder.backend != "st" and not os.environ.get("USE_SYNTHETIC"):
                    _log(f"[{cfg.domain}] *** WARNING: '{emb_name}' fell back to "
                         f"{embedder.backend.upper()} (sentence-transformers missing/failed). "
                         f"Rows are labeled '{emb_name.split('/')[-1]}' but are NOT this model. ***")
            mat = embedder.encode(texts)
            qvec = embedder.encode([ex["question"]])[0]
            bm25 = make_bm25(texts) if need_bm25 else None
            prepared.append((ex, corpus, mat, qvec, bm25))

        for spec in specs:
            row = run_one(cfg, spec, prepared, fingerprint)
            if row:
                rows.append(row)

    git_sync(f"sweep pass complete: {len(rows)} experiments written")
    all_rows = read_results_csv(cfg.results_csv)
    print(f"\n[{cfg.domain}] this pass: {len(rows)} experiments | csv total: {len(all_rows)} rows "
          f"-> {_csv_paths(cfg.results_csv)[0]}")
    if all_rows:
        same_n = [r for r in all_rows if str(r.get("n_examples")) == str(n)] or all_rows
        def _score(r):
            try:
                return sum(float(r[m]) for m in METRIC_KEYS)
            except Exception:
                return -1.0
        best = max(same_n, key=_score)   # compare like with like (N=25 vs N=200)
        print(f"[{cfg.domain}] best so far: {best.get('exp_id')} emb={best.get('embedder')} "
              f"chunk={best.get('chunk_config')} gen={best.get('gen_model')} "
              f"ret={best.get('retrieval')} rr={best.get('rerank')} order={best.get('context_order')} "
              f"| AUROC(adh)={best.get('adherence_auroc','-')}")
    return rows


# =========================================================================== #
# CUSTOMER SUPPORT DOMAIN CONFIG + SYNTHETIC DATA
# =========================================================================== #
def synthetic():
    docs = [
        {"doc_id": "d0", "text": ("To reset your password, open the sign in page and click the Forgot password link below the "
                                  "login form. Enter the email address associated with your account and submit the form. You "
                                  "will receive an email containing a secure reset link, which is valid for one hour. Click the "
                                  "link and choose a new password that meets the strength requirements.")},
        {"doc_id": "d1", "text": ("If a network connection drops, first restart the router and check the cables. Run the built in "
                                  "network troubleshooter to reset the adapter. If the problem persists, update the network "
                                  "driver from the vendor site.")},
        {"doc_id": "d2", "text": ("Two factor authentication adds a second layer of security. After entering your password you "
                                  "provide a one time code from an authenticator app. Enable it under the Security section of "
                                  "Account Settings.")},
        {"doc_id": "d3", "text": ("Billing is handled from the Billing page. We accept major credit cards and support monthly "
                                  "renewals. A failed payment enters a grace period before features are suspended.")},
    ]
    return [
        {"question": "How do I reset my account password?", "documents": docs,
         "reference": "Open the sign in page, click Forgot password, enter your email, and use the secure reset link.",
         "gold": {"context_relevance": 0.30, "context_utilization": 0.28, "completeness": 1.0, "adherence": 1}},
        {"question": "How do I troubleshoot a network connection issue?", "documents": docs,
         "reference": "Restart the router, check cables, run the network troubleshooter, and update the network driver.",
         "gold": {"context_relevance": 0.26, "context_utilization": 0.24, "completeness": 1.0, "adherence": 1}},
        {"question": "How do I turn on two factor authentication?", "documents": docs,
         "reference": "Enable two factor authentication under the Security section of Account Settings.",
         "gold": {"context_relevance": 0.20, "context_utilization": 0.18, "completeness": 0.8, "adherence": 0}},
    ]

CFG = DomainConfig(
    domain="customer_support",
    dataset_config="techqa",
    embedders=("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"),
    chunk_configs=(
        {"label": "whole_doc",    "strategy": "whole_doc"},
        {"label": "sentence_3o1", "strategy": "sentence", "size": 3, "overlap": 1},
        {"label": "sliding_5o2",  "strategy": "sliding",  "size": 5, "overlap": 2},
        {"label": "fixed_120w",   "strategy": "fixed",    "size": 120, "overlap": 20},
    ),
    k_retrieve=10, k_final=5,
    results_csv="results_customer_support.csv",
)

if __name__ == "__main__":
    run_experiment(CFG, synthetic)
