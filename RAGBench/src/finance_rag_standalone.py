"""
finance_rag_standalone.py — Finance (RAGBench finqa) standalone pipeline.

Persistence / infrastructure layer is identical to the CS + GK sweeps:
  * stable experiment IDs from a config-signature registry (FIN-001 ... FIN-096)
  * dataset fingerprinting (completion is keyed on BOTH n_examples and fingerprint)
  * checkpoint + resume every CHECKPOINT_EVERY examples (local -> Drive -> GitHub)
  * persistent generation/judge caches with SHA1 keys (never tuples, never cached None)
  * append-only results CSV with newest-row-per-exp_id de-duplication at report time
  * OpenRouter primary transport, Groq fallback (key rotation), HF Inference last resort
  * EXP_ONLY / FORCE_RERUN targeted reruns
  * judge-outage guard: aborts an experiment instead of writing 200 heuristic-scored rows

UNCHANGED from the previous version of this file:
  * the RAGBench dataset loader
  * the verbatim RAGBench Appendix 7.4 judge prompt
  * TRACe metric computation (compute_trace / gold_eval)
  * the experiment matrix (dense AND hybrid retained -> 96 experiments)
  * all domain-specific retrieval / generation logic

Run:
  python finance_rag_standalone.py finance
  USE_SYNTHETIC=1 python finance_rag_standalone.py finance     (offline smoke test, no API keys)
  python finance_rag_standalone.py finance --report            (re-export the report CSVs only)
"""
from __future__ import annotations
import os
import re
import csv
import ast
import json
import math
import sys
import time
import random
import hashlib
import shutil
import subprocess
import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, Any

import numpy as np

# Force line-buffered stdout so progress shows while the sweep is running
# (Windows + piping otherwise looks like a silent hang / infinite loop).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _ts() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _plog(msg: str) -> None:
    """Progress log with a wall-clock stamp (long sweeps are unreadable without one)."""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


REJECTION = "i cannot answer the question because of insufficient information in documents"


# --------------------------------------------------------------------------- #
# Env helpers
# --------------------------------------------------------------------------- #
def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default)).strip())
    except Exception:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, "1" if default else "")).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


# --------------------------------------------------------------------------- #
# Paths — LOCAL is the fast scratch leg, DRIVE is the durable leg, GIT is the
# third leg. LOCAL_ROOT must NOT live on Drive or every save is written twice.
# --------------------------------------------------------------------------- #
LOCAL_ROOT = (os.environ.get("LOCAL_ROOT") or os.environ.get("SAVE_PATH") or "./_rag_local").rstrip("/")
DRIVE_ROOT = (os.environ.get("DRIVE_ROOT") or "").rstrip("/")
REPO_ROOT = (os.environ.get("REPO_ROOT") or "").rstrip("/")
ENABLE_GIT = _env_flag("ENABLE_GIT", False)
GIT_PAT = os.environ.get("GIT_PAT_V", "") or os.environ.get("GH_PAT_NEW", "")
REPO_OWNER = os.environ.get("REPO_OWNER", "veenulearns-lab")
REPO_NAME = os.environ.get("REPO_NAME", "RAGBench-Capstone-Batch26")

CHECKPOINT_EVERY = max(1, _env_int("CHECKPOINT_EVERY", 5))
GIT_PUSH_EVERY_N_CKPT = max(1, _env_int("GIT_PUSH_EVERY_N_CKPT", 5))
GUARD_PAUSE = _env_float("GUARD_PAUSE", 0.0)
JUDGE_ABORT_AFTER = _env_int("JUDGE_ABORT_AFTER", 10)
SAVE_ANSWERS = _env_flag("SAVE_ANSWERS", True)
MIN_VALID_FRAC = _env_float("MIN_VALID_FRAC", 0.80)


class Paths:
    """Everything the run reads or writes, for one domain."""

    def __init__(self, domain: str):
        self.domain = domain
        self.local = os.path.join(LOCAL_ROOT, domain)
        self.drive = os.path.join(DRIVE_ROOT, domain) if DRIVE_ROOT else ""
        self.ckpt_local = os.path.join(self.local, "ckpt")
        os.makedirs(self.ckpt_local, exist_ok=True)
        if self.drive:
            try:
                os.makedirs(os.path.join(self.drive, "ckpt"), exist_ok=True)
            except Exception as e:
                _log(f"[paths] WARNING: Drive leg unavailable ({type(e).__name__}: {e}); local only.")
                self.drive = ""

    # file names are stable so the notebook report cell can find them
    def f(self, name: str) -> str:
        return os.path.join(self.local, name)

    @property
    def results_csv(self) -> str:
        return self.f(f"results_{self.domain}.csv")

    @property
    def registry(self) -> str:
        return self.f(f"experiments_{self.domain}.json")

    @property
    def state(self) -> str:
        return self.f(f"state_{self.domain}.json")

    @property
    def gen_cache(self) -> str:
        return self.f("cache_gen.json")

    @property
    def judge_cache(self) -> str:
        return self.f("cache_judge.json")

    def ckpt(self, exp_id: str) -> str:
        return os.path.join(self.ckpt_local, f"{exp_id}.json")

    def mirror(self, local_path: str) -> None:
        """Copy one local file onto the Drive leg, preserving the sub-path."""
        if not self.drive:
            return
        try:
            rel = os.path.relpath(local_path, self.local)
            dst = os.path.join(self.drive, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(local_path, dst)
        except Exception as e:
            _log(f"[drive] WARNING: mirror failed for {os.path.basename(local_path)} "
                 f"({type(e).__name__}: {e})")


def _atomic_write(path: str, text: str) -> None:
    """Write via a temp file + rename so a killed Colab session cannot leave a
    half-written JSON that breaks the next resume."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _restore_from_drive(paths: Paths) -> None:
    """Fresh Colab VM: pull state/caches/checkpoints back from Drive before resuming."""
    if not paths.drive or not os.path.isdir(paths.drive):
        return
    restored = 0
    for root, _dirs, files in os.walk(paths.drive):
        for name in files:
            src = os.path.join(root, name)
            rel = os.path.relpath(src, paths.drive)
            dst = os.path.join(paths.local, rel)
            if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
                continue
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                restored += 1
            except Exception:
                pass
    if restored:
        _log(f"[drive] restored {restored} file(s) from {paths.drive}")


# --------------------------------------------------------------------------- #
# Git checkpoint sync (third leg)
# --------------------------------------------------------------------------- #
_git_fail_logged = False


def git_sync(paths: Paths, message: str) -> None:
    global _git_fail_logged
    if not (ENABLE_GIT and REPO_ROOT and GIT_PAT and os.path.isdir(os.path.join(REPO_ROOT, ".git"))):
        return
    try:
        dst = os.path.join(REPO_ROOT, "results", paths.domain)
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(paths.local):
            src = os.path.join(paths.local, name)
            if os.path.isfile(src) and not name.endswith(".tmp"):
                shutil.copy2(src, os.path.join(dst, name))
        remote = f"https://{GIT_PAT}@github.com/{REPO_OWNER}/{REPO_NAME}.git"
        run = lambda *a: subprocess.run(a, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
        run("git", "add", "-A", os.path.relpath(dst, REPO_ROOT))
        c = run("git", "-c", "user.email=colab@local", "-c", f"user.name={REPO_OWNER}",
                "commit", "-m", message)
        if "nothing to commit" in (c.stdout + c.stderr).lower():
            return
        p = run("git", "push", remote, "HEAD")
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout)[:200])
        _log(f"[git] pushed: {message}")
    except Exception as e:
        if not _git_fail_logged:
            _log(f"[git] WARNING: sync disabled for this run ({type(e).__name__}: {str(e)[:160]})")
            _git_fail_logged = True


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
# Chunking — every strategy, per document (provenance preserved)   [UNCHANGED]
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
# Retrieval — dense cosine + BM25 hybrid RRF                      [UNCHANGED]
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
# Reranking + context ordering                                    [UNCHANGED]
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


# =========================================================================== #
# TRANSPORT: OpenRouter primary -> Groq fallback -> HF Inference last resort
# =========================================================================== #
PROVIDER_ORDER = [p.strip().lower() for p in
                  os.environ.get("PROVIDER_ORDER", "openrouter,groq,hf").split(",") if p.strip()]

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
HF_TOKEN = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or "").strip()

GROQ_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
_single = os.environ.get("GROQ_API_KEY", "").strip()
if _single and _single not in GROQ_KEYS:
    GROQ_KEYS.insert(0, _single)

_key_idx = 0
_keys_exhausted_until = 0.0          # timestamp when ALL Groq keys' TPD resets
_provider_dead: Dict[str, str] = {}  # provider -> reason (hard-disabled for this run)
_provider_calls: Dict[str, int] = {}

# Canonical model name -> per-provider model id. Canonical names stay Groq-style
# because that is what every locked config and every results CSV already records.
MODEL_ROUTES: Dict[str, Dict[str, str]] = {
    "llama-3.3-70b-versatile": {
        "openrouter": "meta-llama/llama-3.3-70b-instruct",
        "groq": "llama-3.3-70b-versatile",
        "hf": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "llama-3.1-8b-instant": {
        "openrouter": "meta-llama/llama-3.1-8b-instruct",
        "groq": "llama-3.1-8b-instant",
        "hf": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "openai/gpt-oss-120b": {
        "openrouter": "openai/gpt-oss-120b",
        "groq": "openai/gpt-oss-120b",
        "hf": "openai/gpt-oss-120b",
    },
    "openai/gpt-oss-20b": {
        "openrouter": "openai/gpt-oss-20b",
        "groq": "openai/gpt-oss-20b",
        "hf": "openai/gpt-oss-20b",
    },
    "qwen/qwen3-32b": {
        "openrouter": "qwen/qwen3-32b",
        "groq": "qwen/qwen3-32b",
        "hf": "Qwen/Qwen3-32B",
    },
}


def _route(model: str, provider: str) -> str:
    return MODEL_ROUTES.get(model, {}).get(provider, model)


def _cooldown() -> None:
    delay = _env_float("GROQ_COOLDOWN", 0.5)
    if delay > 0:
        time.sleep(delay)


def _parse_retry_after(err) -> Optional[float]:
    """Extract wait time from a rate-limit error message (e.g. 'try again in 2m45.024s')."""
    m = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", str(err), re.IGNORECASE)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2))
    m = re.search(r"retry[- ]after[\"'\s:]+(\d+)", str(err), re.IGNORECASE)
    return float(m.group(1)) if m else None


def _is_tpd_limit(err) -> bool:
    m = str(err).lower()
    return "tokens per day" in m or "tpd" in m


def _is_transient(err: Exception) -> bool:
    m = str(err).lower()
    # Keep matches tight — broad "rate"/"connection" caused needless retry loops.
    return any(t in m for t in ("429", "rate limit", "timeout", "timed out", "503", "502", "500",
                                "overloaded", "temporarily unavailable", "connection reset",
                                "connection aborted", "server error", "bad gateway"))


def _is_credit_or_auth(err: Exception) -> bool:
    """402 / 401 / 'insufficient credits' -> the provider is out for this run.

    This is exactly the OpenRouter failure that silently produced 16 CS experiments
    with near-zero judge coverage. Retrying it is pointless; fail over instead.
    """
    m = str(err).lower()
    return any(t in m for t in ("402", "401", "403", "insufficient credit", "insufficient_quota",
                                "invalid api key", "no auth credentials", "unauthorized",
                                "payment required", "requires more credits"))


def _kill_provider(provider: str, reason: str) -> None:
    if provider not in _provider_dead:
        _provider_dead[provider] = reason
        _log(f"[transport] *** {provider.upper()} DISABLED for this run: {reason[:160]} *** "
             f"falling back to {[p for p in PROVIDER_ORDER if p not in _provider_dead] or 'NOTHING'}")


def _available_providers() -> List[str]:
    out = []
    for p in PROVIDER_ORDER:
        if p in _provider_dead:
            continue
        if p == "openrouter" and not OPENROUTER_API_KEY:
            continue
        if p == "groq" and not GROQ_KEYS:
            continue
        if p == "hf" and not HF_TOKEN:
            continue
        out.append(p)
    return out


def _call_openrouter(model: str, prompt: str, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY,
                    timeout=_env_float("REQUEST_TIMEOUT", 120.0))
    r = client.chat.completions.create(
        model=_route(model, "openrouter"), temperature=0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        extra_headers={"HTTP-Referer": f"https://github.com/{REPO_OWNER}/{REPO_NAME}",
                       "X-Title": "RAGBench-Capstone-Batch26"},
    )
    if not getattr(r, "choices", None):
        raise RuntimeError(f"OpenRouter returned no choices: {str(r)[:200]}")
    return (r.choices[0].message.content or "").strip()


def _call_groq(model: str, prompt: str, max_tokens: int) -> str:
    from groq import Groq
    key = GROQ_KEYS[_key_idx % len(GROQ_KEYS)]
    r = Groq(api_key=key, timeout=_env_float("REQUEST_TIMEOUT", 120.0)).chat.completions.create(
        model=_route(model, "groq"), temperature=0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return (r.choices[0].message.content or "").strip()


def _call_hf(model: str, prompt: str, max_tokens: int) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN, timeout=_env_float("REQUEST_TIMEOUT", 120.0))
    r = client.chat_completion(model=_route(model, "hf"), max_tokens=max_tokens, temperature=0.0,
                               messages=[{"role": "user", "content": prompt}])
    return (r.choices[0].message.content or "").strip()


def rotate_key() -> None:
    global _key_idx
    if GROQ_KEYS:
        _key_idx = (_key_idx + 1) % len(GROQ_KEYS)


class TransportError(RuntimeError):
    """Every provider refused the call."""


def llm_chat(model: str, prompt: str, max_tokens: int, tag: str = "llm") -> str:
    """One prompt -> one string, trying each live provider with backoff.

    Provider order is OpenRouter -> Groq -> HF. A credit/auth failure kills the
    provider for the rest of the run rather than being retried 8 times per call.
    """
    global _keys_exhausted_until
    providers = _available_providers()
    if not providers:
        raise TransportError("no live provider: set OPENROUTER_API_KEY, GROQ_API_KEY(S) or HF_TOKEN")

    last_err: Optional[Exception] = None
    for provider in providers:
        max_attempts = _env_int("MAX_ATTEMPTS", 6)
        delay = 4.0
        for attempt in range(max_attempts):
            try:
                if provider == "groq" and _keys_exhausted_until > time.time():
                    wait = _keys_exhausted_until - time.time()
                    if wait > 300:
                        raise RuntimeError(f"groq TPD exhausted, {wait:.0f}s remaining")
                    _log(f"[{tag}] groq TPD cooldown: waiting {wait:.0f}s ...")
                    time.sleep(wait + 2)
                    _keys_exhausted_until = 0.0

                out = {"openrouter": _call_openrouter, "groq": _call_groq, "hf": _call_hf}[provider](
                    model, prompt, max_tokens)
                _provider_calls[provider] = _provider_calls.get(provider, 0) + 1
                _cooldown()
                return out

            except Exception as e:  # noqa: BLE001 — provider SDKs raise many types
                last_err = e
                if _is_credit_or_auth(e):
                    _kill_provider(provider, f"{type(e).__name__}: {str(e)[:140]}")
                    break
                if attempt >= max_attempts - 1 or not _is_transient(e):
                    _log(f"[{tag}] {provider} gave up after {attempt + 1} attempt(s): "
                         f"{type(e).__name__}: {str(e)[:140]}")
                    break
                if provider == "groq":
                    rotate_key()
                    if _is_tpd_limit(e):
                        retry_after = _parse_retry_after(e) or 180.0
                        if attempt >= len(GROQ_KEYS):
                            _keys_exhausted_until = time.time() + retry_after
                            _log(f"[{tag}] all {len(GROQ_KEYS)} groq keys TPD-exhausted "
                                 f"({retry_after:.0f}s)")
                            break
                        time.sleep(min(delay, 10.0))
                        continue
                wait = _parse_retry_after(e) or delay
                time.sleep(min(wait + random.uniform(0, 1.5), 120.0))
                delay = min(delay * 2, 120.0)

    raise TransportError(f"all providers failed for {model}: "
                         f"{type(last_err).__name__}: {str(last_err)[:200]}")


# =========================================================================== #
# Persistent caches (SHA1 keys — never tuples, and None is NEVER persisted)
# =========================================================================== #
_gen_cache: Dict[str, str] = {}
_judge_cache: Dict[str, dict] = {}
_judge_miss_run: set = set()   # in-memory only, so a later run can retry
_cache_dirty = False


def _ck(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def load_caches(paths: Paths) -> None:
    global _gen_cache, _judge_cache
    _gen_cache = {k: v for k, v in _read_json(paths.gen_cache, {}).items() if isinstance(v, str)}
    raw = _read_json(paths.judge_cache, {})
    # Defensive: drop any cached nulls left behind by an outage-era run so they
    # are retried instead of being replayed as "judge unavailable" forever.
    _judge_cache = {k: v for k, v in raw.items() if isinstance(v, dict) and v}
    dropped = len(raw) - len(_judge_cache)
    _log(f"[cache] gen={len(_gen_cache)} judge={len(_judge_cache)}"
         + (f" (dropped {dropped} null judge entries)" if dropped else ""))


def save_caches(paths: Paths) -> None:
    global _cache_dirty
    if not _cache_dirty:
        return
    _atomic_write(paths.gen_cache, json.dumps(_gen_cache))
    _atomic_write(paths.judge_cache, json.dumps(_judge_cache))
    paths.mirror(paths.gen_cache)
    paths.mirror(paths.judge_cache)
    _cache_dirty = False


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def make_generator(model: str) -> Callable[[str, List[str]], str]:
    if not os.environ.get("USE_SYNTHETIC"):
        def gen(question: str, contexts: List[str]) -> str:
            global _cache_dirty
            ck = _ck("gen", model, question, *contexts)
            if ck in _gen_cache:
                return _gen_cache[ck]
            ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
            prompt = (f"Answer ONLY from the context. If insufficient, reply exactly:\n'{REJECTION}'.\n\n"
                      f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:")
            try:
                out = llm_chat(model, prompt, max_tokens=300, tag="gen")
            except TransportError as e:
                # Degrade instead of crashing, but make it visible and DO NOT cache:
                # a cached degraded answer would silently poison every later rerun.
                _log(f"[gen] TRANSPORT FAILURE: {str(e)[:180]}")
                return contexts[0] if contexts else REJECTION
            _gen_cache[ck] = out
            _cache_dirty = True
            return out
        return gen

    def gen_offline(question: str, contexts: List[str]) -> str:
        global _cache_dirty
        if not contexts:
            return REJECTION
        ck = _ck("offline", question, *contexts)
        if ck in _gen_cache:
            return _gen_cache[ck]
        out = sorted(contexts, key=lambda c: -coverage(question, c))[0]
        _gen_cache[ck] = out
        _cache_dirty = True
        return out
    return gen_offline


# --------------------------------------------------------------------------- #
# Verbatim RAGBench Appendix 7.4 judge prompt        [UNCHANGED — DO NOT EDIT]
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
# Sentence-key helpers + judge call                                [UNCHANGED]
# --------------------------------------------------------------------------- #
def _skey(i: int) -> str:
    """Base-26 sentence key: a..z, then aa, ab, ... (handles >26 sentences)."""
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


def trace_via_judge(question: str, keyed: List[Tuple[str, str]], answer: str,
                    model: str) -> Optional[dict]:
    global _cache_dirty
    if os.environ.get("USE_SYNTHETIC"):
        return None

    ck = _ck("judge", model, question, *[k for k, _ in keyed], answer)
    if ck in _judge_cache:
        return _judge_cache[ck]
    if ck in _judge_miss_run:
        return None

    docs = "\n".join(f"{k}. {s}" for k, s in keyed)
    ans = "\n".join(f"{_skey(i)}. {s}" for i, s in enumerate(split_sentences(answer)))
    prompt = JUDGE_PROMPT.format(documents=docs, question=question, answer=ans)

    for parse_attempt in range(2):
        try:
            raw = llm_chat(model, prompt, max_tokens=2048, tag="judge")
            if "```" in raw:
                raw = raw.split("```")[1]
                raw = raw[4:] if raw.startswith("json") else raw
            raw = raw[raw.find("{"): raw.rfind("}") + 1]
            out = json.loads(raw)
            if not isinstance(out, dict):
                raise ValueError("judge returned non-object JSON")
            _judge_cache[ck] = out
            _cache_dirty = True
            return out
        except TransportError as e:
            _log(f"[judge] TRANSPORT FAILURE: {str(e)[:180]}")
            break
        except Exception as e:
            _log(f"[judge] unparseable response (attempt {parse_attempt + 1}): "
                 f"{type(e).__name__}: {str(e)[:120]}")

    # Miss is remembered for THIS run only and never written to cache_judge.json,
    # so a rerun after an outage does not replay the outage.
    _judge_miss_run.add(ck)
    return None


def compute_trace(question: str, chunks: List[Dict], answer: str, reference: str,
                  model: str) -> Dict[str, float]:
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
    # offline sentence-level heuristic fallback
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
# AUROC / RMSE against RAGBench gold labels                       [UNCHANGED]
# --------------------------------------------------------------------------- #
def gold_metrics(row: dict) -> Dict[str, Optional[float]]:
    def g(*names):
        for n in names:
            if n in row and row[n] is not None:
                try:
                    return float(row[n])
                except Exception:
                    pass
        return None
    return {
        "context_relevance": g("relevance_score", "gpt3_context_relevance", "context_relevance"),
        "context_utilization": g("utilization_score", "gpt3_utilization", "context_utilization"),
        "completeness": g("completeness_score", "adherence_completeness", "completeness"),
        "adherence": g("adherence_score", "gpt3_adherence", "adherence"),
    }


def gold_eval(preds: List[dict], golds: List[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        from sklearn.metrics import roc_auc_score, mean_squared_error
    except Exception:
        return out
    for m in ("context_relevance", "context_utilization", "completeness"):
        pairs = [(p[m], g[m]) for p, g in zip(preds, golds) if g.get(m) is not None]
        if pairs:
            pv, gv = [a for a, _ in pairs], [b for _, b in pairs]
            out[f"{m}_rmse"] = round(math.sqrt(mean_squared_error(gv, pv)), 4)
    adh = [(p["adherence"], g["adherence"]) for p, g in zip(preds, golds) if g.get("adherence") is not None]
    labels = [int(round(b)) for _, b in adh]
    if adh and len(set(labels)) > 1:
        try:
            out["adherence_auroc"] = round(roc_auc_score(labels, [a for a, _ in adh]), 4)
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# Config + data loading
# --------------------------------------------------------------------------- #
@dataclass
class DomainConfig:
    domain: str
    exp_prefix: str
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
    results_csv: str = "results.csv"   # kept for backwards compat; Paths owns the real path

    def __post_init__(self):
        # RULE: the judge must be a DIFFERENT and STRONGER model than the generator.
        # No self-grading, and a weaker judge cannot reliably grade a stronger generator.
        _RANK = {"allam-2-7b": 1, "llama-3.1-8b-instant": 2, "openai/gpt-oss-20b": 3,
                 "qwen/qwen3-32b": 4, "qwen/qwen3.6-27b": 4,
                 "llama-3.3-70b-versatile": 5, "openai/gpt-oss-120b": 6}
        for g in (self.gen_models or (self.gen_model,)):
            if self.judge_model == g:
                raise ValueError(f"[{self.domain}] judge_model must DIFFER from generator "
                                 f"'{g}' - a model cannot grade its own output.")
            gr, jr = _RANK.get(g), _RANK.get(self.judge_model)
            if gr and jr and jr <= gr:
                raise ValueError(f"[{self.domain}] judge '{self.judge_model}' (tier {jr}) must be "
                                 f"STRONGER than generator '{g}' (tier {gr}).")


def apply_runtime_overrides(cfg: DomainConfig) -> DomainConfig:
    """Env knobs. The matrix itself is NOT trimmed unless you ask for it.
    N_EXAMPLES=5     -> sample count
    DATASET_CONFIG=x -> override the RAGBench subset
    QUICK_SWEEP=1    -> 1 embedder, 1 chunk, dense only, no rerank, forward order
    DENSE_ONLY=1     -> drop hybrid (NOT used for biomedical/finance: both are kept)
    """
    if os.environ.get("N_EXAMPLES"):
        try:
            cfg.n_examples = max(1, int(os.environ["N_EXAMPLES"]))
        except ValueError:
            pass
    if os.environ.get("DATASET_CONFIG"):
        cfg.dataset_config = os.environ["DATASET_CONFIG"].strip()
    if os.environ.get("GEN_MODEL"):
        cfg.gen_model = os.environ["GEN_MODEL"].strip()
        cfg.gen_models = ()
    if os.environ.get("JUDGE_MODEL"):
        cfg.judge_model = os.environ["JUDGE_MODEL"].strip()
    if _env_flag("DENSE_ONLY"):
        cfg.retrievals = ("dense",)
        _log(f"[{cfg.domain}] DENSE_ONLY=1 -> hybrid dropped from the matrix")
    if _env_flag("QUICK_SWEEP"):
        cfg.embedders = cfg.embedders[:1]
        cfg.chunk_configs = cfg.chunk_configs[:1]
        cfg.retrievals = ("dense",)
        cfg.rerank_options = ("none",)
        cfg.context_orders = ("forward",)
        _log(f"[{cfg.domain}] QUICK_SWEEP=1 -> emb={len(cfg.embedders)} "
             f"chunk={len(cfg.chunk_configs)} ret={cfg.retrievals} "
             f"rr={cfg.rerank_options} order={cfg.context_orders}")
    return cfg


def load_examples(cfg: DomainConfig, synthetic_fn: Callable[[], List[Dict]]) -> List[Dict]:
    if not os.environ.get("USE_SYNTHETIC"):
        try:
            from datasets import load_dataset
            _log(f"[data] loading HuggingFace rungalileo/ragbench/{cfg.dataset_config} ...")
            ds = load_dataset("rungalileo/ragbench", cfg.dataset_config, split="test")
            n = min(cfg.n_examples, len(ds))
            _log(f"[data] HuggingFace rungalileo/ragbench/{cfg.dataset_config} "
                 f"test={len(ds)} -> sampling {n}")
            idx = sorted(random.Random(cfg.seed).sample(range(len(ds)), n))
            out = []
            for row in ds.select(idx):
                docs = row["documents"]
                if isinstance(docs, str):
                    try:
                        docs = ast.literal_eval(docs)
                    except Exception:
                        docs = [docs]
                out.append({"question": row["question"],
                            "documents": [{"doc_id": f"d{i}", "text": t} for i, t in enumerate(docs)],
                            "reference": row.get("response", "") or "",
                            "gold": gold_metrics(row)})
            if out:
                return out
        except Exception as e:
            _log(f"[data] RAGBench load failed ({type(e).__name__}: {e}); using synthetic set.")
    # Synthetic set also honours N_EXAMPLES so an offline smoke run produces a
    # DIFFERENT fingerprint from the real N=200 run and can never mark it done.
    return synthetic_fn()[:max(1, cfg.n_examples)]


# =========================================================================== #
# Dataset fingerprint + experiment registry (stable IDs)
# =========================================================================== #
def dataset_fingerprint(cfg: DomainConfig, examples: List[Dict]) -> str:
    """Identifies the exact evaluation set. Completion is keyed on BOTH the
    example count and this value, so an N=1 smoke run can never mark an
    experiment 'done' for N=200 (the CS-sweep bug)."""
    h = hashlib.sha1()
    h.update(f"{cfg.domain}|{cfg.dataset_config}|{cfg.seed}|{len(examples)}".encode())
    for ex in examples:
        h.update(str(ex.get("question", "")).strip().encode("utf-8", "ignore"))
        h.update(f"|{len(ex.get('documents', []))}|".encode())
    return h.hexdigest()[:16]


def matrix(cfg: DomainConfig) -> List[Dict[str, str]]:
    """Canonical enumeration order — this is what fixes the ID numbering.
    Outer loops are the expensive ones (embedder, chunking) so a resumed run
    re-encodes as little as possible."""
    gen_models = cfg.gen_models or (cfg.gen_model,)
    out = []
    for emb in cfg.embedders:
        for cc in cfg.chunk_configs:
            for gen in gen_models:
                for ret in cfg.retrievals:
                    for rr in cfg.rerank_options:
                        for order in cfg.context_orders:
                            out.append({"embedder": emb, "chunk_config": cc["label"],
                                        "gen_model": gen, "retrieval": ret,
                                        "rerank": rr, "context_order": order,
                                        "judge_model": cfg.judge_model})
    return out


def _sig(spec: Dict[str, str]) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def load_or_build_registry(cfg: DomainConfig, paths: Paths) -> Dict[str, str]:
    """signature -> exp_id. Existing IDs are never renumbered; new configs get
    the next free number, so adding a chunker later does not shift old IDs."""
    reg = _read_json(paths.registry, {})
    used = {int(v.split("-")[1]) for v in reg.values() if "-" in v}
    nxt = max(used) + 1 if used else 1
    added = 0
    for spec in matrix(cfg):
        s = _sig(spec)
        if s not in reg:
            reg[s] = f"{cfg.exp_prefix}-{nxt:03d}"
            nxt += 1
            added += 1
    if added or not os.path.exists(paths.registry):
        _atomic_write(paths.registry, json.dumps(reg, indent=1, sort_keys=True))
        paths.mirror(paths.registry)
        _log(f"[registry] {len(reg)} experiment IDs ({added} new) -> {paths.registry}")
    return reg


def load_state(paths: Paths) -> Dict[str, Any]:
    return _read_json(paths.state, {})


def save_state(paths: Paths, state: Dict[str, Any]) -> None:
    _atomic_write(paths.state, json.dumps(state, indent=1, sort_keys=True))
    paths.mirror(paths.state)


def is_done(state: Dict[str, Any], exp_id: str, n: int, fp: str) -> bool:
    rec = state.get(exp_id)
    return bool(rec and rec.get("n_examples") == n and rec.get("fingerprint") == fp)


# --------------------------------------------------------------------------- #
# Targeted rerun selection
# --------------------------------------------------------------------------- #
def _exp_only_set(cfg: DomainConfig) -> Optional[set]:
    raw = os.environ.get("EXP_ONLY", "").strip()
    if not raw:
        return None
    out = set()
    for tok in re.split(r"[,\s]+", raw):
        if not tok:
            continue
        tok = tok.strip().upper()
        if tok.isdigit():
            tok = f"{cfg.exp_prefix}-{int(tok):03d}"
        elif "-" in tok:
            pre, _, num = tok.partition("-")
            if num.isdigit():
                tok = f"{cfg.exp_prefix}-{int(num):03d}"
        out.add(tok)
    return out or None


# =========================================================================== #
# Checkpointing
# =========================================================================== #
def _ckpt_load(paths: Paths, exp_id: str, n: int, fp: str) -> Optional[dict]:
    d = _read_json(paths.ckpt(exp_id), None)
    if not isinstance(d, dict):
        return None
    if d.get("fingerprint") != fp or d.get("n_examples") != n:
        _log(f"[{exp_id}] stale checkpoint discarded "
             f"(fp {str(d.get('fingerprint'))[:8]} vs {fp[:8]}, "
             f"n {d.get('n_examples')} vs {n})")
        return None
    return d


def _ckpt_save(paths: Paths, exp_id: str, payload: dict) -> None:
    p = paths.ckpt(exp_id)
    _atomic_write(p, json.dumps(payload))
    paths.mirror(p)


# =========================================================================== #
# Append-only results CSV
# =========================================================================== #
RESULT_FIELDS = ["exp_id", "timestamp", "domain", "dataset_config", "n_examples", "fingerprint",
                 "judge_model", "embedder", "chunk_config", "gen_model", "retrieval", "rerank",
                 "context_order", "context_relevance", "context_utilization", "completeness",
                 "adherence", "judge_used", "context_relevance_rmse", "context_utilization_rmse",
                 "completeness_rmse", "adherence_auroc", "elapsed_s"]


def append_result(paths: Paths, row: Dict[str, Any]) -> None:
    """Append-only: never truncate. De-duplication (newest row per exp_id) happens
    at report time, so a rerun can never destroy the earlier evidence."""
    path = paths.results_csv
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_FIELDS})
    paths.mirror(path)


# =========================================================================== #
# Report export (16-column viva layout + judge-coverage quarantine)
# =========================================================================== #
REPORT_COLUMNS = [
    ("Judge Model", "judge_model"), ("Embedding Model", "embedder"),
    ("Chunking Strategy", "chunk_config"), ("Generator LLM", "gen_model"),
    ("Retrieval Method", "retrieval"), ("Re-ranking Method", "rerank"),
    ("Context Ordering", "context_order"),
    ("TRACe Context Relevance \u2191", "context_relevance"),
    ("TRACe Context Utilization \u2191", "context_utilization"),
    ("TRACe Completeness \u2191", "completeness"),
    ("TRACe Adherence \u2191", "adherence"),
    ("Valid_Judge_response", "_valid"),
    ("Context Relevance RMSE \u2193", "context_relevance_rmse"),
    ("Context Utilization RMSE \u2193", "context_utilization_rmse"),
    ("Completeness RMSE \u2193", "completeness_rmse"),
    ("Adherence AUROC \u2191", "adherence_auroc"),
]


def export_report(cfg: DomainConfig, paths: Paths, n_filter: Optional[int] = None,
                  sort_by: str = "adherence_auroc") -> None:
    try:
        import pandas as pd
    except Exception as e:
        _log(f"[report] pandas unavailable ({e}); skipping export.")
        return
    if not os.path.exists(paths.results_csv):
        _log("[report] no results CSV yet; nothing to export.")
        return

    df = pd.read_csv(paths.results_csv)
    _log(f"[report] master rows: {len(df)}")
    if n_filter is not None:
        df = df[df["n_examples"].astype(str) == str(n_filter)]
    if df.empty:
        _log("[report] no rows after filtering; nothing to export.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    n0 = len(df)
    df = df.sort_values("timestamp").drop_duplicates("exp_id", keep="last").reset_index(drop=True)
    if n0 != len(df):
        _log(f"[report] de-duplicated {n0} -> {len(df)} (newest row per exp_id)")

    df["judge_used"] = pd.to_numeric(df["judge_used"], errors="coerce").fillna(0.0)
    df["n_examples"] = pd.to_numeric(df["n_examples"], errors="coerce").fillna(0).astype(int)
    df["_valid"] = ((df["judge_used"] * df["n_examples"]).round().astype(int).astype(str)
                    + "/" + df["n_examples"].astype(str))
    df["Status"] = ["report-grade" if u >= MIN_VALID_FRAC else "LOW JUDGE COVERAGE"
                    for u in df["judge_used"]]
    df["_rank"] = (df["Status"] != "report-grade").astype(int)
    df["_s"] = pd.to_numeric(df.get(sort_by), errors="coerce")
    df = df.sort_values(["_rank", "_s"], ascending=[True, False], na_position="last")

    out = pd.DataFrame()
    out["Experiment ID"] = df["exp_id"]
    for label, src in REPORT_COLUMNS:
        out[label] = df[src] if src in df.columns else ""
    out["Status"] = df["Status"]

    ok = out[out.Status == "report-grade"]
    bad = out[out.Status != "report-grade"]
    for frame, name in ((out, "_ALL"), (ok.drop(columns="Status"), ""), (bad, "_EXCLUDED")):
        p = paths.f(f"report_{cfg.domain}{name}.csv")
        frame.to_csv(p, index=False, encoding="utf-8-sig")  # BOM keeps arrows readable in Excel
        paths.mirror(p)

    _log(f"[report] report-grade: {len(ok)} | low coverage: {len(bad)} | total: {len(out)}")
    if len(bad):
        _log("[report] excluded: " + ", ".join(
            f"{e}({v})" for e, v in zip(bad["Experiment ID"], bad["Valid_Judge_response"])))


# =========================================================================== #
# The sweep
# =========================================================================== #
METRIC_KEYS = ["context_relevance", "context_utilization", "completeness", "adherence"]


class JudgeOutage(RuntimeError):
    """Raised when the judge stops responding, so we halt instead of silently
    filling 200 rows with heuristic scores (the CS-034..048 failure)."""


def run_experiment(cfg: DomainConfig, synthetic_fn: Callable[[], List[Dict]]):
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    cfg = apply_runtime_overrides(cfg)

    paths = Paths(cfg.domain)
    _restore_from_drive(paths)
    load_caches(paths)

    examples = load_examples(cfg, synthetic_fn)
    n = len(examples)
    fp = dataset_fingerprint(cfg, examples)
    mode = "synthetic" if os.environ.get("USE_SYNTHETIC") else cfg.dataset_config
    have_gold = any(any(v is not None for v in ex.get("gold", {}).values()) for ex in examples)
    gen_models = cfg.gen_models or (cfg.gen_model,)

    reg = load_or_build_registry(cfg, paths)
    state = load_state(paths)
    specs = matrix(cfg)
    only = _exp_only_set(cfg)
    force = _env_flag("FORCE_RERUN")

    pending = []
    for spec in specs:
        exp_id = reg[_sig(spec)]
        if only is not None and exp_id not in only:
            continue
        if is_done(state, exp_id, n, fp) and not force:
            continue
        pending.append((exp_id, spec))

    _log("=" * 78)
    _log(f"[{cfg.domain}] {n} examples ({mode}) · fingerprint={fp} · device={DEVICE} "
         f"· gold_labels={have_gold}")
    _log(f"[{cfg.domain}] judge={cfg.judge_model} · generators={list(gen_models)} "
         f"· providers={_available_providers() or ['OFFLINE']}")
    _log(f"[{cfg.domain}] matrix={len(specs)} configs · pending={len(pending)} "
         f"· done={len(specs) - len(pending)}"
         + (f" · EXP_ONLY={sorted(only)}" if only else "")
         + (" · FORCE_RERUN" if force else ""))
    _log(f"[{cfg.domain}] local={paths.local} · drive={paths.drive or 'OFF'} "
         f"· git={'ON' if (ENABLE_GIT and GIT_PAT and REPO_ROOT) else 'OFF'}")
    _log("=" * 78)

    if not pending:
        _log(f"[{cfg.domain}] nothing to do — all requested experiments are already complete.")
        export_report(cfg, paths)
        return []

    rows: List[Dict[str, Any]] = []
    ckpt_count = 0
    t_run = time.time()

    # Group by (embedder, chunk) so embeddings are built once per block.
    blocks: Dict[Tuple[str, str], List[Tuple[str, Dict[str, str]]]] = {}
    for exp_id, spec in pending:
        blocks.setdefault((spec["embedder"], spec["chunk_config"]), []).append((exp_id, spec))

    cc_by_label = {cc["label"]: cc for cc in cfg.chunk_configs}
    done_i = 0

    for (emb_name, cc_label), block in blocks.items():
        cc = cc_by_label[cc_label]
        need_bm25 = any(s["retrieval"] == "hybrid" for _, s in block)
        _plog(f"[{cfg.domain}] block emb={emb_name.split('/')[-1]} chunk={cc_label} "
              f"-> {len(block)} experiment(s)")

        prepared = []
        embedder = None
        for ei, ex in enumerate(examples, 1):
            corpus = build_corpus(ex["documents"], cc)
            if not corpus:
                continue
            texts = [c["text"] for c in corpus]
            if embedder is None:
                _plog(f"[{cfg.domain}] loading embedder {emb_name} ...")
                embedder = get_embedder(emb_name, corpus_hint=texts)
                _plog(f"[{cfg.domain}] embedder ready (backend={embedder.backend})")
                if embedder.backend != "st" and not os.environ.get("USE_SYNTHETIC"):
                    _log(f"[{cfg.domain}] *** WARNING: '{emb_name}' fell back to "
                         f"{embedder.backend.upper()} (sentence-transformers missing/failed). "
                         f"Rows are labeled '{emb_name.split('/')[-1]}' but are NOT this model. ***")
            mat = embedder.encode(texts)
            qvec = embedder.encode([ex["question"]])[0]
            bm25 = make_bm25(texts) if need_bm25 else None
            prepared.append((ex, corpus, mat, qvec, bm25))
            if ei == 1 or ei == len(examples) or ei % 25 == 0:
                _plog(f"[{cfg.domain}]   encoded {ei}/{len(examples)} (chunks={len(corpus)})")

        for exp_id, spec in block:
            done_i += 1
            t0 = time.time()
            generator = make_generator(spec["gen_model"])
            n_prep = len(prepared)
            synth = bool(os.environ.get("USE_SYNTHETIC"))

            ck = _ckpt_load(paths, exp_id, n, fp)
            preds = list(ck.get("preds") or []) if ck else []
            answers = list(ck.get("answers") or []) if ck else []
            preds = (preds + [None] * n_prep)[:n_prep]
            answers = (answers + [None] * n_prep)[:n_prep]
            golds = [ex.get("gold", {}) for ex, *_ in prepared]   # deterministic, never cached

            def _ckpt_now() -> None:
                _ckpt_save(paths, exp_id, {"exp_id": exp_id, "fingerprint": fp, "n_examples": n,
                                           "spec": spec, "preds": preds, "answers": answers,
                                           "updated": _ts()})

            # An example that was scored by the offline heuristic is NOT finished work:
            # on resume we re-judge it. This is what makes outage recovery a plain rerun
            # instead of "delete cache_judge.json and hope".
            todo_idx = [i for i, p in enumerate(preds)
                        if p is None or (not synth and float(p.get("_judge_used", 0.0)) < 1.0)]
            scored = sum(1 for p in preds if p is not None)
            retries = sum(1 for i in todo_idx if preds[i] is not None)
            if scored:
                _plog(f"[{cfg.domain}] {exp_id} RESUMING — {scored}/{n_prep} on disk, "
                      f"{len(todo_idx)} to compute ({retries} unjudged retr{'y' if retries == 1 else 'ies'})")
            if not todo_idx and scored == n_prep:
                _plog(f"[{cfg.domain}] {exp_id} checkpoint already complete; scoring from disk")

            _plog(f"[{cfg.domain}] {exp_id} ({done_i}/{len(pending)}) "
                  f"emb={emb_name.split('/')[-1]} chunk={cc_label} gen={spec['gen_model'].split('/')[-1]} "
                  f"ret={spec['retrieval']} rr={spec['rerank']} order={spec['context_order']}")

            consecutive_judge_misses = 0
            aborted = False

            for step, si in enumerate(todo_idx, 1):
                ex, corpus, mat, qvec, bm25 = prepared[si]
                if spec["retrieval"] == "hybrid":
                    ranked = hybrid_rrf_rank(ex["question"], qvec, mat, bm25)
                else:
                    ranked = dense_rank(qvec, mat)
                got = [corpus[i] for i in ranked[:min(cfg.k_retrieve, len(corpus))]]
                got = cross_encoder_rerank(ex["question"], got, cfg.reranker_model, cfg.k_final) \
                    if spec["rerank"] == "cross_encoder" else got[:cfg.k_final]
                got = order_context(got, spec["context_order"])

                answer = generator(ex["question"], [c["text"] for c in got])
                p = compute_trace(ex["question"], got, answer, ex["reference"], cfg.judge_model)
                preds[si] = p
                if SAVE_ANSWERS:
                    answers[si] = str(answer)[:1000]

                if p.get("_judge_used", 0.0) >= 1.0:
                    consecutive_judge_misses = 0
                else:
                    consecutive_judge_misses += 1

                if step % CHECKPOINT_EVERY == 0 or step == len(todo_idx):
                    _ckpt_now()
                    save_caches(paths)
                    ckpt_count += 1
                    have = [q for q in preds if q is not None]
                    jc = float(np.mean([q.get("_judge_used", 0.0) for q in have])) if have else 0.0
                    _plog(f"[{cfg.domain}]   {exp_id} {step}/{len(todo_idx)} "
                          f"(judge coverage {jc:.0%}) [ckpt]")
                    if ckpt_count % GIT_PUSH_EVERY_N_CKPT == 0:
                        git_sync(paths, f"{cfg.domain} {exp_id} @ {step}/{len(todo_idx)}")

                if not synth and JUDGE_ABORT_AFTER > 0 and consecutive_judge_misses >= JUDGE_ABORT_AFTER:
                    _ckpt_now()
                    save_caches(paths)
                    aborted = True
                    break

            if aborted:
                raise JudgeOutage(
                    f"{exp_id}: {consecutive_judge_misses} consecutive judge failures. The row "
                    f"was NOT written and the experiment is still pending — progress is "
                    f"checkpointed and unjudged examples will be retried on the next run. Fix "
                    f"the judge transport (providers available: {_available_providers() or 'none'}) "
                    f"and rerun; set JUDGE_ABORT_AFTER=0 to disable this guard.")

            done_pairs = [(p, g) for p, g in zip(preds, golds) if p is not None]
            if not done_pairs:
                continue
            preds_ok = [p for p, _ in done_pairs]
            golds_ok = [g for _, g in done_pairs]

            avg = {k: round(float(np.mean([p[k] for p in preds_ok])), 3) for k in METRIC_KEYS}
            judge_used = round(float(np.mean([p.get("_judge_used", 0.0) for p in preds_ok])), 3)
            row = {"exp_id": exp_id, "timestamp": _ts(), "domain": cfg.domain,
                   "dataset_config": mode, "n_examples": n, "fingerprint": fp,
                   "judge_model": cfg.judge_model,
                   "embedder": emb_name.split("/")[-1], "chunk_config": cc_label,
                   "gen_model": spec["gen_model"], "retrieval": spec["retrieval"],
                   "rerank": spec["rerank"], "context_order": spec["context_order"],
                   **avg, "judge_used": judge_used,
                   "elapsed_s": round(time.time() - t0, 1)}
            if have_gold:
                row.update(gold_eval(preds_ok, golds_ok))

            append_result(paths, row)
            rows.append(row)
            # The row is always written (it is evidence), but an experiment is only
            # marked DONE when the real judge actually scored it. Low-coverage rows
            # stay pending so a plain rerun picks them up once the judge is back.
            if judge_used >= MIN_VALID_FRAC or synth:
                state[exp_id] = {"n_examples": n, "fingerprint": fp, "judge_used": judge_used,
                                 "completed": _ts()}
                save_state(paths, state)
            else:
                state.pop(exp_id, None)
                save_state(paths, state)
                _log(f"[{cfg.domain}] {exp_id} left PENDING: judge coverage {judge_used:.0%} "
                     f"< {MIN_VALID_FRAC:.0%}. Row written and quarantined; rerun to re-judge.")
            save_caches(paths)

            gold_str = f"  | AUROC(adh)={row['adherence_auroc']}" if "adherence_auroc" in row else ""
            flag = "" if judge_used >= MIN_VALID_FRAC else "  << LOW JUDGE COVERAGE"
            _log(f"  [{done_i}/{len(pending)}] {exp_id} emb={row['embedder']:20s} "
                 f"chunk={cc_label:14s} gen={spec['gen_model'].split('/')[-1]:22s} "
                 f"ret={spec['retrieval']:6s} rr={spec['rerank']:13s} "
                 f"order={spec['context_order']:7s} -> {avg} judge={judge_used:.0%}"
                 f"{gold_str}{flag}")

            if GUARD_PAUSE > 0:
                time.sleep(GUARD_PAUSE)

        # free the block's embeddings before the next one
        prepared = []

    save_caches(paths)
    git_sync(paths, f"{cfg.domain} sweep: {len(rows)} experiment(s) completed")
    export_report(cfg, paths)

    _log(f"\n[{cfg.domain}] wrote {len(rows)} row(s) to {paths.results_csv} "
         f"in {(time.time() - t_run) / 60:.1f} min")
    if _provider_calls:
        _log(f"[{cfg.domain}] provider usage: {_provider_calls}"
             + (f" · disabled: {list(_provider_dead)}" if _provider_dead else ""))
    if rows:
        best = max(rows, key=lambda r: sum(r[m] for m in METRIC_KEYS))
        _log(f"[{cfg.domain}] best of this run (mean TRACe): {best['exp_id']} "
             f"emb={best['embedder']} chunk={best['chunk_config']} gen={best['gen_model']} "
             f"ret={best['retrieval']} rr={best['rerank']} order={best['context_order']} "
             f"avg={sum(best[m] for m in METRIC_KEYS) / 4:.3f}")
        judged = float(np.mean([r.get("judge_used", 0.0) for r in rows]))
        if not os.environ.get("USE_SYNTHETIC") and judged < 1.0:
            _log(f"[{cfg.domain}] WARNING: only {judged:.0%} of samples were scored by the real "
                 f"judge; the rest fell back to the offline heuristic (a DIFFERENT metric "
                 f"family). Check judge_used before reporting these numbers.")
    return rows


def main(cfg: DomainConfig, synthetic_fn: Callable[[], List[Dict]]) -> None:
    argv = [a for a in sys.argv[1:]]
    if argv and not argv[0].startswith("-") and argv[0] != cfg.domain:
        _log(f"[warn] argv domain '{argv[0]}' != script domain '{cfg.domain}'; "
             f"using '{cfg.domain}'.")
    if "--report" in argv or _env_flag("REPORT_ONLY"):
        cfg = apply_runtime_overrides(cfg)
        export_report(cfg, Paths(cfg.domain))
        return
    try:
        run_experiment(cfg, synthetic_fn)
    except JudgeOutage as e:
        _log("")
        _log("!" * 78)
        _log(f"HALTED: {e}")
        _log("!" * 78)
        sys.exit(2)


# =========================================================================== #
# FINANCE DOMAIN CONFIG + SYNTHETIC DATA
# =========================================================================== #
def synthetic():
    docs = [
        {"doc_id": "d0", "text": ("The income statement reports financial performance over a period by summarizing revenues "
                                  "and expenses. Revenue is recognized when goods or services are transferred to the customer, "
                                  "not necessarily when cash is received. Gross profit is calculated as revenue minus the cost "
                                  "of goods sold. Operating income further subtracts operating expenses such as salaries and "
                                  "rent. Net income is what remains after interest and taxes are deducted.")},
        {"doc_id": "d1", "text": ("The balance sheet presents financial position at a single point in time. It lists assets, "
                                  "liabilities, and shareholders equity, following the rule that assets equal liabilities plus "
                                  "equity. Current assets and liabilities are those expected to be settled within one year.")},
        {"doc_id": "d2", "text": ("The cash flow statement tracks cash moving into and out of a business across operating, "
                                  "investing, and financing activities. A company can be profitable yet run short of cash if "
                                  "receivables are not collected. Free cash flow is used to assess financial flexibility.")},
        {"doc_id": "d3", "text": ("Depreciation allocates the cost of a tangible asset over its useful life. It reduces reported "
                                  "net income but is a non cash expense that is added back on the cash flow statement.")},
    ]
    return [
        {"question": "How is revenue reported on the income statement?", "documents": docs,
         "reference": "Revenue is recognized when goods or services are transferred; gross profit is revenue minus cost of goods sold and net income remains after interest and taxes.",
         "gold": {"context_relevance": 0.28, "context_utilization": 0.26, "completeness": 1.0, "adherence": 1}},
        {"question": "What does the balance sheet show?", "documents": docs,
         "reference": "The balance sheet shows assets, liabilities and shareholders equity at a point in time, where assets equal liabilities plus equity.",
         "gold": {"context_relevance": 0.22, "context_utilization": 0.22, "completeness": 1.0, "adherence": 1}},
        {"question": "Why can a profitable company run short of cash?", "documents": docs,
         "reference": "A profitable company can run short of cash if its receivables are not collected.",
         "gold": {"context_relevance": 0.18, "context_utilization": 0.18, "completeness": 0.7, "adherence": 0}},
    ]


# Matrix: 2 embedders x 4 chunkers x 1 generator x 2 retrievals x 2 rerank x 3 orders
#       = 96 experiments -> FIN-001 .. FIN-096.  Dense AND hybrid are both kept.
CFG = DomainConfig(
    domain="finance",
    exp_prefix="FIN",
    dataset_config=os.environ.get("DATASET_CONFIG", "finqa"),
    embedders=("ProsusAI/finbert", "BAAI/bge-small-en-v1.5"),
    chunk_configs=(
        {"label": "whole_doc",      "strategy": "whole_doc"},
        {"label": "sentence_4o1",   "strategy": "sentence", "size": 4, "overlap": 1},
        {"label": "fixed_150w",     "strategy": "fixed", "size": 150, "overlap": 30},
        {"label": "large_sem_1100", "strategy": "large_semantic", "max_chars": 1100},
    ),
    retrievals=("dense", "hybrid"),
    rerank_options=("none", "cross_encoder"),
    context_orders=("forward", "reverse", "sides"),
    gen_model="llama-3.3-70b-versatile",
    judge_model="openai/gpt-oss-120b",
    k_retrieve=12, k_final=5,
    results_csv="results_finance.csv",
)

if __name__ == "__main__":
    main(CFG, synthetic)
