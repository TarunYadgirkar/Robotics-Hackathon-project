"""Embedding providers for the decision brain.

One interface, three implementations, tried in order:

    LocalMiniLM      all-MiniLM-L6-v2, 23M params, ~4ms, offline.  DEFAULT.
    FireworksQwen3   qwen3-embedding-8b via API.  Availability fallback only.
    None             -> caller falls back to the difflib path in coverage.py.

Measured on the 26-query brain benchmark (12 must match a real task, 14 must
abstain), scoring the tier decision:

    coverage + difflib (the original)     50.0%   TP=0  FN=12 FP=1 TN=13
    coverage + MiniLM word-similarity     84.6%   TP=9  FN=3  FP=1 TN=13
    MiniLM whole-string                   92.3%   TP=10 FN=2  FP=0 TN=14   <- decision rule
    qwen3-embedding-8b whole-string       92.3%   TP=11 FN=1  FP=1 TN=13

An 8B cloud model scores the same as a 23M local one, so Fireworks is here for
availability, NOT accuracy -- do not claim otherwise.  MiniLM is the default
because it is the only provider with FP=0: it never claims to know a task the
corpus does not contain, which is the one error that would invert BEAT 3.

n=26, so a one-query difference is noise.  The defensible claim is that model
capacity is not the bottleneck on a 50-item closed vocabulary.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_FILE = CACHE_DIR / "embeddings.json"
CALL_LOG = CACHE_DIR / "calls.jsonl"

FIREWORKS_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/embeddings"
FIREWORKS_TIMEOUT_S = 3.0

_LOCAL_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def _key(provider_name, text):
    return hashlib.sha256(f"{provider_name}\x00{text}".encode()).hexdigest()


class _Cache:
    """Disk cache keyed by sha256(provider + text).

    Rehearse the demo and every rehearsed query is served from disk at judging
    time with zero network calls.  Commit this file.
    """

    def __init__(self):
        self._data = {}
        self._dirty = False
        if CACHE_FILE.exists():
            try:
                self._data = json.loads(CACHE_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key):
        hit = self._data.get(key)
        return np.asarray(hit, dtype=np.float32) if hit is not None else None

    def put(self, key, vec):
        self._data[key] = [round(float(x), 6) for x in vec]
        self._dirty = True

    def flush(self):
        if not self._dirty:
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(self._data))
        self._dirty = False


_CACHE = _Cache()


def _log_call(provider, n_texts, n_cached, latency_s, error=None):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with CALL_LOG.open("a") as fh:
            fh.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "provider": provider,
                "n_texts": n_texts,
                "n_cache_hits": n_cached,
                "latency_s": round(latency_s, 3),
                "error": error,
            }) + "\n")
    except OSError:
        pass  # logging must never break the demo


def _normalize(mat):
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


class _Provider:
    """Shared cache-then-fetch logic.  Subclasses implement _embed_uncached."""

    name = "base"

    def encode(self, texts):
        texts = list(texts)
        vectors = [None] * len(texts)
        missing_idx = []
        for i, text in enumerate(texts):
            hit = _CACHE.get(_key(self.name, text))
            if hit is None:
                missing_idx.append(i)
            else:
                vectors[i] = hit

        n_cached = len(texts) - len(missing_idx)
        started = time.time()
        if missing_idx:
            fresh = self._embed_uncached([texts[i] for i in missing_idx])
            for slot, i in enumerate(missing_idx):
                vectors[i] = fresh[slot]
                _CACHE.put(_key(self.name, texts[i]), fresh[slot])
            _CACHE.flush()
        _log_call(self.name, len(texts), n_cached, time.time() - started)
        return _normalize(np.vstack([v.reshape(1, -1) for v in vectors]))

    def _embed_uncached(self, texts):
        raise NotImplementedError


class LocalMiniLM(_Provider):
    name = "local-minilm"

    def __init__(self):
        from sentence_transformers import SentenceTransformer  # raises if absent

        self._model = SentenceTransformer(_LOCAL_MODEL_ID)

    def _embed_uncached(self, texts):
        return _normalize(self._model.encode(texts, normalize_embeddings=True))


class FireworksQwen3(_Provider):
    name = "fireworks-qwen3-8b"

    def __init__(self):
        self._api_key = os.environ.get("FIREWORKS_API_KEY")
        if not self._api_key:
            raise RuntimeError("FIREWORKS_API_KEY not set")

    def _embed_uncached(self, texts):
        import urllib.request

        body = json.dumps({"model": FIREWORKS_MODEL, "input": texts}).encode()
        request = urllib.request.Request(
            FIREWORKS_URL, body,
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=FIREWORKS_TIMEOUT_S) as response:
            payload = json.load(response)
        rows = sorted(payload["data"], key=lambda d: d["index"])
        return _normalize(np.array([r["embedding"] for r in rows], dtype=np.float32))


def get_provider(prefer="auto"):
    """Return an embedding provider, or None if none is available.

    None is a supported outcome: coverage.py degrades to its difflib path so a
    teammate without torch installed still gets a runnable (if weaker) brain.
    """
    order = {
        "auto": (LocalMiniLM, FireworksQwen3),
        "local": (LocalMiniLM,),
        "fireworks": (FireworksQwen3,),
        "none": (),
    }[prefer]

    for cls in order:
        try:
            return cls()
        except Exception as exc:  # noqa: BLE001 - any failure means "try the next one"
            _log_call(cls.name, 0, 0, 0.0, error=f"{type(exc).__name__}: {exc}")
    return None
