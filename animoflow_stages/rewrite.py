"""Multilingual prompt rewriter — Qwen2.5-1.5B-Instruct + RAFSL few-shot from
the local HumanML3D caption corpus.

This is the SHARED core (stage library). Two consumers:
  * animoflow-api/api/main.py `_run_pipeline` — rewrites before workflow
    compilation on every API-driven job (webUI, Blender addon, HF Space);
    imports via the api/rewriter.py re-export shim.
  * nodes/prompt_rewrite_node.py (AnimoFlow_PromptRewrite) — GUI-workflow
    rewrite for users driving the ComfyUI graph directly. GUI-only: API-
    compiled DAGs never contain the node (the API already rewrote), so
    there is no double rewrite.

Phase-0 verified (2026-06-15): 90% pass rate across 30 prompts × 10 language
families with this exact template + model. ZeroGPU server-only latency 0.32s
mean / 0.16s p50 on zero-a10g.

Weights are baked into the animoflow-app Docker image at build time:
    /opt/rewriter/qwen/        — Qwen/Qwen2.5-1.5B-Instruct snapshot
    /opt/rewriter/minilm/      — sentence-transformers retriever snapshot
    /opt/rewriter/corpus/      — captions.json + embeddings.npy
Everywhere else (OSS-local, ComfyUI GUI) the assets lazy-download from HF
Hub on the first non-skipped rewrite (~3.2 GB one-time). The cheap skip
heuristic runs before any load, so HumanML3D-style English prompts never
touch the weights.

Env vars (override the bake-time defaults if needed):
    REWRITER_MODEL_REPO        — defaults to "Qwen/Qwen2.5-1.5B-Instruct"
    REWRITER_RETRIEVER_REPO    — defaults to "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    REWRITER_DATA_DIR          — defaults to "/opt/rewriter"
    REWRITER_DISABLED          — if "1", every rewrite() call is a no-op (pass-through)
    REWRITER_DEVICE            — pin the LLM device: "cpu" | "cuda" | "mps" (unset = auto)
"""

from __future__ import annotations

import functools
import logging
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

MODEL_REPO = os.environ.get("REWRITER_MODEL_REPO", "Qwen/Qwen2.5-1.5B-Instruct")
RETRIEVER_REPO = os.environ.get(
    "REWRITER_RETRIEVER_REPO",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
DATA_DIR = Path(os.environ.get("REWRITER_DATA_DIR", "/opt/rewriter"))
CORPUS_REPO = os.environ.get("REWRITER_CORPUS_REPO", "AnimoFlow/rewriter-corpus")
CORPUS_REVISION = os.environ.get("REWRITER_CORPUS_REVISION", "5b98da4d2db075d40afd59badf51f2eefe1db5e2")

# Pull in the @spaces.GPU decorator if we're inside the animoflow-app
# image (or HF Space) — otherwise pass-through. animoflow-app puts its own
# directory on sys.path at boot (see animoflow-app/app.py:114-115) so
# spaces_compat is importable from here at runtime.
try:
    from spaces_compat import GPU  # type: ignore[import-not-found]
    _GPU_DECORATOR_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised in OSS local / tests
    _GPU_DECORATOR_AVAILABLE = False

    def GPU(*decorator_args, **decorator_kwargs):  # type: ignore[no-redef]
        if (
            len(decorator_args) == 1
            and callable(decorator_args[0])
            and not decorator_kwargs
        ):
            return decorator_args[0]

        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        return decorator


# ---------------------------------------------------------------------------
# Prompt template — same as probe + tweaks from Phase-0 failure analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a prompt rewriter for a text-to-motion model that was trained on the HumanML3D dataset. Convert the user's input (in any language, any phrasing) into ONE short English caption that describes the motion only, in the exact style of HumanML3D captions.

HumanML3D style rules:
- If the input is already a complete English motion description starting with "a person", "the person", "someone", or "the figure", return it UNCHANGED.
- Do NOT add details (direction, speed, manner, location, emotion) that are not present in the input. A faithful 4-token caption is better than an embellished 12-token one.
- ALWAYS begin with "a person ..." (or "the person ...", "someone ..."). Never omit the subject.
- English only. Present tense.
- Describe motion only — strip greetings, polite framing, "please show me", emotional context.
- Use motion verbs (walk, run, jump, kick, bend, sit, stand, raise, lower, turn, step).
- Up to 30 tokens, in a single sentence. Brevity is fine if the input is brief.
- No quotes, no commentary, no preamble. Output only the caption."""

USER_TEMPLATE = """Examples of HumanML3D-style captions in the right style:
{examples}

Now rewrite this user input as ONE HumanML3D-style caption:
{user_input}"""


# ---------------------------------------------------------------------------
# Skip heuristic (very cheap)
# ---------------------------------------------------------------------------

_PERSON_PREFIX = re.compile(r"^\s*(a person|the person|someone|the figure)\b", re.IGNORECASE)
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_MOTION_VERB_HINT = re.compile(
    r"\b(walk|run|jump|kick|step|bend|sit|stand|raise|lower|turn|crouch|kneel|wave|dance|throw|pick|lift|push|pull|squat|spin|leap|stretch)",
    re.IGNORECASE,
)


def _looks_already_humanml3d(text: str) -> bool:
    if not text:
        return False
    if _NON_ASCII.search(text):  # any non-ASCII → not English
        return False
    if not _PERSON_PREFIX.search(text):
        return False
    if not _MOTION_VERB_HINT.search(text):
        return False
    tokens = re.findall(r"[A-Za-z']+", text)
    return len(tokens) <= 30


# ---------------------------------------------------------------------------
# Lazy-loaded singleton
# ---------------------------------------------------------------------------

_INSTANCE = None
_LOAD_LOCK = threading.Lock()
_LOAD_FAILED: Exception | None = None
_LOAD_LATENCY_S: float | None = None


def _get_instance():
    """Return the warm Rewriter singleton, loading it on first call.

    Loud-failure policy (no-silent-fallback policy):
    if loading fails, every subsequent call re-raises the same loud error.
    """
    global _INSTANCE, _LOAD_FAILED, _LOAD_LATENCY_S
    if _LOAD_FAILED is not None:
        raise RuntimeError(f"rewriter unavailable: {_LOAD_FAILED}") from _LOAD_FAILED
    if _INSTANCE is not None:
        return _INSTANCE
    with _LOAD_LOCK:
        if _LOAD_FAILED is not None:
            raise RuntimeError(f"rewriter unavailable: {_LOAD_FAILED}") from _LOAD_FAILED
        if _INSTANCE is not None:
            return _INSTANCE
        try:
            t0 = time.time()
            _INSTANCE = _Rewriter()
            _LOAD_LATENCY_S = time.time() - t0
            log.info(
                "[rewriter] singleton ready in %.1fs (model=%s device=%s)",
                _LOAD_LATENCY_S, MODEL_REPO, _INSTANCE.device,
            )
        except Exception as e:
            _LOAD_FAILED = e
            log.exception("[rewriter] FAILED to load — surfacing loud error to clients")
            raise RuntimeError(f"rewriter unavailable: {e}") from e
    return _INSTANCE


class _Rewriter:
    """Internal class — use module-level rewrite() instead."""

    def __init__(self):
        import json
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        on_zerogpu = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("true", "1")
        # REWRITER_DEVICE pins the device explicitly (cpu | cuda | mps).
        # Deployment need: on a shared small GPU (e.g. a 16 GB card where
        # Wan video needs every byte) the auto-pick below would park Qwen's
        # ~3 GB on CUDA permanently. Unset → existing auto behavior.
        forced = os.environ.get("REWRITER_DEVICE", "").strip().lower()
        if forced in ("cpu", "cuda", "mps"):
            self.device = forced
            dtype = "float32" if forced == "cpu" else "bfloat16"
        elif on_zerogpu:
            # On ZeroGPU, model loads on CPU at boot — moved to CUDA inside
            # the @GPU-decorated rewrite path.
            self.device = "cpu"
            dtype = "bfloat16"
        elif torch.cuda.is_available():
            self.device = "cuda"
            dtype = "bfloat16"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
            dtype = "bfloat16"
        else:
            self.device = "cpu"
            dtype = "float32"

        self._on_zerogpu = on_zerogpu
        self._moved_to_cuda = False
        self.model_id = MODEL_REPO

        # Tokenizer + model
        local_model_dir = DATA_DIR / "qwen"
        model_src = str(local_model_dir) if local_model_dir.is_dir() else MODEL_REPO
        log.info("[rewriter] loading LLM from %s on %s (%s)", model_src, self.device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_src)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_src,
            dtype=getattr(torch, dtype),
            device_map=self.device,
        ).eval()

        # Retriever
        local_retriever_dir = DATA_DIR / "minilm"
        retriever_src = str(local_retriever_dir) if local_retriever_dir.is_dir() else RETRIEVER_REPO
        log.info("[rewriter] loading retriever from %s on cpu", retriever_src)
        # MiniLM is small (~110 MB) — keep it on CPU regardless to leave GPU
        # bandwidth entirely for the LLM. Adds <30 ms per rewrite vs GPU.
        self.retriever = SentenceTransformer(retriever_src, device="cpu")

        # Corpus — either baked (Dockerfile-based deploys) or lazy-fetched
        # from the HF dataset repo (Gradio SDK Spaces ignore Dockerfile RUN
        # steps, so we download on first init there). Loud-fail on miss.
        corpus_dir = DATA_DIR / "corpus"
        captions_path = corpus_dir / "captions.json"
        embeddings_path = corpus_dir / "embeddings.npy"
        if not captions_path.is_file() or not embeddings_path.is_file():
            from huggingface_hub import snapshot_download
            log.info(
                "[rewriter] corpus not at %s — downloading %s@%s from HF Hub",
                corpus_dir, CORPUS_REPO, CORPUS_REVISION,
            )
            try:
                downloaded = snapshot_download(
                    repo_id=CORPUS_REPO,
                    revision=CORPUS_REVISION,
                    repo_type="dataset",
                    allow_patterns=["captions.json", "embeddings.npy"],
                )
            except Exception as e:
                raise RuntimeError(
                    f"rewriter corpus download failed from {CORPUS_REPO}@{CORPUS_REVISION}: {e}"
                ) from e
            corpus_dir = Path(downloaded)
            captions_path = corpus_dir / "captions.json"
            embeddings_path = corpus_dir / "embeddings.npy"
            if not captions_path.is_file() or not embeddings_path.is_file():
                raise FileNotFoundError(
                    f"rewriter corpus snapshot landed at {corpus_dir} but missing "
                    "captions.json or embeddings.npy."
                )
        log.info("[rewriter] loading corpus from %s", corpus_dir)
        self.captions: list[str] = json.loads(captions_path.read_text())
        # fp16 on disk → fp32 in RAM (cosine math is fp32)
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        log.info(
            "[rewriter] %d captions × %d-d embeddings ready",
            len(self.captions), self.embeddings.shape[1],
        )

        # torch + numpy refs cached for inference path
        self._torch = torch
        self._np = np

    # -- core inference -----------------------------------------------------

    def _retrieve_topk(self, query: str, k: int = 3) -> list[str]:
        np = self._np
        q = self.retriever.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.embeddings @ q
        idxs = np.argpartition(-sims, k)[:k]
        idxs = idxs[np.argsort(-sims[idxs])]
        return [self.captions[i] for i in idxs]

    def _generate(self, prompt_ids, max_new_tokens: int):
        torch = self._torch
        # ZeroGPU: move model to CUDA inside the @GPU call.
        if self._on_zerogpu and not self._moved_to_cuda:
            self.model = self.model.to("cuda")
            self._moved_to_cuda = True
        device = "cuda" if (self._on_zerogpu or self.device == "cuda") else self.device
        with torch.no_grad():
            return self.model.generate(
                prompt_ids.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

    def run(self, user_input: str, max_new_tokens: int = 48) -> "RewriteResult":
        t0 = time.time()
        examples = self._retrieve_topk(user_input, k=3)
        examples_block = "\n".join(f"- {e}" for e in examples)
        user_msg = USER_TEMPLATE.format(examples=examples_block, user_input=user_input)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=False,
        )
        if hasattr(encoded, "shape"):
            prompt_ids = encoded
        else:
            prompt_ids = encoded["input_ids"]
        input_len = prompt_ids.shape[-1]
        out = self._generate(prompt_ids, max_new_tokens)
        new_tokens = out[0, input_len:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        decoded = decoded.strip('"').strip("'").lstrip("-").strip()
        # Take only the first non-empty line
        for line in decoded.split("\n"):
            line = line.strip()
            if line:
                decoded = line
                break
        return RewriteResult(
            rewritten=decoded,
            examples=examples,
            latency_s=time.time() - t0,
            input_tokens=int(input_len),
            output_tokens=int(new_tokens.shape[0]),
            skipped=False,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RewriteResult:
    """Plain data class — small intentional surface, easy to add to job records."""

    __slots__ = ("rewritten", "examples", "latency_s", "input_tokens", "output_tokens", "skipped")

    def __init__(
        self,
        rewritten: str,
        examples: list[str],
        latency_s: float,
        input_tokens: int,
        output_tokens: int,
        skipped: bool,
    ):
        self.rewritten = rewritten
        self.examples = examples
        self.latency_s = latency_s
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.skipped = skipped

    def to_dict(self) -> dict:
        return {
            "rewritten": self.rewritten,
            "examples": self.examples,
            "latency_s": self.latency_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "skipped": self.skipped,
        }


Mode = Literal["auto", "force", "skip"]


# GPU-decorated only on ZeroGPU; pass-through everywhere else.
# duration=60s leaves headroom for cold-first-call model move-to-cuda +
# initial CUDA-kernel compile (verified mean 0.32s server-only on warm
# calls, but the first @GPU call after a Space wake-up has to do an
# eager .to('cuda') and pay one-time cuDNN/JIT compilation). Tighter
# budgets (15s) trigger ZeroGPU's `GPU task aborted` mid-load.
#
# IMPORTANT: only pass picklable args (no module/instance refs). ZeroGPU
# forks a worker that inherits the warm singleton via copy-on-write, so
# we look up the instance inside the function instead of passing it.
@GPU(duration=60)
def _do_rewrite_gpu(user_input: str) -> RewriteResult:
    return _get_instance().run(user_input)


def warmup() -> None:
    """Eager-load the Rewriter in the parent orchestrator process.

    Call this once from the boot path (animoflow-app/app.py after the
    FastAPI app is imported but before the first request). Loading here
    means the @GPU fork inherits a fully-loaded model via copy-on-write
    on every subsequent call — same trick the MDM registry uses
    (animoflow-app/animoflow_models/registry.py:1).

    Safe to call multiple times — _get_instance() is idempotent under
    the load lock and re-raises the prior failure if loading errored.
    """
    try:
        _get_instance()
    except Exception as e:
        # Don't kill orchestrator boot — let the first /v1/jobs request
        # surface the loud-fail error via the normal _run_pipeline path.
        log.warning("[rewriter] warmup failed (will retry per-request): %s", e)


# LRU cache around the GPU-decorated call so identical (prompt, mode) tuples
# don't re-burn ZeroGPU. Per-process. ~100 KB at maxsize=1024.
@lru_cache(maxsize=1024)
def _cached_rewrite(user_input: str) -> tuple:
    """Return RewriteResult fields as a hashable tuple — lru_cache needs hashable args + returns."""
    result = _do_rewrite_gpu(user_input)
    return (
        result.rewritten,
        tuple(result.examples),
        result.latency_s,
        result.input_tokens,
        result.output_tokens,
    )


def rewrite(prompt: str, mode: Mode = "auto") -> RewriteResult:
    """Rewrite a user prompt into a HumanML3D-style English caption.

    mode:
      - "auto"  (default): run unless the cheap heuristic says the input is
                already HumanML3D-style English.
      - "force": always run.
      - "skip":  no-op — return the input unchanged with skipped=True.

    Raises RuntimeError loudly if the rewriter failed to load or inference
    raises. Callers should fail the job with `error="rewriter failed: …"`
    rather than silently passing the original prompt through (per the
    no-silent-fallback policy).
    """
    if os.environ.get("REWRITER_DISABLED", "0") == "1" or mode == "skip":
        return RewriteResult(
            rewritten=prompt,
            examples=[],
            latency_s=0.0,
            input_tokens=0,
            output_tokens=0,
            skipped=True,
        )
    if mode == "auto" and _looks_already_humanml3d(prompt):
        return RewriteResult(
            rewritten=prompt,
            examples=[],
            latency_s=0.0,
            input_tokens=0,
            output_tokens=0,
            skipped=True,
        )

    cached = _cached_rewrite(prompt)
    return RewriteResult(
        rewritten=cached[0],
        examples=list(cached[1]),
        latency_s=cached[2],
        input_tokens=cached[3],
        output_tokens=cached[4],
        skipped=False,
    )


def health() -> dict:
    """Used for /v1/health to report rewriter readiness without forcing a load."""
    return {
        "ready": _INSTANCE is not None,
        "failed": str(_LOAD_FAILED) if _LOAD_FAILED else None,
        "load_latency_s": _LOAD_LATENCY_S,
        "model_id": MODEL_REPO,
        "data_dir": str(DATA_DIR),
        "gpu_decorator_available": _GPU_DECORATOR_AVAILABLE,
    }
