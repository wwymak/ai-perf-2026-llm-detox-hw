# Task 8 Custom Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PPO reward in `tasks/task8_custom_reward.py` that ranks benign, on-topic, English answers above the prompt-independent template attractors that collapsed Tasks 6 and 7.

**Architecture:** A `reward_score(texts, prompts=None) -> list[float]` function with two layers: hard *vetoes* (wrong-language, severe-repetition) that floor the reward, and a weighted *soft sum* (saturated Detoxify + RM↔Detoxify agreement + embedding prompt-relevance + a repetition/length tail) over survivors. All knobs come from an env-var `CONFIG` dataclass so each docker PPO run flips flags via `-e` without code edits. Heavy models (fasttext, Detoxify, RM, sentence-embedder) are lazy-loaded once at module scope.

**Tech Stack:** Python 3.13, `uv` for deps, pytest. Reuses `src/toxic_rl/detoxify_reward.py::DetoxifyReward` and `src/toxic_rl/reward_model.py::TrainedRewardModel`. New deps: `fasttext-langdetect`, `sentence-transformers`.

## Global Constraints

- Python ≥ 3.13; manage deps with `uv add` (never `pip install`); run code with `uv run`.
- Functions require docstrings AND type hints (clear and concise) — not optional.
- TDD is the default; run the full suite before and after multi-file changes and report the passing count.
- Tests are minimal (POC repo): veto fixtures, ordering invariant, no-crash only. Do not add exhaustive per-knob tests.
- `reward_score` signature is fixed: `(texts: Sequence[str], prompts: Sequence[str] | None = None) -> list[float]`, with `reward_score.prompt_conditioned = True`. Higher = better.
- The reward worker calls the function per-completion; never raise from a scoring path (an exception kills the PPO run) — degrade to neutral and log once.
- Reuse existing wrappers: `DetoxifyReward(axis="toxicity").score(texts) -> list[float]`; `TrainedRewardModel(model_dir).score(texts, prompts=None) -> list[float]` with a `.prompt_conditioned` bool attribute.
- Output range ~`[-1, 1]` (default `CLIP_LO=-1.0`, `CLIP_HI=1.0`, `FLOOR=-1.0`) so KL pressure stays comparable to the inv:detoxify / rm runs.
- Documentation lives in `docs/task8_custom_reward.md`. **The README stays unchanged.**
- Do NOT run git commits — the user commits manually. The `git add`/`commit` steps below are for the user to run; the implementer stages and writes the message but stops short of committing unless the user says otherwise.

**Confirmed facts (verified against the codebase, do not re-investigate):**
- `src/toxic_rl/prompts.py::build_prompt_records` populates `extra_info["prompt_text"]` with the raw user prompt, so prompts DO flow to the reward worker via `compute_score`. The custom dispatcher in `src/toxic_rl/verl_reward.py` passes them as `prompts=[prompt_text]` because `reward_score.prompt_conditioned = True`.
- The RM is unbounded Bradley-Terry; existing `composite_rm` centers it at μ=3.0, σ=2.0. Reuse those as `RM_MU`/`RM_SIGMA` defaults.
- `_trigram_repeat_score(text)` and `_hit_cap_score(text)` already exist in `src/toxic_rl/verl_reward.py` as reference implementations for the repetition/length-cap logic.

---

## File Structure

- `tasks/task8_custom_reward.py` — the reward. Holds `CONFIG`, model-cache loaders, each signal function, and `reward_score`. One file (matches the task's single-file deliverable + the verl import path `custom:tasks.task8_custom_reward`).
- `tests/test_task8_custom_reward.py` — minimal pytest, CPU-only, heavy models monkeypatched.
- `docs/task8_custom_reward.md` — config reference, modified docker run block, run instructions.
- `submissions/task8_writeup.md` — written after the PPO run (manual, post-implementation).

---

### Task 1: Dependencies + CONFIG dataclass

**Files:**
- Modify: `pyproject.toml` (via `uv add`)
- Modify: `tasks/task8_custom_reward.py` (replace the `NotImplementedError` stub)
- Test: `tests/test_task8_custom_reward.py`

**Interfaces:**
- Produces: `CONFIG` — a module-level instance of a frozen dataclass `RewardConfig` read from env vars once at import. Fields (all `DETOX_`-prefixed env vars): `veto_lang: bool`, `veto_rep: bool`, `lang_min_conf: float`, `rep_floor_ratio: float`, `min_tokens: int`, `use_embedding: bool`, `sat_threshold: float`, `rm_mu: float`, `rm_sigma: float`, `echo_threshold: float`, `w_base: float`, `w_agree: float`, `w_rel: float`, `mild_rep_w: float`, `len_cap_w: float`, `clip_lo: float`, `clip_hi: float`, `floor: float`.
- Produces: `_env_bool(name, default) -> bool`, `_env_float(name, default) -> float`, `_env_int(name, default) -> int` helpers.

- [ ] **Step 1: Add dependencies**

Run: `uv add fasttext-langdetect sentence-transformers`
Expected: both resolve and land in `pyproject.toml` `[project.dependencies]`.

- [ ] **Step 2: Write the failing test for CONFIG defaults + env override**

Replace the contents of `tests/test_task8_custom_reward.py` (create it) with:

```python
"""Minimal POC tests for the task 8 custom reward."""
from __future__ import annotations

import importlib

import pytest


def _reload(monkeypatch, **env):
    """Reload the reward module with the given DETOX_ env vars set."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import tasks.task8_custom_reward as m
    return importlib.reload(m)


def test_config_defaults(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("DETOX_"):
            monkeypatch.delenv(k, raising=False)
    m = _reload(monkeypatch)
    assert m.CONFIG.veto_lang is True
    assert m.CONFIG.rep_floor_ratio == pytest.approx(0.4)
    assert m.CONFIG.rm_mu == pytest.approx(3.0)
    assert m.CONFIG.clip_lo == pytest.approx(-1.0)
    assert m.CONFIG.clip_hi == pytest.approx(1.0)


def test_config_env_override(monkeypatch):
    m = _reload(monkeypatch, DETOX_VETO_LANG="0", DETOX_W_REL="0.7")
    assert m.CONFIG.veto_lang is False
    assert m.CONFIG.w_rel == pytest.approx(0.7)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_task8_custom_reward.py -v`
Expected: FAIL — `reward_score` currently raises `NotImplementedError` on import-time use, and `CONFIG` does not exist.

- [ ] **Step 4: Implement CONFIG + env helpers**

Replace the body of `tasks/task8_custom_reward.py` below the module docstring. Keep the existing docstring. Add:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


def _env_bool(name: str, default: bool) -> bool:
    """Read a 0/1 (or true/false) env var, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default``."""
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default``."""
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


@dataclass(frozen=True)
class RewardConfig:
    """All reward knobs, each overridable via a ``DETOX_``-prefixed env var.

    A weight of 0 zeroes the term and (where relevant) skips the model load.
    """

    veto_lang: bool
    veto_rep: bool
    lang_min_conf: float
    rep_floor_ratio: float
    min_tokens: int
    use_embedding: bool
    sat_threshold: float
    rm_mu: float
    rm_sigma: float
    echo_threshold: float
    w_base: float
    w_agree: float
    w_rel: float
    mild_rep_w: float
    len_cap_w: float
    clip_lo: float
    clip_hi: float
    floor: float

    @classmethod
    def from_env(cls) -> "RewardConfig":
        """Build a config from ``DETOX_`` env vars (or defaults)."""
        return cls(
            veto_lang=_env_bool("DETOX_VETO_LANG", True),
            veto_rep=_env_bool("DETOX_VETO_REP", True),
            lang_min_conf=_env_float("DETOX_LANG_MIN_CONF", 0.5),
            rep_floor_ratio=_env_float("DETOX_REP_FLOOR_RATIO", 0.4),
            min_tokens=_env_int("DETOX_MIN_TOKENS", 4),
            use_embedding=_env_bool("DETOX_USE_EMBEDDING", True),
            sat_threshold=_env_float("DETOX_SAT_THRESHOLD", 0.9),
            rm_mu=_env_float("DETOX_RM_MU", 3.0),
            rm_sigma=_env_float("DETOX_RM_SIGMA", 2.0),
            echo_threshold=_env_float("DETOX_ECHO_THRESHOLD", 0.6),
            w_base=_env_float("DETOX_W_BASE", 0.5),
            w_agree=_env_float("DETOX_W_AGREE", 0.25),
            w_rel=_env_float("DETOX_W_REL", 0.25),
            mild_rep_w=_env_float("DETOX_MILD_REP_W", 0.3),
            len_cap_w=_env_float("DETOX_LEN_CAP_W", 0.2),
            clip_lo=_env_float("DETOX_CLIP_LO", -1.0),
            clip_hi=_env_float("DETOX_CLIP_HI", 1.0),
            floor=_env_float("DETOX_FLOOR", -1.0),
        )


CONFIG = RewardConfig.from_env()
```

Leave the existing `reward_score` stub and the `reward_score.prompt_conditioned = True` line for now (the next tasks replace the body).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_task8_custom_reward.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Validate**

Run: `uvx ruff check tasks/task8_custom_reward.py tests/test_task8_custom_reward.py`
Expected: no errors.

- [ ] **Step 7: Stage (user commits)**

```bash
git add pyproject.toml uv.lock tasks/task8_custom_reward.py tests/test_task8_custom_reward.py
# user runs: git commit -m "feat(task8): reward CONFIG dataclass + env knobs"
```

---

### Task 2: Pure helper signals (no models)

**Files:**
- Modify: `tasks/task8_custom_reward.py`
- Test: `tests/test_task8_custom_reward.py`

**Interfaces:**
- Consumes: `CONFIG` from Task 1.
- Produces:
  - `_distinct_trigram_ratio(text: str) -> float` — `1.0` when <5 words (no penalty signal), else `len(set(trigrams))/len(trigrams)`.
  - `severe_repetition(text: str) -> bool` — `True` when `_distinct_trigram_ratio < CONFIG.rep_floor_ratio`.
  - `_mild_rep_penalty(text: str) -> float` in `[0,1]` — `1.0 - _distinct_trigram_ratio(text)` (soft tail).
  - `_hit_cap_penalty(text: str, char_threshold: int = 240) -> float` — `1.0` if `len(text) >= char_threshold` else `0.0`.
  - `_token_overlap(prompt: str, text: str) -> float` in `[0,1]` — `|prompt_tokens ∩ text_tokens| / |text_tokens|` (fraction of the *completion* that is prompt-echo), lowercased word split; `0.0` if either side empty.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_task8_custom_reward.py`:

```python
def test_severe_repetition_detects_loop(monkeypatch):
    m = _reload(monkeypatch)
    looped = "TabControl\n" * 8
    assert m.severe_repetition(looped) is True
    assert m.severe_repetition("the cat sat on a warm mat by the door today") is False


def test_token_overlap_flags_echo(monkeypatch):
    m = _reload(monkeypatch)
    prompt = "describe my coworker who keeps interrupting me"
    echo = "describe my coworker who keeps interrupting me"
    fresh = "they listen patiently and rarely speak over anyone in meetings"
    assert m._token_overlap(prompt, echo) > 0.9
    assert m._token_overlap(prompt, fresh) < 0.4


def test_hit_cap_penalty(monkeypatch):
    m = _reload(monkeypatch)
    assert m._hit_cap_penalty("x" * 240) == 1.0
    assert m._hit_cap_penalty("short") == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_task8_custom_reward.py -v`
Expected: FAIL — `severe_repetition`, `_token_overlap`, `_hit_cap_penalty` undefined.

- [ ] **Step 3: Implement the pure helpers**

Add to `tasks/task8_custom_reward.py` (after `CONFIG`):

```python
def _distinct_trigram_ratio(text: str) -> float:
    """Fraction of distinct word-trigrams; 1.0 when too short to judge."""
    words = text.split()
    if len(words) < 5:
        return 1.0
    tri = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if not tri:
        return 1.0
    return len(set(tri)) / len(tri)


def severe_repetition(text: str) -> bool:
    """True when trigram diversity is below the veto floor (degenerate loop)."""
    return _distinct_trigram_ratio(text) < CONFIG.rep_floor_ratio


def _mild_rep_penalty(text: str) -> float:
    """Soft repetition penalty in [0,1] for near-misses the veto doesn't catch."""
    return 1.0 - _distinct_trigram_ratio(text)


def _hit_cap_penalty(text: str, char_threshold: int = 240) -> float:
    """1.0 if the completion likely ran to the token cap, else 0.0."""
    return 1.0 if len(text) >= char_threshold else 0.0


def _token_overlap(prompt: str, text: str) -> float:
    """Fraction of completion words that also appear in the prompt (echo signal)."""
    p = set(prompt.lower().split())
    t = text.lower().split()
    if not p or not t:
        return 0.0
    hits = sum(1 for w in t if w in p)
    return hits / len(t)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_task8_custom_reward.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Validate + stage**

Run: `uvx ruff check tasks/task8_custom_reward.py`
Expected: no errors.

```bash
git add tasks/task8_custom_reward.py tests/test_task8_custom_reward.py
# user commits: "feat(task8): pure repetition/echo/length helpers"
```

---

### Task 3: Model-backed signals + lazy cache

**Files:**
- Modify: `tasks/task8_custom_reward.py`
- Test: `tests/test_task8_custom_reward.py`

**Interfaces:**
- Consumes: `CONFIG`; `DetoxifyReward`, `TrainedRewardModel`; the pure helpers from Task 2.
- Produces:
  - `_models()` -> a module-level cache dict with keys `"detox"`, `"rm"`, `"embedder"`, `"fasttext"`, populated lazily on first access. RM dir read from env `DETOX_RM_DIR` (default `checkpoints/rm`). Embedder loaded only if `CONFIG.use_embedding`.
  - `is_english(text: str) -> bool` — `True` unless fasttext is confident (≥ `CONFIG.lang_min_conf`) the top language ≠ `en`; newlines sanitized first; any error → `True` (no veto).
  - `saturated_detox(detox_score: float) -> float` in `[0,1]` — `min(1.0, (1.0 - detox_score) / CONFIG.sat_threshold)`.
  - `rm_detox_agreement(rm_raw: float, detox_score: float) -> float` in `[0,1]` — `1.0 - abs(sigmoid((rm_raw - CONFIG.rm_mu)/CONFIG.rm_sigma) - (1.0 - detox_score))`.
  - `relevance(prompt: str, text: str, embedder) -> float` — `cosine(embed(prompt), embed(text))` minus an echo penalty when `_token_overlap(prompt, text) > CONFIG.echo_threshold`; `0.0` if no prompt or no embedder.

These functions take *scores* (not texts) where possible so tests need no model. `is_english` and `relevance` touch models, so tests monkeypatch `_models`.

- [ ] **Step 1: Write failing tests (model math, no real models)**

Append:

```python
import math


def test_saturated_detox_flattens(monkeypatch):
    m = _reload(monkeypatch)
    # very benign (detox≈0) saturates to 1.0
    assert m.saturated_detox(0.0) == pytest.approx(1.0)
    # still benign past threshold stays clamped at 1.0
    assert m.saturated_detox(0.05) == pytest.approx(1.0)
    # toxic gives a low base
    assert m.saturated_detox(0.9) < 0.2


def test_agreement_peaks_when_both_benign(monkeypatch):
    m = _reload(monkeypatch)
    # RM well above mu (benign per RM) + detox low (benign) → agree ≈ 1
    high = m.rm_detox_agreement(rm_raw=7.0, detox_score=0.02)
    # RM benign but detox says toxic → disagreement → lower
    low = m.rm_detox_agreement(rm_raw=7.0, detox_score=0.95)
    assert high > 0.9
    assert low < high
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_task8_custom_reward.py -v`
Expected: FAIL — `saturated_detox`, `rm_detox_agreement` undefined.

- [ ] **Step 3: Implement the model-backed signals + cache**

Add to `tasks/task8_custom_reward.py`:

```python
import math

_MODELS: dict[str, object] = {}


def _warn_once(key: str, msg: str) -> None:
    """Print a warning the first time ``key`` is seen this process."""
    if _MODELS.get(f"__warned_{key}"):
        return
    print(f"[task8_custom_reward] {msg}")
    _MODELS[f"__warned_{key}"] = True


def _models() -> dict[str, object]:
    """Lazily load and cache detox / RM / embedder / fasttext once per process."""
    if "detox" not in _MODELS:
        from src.toxic_rl.detoxify_reward import DetoxifyReward

        _MODELS["detox"] = DetoxifyReward(axis="toxicity")
    if "rm" not in _MODELS:
        from src.toxic_rl.reward_model import TrainedRewardModel

        rm_dir = os.environ.get("DETOX_RM_DIR", "checkpoints/rm")
        _MODELS["rm"] = TrainedRewardModel(rm_dir)
    if "fasttext" not in _MODELS:
        from ftlangdetect import detect  # fasttext-langdetect

        _MODELS["fasttext"] = detect
    if CONFIG.use_embedding and "embedder" not in _MODELS:
        from sentence_transformers import SentenceTransformer

        _MODELS["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODELS


def is_english(text: str) -> bool:
    """True unless fasttext is confident (≥ lang_min_conf) the text is non-English."""
    try:
        detect = _models()["fasttext"]
        clean = text.replace("\n", " ").strip()
        if not clean:
            return True
        res = detect(text=clean, low_memory=False)
        return not (res["lang"] != "en" and res["score"] >= CONFIG.lang_min_conf)
    except Exception as exc:  # never crash the rollout
        _warn_once("lang", f"language detect failed ({exc!r}); skipping veto")
        return True


def saturated_detox(detox_score: float) -> float:
    """Benign anchor in [0,1]; flattens to 1.0 once clearly benign."""
    benign = 1.0 - detox_score
    return min(1.0, benign / CONFIG.sat_threshold)


def rm_detox_agreement(rm_raw: float, detox_score: float) -> float:
    """Agreement in [0,1]: high when RM and Detoxify both call the text benign."""
    rm_b = 1.0 / (1.0 + math.exp(-(rm_raw - CONFIG.rm_mu) / CONFIG.rm_sigma))
    detox_b = 1.0 - detox_score
    return 1.0 - abs(rm_b - detox_b)


def relevance(prompt: str, text: str, embedder) -> float:
    """Embedding cosine(prompt, text) minus an echo penalty for prompt-parroting."""
    if not prompt or embedder is None or not text.strip():
        return 0.0
    try:
        import numpy as np

        vecs = embedder.encode([prompt, text], normalize_embeddings=True)
        cos = float(np.dot(vecs[0], vecs[1]))
    except Exception as exc:
        _warn_once("relevance", f"relevance failed ({exc!r}); returning 0")
        return 0.0
    if _token_overlap(prompt, text) > CONFIG.echo_threshold:
        cos -= 0.5  # parroting the prompt is not relevance
    return cos
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_task8_custom_reward.py -v`
Expected: PASS (7 passed). (The model-loading paths aren't exercised yet — only the pure math.)

- [ ] **Step 5: Validate + stage**

Run: `uvx ruff check tasks/task8_custom_reward.py`
Expected: no errors.

```bash
git add tasks/task8_custom_reward.py tests/test_task8_custom_reward.py
# user commits: "feat(task8): model-backed signals + lazy cache"
```

---

### Task 4: Assemble `reward_score` (vetoes + soft sum)

**Files:**
- Modify: `tasks/task8_custom_reward.py`
- Test: `tests/test_task8_custom_reward.py`

**Interfaces:**
- Consumes: every signal from Tasks 2–3.
- Produces: the final `reward_score(texts, prompts=None) -> list[float]` (replaces the stub), `reward_score.prompt_conditioned = True`.

- [ ] **Step 1: Write the failing tests — vetoes, ordering, no-crash**

Append to `tests/test_task8_custom_reward.py`:

```python
class _FakeDetox:
    """Returns a fixed toxicity per text via a lookup, default benign."""

    def __init__(self, table=None):
        self.table = table or {}

    def score(self, texts):
        return [self.table.get(t, 0.02) for t in texts]


class _FakeRM:
    prompt_conditioned = True

    def score(self, texts, prompts=None):
        return [6.0 for _ in texts]  # benign per RM


class _FakeEmbedder:
    """Cosine ≈ 1 when prompt and text share words, else ≈ 0 (toy)."""

    def encode(self, pair, normalize_embeddings=True):
        import numpy as np

        a, b = pair
        sa, sb = set(a.lower().split()), set(b.lower().split())
        sim = len(sa & sb) / (len(sb) or 1)
        # encode prompt as [1,0], text as [sim, sqrt(1-sim^2)] → dot = sim
        return np.array([[1.0, 0.0], [sim, (1 - sim * sim) ** 0.5]])


@pytest.fixture
def patched(monkeypatch):
    m = _reload(monkeypatch)

    def fake_models():
        # detox table: the attractors are benign-looking to Detoxify (that's the point)
        m._MODELS["detox"] = _FakeDetox()
        m._MODELS["rm"] = _FakeRM()
        m._MODELS["embedder"] = _FakeEmbedder() if m.CONFIG.use_embedding else None
        m._MODELS["fasttext"] = lambda text, low_memory=False: (
            {"lang": "zh", "score": 0.99}
            if any("一" <= ch <= "鿿" for ch in text)
            else {"lang": "en", "score": 0.99}
        )
        return m._MODELS

    monkeypatch.setattr(m, "_models", fake_models)
    return m


def test_language_veto_floors_chinese(patched):
    out = patched.reward_score(
        ["感谢您提供了这个反馈。"], prompts=["tell me what you think of politicians"]
    )
    assert out[0] == pytest.approx(patched.CONFIG.floor)


def test_repetition_veto_floors_loop(patched):
    out = patched.reward_score(
        ["TabControl\n" * 8], prompts=["describe my coworker"]
    )
    assert out[0] == pytest.approx(patched.CONFIG.floor)


def test_benign_ontopic_outscores_attractors(patched):
    prompt = "describe my coworker who keeps interrupting me"
    good = "my coworker often interrupts me in meetings and i find it hard to finish my point"
    attractors = [
        "感谢您提供了这个反馈。",          # wrong language
        "TabControl\n" * 8,               # repetition loop
        "You are a helpful assistant. Respond to the user thoughtfully and kindly.",  # echo
    ]
    good_score = patched.reward_score([good], prompts=[prompt])[0]
    for a in attractors:
        a_score = patched.reward_score([a], prompts=[prompt])[0]
        assert good_score > a_score, f"good !> attractor {a!r}"


def test_no_crash_on_malformed(patched):
    out = patched.reward_score(["", "   ", "ok"], prompts=None)
    assert len(out) == 3
    assert all(isinstance(x, float) and x == x for x in out)  # finite, no NaN
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_task8_custom_reward.py -v`
Expected: FAIL — `reward_score` still raises `NotImplementedError`.

- [ ] **Step 3: Implement `reward_score`**

Replace the stub `reward_score` (and keep the trailing `prompt_conditioned` line) with:

```python
def reward_score(
    texts: Sequence[str],
    prompts: Sequence[str] | None = None,
) -> list[float]:
    """Return one reward per completion (higher = better).

    Hard vetoes (wrong language, severe repetition) floor the reward; survivors
    get a weighted sum of saturated Detoxify, RM↔Detoxify agreement, embedding
    prompt-relevance, and a repetition/length tail. All knobs come from CONFIG.

    Args:
        texts: completions to score.
        prompts: same-length originating prompts (or None → relevance/agreement
            degrade to neutral).

    Returns:
        list[float] of len(texts); each clipped to [CONFIG.clip_lo, CONFIG.clip_hi].
    """
    n = len(texts)
    if prompts is None:
        prompts = [""] * n

    models = _models()
    detox = models["detox"]
    rm = models["rm"]
    embedder = models.get("embedder") if CONFIG.use_embedding else None

    try:
        detox_scores = detox.score(list(texts))
    except Exception as exc:
        _warn_once("detox", f"detox failed ({exc!r}); using 0.0")
        detox_scores = [0.0] * n
    try:
        rm_scores = rm.score(list(texts), prompts=list(prompts)) if getattr(
            rm, "prompt_conditioned", False
        ) else rm.score(list(texts))
    except Exception as exc:
        _warn_once("rm", f"rm failed ({exc!r}); using rm_mu")
        rm_scores = [CONFIG.rm_mu] * n

    out: list[float] = []
    for text, prompt, dscore, rscore in zip(texts, prompts, detox_scores, rm_scores):
        # --- degenerate / empty ---
        if not text or not text.strip():
            out.append(CONFIG.floor)
            continue
        # --- hard vetoes ---
        if CONFIG.veto_rep and severe_repetition(text):
            out.append(CONFIG.floor)
            continue
        if CONFIG.veto_lang and len(text.split()) >= CONFIG.min_tokens and not is_english(text):
            out.append(CONFIG.floor)
            continue
        # --- soft sum ---
        base = saturated_detox(float(dscore))
        agree = rm_detox_agreement(float(rscore), float(dscore))
        rel = relevance(prompt, text, embedder) if CONFIG.w_rel else 0.0
        tail = CONFIG.mild_rep_w * _mild_rep_penalty(text) + CONFIG.len_cap_w * _hit_cap_penalty(text)
        r = CONFIG.w_base * base + CONFIG.w_agree * agree + CONFIG.w_rel * rel - tail
        out.append(max(CONFIG.clip_lo, min(CONFIG.clip_hi, r)))
    return out


reward_score.prompt_conditioned = True
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_task8_custom_reward.py -v`
Expected: PASS (11 passed).

- [ ] **Step 5: Full suite**

Run: `uv run pytest -q`
Expected: all prior tests still pass; report the count.

- [ ] **Step 6: Validate + stage**

Run: `uvx ruff check tasks/task8_custom_reward.py && uvx ruff format --check tasks/task8_custom_reward.py`
Expected: no errors.

```bash
git add tasks/task8_custom_reward.py tests/test_task8_custom_reward.py
# user commits: "feat(task8): assemble reward_score (vetoes + soft sum)"
```

---

### Task 5: Documentation

**Files:**
- Create: `docs/task8_custom_reward.md`

**Interfaces:** none (docs only). README stays unchanged.

- [ ] **Step 1: Write the doc**

Create `docs/task8_custom_reward.md` with:
- A one-paragraph description of the reward (vetoes + soft sum, why — point at the Task 6/7 attractors).
- A **config table**: every `DETOX_` env var, its default, and one line on what it does (copy from `RewardConfig.from_env`).
- The **modified docker run block** for the Task 8 PPO run — copy the README's Task 8 block verbatim but change the install line to:
  `pip install -q verl==0.8.0 detoxify fasttext-langdetect sentence-transformers 2>&1 | tail -1` and add an example of flipping a knob, e.g. `-e DETOX_USE_EMBEDDING=0 -e DETOX_W_REL=0.4`.
- A **note on model caching**: the embedder downloads to the bind-mounted `~/.cache/huggingface` on first use; fasttext's `lid.176` model downloads to its own path inside the container (pre-warm by running the reward once on a dummy string if cold).
- A **"what to put in the writeup"** pointer to `submissions/task8_writeup.md`.

- [ ] **Step 2: Verify it renders + links resolve**

Run: `uvx --from markdown-it-py markdown-it docs/task8_custom_reward.md > /dev/null`
Expected: no parse error. (If `markdown-it-py` is unavailable, just visually confirm the file.)

- [ ] **Step 3: Stage**

```bash
git add docs/task8_custom_reward.md
# user commits: "docs(task8): custom reward config + run instructions"
```

---

## Post-implementation (manual, requires GPU + docker — not part of TDD loop)

These are run by the user on the GPU VM, not the implementing agent:

1. Launch the Task 8 PPO run using the docker block in `docs/task8_custom_reward.md` (`TOXIC_REWARD=custom:tasks.task8_custom_reward`).
2. Merge FSDP shards → HF (`checkpoints/ppo_custom_merged`).
3. Eval: `python -m tasks.task7_ppo_rm_eval --ppo-dir checkpoints/ppo_custom_merged --out submissions/task8_ppo_custom_eval.json`.
4. Write `submissions/task8_ppo_custom_eval.txt` and `submissions/task8_writeup.md` (what collapsed into what, final design, why).
```
