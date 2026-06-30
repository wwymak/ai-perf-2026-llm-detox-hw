"""Task 8 — Design and ship a custom reward [15 points].

Tasks 6 and 7 showed you two attractors: ``inv:detoxify`` collapses
the policy onto an OpenAI-style refusal template; ``rm:<your-RM>``
collapses onto a different template (in our runs, a Mandarin-greeting
or "I'm sorry could you provide more context" attractor). Design a
reward function that *can't be saturated by a single template*.

The function below is loaded by the verl reward worker when you launch
PPO with ``TOXIC_REWARD=custom:tasks.task8_custom_reward``. It runs in
the same docker container as the rollout. Detoxify, your trained RM,
and any other reward source are importable here.

Some hints (pick any combination, or invent your own):

  * **Saturating Detoxify above a threshold.** Once a completion is
    "clearly benign", uniform reward removes the incentive to push
    toward template attractors.
  * **Penalising repetition.** Trigram-repetition penalty bites where
    the policy starts looping on a phrase.
  * **Penalising length-cap hits.** If the policy learns to always
    run to the token cap, penalise that signal.
  * **Prompt-relevance signal.** A response that ignores the prompt
    can still score high on Detoxify by accident. Bag-of-words
    overlap or embedding similarity ties the reward to the prompt.
    Beware trivial echoing — bake a check against that.
  * **Blending or gating with your RM.** Detoxify and your RM
    disagree in interesting ways; their disagreement is signal.

The score function returns a list of floats — one reward per
completion, in the same order as the input ``texts`` list. Higher =
better.

Submit your final reward design + writeup in:

  * this file (the implementation)
  * ``submissions/task8_writeup.md`` (what you tried, what collapsed
    into what, what your final design looks like, why)
"""

from __future__ import annotations

import math
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

        _MODELS["embedder"] = SentenceTransformer(
            "Alibaba-NLP/gte-multilingual-base", trust_remote_code=True
        )
    return _MODELS


def is_english(text: str) -> bool:
    """Return True unless fasttext is confident the text is non-English.

    Sanitizes newlines first.  Any error defaults to True (no veto).
    """
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
    """Benign anchor in [0,1] that flattens to 1.0 once clearly benign.

    Computes ``min(1.0, (1.0 - detox_score) / CONFIG.sat_threshold)``.
    """
    benign = 1.0 - detox_score
    return min(1.0, benign / CONFIG.sat_threshold)


def rm_detox_agreement(rm_raw: float, detox_score: float) -> float:
    """Agreement score in [0,1]: high when RM and Detoxify both call text benign.

    Uses ``1.0 - abs(sigmoid((rm_raw - rm_mu) / rm_sigma) - (1.0 - detox_score))``.
    """
    rm_b = 1.0 / (1.0 + math.exp(-(rm_raw - CONFIG.rm_mu) / CONFIG.rm_sigma))
    detox_b = 1.0 - detox_score
    return 1.0 - abs(rm_b - detox_b)


def relevance(prompt: str, text: str, embedder: object) -> float:
    """Embedding cosine similarity minus an echo penalty for prompt-parroting.

    Returns 0.0 when prompt is empty, embedder is None, or text is blank.
    Subtracts 0.5 when ``_token_overlap`` exceeds ``CONFIG.echo_threshold``.
    """
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
        rm_scores = (
            rm.score(list(texts), prompts=list(prompts))
            if getattr(rm, "prompt_conditioned", False)
            else rm.score(list(texts))
        )
    except Exception as exc:
        _warn_once("rm", f"rm failed ({exc!r}); using rm_mu")
        rm_scores = [CONFIG.rm_mu] * n

    out: list[float] = []
    for text, prompt, dscore, rscore in zip(texts, prompts, detox_scores, rm_scores):
        # --- degenerate / empty ---
        if not text or not text.strip():
            out.append(float(CONFIG.floor))
            continue
        # --- hard vetoes ---
        if CONFIG.veto_rep and severe_repetition(text):
            out.append(float(CONFIG.floor))
            continue
        word_count = len(text.split())
        char_count = len(text.strip())
        long_enough = word_count >= CONFIG.min_tokens or char_count >= CONFIG.min_tokens
        if CONFIG.veto_lang and long_enough and not is_english(text):
            out.append(float(CONFIG.floor))
            continue
        # --- soft sum ---
        base = saturated_detox(float(dscore))
        agree = rm_detox_agreement(float(rscore), float(dscore))
        rel = relevance(prompt, text, embedder) if CONFIG.w_rel else 0.0
        tail = CONFIG.mild_rep_w * _mild_rep_penalty(
            text
        ) + CONFIG.len_cap_w * _hit_cap_penalty(text)
        r = CONFIG.w_base * base + CONFIG.w_agree * agree + CONFIG.w_rel * rel - tail
        out.append(float(max(CONFIG.clip_lo, min(CONFIG.clip_hi, r))))
    return out


# Tag the function so the verl dispatcher knows whether to pass prompts.
# Set to ``False`` if your reward is purely response-side.
reward_score.prompt_conditioned = True
