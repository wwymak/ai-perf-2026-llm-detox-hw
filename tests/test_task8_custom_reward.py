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
    out = patched.reward_score(["TabControl\n" * 8], prompts=["describe my coworker"])
    assert out[0] == pytest.approx(patched.CONFIG.floor)


def test_benign_ontopic_outscores_attractors(patched):
    prompt = "describe my coworker who keeps interrupting me"
    good = "my coworker often interrupts me in meetings and i find it hard to finish my point"
    attractors = [
        "感谢您提供了这个反馈。",  # wrong language
        "TabControl\n" * 8,  # repetition loop
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
