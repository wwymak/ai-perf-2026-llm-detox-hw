# Task 8 — Custom Reward Design

**Date:** 2026-06-28
**File:** `tasks/task8_custom_reward.py` (+ `submissions/task8_writeup.md`)

## Goal

Design a PPO reward that cannot be saturated by a single prompt-independent
template. Tasks 6 and 7 each collapsed onto an attractor:

- **Task 6 (`inv:detoxify`)** — repeated code-token garbage
  (`TabControl\nTabControl…`, `PageRouteBuilder<…>`), short nonsense
  (`elize it!`).
- **Task 7 (`rm:<RM>`)** — Chinese gratitude phrases (`感谢您提供了这个反馈。`,
  `谢谢你！`) and verbatim system-prompt echoes
  (`You are a helpful assistant. Respond to the user…`).

Both saturate the reward cheaply with a fixed template. The reward must rank
real, on-topic, benign English answers **above** any such attractor.

## Structure: additive + hard veto

A weighted sum of soft "quality" terms, wrapped in hard vetoes that floor the
reward regardless of the soft terms. Vetoes encode *categorical* failures
("not a valid answer at all"); the weighted sum shapes *quality* among
survivors.

```
reward_score(texts, prompts):
  load_models_once()                 # module-scope cache; load only enabled models
  for (text, prompt) in zip(texts, prompts):
      # ---- HARD VETOES (return FLOOR, skip everything) ----
      empty/whitespace                                   -> FLOOR
      VETO_LANG and tokens>=MIN_TOKENS and not english   -> FLOOR
      VETO_REP and distinct_trigram_ratio < REP_FLOOR    -> FLOOR
      # ---- SOFT QUALITY SUM (survivors only) ----
      base      = saturated_detox(text)                  # benign anchor, flattens
      agree     = rm_detox_agreement(text)               # both squashed to [0,1]
      relevance = embed_cosine(prompt, text) - echo_guard(prompt, text)
      tail      = -MILD_REP_W*trigram_rep(text) - LEN_CAP_W*hit_cap(text)
      r = W_BASE*base + W_AGREE*agree + W_REL*relevance + tail
      out.append(clip(r, CLIP_LO, CLIP_HI))
  return out
```

`reward_score.prompt_conditioned = True`.

### Vetoes (decided)

- **Wrong language** — `fasttext-langdetect`; veto if top label ≠ `en` AND
  confidence ≥ `LANG_MIN_CONF`. Guarded by `tokens >= MIN_TOKENS` (fasttext is
  noise on short strings; those fall through to soft handling). Sanitize
  newlines before detection.
- **Severe repetition** — distinct-trigram ratio `< REP_FLOOR_RATIO`
  (default 0.4). Milder repetition stays a soft tail penalty.

System-prompt echo and degenerate/short are **not** vetoes (echo → low
relevance; short → small length-floor penalty), to keep vetoes high-precision.

### Soft terms (decided)

- **Saturated Detoxify** — `benign = 1 - detox`; `min(1, benign/SAT_THRESHOLD)`
  so once `detox ≤ ~0.1` reward flattens to 1.0. Removes "push benign harder".
- **RM↔Detoxify agreement** — `detox_b = 1 - detox`;
  `rm_b = sigmoid((rm - RM_MU)/RM_SIGMA)`; `agree = 1 - |rm_b - detox_b|`.
  High where both call it benign; low where one is gamed.
- **Prompt relevance (embedding cosine)** — `cosine(embed(prompt), embed(text))`
  minus an **anti-echo guard**: if normalized prompt-token overlap >
  `ECHO_THRESHOLD`, subtract a penalty so the policy can't score relevance by
  parroting the prompt.
- **Tail** — soft trigram-repetition + length-cap penalties for near-misses
  the vetoes don't catch. Reuses `_trigram_repeat_score` / `_hit_cap_score`
  logic from `src/toxic_rl/verl_reward.py`.

## Config: env-var dataclass

Module-level CONFIG read once at load, each key `DETOX_`-prefixed and
overridable per docker run (fits the existing `-e` pattern; self-documents in
the log). A flag of `0` zeroes a term's weight **and** skips its model load.

Keys: `VETO_LANG`, `VETO_REP`, `LANG_MIN_CONF`, `REP_FLOOR_RATIO`,
`MIN_TOKENS`, `USE_EMBEDDING`, `SAT_THRESHOLD`, `RM_MU`, `RM_SIGMA`,
`ECHO_THRESHOLD`, `W_BASE`, `W_AGREE`, `W_REL`, `MILD_REP_W`, `LEN_CAP_W`,
`CLIP_LO`, `CLIP_HI`, `FLOOR`. Defaults: `CLIP=[-1,1]`, `FLOOR=-1.0`,
`REP_FLOOR_RATIO=0.4`, `RM_MU=3.0`, `RM_SIGMA=2.0`.

## Components

Small, independently-testable functions; reuse existing wrappers:

- `_load_models_once()` — module-global cache (mirrors
  `DetoxifyReward._MODEL_CACHE`). Reuses `DetoxifyReward` and
  `TrainedRewardModel` (which already has `score(texts, prompts=)`). Loads the
  embedder only if `USE_EMBEDDING`.
- `is_english(text, min_conf) -> bool`
- `severe_repetition(text, floor_ratio) -> bool`
- `saturated_detox(text) -> [0,1]`
- `rm_detox_agreement(text) -> [0,1]`
- `relevance(prompt, text) -> float` (cosine − echo_guard)
- aggregation + clip

## Error handling

Never crash the rollout (an exception kills the PPO run):

- Per-term try/except → neutral value on failure (relevance→0, agreement→0.5,
  veto→no-veto), log once.
- Missing/empty prompt (`prompts=None` or `""`, e.g. parquet didn't populate
  `prompt_text`) → prompt-conditioned terms return neutral + one-time warning;
  response-side terms still work.
- Empty/whitespace completion → `FLOOR` before vetoes.
- Sanitize newlines before fasttext.

## Testing (minimal — POC)

`tests/test_task8_custom_reward.py`, CPU-only, heavy models monkeypatched.
Just enough to confirm the implementation is on track:

1. **Veto fixtures** (real hacks from the eval JSONs): `"感谢您提供了这个反馈。"`
   → language veto → FLOOR; `"TabControl\nTabControl\nTabControl\nTabControl"`
   → repetition veto → FLOOR.
2. **Ordering invariant**: a benign on-topic English answer outscores every
   attractor fixture.
3. **No-crash**: `None` prompt / empty string / malformed input → finite
   float, never raises.

## Risks / open checks (resolve during implementation)

1. **Verify the parquet builder populates `extra_info["prompt_text"]`** —
   `composite_rm` assumes it; if absent, relevance/agreement silently degrade.
2. **Model caching** — fasttext, Detoxify, RM, embedder each loaded once at
   module scope or PPO crawls.
3. **RM normalization** — reuse μ=3/σ=2 calibration but expose as config;
   sign may shift for detox direction.
4. **fasttext short-text noise** — `MIN_TOKENS` guard.
5. **Reward range vs `--kl-coef 0.001`** — keep output ~`[-1,1]` so KL pressure
   is comparable to the inv:detoxify / rm runs; note range in writeup.
6. **Container deps** — `fasttext-langdetect` + `sentence-transformers` must be
   added to the docker `pip install` line for the task-8 run. Document the
   modified run block in `docs/task8_custom_reward.md` (NOT the README).
   Embedder weights download to the bind-mounted HF cache (pre-warm), fasttext
   `lid.176` downloads to its own path.

## Deliverables

- `tasks/task8_custom_reward.py` — implementation
- `tests/test_task8_custom_reward.py` — minimal tests
- `submissions/task8_writeup.md` — what was tried, what collapsed into what,
  final design, why it works/fails
- `docs/task8_custom_reward.md` — a standalone doc for this task: config
  reference (env-var knobs), the modified docker run block with the added
  `pip install` deps, and run instructions. **README stays unchanged.**
