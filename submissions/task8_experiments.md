
9 new PPO runs, all from Qwen/Qwen2.5-0.5B, one knob varied per run so each eval delta is attributable to a single mechanism. The two existing runs (all_defaults, W_REL=0.5) serve as reference points.

- Group A — veto knockouts (A1 lang-off, A2 rep-off, A3 rep-floor-strict): expect the Task 7 Mandarin and Task 6 gibberish attractors to return when each veto is removed — the cleanest causal demo.
- Group B — soft-term leave-one-out (B1 base-off, B2 agree-off, B3 rel-off): B3 completes a relevance ladder 0 / 0.25 / 0.5 alongside your existing runs.
- Group C — graded sweeps (C1 sat-threshold, C2 rm-sigma, C3 echo-threshold): shows continuous sensitivity, not just on/off.

The eval already emits the collapse signature (low mean_std + worst_of_k text), so no eval changes needed. The only code change is generalising tasks/merge_adaptor.py's hardcoded paths into a small --run-name CLI; plus the writeup and a docs "Ablation runs" subsection. I added a cheap local-reward sanity check to confirm each knob behaves before spending GPU, and a fallback to drop Group C if budget runs short.
Error: You are not in plan mode. To enter plan mode, call the EnterPlanMode tool first. If your plan was already approved, continue with implementation.

It looks like plan mode was already exited (the AskUserQuestion answers transitioned us out). The plan is written and complete at:

/Users/wendy.mak/.claude/plans/looking-at-users-wendy-mak-work-experime-expressive-blum.md

Since I'm no longer in plan mode, I'll pause here rather than start executing — these are GPU-heavy PPO runs and, per your global instructions, I won't kick off work (or any commits) without explicit go-ahead.

The proposed experiment set, at a glance:

┌───────┬───────────────────┬─────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────┐
│ Group │        Run        │              Override               │                               Demonstrates                                │
├───────┼───────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ A     │ veto_lang_off     │ DETOX_VETO_LANG=0                   │ Task 7 Mandarin attractor returns                                         │
├───────┼───────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ A     │ veto_rep_off      │ DETOX_VETO_REP=0                    │ Task 6 gibberish/loop attractor returns                                   │
├───────┼───────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ A     │ rep_floor_strict  │ DETOX_REP_FLOOR_RATIO=0.7           │ over-strict veto floors valid answers                                     │
├───────┼───────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ B     │ w_base_off        │ DETOX_W_BASE=0                      │ benign-anchor contribution                                                │
├───────┼───────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ B     │ w_agree_off       │ DETOX_W_AGREE=0                     │ RM↔Detoxify agreement contribution                                        │
├───────┼───────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ B     │ w_rel_off         │ DETOX_W_REL=0 DETOX_USE_EMBEDDING=0 │ strongest uniform-refusal collapse; completes relevance ladder 0/0.25/0.5 │
├───────┼───────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ C     │ sat_threshold_low │ DETOX_SAT_THRESHOLD=0.3             │ saturation-point sensitivity                                              │
├───────┼───────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ C     │ rm_sigma_wide     │ DETOX_RM_SIGMA=4.0                  │ agreement softness (brackets B2)                                          │
├───────┼───────────────────┼─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ C     │ echo_thresh_low   │ DETOX_ECHO_THRESHOLD=0.3            │ anti-echo aggressiveness                                                  │
└───────┴───────────────────┴─────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────┘

Each run: launch PPO (identical hyperparameters, only the -e DETOX_… line changes) → merge to HF → eval with task7_ppo_rm_eval → record support_rate/mean_max/mean_std + worst_of_k attractor text.
