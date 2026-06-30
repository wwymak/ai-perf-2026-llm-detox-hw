#!/usr/bin/env bash
# Task 8 — custom-reward knob-isolation study.
# Runs 9 PPO experiments, one DETOX_ knob varied per run, all from the same
# base actor (Qwen/Qwen2.5-0.5B) so results are comparable to the two existing
# runs (all_defaults, W_REL=0.5). Each run: PPO -> verl.model_merger -> eval.
#
# Usage:
#   bash submissions/task8_knob_study.sh                 # run all 9
#   bash submissions/task8_knob_study.sh veto_lang_off   # run just one
#   SANITY_ONLY=1 bash submissions/task8_knob_study.sh   # cheap CPU sanity, no GPU
#
# Run from the project root.
set -euo pipefail

IMG="verlai/verl:vllm023.dev1"
ACTOR="Qwen/Qwen2.5-0.5B"

# run_name -> DETOX_ override env-lines (space-separated "-e VAR=VAL" tokens).
declare -A OVERRIDES=(
  [veto_lang_off]="-e DETOX_VETO_LANG=0"
  [veto_rep_off]="-e DETOX_VETO_REP=0"
  [rep_floor_strict]="-e DETOX_REP_FLOOR_RATIO=0.7"
  [w_base_off]="-e DETOX_W_BASE=0"
  [w_agree_off]="-e DETOX_W_AGREE=0"
  [w_rel_off]="-e DETOX_W_REL=0 -e DETOX_USE_EMBEDDING=0"
  [sat_threshold_low]="-e DETOX_SAT_THRESHOLD=0.3"
  [rm_sigma_wide]="-e DETOX_RM_SIGMA=4.0"
  [echo_thresh_low]="-e DETOX_ECHO_THRESHOLD=0.3"
)
# Deterministic run order (Group A vetoes, Group B soft-terms, Group C sweeps).
ORDER=(veto_lang_off veto_rep_off rep_floor_strict \
       w_base_off w_agree_off w_rel_off \
       sat_threshold_low rm_sigma_wide echo_thresh_low)

# ---------------------------------------------------------------------------
# Cheap CPU sanity: confirm each toggled knob actually changes the reward on a
# known attractor fixture BEFORE spending GPU. No docker needed.
# ---------------------------------------------------------------------------
sanity() {
  echo "== sanity: language veto on vs off (Mandarin gratitude fixture) =="
  python -c "from tasks.task8_custom_reward import reward_score; print('veto ON ->', reward_score(['感谢您提供了这个反馈。'], ['x']))"
  DETOX_VETO_LANG=0 python -c "from tasks.task8_custom_reward import reward_score; print('veto OFF->', reward_score(['感谢您提供了这个反馈。'], ['x']))"

  echo "== sanity: repetition veto on vs off (looping fixture) =="
  python -c "from tasks.task8_custom_reward import reward_score; print('veto ON ->', reward_score(['TabControl\nTabControl\nTabControl\nTabControl'], ['x']))"
  DETOX_VETO_REP=0 python -c "from tasks.task8_custom_reward import reward_score; print('veto OFF->', reward_score(['TabControl\nTabControl\nTabControl\nTabControl'], ['x']))"
}

if [[ "${SANITY_ONLY:-0}" == "1" ]]; then
  sanity
  exit 0
fi

# ---------------------------------------------------------------------------
# One full experiment: PPO -> merge -> eval.
# ---------------------------------------------------------------------------
run_one() {
  local name="$1"
  local extra="${OVERRIDES[$name]}"
  local out="outputs/ppo_custom_${name}"
  local merged="checkpoints/ppo_custom_merged_${name}"
  local log="submissions/task8_log_${name}.txt"
  local eval_json="submissions/task8_ppo_custom_eval_${name}.json"

  echo "############################################################"
  echo "# RUN: ${name}   overrides: ${extra}"
  echo "############################################################"

  # --- 1. PPO (only the DETOX_ override + out dir + log differ from baseline) ---
  sudo docker run --rm --gpus all --ipc=host \
    -v "$(pwd)":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -v "$HOME/.cache/torch":/root/.cache/torch \
    -e TOXIC_REWARD=custom:tasks.task8_custom_reward \
    ${extra} \
    -e HYDRA_FULL_ERROR=1 \
    -e PYTHONPATH=/workspace \
    -w /workspace \
    "${IMG}" \
    bash -c "pip install -q verl==0.8.0 detoxify fasttext-langdetect sentence-transformers 2>&1 | tail -1 && \
             python -m src.toxic_rl.verl_runner --algo ppo \
               --train-parquet data/train.parquet \
               --val-parquet data/val.parquet \
               --actor-path ${ACTOR} \
               --out ${out} \
               --reward custom:tasks.task8_custom_reward \
               --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
               --rollout-n 8 --max-response-length 64 \
               --rollout-gpu-mem 0.25 \
               --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
               --save-freq 20 --test-freq 10" \
    2>&1 | tee "${log}"

  # --- 2. Merge FSDP shards -> HF checkpoint ---
  sudo docker run --rm --gpus all --ipc=host \
    -v "$(pwd)":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -w /workspace \
    "${IMG}" \
    bash -c "pip install -q verl==0.8.0 2>&1 | tail -1 && \
             python -m verl.model_merger merge --backend fsdp \
               --local_dir /workspace/${out}/global_step_100/actor \
               --target_dir /workspace/${merged}"
  sudo chmod 644 "${merged}/model.safetensors"

  # --- 3. Eval (greedy + K=16 diagnostic + worst-of-16) ---
  python -m tasks.task7_ppo_rm_eval \
    --ppo-dir "${merged}" \
    --out "${eval_json}"

  echo "# DONE ${name}: eval -> ${eval_json}"
}

# Optional pre-warm of fasttext lid.176 so the first GPU run doesn't cold-start.
python -c "from tasks.task8_custom_reward import reward_score; reward_score(['hello world'], ['hi'])" || true

if [[ $# -ge 1 ]]; then
  for name in "$@"; do run_one "$name"; done
else
  for name in "${ORDER[@]}"; do run_one "$name"; done
fi
