# Task 8: Custom Reward Configuration and PPO Execution

## Reward Design

The custom reward function (`tasks/task8_custom_reward.py`) combines hard vetoes with a weighted soft sum to resist single-template collapse. It applies strict filters for degenerate repetition (trigram diversity below a floor) and non-English text (via fasttext language detection), then scores survivors using saturated Detoxify (capped at a benign threshold to remove saturation incentive), RM↔Detoxify agreement (high when both agree the text is benign), and embedding-based prompt-relevance (with an echo-parroting penalty). The design targets the specific attractors observed in Tasks 6 and 7: the OpenAI-style refusal template that saturates Detoxify in Task 6, and the Mandarin-greeting or "context-requesting" template from the learned RM in Task 7. By combining multiple incommensurable signals (lexical saturation, learned preference, semantic relevance, and diversity) with environment-driven weight tuning, the reward prevents the policy from gaming any single signal.

## Configuration

Every `DETOX_` environment variable is listed below with its default value and one-line description:

| Env Var | Default | Description |
|---------|---------|-------------|
| `DETOX_VETO_LANG` | `True` | Hard veto: reject completions detected as non-English (when confidence ≥ `DETOX_LANG_MIN_CONF`) |
| `DETOX_LANG_MIN_CONF` | `0.5` | Language detection confidence threshold; vetoed only if `score ≥ this` and language ≠ English |
| `DETOX_VETO_REP` | `True` | Hard veto: reject completions with trigram diversity below `DETOX_REP_FLOOR_RATIO` (degenerate repetition) |
| `DETOX_REP_FLOOR_RATIO` | `0.4` | Distinct-trigram ratio floor; `veto_rep=True` floors any text with `distinct_trigrams / total_trigrams < this` |
| `DETOX_MIN_TOKENS` | `4` | Minimum token count for language veto to apply (short texts skip the language check) |
| `DETOX_USE_EMBEDDING` | `True` | Load and use the sentence-transformer embedder for prompt-relevance scoring |
| `DETOX_SAT_THRESHOLD` | `0.9` | Detoxify saturation point; `saturated_detox = min(1.0, (1 - detox_score) / this)` |
| `DETOX_RM_MU` | `3.0` | RM score sigmoid center; used in `rm_detox_agreement` computation |
| `DETOX_RM_SIGMA` | `2.0` | RM score sigmoid scale; wider = softer discrimination |
| `DETOX_ECHO_THRESHOLD` | `0.6` | Token-overlap ratio above which the text is flagged as parroting the prompt |
| `DETOX_W_BASE` | `0.5` | Weight for saturated-Detoxify term |
| `DETOX_W_AGREE` | `0.25` | Weight for RM↔Detoxify agreement term |
| `DETOX_W_REL` | `0.25` | Weight for embedding-relevance term |
| `DETOX_MILD_REP_W` | `0.3` | Weight on soft repetition penalty (not a hard veto, just a score deduction) |
| `DETOX_LEN_CAP_W` | `0.2` | Weight on length-cap-hit penalty (penalises reaching the token limit) |
| `DETOX_CLIP_LO` | `-1.0` | Lower reward bound; all scores clipped to `[DETOX_CLIP_LO, DETOX_CLIP_HI]` |
| `DETOX_CLIP_HI` | `1.0` | Upper reward bound |
| `DETOX_FLOOR` | `-1.0` | Reward for completions that fail hard vetoes |
| `DETOX_RM_DIR` | `checkpoints/rm` | Path to the trained reward model directory |

## Docker Run Block for Task 8 PPO

Copy the following block and run from the project root. This launches verl's PPO trainer with the custom reward, mounting the repo, Hugging Face cache, and torch cache into the container.

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v $(pwd):/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/torch:/root/.cache/torch \
  -e TOXIC_REWARD=custom:tasks.task8_custom_reward \
  -e DETOX_USE_EMBEDDING=0 \
  -e DETOX_W_REL=0.4 \
  -e HYDRA_FULL_ERROR=1 \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  verlai/verl:vllm023.dev1 \
  bash -c "pip install -q verl==0.8.0 detoxify fasttext-langdetect sentence-transformers 2>&1 | tail -1 && \
           python -m src.toxic_rl.verl_runner --algo ppo \
             --train-parquet data/train.parquet \
             --val-parquet data/val.parquet \
             --actor-path Qwen/Qwen2.5-0.5B \
             --out outputs/ppo_custom \
             --reward custom:tasks.task8_custom_reward \
             --total-steps 100 --train-batch-size 16 --ppo-mini-batch-size 8 \
             --rollout-n 8 --max-response-length 64 \
             --rollout-gpu-mem 0.25 \
             --actor-lr 2e-6 --critic-lr 1e-5 --kl-coef 0.001 \
             --save-freq 20 --test-freq 10" \
  2>&1 | tee submissions/task8_log.txt
```

**How to override knobs:** add or change `-e DETOX_<VAR>=<value>` lines. For example, to disable repetition penalty and raise the relevance weight:

```bash
-e DETOX_MILD_REP_W=0.0 \
-e DETOX_W_REL=0.5 \
```

## Model Caching

The custom reward loads three model artifacts on first use:

1. **Detoxify** (off-the-shelf toxic-bert classifier): shipped with the `detoxify` package; no external download needed.

2. **Sentence-Transformer embedder** (`all-MiniLM-L6-v2`): downloaded on first use to `~/.cache/huggingface/` (inside the container, this is `/root/.cache/huggingface/` bound-mounted from the host). Subsequent runs reuse the cached weights. Set `DETOX_USE_EMBEDDING=0` to skip loading.

3. **fasttext language-detection model** (`lid.176`): the `fasttext-langdetect` package downloads this model to a hardcoded path inside the container on first use. Cold start adds ~30 seconds; subsequent runs are fast. To pre-warm the model before a long PPO run, execute the reward once on a dummy string:

   ```bash
   python -c "from tasks.task8_custom_reward import reward_score; reward_score(['hello world'])"
   ```

   This ensures the fasttext model is cached before the verl docker container starts.

## Writeup

After the PPO run completes and the policy is merged to HF format, run:

```bash
python -m tasks.task7_ppo_rm_eval \
    --ppo-dir checkpoints/ppo_custom_merged \
    --out submissions/task8_ppo_custom_eval.json
```

Then write your findings in **`submissions/task8_writeup.md`**. Include:

- What patterns collapsed into what during your experimental iterations
- The final reward design and why you chose this combination of signals
- Whether the custom reward succeeded in avoiding template collapse, and if so, what the new attractor looks like (if any)
- Any observations about knob sensitivity or failure modes you discovered
