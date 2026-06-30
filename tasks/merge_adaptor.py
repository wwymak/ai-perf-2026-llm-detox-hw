from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    dtype=torch.float32,
    device_map="cuda",
)

m = PeftModel.from_pretrained(
    base, str("/home/wwymak/ai-perf-2026-llm-detox-hw/checkpoints/sft")
)
m = m.merge_and_unload()
m.save_pretrained("/home/wwymak/ai-perf-2026-llm-detox-hw/checkpoints/merged-sft")
