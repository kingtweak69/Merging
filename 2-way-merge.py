import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os

MODEL_A = "llmfan46/Qwen3.6-35B-A3B-uncensored-heretic"
MODEL_B = "DavidAU/Qwen3.5-35B-A3B-HighIQ-Prototype-II"
OUTPUT_PATH = "/workspace/qwen-merge"

def get_weights(layer_idx):
    if layer_idx < 14:
        return (0.65, 0.35)
    elif layer_idx < 28:
        return (0.70, 0.30)
    else:
        return (0.75, 0.25)


print("Loading MODEL_A…")
model_a = AutoModelForCausalLM.from_pretrained(
    BASE_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("Loading MODEL_B…")
model_b = AutoModelForCausalLM.from_pretrained(
    CODER_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

print("Merging…")
state_a = model_a.state_dict()
state_b = model_b.state_dict()

merged_state = {}
skipped = []

for key in state_a.keys():
    output_layer_idx = None
    for i in range(40):
        if f".{i}." in key:
            output_layer_idx = i
            break

    if output_layer_idx is not None:
        wa, wb = get_weights(output_layer_idx)
        coder_layer_idx = output_layer_idx + CODER_OFFSET
        coder_key = key.replace(f".{output_layer_idx}.", f".{coder_layer_idx}.")

        has_b = coder_key in state_b and state_b[coder_key].shape == state_a[key].shape

        if has_b:
            merged_state[key] = (
                wa * state_a[key].float() +
                wb * state_b[coder_key].float()
            ).to(torch.bfloat16)
        else:
            merged_state[key] = state_a[key]
            skipped.append(key)
    else:
        merged_state[key] = state_a[key]

print(f"Skipped {len(skipped)} tensors")
print("Loading merged weights…")
model_a.load_state_dict(merged_state)

print(f"Saving to {OUTPUT_PATH}…")
os.makedirs(OUTPUT_PATH, exist_ok=True)
model_a.save_pretrained(OUTPUT_PATH, safe_serialization=True, max_shard_size="10GB")

tokenizer = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)

del model_a, model_b, state_a, state_b, merged_state
gc.collect()

print(f"Done. Saved to {OUTPUT_PATH}")
