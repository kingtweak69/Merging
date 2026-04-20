import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os

QWOPUS_ID = “huihui-ai/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated”
HERETIC_ID = “Youssofal/Qwen3.6-35B-A3B-Abliterated-Heretic-BF16”
CODER_ID = “trohrbaugh/Qwen3-Coder-Next-heretic”
OUTPUT_PATH = “/content/drive/MyDrive/tweakbot-v2-trohr”

CODER_OFFSET = 8

def get_weights(layer_idx):
if layer_idx < 14:
return (0.48, 0.32, 0.20)
elif layer_idx < 28:
return (0.40, 0.28, 0.32)
else:
return (0.30, 0.22, 0.48)

print(“Loading Qwopus…”)
model_a = AutoModelForCausalLM.from_pretrained(
QWOPUS_ID,
torch_dtype=torch.bfloat16,
device_map=“cpu”,
trust_remote_code=True,
low_cpu_mem_usage=True,
)

print(“Loading Heretic 3.6…”)
model_b = AutoModelForCausalLM.from_pretrained(
HERETIC_ID,
torch_dtype=torch.bfloat16,
device_map=“cpu”,
trust_remote_code=True,
low_cpu_mem_usage=True,
)

print(“Loading trohrbaugh Coder-Next heretic…”)
model_c = AutoModelForCausalLM.from_pretrained(
CODER_ID,
torch_dtype=torch.bfloat16,
device_map=“cpu”,
trust_remote_code=True,
low_cpu_mem_usage=True,
)

print(“Merging…”)
state_a = model_a.state_dict()
state_b = model_b.state_dict()
state_c = model_c.state_dict()

merged_state = {}
skipped = []

for key in state_b.keys():
output_layer_idx = None
for i in range(40):
if f”.{i}.” in key:
output_layer_idx = i
break

```
if output_layer_idx is not None:
    wa, wb, wc = get_weights(output_layer_idx)
    coder_layer_idx = output_layer_idx + CODER_OFFSET
    coder_key = key.replace(f".{output_layer_idx}.", f".{coder_layer_idx}.")

    has_a = key in state_a and state_a[key].shape == state_b[key].shape
    has_c = coder_key in state_c and state_c[coder_key].shape == state_b[key].shape

    if has_a and has_c:
        merged_state[key] = (
            wa * state_a[key].float() +
            wb * state_b[key].float() +
            wc * state_c[coder_key].float()
        ).to(torch.bfloat16)
    elif has_a:
        total = wa + wb
        merged_state[key] = (
            (wa / total) * state_a[key].float() +
            (wb / total) * state_b[key].float()
        ).to(torch.bfloat16)
        skipped.append(key)
    else:
        merged_state[key] = state_b[key]
        skipped.append(key)
else:
    if key in state_a and state_a[key].shape == state_b[key].shape:
        merged_state[key] = (
            0.5 * state_a[key].float() +
            0.5 * state_b[key].float()
        ).to(torch.bfloat16)
    else:
        merged_state[key] = state_b[key]
```

print(f”Skipped {len(skipped)} tensors”)
print(“Loading merged weights…”)
model_b.load_state_dict(merged_state)

print(f”Saving to {OUTPUT_PATH}…”)
os.makedirs(OUTPUT_PATH, exist_ok=True)
model_b.save_pretrained(OUTPUT_PATH, safe_serialization=True, max_shard_size=“10GB”)

tokenizer = AutoTokenizer.from_pretrained(HERETIC_ID, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)

del model_a, model_b, model_c, state_a, state_b, state_c, merged_state
gc.collect()

print(f”Done. Saved to {OUTPUT_PATH}”)
