"""Step 2 (GPU, HF transformers): mean residual-stream activation over the response tokens, at the output of every
decoder layer (same as the persona repo's post-MLP residual). One row per response, same order as the jsonl.
Output: outputs/<model>/activations/{lang}.pt -> bf16 tensor (n_responses, n_layers, d_model)."""
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from common import MODEL, OUT, read_jsonl

BATCH = 16
MAX_LEN = 2048

tok = AutoTokenizer.from_pretrained(MODEL)
cfg = AutoConfig.from_pretrained(MODEL)
# multimodal checkpoints (Gemma 4, Qwen3.5) must load with their own class, or text weights can silently stay random
Auto = AutoModelForImageTextToText if cfg.architectures[0].endswith("ForConditionalGeneration") else AutoModelForCausalLM
model = Auto.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
tcfg = cfg.get_text_config()
layers = next(m for name, m in model.named_modules()  # the text decoder's layer list, wherever the wrapper puts it
              if isinstance(m, torch.nn.ModuleList) and len(m) == tcfg.num_hidden_layers
              and not any(v in name for v in ("vision", "visual", "audio")))
D = tcfg.hidden_size
cur = {}  # filled per batch: response mask + output buffer


def make_hook(i):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        m = cur["mask"]
        cur["acts"][:, i] = ((h.float() * m).sum(1) / m.sum(1).clamp(min=1)).to(torch.bfloat16)
    return hook


for i, layer in enumerate(layers):
    layer.register_forward_hook(make_hook(i))


def encode(r):
    prompt = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False).input_ids
    a = tok(r["response"], add_special_tokens=False).input_ids
    return (p + a)[:MAX_LEN], ([0] * len(p) + [1] * len(a))[:MAX_LEN]


for f in sorted((OUT / "responses").glob("*.jsonl")):
    out_file = OUT / "activations" / f"{f.stem}.pt"
    if out_file.exists():
        continue
    rows = read_jsonl(f)
    enc = [encode(r) for r in rows]
    order = sorted(range(len(enc)), key=lambda j: len(enc[j][0]))  # length-sorted batches = less padding
    result = torch.zeros(len(rows), len(layers), D, dtype=torch.bfloat16)
    for b in tqdm(range(0, len(order), BATCH), desc=f.stem, leave=False):
        idx = order[b:b + BATCH]
        T = max(len(enc[j][0]) for j in idx)
        ids = torch.full((len(idx), T), tok.pad_token_id)
        attn = torch.zeros(len(idx), T, dtype=torch.long)
        mask = torch.zeros(len(idx), T, 1)
        for k, j in enumerate(idx):  # right padding
            n = len(enc[j][0])
            ids[k, :n] = torch.tensor(enc[j][0])
            attn[k, :n] = 1
            mask[k, :n, 0] = torch.tensor(enc[j][1], dtype=torch.float)
        cur["mask"] = mask.cuda()
        cur["acts"] = torch.empty(len(idx), len(layers), D, dtype=torch.bfloat16, device="cuda")
        with torch.inference_mode():
            model(input_ids=ids.cuda(), attention_mask=attn.cuda(), logits_to_keep=1)  # logits for 1 position only
        result[idx] = cur["acts"].cpu()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_file)
    print(f"{f.stem}: {tuple(result.shape)}")
