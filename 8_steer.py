"""Step 8 (GPU, HF transformers): steer with a PC of a language space (STEER_SET / STEER_PC in common.py: all, strong or
weak languages; PC1 by default). No system prompt; the same 10 questions asked in English, Chinese, Turkish and Hindi
(data/steer_questions.json, hand-translated). Add alpha * sd * PC to the residual stream at the output of the middle
layer (the layer the space was built from), with the PC oriented so English projects onto + (English need not be in the set).
sd = standard deviation of the set's language vectors' scores on that PC, so alpha is in "language spread" units.
  MODES["all"]: add at every position (question + answer)
  MODES["gen"]: add only from the last prompt token on (the position that predicts the first answer token) = answer only
Greedy decoding so alpha=0 is the same baseline for both modes. Output: outputs/<model>/steer[_pcN]/samples.jsonl"""
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DATA, MODEL, OUT, STEER, STEER_PC, STEER_SET, STRONG, WEAK, read_jsonl, write_jsonl

ALPHAS = [-8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8]  # ±3/5/6 added after a first look: the switch is between 4 and 8
MODES = ["all", "gen"]
QLANGS = ["english", "chinese", "turkish", "hindi"]
MAX_NEW = 320

# the PC of the set's language vectors at the middle layer (same as 5_pca.py / 10_subsets.py), oriented so English is +
d = torch.load(OUT / "vectors.pt")
frac = dict(zip(d["langs_all"], d["frac3"]))
keep = [i for i, l in enumerate(d["langs"]) if STEER_SET == "all" or (frac[l] >= STRONG if STEER_SET == "strong" else frac[l] < WEAK)]
names = [d["langs"][i] for i in keep]
LAYER = d["vec"].shape[1] // 2
X_all = d["vec"][:, LAYER].float()
X = X_all[keep]
mu = X.mean(0)
pc1 = torch.linalg.svd(X - mu, full_matrices=False).Vh[STEER_PC]  # named pc1 below, but it is whichever PC is steered
z = (X - mu) @ pc1
if (X_all[d["langs"].index("english")] - mu) @ pc1 < 0:
    pc1, z = -pc1, -z
sd = z.std().item()
print(f"{STEER_SET} ({len(keep)} languages), layer {LAYER}, PC{STEER_PC + 1} sd {sd:.1f}, |mean language vector| {mu.norm():.1f}")
print(f"PC{STEER_PC + 1} ends:", [names[i] for i in z.argsort()[:5]], "...", [names[i] for i in z.argsort()[-5:]])

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
layer = model.model.layers[LAYER]
vec = pc1.to("cuda", torch.bfloat16)
cur = {"add": None, "mode": "all"}


def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if cur["add"] is not None:
        if cur["mode"] == "all" or h.shape[1] == 1:  # decode steps are 1 token wide
            h = h + cur["add"]
        else:  # prefill in "gen" mode: only the last prompt position (left padding, so it's -1 for every row)
            h = h.clone()
            h[:, -1] += cur["add"]
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


layer.register_forward_hook(hook)

qs = json.loads((DATA / "steer_questions.json").read_text(encoding="utf-8"))
prompts = [(ql, q["id"], q[ql]) for ql in QLANGS for q in qs]
enc = tok([tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) for _, _, p in prompts],
          return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

# residual norm at this layer on these prompts, for scale
norms = []
cur["add"] = None
probe = layer.register_forward_hook(lambda m, i, o: norms.append((o[0] if isinstance(o, tuple) else o)[:, -1].float().norm(dim=-1).mean().item()))
with torch.inference_mode():
    model(**enc, logits_to_keep=1)
probe.remove()
print(f"residual norm at the last prompt token: {norms[0]:.1f}; 1 sd of steering = {sd / norms[0]:.0%} of it")

f = STEER / "samples.jsonl"
rows = read_jsonl(f) if f.exists() else []  # rerunning only adds (mode, alpha) pairs not done yet
done = {(r["mode"], r["alpha"]) for r in rows}
for mode in MODES:
    for a in tqdm([a for a in ALPHAS if (mode, a) not in done], desc=mode):
        if a == 0 and mode != MODES[0]:
            rows += [dict(r, mode=mode) for r in rows if r["alpha"] == 0 and r["mode"] == MODES[0]]
            continue
        cur["mode"], cur["add"] = mode, (a * sd * vec if a else None)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None)
        for (ql, qid, p), o in zip(prompts, out[:, enc.input_ids.shape[1]:]):
            n = int((o != tok.pad_token_id).sum())
            rows.append({"mode": mode, "alpha": a, "qlang": ql, "q_id": qid, "question": p,
                         "response": tok.decode(o, skip_special_tokens=True), "n_tokens": n, "capped": n >= MAX_NEW})
rows.sort(key=lambda r: (r["mode"], r["alpha"]))
write_jsonl(f, rows)
print(f"{len(rows)} samples -> {f}")
