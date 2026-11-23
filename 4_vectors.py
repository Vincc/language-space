"""Step 4 (CPU): per-language mean vectors from fully-in-language responses (score == 3), like the persona repo's role vectors.
Also keeps the unfiltered mean of ALL responses for EVERY language, to see where the model lands for languages it can't produce.
Output: outputs/<model>/vectors.pt ->
  {"langs", "n3", "vec" (n_kept, L, D),          # score-3 means, languages with >= MIN_COUNT score-3 responses
   "langs_all", "frac3", "vec_all" (n_all, L, D)} # unfiltered means, every language"""
import json

import torch

from common import OUT, languages

MIN_COUNT = 50  # out of 300 responses

d = {"langs": [], "n3": [], "vec": [], "langs_all": [], "frac3": [], "vec_all": []}
for l in languages():
    act_f, score_f = OUT / "activations" / f"{l['id']}.pt", OUT / "scores" / f"{l['id']}.json"
    if not (act_f.exists() and score_f.exists()):
        continue
    acts = torch.load(act_f).float()
    keep = torch.tensor([s == 3 for s in json.loads(score_f.read_text())])
    d["langs_all"].append(l["id"])
    d["frac3"].append(keep.float().mean().item())
    d["vec_all"].append(acts.mean(0))
    if keep.sum() >= MIN_COUNT:
        d["langs"].append(l["id"])
        d["n3"].append(int(keep.sum()))
        d["vec"].append(acts[keep].mean(0))
d["vec"], d["vec_all"] = torch.stack(d["vec"]).bfloat16(), torch.stack(d["vec_all"]).bfloat16()
dropped = sorted(set(d["langs_all"]) - set(d["langs"]))
print(f"{len(d['langs'])}/{len(d['langs_all'])} languages pass; dropped: {' '.join(dropped)}")
torch.save(d, OUT / "vectors.pt")
