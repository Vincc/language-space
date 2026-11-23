# %% [markdown]
# # Language space within capability groups (CPU, needs vectors.pt + tokstats.json)
# Split the passing languages by how reliably the model writes them (share of 300 responses judged fully in-language):
# STRONG >= 0.9, WEAK < 0.7 (the 0.7-0.9 middle is left out). Redo PCA inside each group and ask:
# 1. how many dimensions does each group's space need?  2. what do its top PCs separate?
# 3. is its PC1 still the resource / tokenization axis?  4. do its PCs line up with the full space's PCs?
# Small groups always look lower-dimensional, so every number is compared with random subsets of the same size.

# %%
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from common import OUT, STRONG, WEAK

N_RANDOM = 200
rho = lambda a, b: spearmanr(a, b)[0]

d = torch.load(OUT / "vectors.pt")
langs = d["langs"]
V = d["vec"].float().numpy()
LAYER = V.shape[1] // 2
X = V[:, LAYER]
frac = dict(zip(d["langs_all"], d["frac3"]))
f3 = np.array([frac[l] for l in langs])
bpt = np.array([json.loads((OUT / "tokstats.json").read_text())["bytes/token"][l] for l in langs])
groups = {"all": np.arange(len(langs)), "strong": np.where(f3 >= STRONG)[0], "weak": np.where(f3 < WEAK)[0]}
full = PCA(10).fit(X - X.mean(0))
rng = np.random.default_rng(0)


def fit(idx, k=10):
    Y = X[idx] - X[idx].mean(0)
    return PCA(min(k, len(idx) - 1)).fit(Y), Y


def dims(p, t):  # number of PCs for a cumulative share t (inf if the k fitted PCs don't get there)
    c = np.cumsum(p.explained_variance_ratio_)
    return int(np.argmax(c >= t)) + 1 if c[-1] >= t else float("inf")


# %% 1. Dimensionality, against random subsets of the same size
for g, idx in groups.items():
    p, _ = fit(idx, len(idx) - 1)
    ev = p.explained_variance_ratio_
    rnd = np.array([PCA(5).fit(X[s] - X[s].mean(0)).explained_variance_ratio_
                    for s in (rng.choice(len(langs), len(idx), replace=False) for _ in range(N_RANDOM))]) if g != "all" else None
    line = f"{g:7s} n={len(idx):2d}  PC1-5 {np.round(ev[:5], 3)}  dims for 70/80/90%: {dims(p, .7)}/{dims(p, .8)}/{dims(p, .9)}"
    if rnd is not None:
        line += f"\n{'':13s}random subsets of {len(idx)}: PC1-5 {np.round(rnd.mean(0), 3)} (PC1 95% range {np.percentile(rnd[:, 0], 2.5):.3f}-{np.percentile(rnd[:, 0], 97.5):.3f})"
    print(line)

# %% 2. What the top PCs separate, and 3. whether PC1 is still the resource / tokenization axis
for g, idx in groups.items():
    p, Y = fit(idx)
    Z = Y @ p.components_[:3].T
    print(f"\n=== {g} ({len(idx)} languages)")
    for j in range(3):
        o = np.argsort(Z[:, j])
        print(f"PC{j + 1} {p.explained_variance_ratio_[j]:.1%}  low: {[langs[idx[i]] for i in o[:5]]}\n{'':10s}high: {[langs[idx[i]] for i in o[::-1][:5]]}")
        print(f"{'':10s}|rho| with bytes/token {abs(rho(Z[:, j], bpt[idx])):.2f}, with fraction in-language {abs(rho(Z[:, j], f3[idx])):.2f}")

# %% 4. Alignment with the full space: |cos| between each group's PC1-3 and the full space's PC1-5
print("\n|cos(group PC, full-space PC)|   columns: full PC1..PC5, then the share of the group PC inside the full top-10")
for g in ["strong", "weak"]:
    p, _ = fit(groups[g])
    for j in range(3):
        c = np.abs(full.components_[:5] @ p.components_[j])
        inside = np.sum((full.components_ @ p.components_[j]) ** 2)
        print(f"{g:7s} PC{j + 1}: " + " ".join(f"{v:.2f}" for v in c) + f"   in full top-10: {inside:.2f}")
ps, pw = fit(groups["strong"])[0], fit(groups["weak"])[0]
print("strong vs weak, |cos| of PC1-3 with each other:\n", np.round(np.abs(ps.components_[:3] @ pw.components_[:3].T), 2))
