# %% [markdown]
# # Cross-model comparison of language space (Qwen3-32B, Qwen3.5-27B, Gemma 4 31B)
# Needs outputs/<model>/vectors.pt and tokstats.json (from 6_tokenization.py) for each model. Middle layer of each model.
# Key test: each model has its own tokenizer. If PC1 follows each model's OWN bytes/token more than the other models',
# PC1 is a tokenizer effect; if all models' PC1 agree with each other regardless, it's something shared (resource level).

# %%
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent / "outputs"
MODELS = ["Qwen3-32B", "Qwen3.5-27B", "gemma-4-31B-it"]
rho = lambda a, b: spearmanr(a, b)[0]
V, T, F = {}, {}, {}  # mid-layer vectors per language, tokenization stats, fraction fully in-language
for m in MODELS:
    d = torch.load(ROOT / m / "vectors.pt")
    L = d["vec"].shape[1] // 2
    V[m] = dict(zip(d["langs"], d["vec"][:, L].float().numpy()))
    F[m] = dict(zip(d["langs_all"], d["frac3"]))
    T[m] = json.loads((ROOT / m / "tokstats.json").read_text())["bytes/token"]

# %% 1. Coverage: which languages each model actually produces (>= 50/300 fully in-language)
for m in MODELS:
    print(f"{m:16s} {len(V[m])} languages pass")
for a in MODELS:
    for b in MODELS:
        if a != b:
            gain = sorted(set(V[b]) - set(V[a]))
            print(f"pass in {b} but not {a} ({len(gain)}): {' '.join(gain)}")
for l in ["hebrew", "welsh", "icelandic", "swahili", "yoruba", "japanese_romaji", "mandarin_pinyin", "sanskrit", "klingon"]:
    print(f"  {l:16s}" + "".join(f"  {m} {F[m].get(l, float('nan')):4.0%}" for m in MODELS))

# %% 2. Same geometry across models? (languages that pass in all three)
shared = sorted(set.intersection(*(set(V[m]) for m in MODELS)))
print(f"\n{len(shared)} languages pass in all three models")


def rsm(m):
    X = np.stack([V[m][l] for l in shared])
    X = X - X.mean(0)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return (X @ X.T)[np.triu_indices(len(shared), 1)]


def pc1(m):
    X = np.stack([V[m][l] for l in shared])
    z = PCA(1).fit_transform(X - X.mean(0))[:, 0]
    return z * np.sign(rho(z, [T[m][l] for l in shared]))  # orient: +PC1 = better tokenized


R = {m: rsm(m) for m in MODELS}
P = {m: pc1(m) for m in MODELS}
for i, a in enumerate(MODELS):
    for b in MODELS[i + 1:]:
        print(f"{a} vs {b}: RSA {rho(R[a], R[b]):.2f}, PC1 agreement {rho(P[a], P[b]):+.2f}")

# %% 3. Tokenizer test: does each model's PC1 follow its OWN tokenizer more than the others'?
print("\n|Spearman(PC1 of row model, bytes/token under column model's tokenizer)|")
print(" " * 16 + "".join(f"{m:>16s}" for m in MODELS))
for a in MODELS:
    print(f"{a:16s}" + "".join(f"{abs(rho(P[a], [T[b][l] for l in shared])):16.2f}" for b in MODELS))
print("tokenizers agree with each other on bytes/token:",
      ", ".join(f"{a}~{b} {rho([T[a][l] for l in shared], [T[b][l] for l in shared]):.2f}"
                for i, a in enumerate(MODELS) for b in MODELS[i + 1:]))

# %% 4. Script / dialect pairs in every model (centered cosine, shared-language centering)
pairs = [("hindi", "urdu"), ("hindi", "hindi_romanized"), ("serbian_cyrillic", "serbian_latin"),
         ("mandarin_simplified", "mandarin_traditional"), ("mandarin_simplified", "mandarin_pinyin"),
         ("mandarin_simplified", "cantonese"), ("malay", "indonesian"), ("arabic", "egyptian_arabic"), ("latin", "italian")]
mu = {m: np.mean([V[m][l] for l in shared], 0) for m in MODELS}
cos = lambda m, a, b: float((V[m][a] - mu[m]) @ (V[m][b] - mu[m]) / np.linalg.norm(V[m][a] - mu[m]) / np.linalg.norm(V[m][b] - mu[m]))
print("\n" + " " * 44 + "".join(f"{m:>16s}" for m in MODELS))
for a, b in pairs:
    print(f"{a:22s}{b:22s}" + "".join(f"{cos(m, a, b):16.2f}" if a in V[m] and b in V[m] else f"{'-':>16s}" for m in MODELS))
