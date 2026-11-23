# %% [markdown]
# # Tokenization control
# Is language-space PC1 (partly) a tokenizer artifact? Scripts the tokenizer handles badly get split into byte-level
# fragments (low bytes/token) and long token sequences, which could shift mean activations no matter the language.
# 1. correlate PC scores with tokenization stats        2. regress the stats out, redo PCA
# 3. within one language, does a response's PC1 follow its own tokenization? how much between-language spread could that explain?
# 4. same language, different script                     5. by layer: PC1 vs tokenization, and geometry vs layer 0 (~token identity)
# Needs responses/, scores/, activations/ (run it on the GPU box; CPU only). Figures go to outputs/<model>/figs/.

# %%
import json

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from transformers import AutoTokenizer

from common import MODEL, OUT, read_jsonl

d = torch.load(OUT / "vectors.pt")
langs = d["langs"]
V = d["vec"].float().numpy()  # (n_lang, n_layers, D)
LAYER = V.shape[1] // 2
tok = AutoTokenizer.from_pretrained(MODEL)
rho = lambda a, b: spearmanr(a, b)[0]


def pca(X, k=5):
    p = PCA(k).fit(X)
    return p.transform(X), p


# %% 1. Tokenization stats per language, over the same score-3 responses the language vectors average
S = {}
for l in langs:
    rows = read_jsonl(OUT / "responses" / f"{l}.jsonl")
    keep = [i for i, s in enumerate(json.loads((OUT / "scores" / f"{l}.json").read_text())) if s == 3]
    ntok = np.array([len(tok(rows[i]["response"], add_special_tokens=False).input_ids) for i in keep])
    nbytes = np.array([len(rows[i]["response"].encode()) for i in keep])
    S[l] = {"keep": keep, "ntok": ntok, "bpt": nbytes / np.maximum(ntok, 1),
            "capped": np.array([rows[i]["finish"] == "length" for i in keep])}
stats = {"bytes/token": np.array([S[l]["bpt"].mean() for l in langs]),
         "tokens/response": np.array([S[l]["ntok"].mean() for l in langs]),
         "frac capped": np.array([S[l]["capped"].mean() for l in langs])}
(OUT / "tokstats.json").write_text(json.dumps({k: dict(zip(langs, v.tolist())) for k, v in stats.items()}))  # for 7_compare.py
for k, v in stats.items():
    o = np.argsort(v)
    print(f"{k:16s} low: {[langs[i] for i in o[:5]]}\n{'':16s} high: {[langs[i] for i in o[-5:]]}")

# %% 2. Do the top PCs track tokenization?
X = V[:, LAYER] - V[:, LAYER].mean(0)
Z, p = pca(X)
print(f"\nlayer {LAYER}, |Spearman(PC score, stat)| across {len(langs)} languages:")
for k, v in stats.items():
    print(f"  {k:16s}" + "".join(f"  PC{j + 1} {abs(rho(Z[:, j], v)):.2f}" for j in range(5)))

# %% 3. Regress the stats out of every dimension, then redo PCA
C = np.column_stack([np.log(stats["bytes/token"]), stats["tokens/response"], stats["frac capped"], np.ones(len(langs))])
R = X - C @ np.linalg.lstsq(C, X, rcond=None)[0]
pc1 = p.components_[0]
print(f"\nshare of variance explained by the 3 stats: all dims {1 - (R ** 2).sum() / (X ** 2).sum():.3f}, "
      f"PC1 direction {1 - ((R @ pc1) ** 2).sum() / ((X @ pc1) ** 2).sum():.3f}  (chance ~ {3 / (len(langs) - 1):.3f})")
ZR, pR = pca(R)
print(f"residual PCA: PC1-5 {np.round(pR.explained_variance_ratio_, 3)} (before: {np.round(p.explained_variance_ratio_, 3)})")
print(f"|cos(PC1, residual PC1)| {abs(pc1 @ pR.components_[0]):.2f}, |Spearman| of language scores {abs(rho(Z[:, 0], ZR[:, 0])):.2f}")
for j in range(3):
    o = np.argsort(ZR[:, j])
    print(f"residual PC{j + 1} low: {[langs[i] for i in o[:5]]}\n{'':13s}high: {[langs[i] for i in o[::-1][:5]]}")

# %% 4. Within language: does one response's PC1 score follow its own tokenization?
zs, lb = [], []  # per-response PC1 score and log bytes/token, demeaned within language
within = []
for l in langs:
    a = torch.load(OUT / "activations" / f"{l}.pt")[S[l]["keep"], LAYER].float().numpy()
    z = (a - V[:, LAYER].mean(0)) @ pc1
    within.append(rho(z, S[l]["bpt"]))
    zs.append(z - z.mean())
    lb.append(np.log(S[l]["bpt"]) - np.log(S[l]["bpt"]).mean())
zs, lb = np.concatenate(zs), np.concatenate(lb)
slope = (zs @ lb) / (lb @ lb)  # pooled within-language slope of PC1 on log bytes/token
pred = slope * np.log(stats["bytes/token"])  # between-language PC1 if tokenization acted only through that slope
print(f"\nwithin-language Spearman(PC1, bytes/token): median {np.nanmedian(within):+.2f}, "
      f"IQR [{np.nanpercentile(within, 25):+.2f}, {np.nanpercentile(within, 75):+.2f}]")
print(f"pooled within-language slope {slope:+.2f} PC1/log-unit; it predicts a between-language PC1 spread (sd) of "
      f"{pred.std():.2f} vs actual {Z[:, 0].std():.2f}")

# %% 5. Same language, different script: tokenization changes, language doesn't
pct = lambda v: (Z[:, 0] < v).mean()
for group in [["hindi", "urdu", "hindi_romanized"], ["serbian_cyrillic", "serbian_latin", "croatian"],
              ["mandarin_simplified", "mandarin_traditional", "mandarin_pinyin", "cantonese"]]:
    for l in group:
        if l in langs:
            i = langs.index(l)
            print(f"{l:22s} bytes/token {stats['bytes/token'][i]:5.2f}   PC1 {Z[i, 0]:+7.1f} (percentile {pct(Z[i, 0]):.0%})")
    print()

# %% 6. By layer: how tokenization-driven is PC1, and how close is the geometry to layer 0 (~token identity)?
def rsm(A):
    A = A - A.mean(0)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    return (A @ A.T)[np.triu_indices(len(A), 1)]


tokdist = -np.abs(np.subtract.outer(np.log(stats["bytes/token"]), np.log(stats["bytes/token"])))[np.triu_indices(len(langs), 1)]
r0 = rsm(V[:, 0])
rows = []
for L in range(V.shape[1]):
    sim = rsm(V[:, L])
    rows.append({"layer": L, "|rho(PC1, bytes/token)|": abs(rho(pca(V[:, L] - V[:, L].mean(0), 1)[0][:, 0], stats["bytes/token"])),
                 "RSA vs layer 0": rho(sim, r0), "RSA vs tokenization similarity": rho(sim, tokdist)})
df = pd.DataFrame(rows)
print(df.iloc[::8].round(2).to_string(index=False))
px.line(df.melt("layer"), x="layer", y="value", color="variable",
        title=f"Tokenization control by layer ({MODEL})").write_html(OUT / "figs" / "tokenization_by_layer.html")
