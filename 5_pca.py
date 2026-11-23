# %% [markdown]
# # Language space: PCA of per-language vectors
# Run cell by cell (VS Code / Jupyter "# %%" cells) or as a script. Needs only outputs/<model>/vectors.pt (~400MB),
# so you can scp that file off the GPU box and run this locally. Figures are written to outputs/<model>/figs/.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.decomposition import PCA

from common import MODEL, OUT, languages

COLOR = "family"  # or "script" / "tier"

d = torch.load(OUT / "vectors.pt")
LAYER = d["vec"].shape[1] // 2  # middle layer (= the persona repo's target layer 32 for Qwen3-32B)
meta = {l["id"]: l for l in languages()}
FIGS = OUT / "figs"
FIGS.mkdir(exist_ok=True)


def fit(X):
    """Mean-center then PCA, as in the persona repo's PCA notebook. X: (n, D) numpy."""
    mu = X.mean(0)
    return PCA().fit(X - mu), mu


def frame(langs, coords):
    df = pd.DataFrame(coords[:, :3], columns=["PC1", "PC2", "PC3"])
    for k in ["name", "family", "script", "tier"]:
        df[k] = [meta[l][k] for l in langs]
    return df


X = d["vec"][:, LAYER].float().numpy()

# %% Variance explained: how many dimensions does language space have?
p, mu = fit(X)
cum = np.cumsum(p.explained_variance_ratio_)
print(f"n={len(X)}  PC1-5: {np.round(p.explained_variance_ratio_[:5], 3)}  "
      f"dims for 70/80/90%: {[int(np.argmax(cum >= t)) + 1 for t in (0.7, 0.8, 0.9)]}")
px.line(y=cum[:40], markers=True, title=f"Cumulative variance explained (layer {LAYER})",
        labels={"x": "component", "y": "cumulative variance"}).write_html(FIGS / f"variance_L{LAYER}.html")

# %% 3D scatter of PC1-3, plus which languages sit at each end of each PC
Z = p.transform(X - mu)
fig = px.scatter_3d(frame(d["langs"], Z), x="PC1", y="PC2", z="PC3", color=COLOR, hover_name="name",
                    hover_data=["script", "tier"], title=f"Language space (layer {LAYER})")
fig.update_traces(marker_size=4)
fig.write_html(FIGS / f"pca3d_L{LAYER}.html")
m = min(6, len(X) // 2)  # languages to show at each end of a PC
for k in range(5):
    o = np.argsort(Z[:, k])
    print(f"PC{k + 1}  low: {[d['langs'][i] for i in o[:m]]}\n     high: {[d['langs'][i] for i in o[::-1][:m]]}")

# %% Layer sweep: how concentrated is language space at each depth?
rows = []
for L in range(d["vec"].shape[1]):
    r = fit(d["vec"][:, L].float().numpy())[0].explained_variance_ratio_
    rows.append({"layer": L, "PC1": r[0], "PC1-3": r[:3].sum(), "PC1-10": r[:10].sum()})
px.line(pd.DataFrame(rows).melt("layer"), x="layer", y="value", color="variable",
        title="Variance explained by top PCs across layers").write_html(FIGS / "layer_sweep.html")

# %% Languages the model can't produce: where do their (unfiltered) responses land?
kept = d["vec"][:, LAYER].float()
m = kept.mean(0)
for l, f, v in zip(d["langs_all"], d["frac3"], d["vec_all"][:, LAYER].float()):
    if l not in d["langs"]:
        sims = torch.nn.functional.cosine_similarity(v - m, kept - m, dim=-1)
        print(f"{l:22s} {f:5.0%} in-language -> nearest: {[d['langs'][i] for i in sims.topk(3).indices]}")

# %% Persona space vs language space (published Qwen3-32B role vectors + Assistant Axis; Qwen3-32B only)
from huggingface_hub import snapshot_download

if MODEL != "Qwen/Qwen3-32B":
    raise SystemExit("persona vectors are only published for Qwen3-32B here; skipping the rest")  # noqa
hf = snapshot_download("lu-christina/assistant-axis-vectors", repo_type="dataset", allow_patterns=["qwen-3-32b/*"])


def load(f):  # published files are either raw (L, D) tensors or dicts with "vector" / "axis"
    t = torch.load(f, map_location="cpu", weights_only=False)
    if isinstance(t, dict):
        t = t.get("vector", t.get("axis"))
    return t.float()


role_files = sorted(Path(hf, "qwen-3-32b/role_vectors").glob("*.pt"))
R = torch.stack([load(f) for f in role_files])[:, LAYER].numpy()  # (n_roles, D)
axis = load(Path(hf, "qwen-3-32b/assistant_axis.pt"))[LAYER].numpy()
default = load(Path(hf, "qwen-3-32b/default_vector.pt"))[LAYER].numpy()
a = axis / np.linalg.norm(axis)

# (a) where do languages sit on the Assistant Axis, relative to the default Assistant and to roles?
proj = pd.DataFrame({"name": [meta[l]["name"] for l in d["langs"]], "family": [meta[l]["family"] for l in d["langs"]],
                     "proj": (X - default) @ a})
role_proj = (R - default) @ a
print(f"Assistant-axis projection (relative to default): languages {proj.proj.mean():.2f} ± {proj.proj.std():.2f}, "
      f"roles {role_proj.mean():.2f} ± {role_proj.std():.2f}")
print("least Assistant-like languages:", proj.nsmallest(8, "proj").name.tolist())
px.strip(proj, x="proj", color="family", hover_name="name",
         title="Language vectors projected on the Assistant Axis (0 = default Assistant)").write_html(FIGS / "assistant_axis_proj.html")

# (b) subspace overlap: fraction of (centered) language variance inside the top-k persona PCs, vs a random k-dim subspace
Rp, _ = fit(R)
Xc = X - X.mean(0)
for k in (1, 3, 10, 50):
    print(f"top-{k} persona PCs capture {((Xc @ Rp.components_[:k].T) ** 2).sum() / (Xc ** 2).sum():.3f} of language variance "
          f"(random subspace: {k / Xc.shape[1]:.4f})")
Lp, _ = fit(X)
Rc = R - R.mean(0)
for k in (1, 3, 10):
    print(f"top-{k} language PCs capture {((Rc @ Lp.components_[:k].T) ** 2).sum() / (Rc ** 2).sum():.3f} of persona variance")

# (c) joint PCA of roles + languages
J = np.concatenate([X, R])
pj, mj = fit(J)
ZJ = pj.transform(J - mj)
px.scatter(x=ZJ[:, 0], y=ZJ[:, 1], color=["language"] * len(X) + ["role"] * len(R),
           hover_name=[meta[l]["name"] for l in d["langs"]] + [f.stem for f in role_files], labels={"x": "PC1", "y": "PC2"},
           title=f"Joint PCA: languages + persona roles (layer {LAYER})").write_html(FIGS / f"joint_pca_L{LAYER}.html")
