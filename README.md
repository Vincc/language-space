# Language space

The persona-space pipeline from [safety-research/assistant-axis](https://github.com/safety-research/assistant-axis)
(Lu et al. 2026), with **languages in place of character roles**. We make Qwen3-32B respond in 175 languages, cache
the mean residual-stream activation over each response at every layer, keep only responses that are fully in the target
language, average them into one vector per language, and run PCA on those vectors.

## Experiment design

| | Persona space (original) | Language space (this repo) |
|---|---|---|
| Unit | 275 character roles + default | 175 languages: 32 high-, 45 mid- and 83 low-resource; 8 classical; 7 constructed. 30 scripts, 27 families |
| Elicitation | 5 role system prompts × 240 questions | 3 "respond only in X" system prompts × 100 English questions = 300 responses per language |
| Activations | mean over response tokens, every layer's output | same (`2_activations.py`) |
| Filter | LLM judge 0–3 role adherence, keep 3s | Claude Haiku 5.5 judge 0–3 "is this actually X in the right script", keep 3s |
| Vectors | mean of score-3 responses per role | same (needs at least 50 of 300) |
| Analysis | PCA, Assistant Axis | PCA, plus a comparison against the published Qwen3-32B persona vectors |

Questions are the first 100 of the persona repo's 240 extraction questions, so inputs match persona space. The questions
stay in English and only the system prompt changes between languages.

**Built-in contrasts in the language list:**
- *Same language, different script*: Hindi / Urdu / romanized Hindi, Serbian Cyrillic / Latin, Punjabi Gurmukhi / Shahmukhi, Mandarin Simplified / Traditional / Pinyin, Japanese / romaji
- *Closely related pairs*: Malay / Indonesian, Bokmål / Nynorsk, Croatian / Bosnian / Serbian, MSA / Egyptian / Levantine / Moroccan Arabic, Mandarin / Cantonese
- *Dead and constructed languages* (Latin, Sanskrit, Classical Chinese, Coptic; Esperanto, Toki Pona, Klingon, Lojban, Quenya)

**Questions `5_pca.py` answers:**
1. How many dimensions does language space have? (variance explained) What do PC1–3 encode: script, family, resource level, or English-likeness?
2. At what depth is language most concentrated? (layer sweep)
3. How does language space relate to persona space? Where do languages project on the Assistant Axis, how much language variance lies inside the top persona PCs (and the reverse), and a joint PCA.
4. When the model fails at a language, which language does it land closest to?

## Cost estimate (175 languages, 52.5k responses)

| Item | Estimate |
|---|---|
| Judge: Haiku 5.5, Batch API, 52.5k × ~1k input tokens | ~$3 |
| GPU: 1× H200 on Vast at ~$3.70/hr (checked 2026-11-18). About 0.5 h setup and model download, ~2–2.5 h generation (24M output tokens), ~2 h activations (34M tokens forward) | ~5 h ≈ $18 |
| Debugging and pilot buffer | ~$10 |
| **Total** | **≈ $30** |

An H100 SXM costs $2.07/hr, but with 32B bf16 weights it has only ~12GB left for KV cache, so generation runs about 3× slower. The total ends up about the same, but you wait longer.
Disk: ~65GB for the model plus ~34GB of per-response activations (52.5k × 64 layers × 5120 × bf16). Rent at least 150GB.

## Running

Put keys in `.env`. Edit constants in `common.py` (no CLI flags). `ONLY` is preset to a 10-language pilot; set it to `None` for the full run. Every step skips outputs that already exist, so after the pilot the full run only does the remaining languages.

```bash
pip install -r requirements.txt
python 1_generate.py       # GPU (vLLM)  -> outputs/Qwen3-32B/responses/   (also fetches data/questions.jsonl)
python 2_activations.py    # GPU (HF)    -> outputs/Qwen3-32B/activations/
python 3_judge.py          # Claude API, can run alongside step 2 -> outputs/Qwen3-32B/scores/
python 4_vectors.py        # CPU         -> outputs/Qwen3-32B/vectors.pt (~400MB, the only file analysis needs)
python 5_pca.py            # or run its "# %%" cells interactively -> outputs/Qwen3-32B/figs/*.html
```

Step 1 needs vLLM and step 2 needs transformers. Kill the step 1 process before starting step 2 so the GPU memory is freed.

Other models: prefix any step with `MODEL=...` (default `Qwen/Qwen3-32B`). Run so far: `Qwen/Qwen3.5-27B` and `google/gemma-4-31B-it`, each writing to `outputs/<model>/`.

```bash
python 6_tokenization.py   # needs responses/scores/activations: is PC1 a tokenizer artifact? -> tokstats.json, figs/tokenization_by_layer.html
python 7_compare.py        # needs each model's vectors.pt + tokstats.json: coverage, shared geometry, cross-tokenizer test
python 8_steer.py          # GPU: steer with ±k·sd·PC1 at the middle layer, no system prompt, 10 questions in en/zh/tr/hi -> steer/samples.jsonl
python 9_steer_judge.py    # Claude API: label each steered answer's language + coherence, print tables -> steer/labels.json
python 10_subsets.py       # CPU: PCA within languages the model writes reliably (>=90%) vs weakly (<70%), vs random subsets
```

Steps 8–9 steer with PC `STEER_PC` (0 = PC1) of language space `STEER_SET` (`all`, `strong` = written ≥90% in-language, `weak` = <70%), both env-overridable like `MODEL`. Output goes to `steer[_<set>][_pcN]/`, e.g. `steer/` = all/PC1, `steer_weak_pc3/` = weak/PC3.

Actual cost of the 3-model run (2026-11-18/19): about $55 on one H200, including about 5 idle hours, plus about $10 of Haiku judge calls.
Throughput was about 5k output tokens/s for generation and about 20 s per language for activations, so each 32B-class model takes about 2 GPU hours.
