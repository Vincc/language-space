"""Step 9 (local, Claude API): label every steered sample with its language, script and coherence, then print
tables of what the steering does. Output: outputs/<model>/steer[_pcN]/labels.json (aligned with samples.jsonl)."""
import json
from collections import Counter

import pandas as pd

from common import STEER, read_jsonl, run_batch

PROMPT = """A chatbot was asked a question and gave the response below. Describe the response's language. Do not judge its content.

<question>
{question}
</question>

<response>
{response}
</response>

Reply with JSON only, with these keys:
"language": the main language of the response, as an English name (e.g. "English", "Mandarin Chinese", "Turkish", "Hindi"). Use "mixed" if no language makes up most of it, and "none" if it is empty or not language.
"other_languages": a list of other languages that appear in more than a word or two (empty list if none)
"script": the main writing system (e.g. "Latin", "Han", "Devanagari", "Arabic")
"coherence": 3 = fluent and sensible, 2 = understandable with errors or drift, 1 = mostly broken (word salad, made-up words, heavy repetition), 0 = gibberish or empty"""

samples = read_jsonl(STEER / "samples.jsonl")
f = STEER / "labels.json"
labels = json.loads(f.read_text(encoding="utf-8")) if f.exists() else [None] * len(samples)
todo = [i for i, l in enumerate(labels) if not (l and l.get("language"))]  # missing or failed labels get retried
if todo:
    texts = run_batch({str(i): {"model": "claude-haiku-5-5", "max_tokens": 150, "thinking": {"type": "disabled"},
                                "messages": [{"role": "user", "content": PROMPT.format(**samples[i])}]}
                       for i in todo})
    for i in todo:
        t = texts.get(str(i)) or ""
        objs = []  # every JSON object in the reply; keep the last (Haiku sometimes corrects itself)
        for j in range(len(t)):
            if t[j] == "{":
                try:
                    objs.append(json.JSONDecoder().raw_decode(t[j:])[0])
                except ValueError:
                    pass
        labels[i] = objs[-1] if objs else None
        if not objs:
            print(f"unparsed label {i}: {t!r}")
    f.write_text(json.dumps(labels, ensure_ascii=False, indent=0), encoding="utf-8")

SAME = {"english": "English", "chinese": "Chinese", "turkish": "Turkish", "hindi": "Hindi"}
df = pd.DataFrame([dict(s, **(l or {})) for s, l in zip(samples, labels)])
df["same_lang"] = [SAME[q] in str(l) for q, l in zip(df.qlang, df.language)]
df["english"] = df.language.eq("English")
for mode in ["all", "gen"]:
    t = df[df["mode"] == mode]
    print(f"\n=== steer {mode} positions: share answered in the question's language | in English | mean coherence | capped")
    print(pd.concat({k: t.pivot_table(index="qlang", columns="alpha", values=k, aggfunc="mean").round(2)
                     for k in ["same_lang", "english", "coherence", "capped"]}).to_string())
    for a in sorted(t.alpha.unique()):
        print(f"alpha {a:+d}: languages {dict(Counter(t[t.alpha == a].language).most_common(6))}")
