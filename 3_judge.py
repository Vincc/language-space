"""Step 3 (local or GPU box, Claude API): score each response 0-3 for "is this actually written in the target language".
Can run while 2_activations.py is running. Output: outputs/<model>/scores/{lang}.json (list aligned with jsonl rows)."""
import json
import re

from common import OUT, languages, read_jsonl, run_batch

PROMPT = """You are checking whether a chatbot response is written in a requested language. Judge only the language, not the content.

Target language: {lang}

<response>
{response}
</response>

Score the response:
0 = not in the target language (written in another language, or empty)
1 = mostly in another language; only some words or sentences are in the target language
2 = mostly in the target language, but with substantial mixing of another language, the wrong script, or so many errors that a fluent reader would struggle
3 = entirely in the target language and its requested script (proper nouns, code, or a few quoted foreign terms are fine)

If the text is mostly made-up words, meaningless strings in the right script, or the same phrase repeated over and over, score it 1 at most: it looks like the target language without actually being it.

Answer with a single digit."""

names = {l["id"]: l["name"] for l in languages()}
rows = {f.stem: read_jsonl(f) for f in sorted((OUT / "responses").glob("*.jsonl"))
        if f.stem in names and not (OUT / "scores" / f"{f.stem}.json").exists()}
print(f"judging {sum(map(len, rows.values()))} responses in {len(rows)} languages")

requests = {
    f"{lang}__{i}": {
        "model": "claude-haiku-5-5",
        "max_tokens": 16,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": PROMPT.format(lang=names[lang], response=r["response"])}],
    }
    for lang, rs in rows.items() for i, r in enumerate(rs)
}
texts = run_batch(requests)

for lang, rs in rows.items():
    scores = []
    for i in range(len(rs)):
        m = re.search(r"[0-3]", texts.get(f"{lang}__{i}") or "")
        scores.append(int(m.group()) if m else None)
    out = OUT / "scores" / f"{lang}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores))
    print(f"{lang}: {sum(s == 3 for s in scores)}/{len(scores)} fully in-language")
