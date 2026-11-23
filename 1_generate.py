"""Step 1 (GPU, vLLM): for every language, 3 system prompts ("respond only in {lang}") x 100 English questions
= 300 responses. Output: outputs/<model>/responses/{lang}.jsonl. Skips files that already exist, so it is safe to rerun."""
from vllm import LLM, SamplingParams

from common import MODEL, OUT, SYS_PROMPTS, languages, questions, write_jsonl

CHUNK = 25  # languages per vLLM call; results are written after each chunk

if __name__ == "__main__":  # needed when vLLM spawns its engine process (e.g. Qwen3.5), which re-imports this file
    qs = questions()
    llm = LLM(MODEL, max_model_len=2048, gpu_memory_utilization=0.92)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

    todo = [l for l in languages() if not (OUT / "responses" / f"{l['id']}.jsonl").exists()]
    print(f"{len(todo)} languages to generate")
    for c in range(0, len(todo), CHUNK):
        jobs = [(l, {"prompt_idx": p, "q_idx": i, "messages": [
            {"role": "system", "content": sp.format(lang=l["name"])}, {"role": "user", "content": q}]})
            for l in todo[c:c + CHUNK] for p, sp in enumerate(SYS_PROMPTS) for i, q in enumerate(qs)]
        prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                   for _, r in jobs]
        outs = llm.generate(prompts, params)
        by_lang = {}
        for (l, r), o in zip(jobs, outs):
            r.update(lang=l["id"], response=o.outputs[0].text, finish=o.outputs[0].finish_reason)
            by_lang.setdefault(l["id"], []).append(r)
        for lang_id, rows in by_lang.items():
            write_jsonl(OUT / "responses" / f"{lang_id}.jsonl", rows)
        print(f"wrote {c + len(by_lang)}/{len(todo)} languages")
