"""Shared constants + tiny helpers. Edit the constants here instead of passing flags."""
import json
import os
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Also tried: Qwen/Qwen3.5-27B, google/gemma-4-31B-it. Override per run with `MODEL=... python 1_generate.py`.
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")
N_QUESTIONS = 100  # first N of the persona repo's 240 extraction questions (asked in English)
SYS_PROMPTS = [  # system prompt naming the language, analogous to the persona repo's role system prompts
    "Respond only in {lang}.",
    "Write your entire response in {lang}, regardless of the language of the question.",
    "Always answer in {lang}. Do not use any other language.",
]
# Pilot: set to a list of language ids to run only those; set back to None for the full run (finished files are skipped).
ONLY = None

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / MODEL.split("/")[-1]
# Capability groups (10_subsets.py, steering): share of 300 responses fully in-language. STRONG >= 0.9, WEAK < 0.7.
STRONG, WEAK = 0.9, 0.7
# Steering (8_steer.py, 9_steer_judge.py): which language space ("all", "strong", "weak") and which PC (0 = PC1).
# Override per run like MODEL: `STEER_SET=weak STEER_PC=2 python 8_steer.py`. all/PC1 lives in steer/, all/PC2 in steer_pc2/.
STEER_SET = os.environ.get("STEER_SET", "all")
STEER_PC = int(os.environ.get("STEER_PC", 0))
STEER = OUT / ("steer" + ("" if STEER_SET == "all" else f"_{STEER_SET}") + ("" if STEER_PC == 0 else f"_pc{STEER_PC + 1}"))
QUESTIONS_URL = "https://raw.githubusercontent.com/safety-research/assistant-axis/master/data/extraction_questions.jsonl"


def languages():
    langs = json.loads((DATA / "languages.json").read_text(encoding="utf-8"))
    return [l for l in langs if ONLY is None or l["id"] in ONLY]


def questions():
    f = DATA / "questions.jsonl"
    if not f.exists():  # fetched rather than vendored: the persona repo has no license file
        f.write_bytes(urllib.request.urlopen(QUESTIONS_URL).read())
    return [json.loads(l)["question"] for l in f.read_text(encoding="utf-8").splitlines()][:N_QUESTIONS]


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def run_batch(requests, poll=30):
    """requests: {custom_id: params dict}. Runs a Claude Message Batch (50% off), returns {custom_id: text or None}."""
    import anthropic

    client = anthropic.Anthropic(max_retries=5)
    items = list(requests.items())
    out = {}
    ids = []  # submit everything first (10k per batch keeps each upload small), then wait on all of them
    for i in range(0, len(items), 10_000):
        ids.append(client.messages.batches.create(requests=[{"custom_id": k, "params": p} for k, p in items[i:i + 10_000]]).id)
        print(f"batch {ids[-1]}: {len(items[i:i + 10_000])} requests")
    for batch_id in ids:
        while (b := client.messages.batches.retrieve(batch_id)).processing_status != "ended":
            print(f"  {batch_id}: {b.request_counts.processing} processing, {b.request_counts.succeeded} done")
            time.sleep(poll)
        for r in client.messages.batches.results(batch_id):
            if r.result.type == "succeeded" and r.result.message.stop_reason != "refusal":
                out[r.custom_id] = next((c.text for c in r.result.message.content if c.type == "text"), None)
            else:
                out[r.custom_id] = None
                print(f"  {r.custom_id}: {r.result.type}")
    return out
