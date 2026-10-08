# Group 2: LLM Decompilation Baseline

## Run

```bash
git clone --recurse-submodules https://github.com/OSU-CSE-5525-AU2026/Group-2.git
cd Group-2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=...
python run.py --n 20 --opt O0
```

## Results

`openai/gpt-oss-120b` via Groq, zero-shot prompt, first 20 HumanEval-Decompile tasks at O0.

| Compile rate | Pass rate | Avg latency | Tokens (in / out) |
|---|---|---|---|
| 90% (18/20) | 80% (16/20) | 18.3s | 17.8k / 39.1k |

Per-task output: `results/20261008-125432_openai_gpt-oss-120b_O0.csv` and `.jsonl`
