"""Baseline pipeline: HumanEval-Decompile assembly -> LLM (via Groq) -> C -> compile -> run tests -> results.

Usage:
    python run.py --model openai/gpt-oss-120b --n 20 --opt O0
"""
import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime

from groq import Groq

DATA = "benchmark/LLM4Decompile/legacy-test/decompile-eval-executable-gcc-obj.json"
TIMEOUT = 10

ZERO_SHOT = (
    "Decompile the following x86-64 assembly (compiled with gcc) into functionally "
    "equivalent C code. The function is named func0. Return only the C function "
    "(plus any #includes it needs) in a single ```c code block.\n\n{asm}"
)


def extract_code(text):
    m = re.search(r"```(?:c|cpp|c\+\+)?\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def evaluate(task, code):
    """Same procedure as LLM4Decompile's evaluation: hoist #includes, append the test main, gcc, run."""
    includes, bodies = [], []
    for src in (task["c_func"], task["c_test"]):
        lines = src.split("\n")
        includes += [l for l in lines if "#include" in l]
        bodies.append("\n".join(l for l in lines if "#include" not in l))
    program = "\n".join(includes) + "\n" + code + "\n" + bodies[1]

    with tempfile.TemporaryDirectory() as d:
        c_file, exe = os.path.join(d, "prog.c"), os.path.join(d, "prog")
        with open(c_file, "w") as f:
            f.write(program)
        try:
            p = subprocess.run(["gcc", c_file, "-o", exe, "-lm"], capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, False, "compile timeout"
        if p.returncode != 0:
            return False, False, p.stderr[-500:]
        try:
            p = subprocess.run([exe], capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return True, False, "run timeout"
        return True, p.returncode == 0, "" if p.returncode == 0 else (p.stderr[-500:] or f"exit {p.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--opt", default="O0")
    args = ap.parse_args()

    tasks = [t for t in json.load(open(DATA)) if t["type"] == args.opt][: args.n]
    client = Groq()
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{args.model.replace('/', '_')}_{args.opt}"
    os.makedirs("results", exist_ok=True)

    rows = []
    for t in tasks:
        start = time.time()
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": ZERO_SHOT.format(asm=t["input_asm_prompt"])}],
            temperature=0,
            max_completion_tokens=8192,
            reasoning_effort="medium",
        )
        latency = time.time() - start
        raw = resp.choices[0].message.content
        code = extract_code(raw)
        compiled, passed, error = evaluate(t, code)
        rows.append({
            "task_id": t["task_id"], "opt": t["type"], "model": args.model, "prompt": "zero_shot",
            "compiled": compiled, "passed": passed, "error": error, "latency_s": round(latency, 2),
            "input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens,
            "generated_code": code, "raw_response": raw,
        })
        print(f"task {t['task_id']:>3}  compiled={compiled!s:5}  passed={passed!s:5}  {latency:.1f}s")

    with open(f"results/{run_id}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(f"results/{run_id}.csv", "w", newline="") as f:
        cols = [k for k in rows[0] if k not in ("generated_code", "raw_response")]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"\n{args.model} | zero_shot | {args.opt} | n={n}")
    print(f"compile rate: {sum(r['compiled'] for r in rows) / n:.1%}")
    print(f"pass rate:    {sum(r['passed'] for r in rows) / n:.1%}")
    print(f"avg latency:  {sum(r['latency_s'] for r in rows) / n:.2f}s")
    print(f"tokens:       {sum(r['input_tokens'] for r in rows)} in / {sum(r['output_tokens'] for r in rows)} out")
    print(f"saved results/{run_id}.jsonl and .csv")


if __name__ == "__main__":
    main()
