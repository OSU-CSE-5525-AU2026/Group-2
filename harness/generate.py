import asyncio
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone

from . import prompts
from .client import ChatClient
from .config import ROOT, Experiment, load_models
from .data import load_tasks
from .records import key, read_jsonl


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


async def generate(exp: Experiment, wait: bool = False) -> None:
    models, endpoints = load_models()
    unknown = [m for m in exp.models if m not in models] + [p for p in exp.prompts if p not in prompts.STRATEGIES]
    if unknown:
        raise SystemExit(f"unknown models or prompts: {unknown}")

    tasks = load_tasks(exp.opt, exp.tasks)
    out = exp.results_dir / "generations.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {key(r) for r in read_jsonl(out)}
    commit = git_commit()
    versions = {p: prompts.version(p) for p in exp.prompts}

    for name in exp.models:
        todo = [
            (p, t, s)
            for p in exp.prompts
            for t in tasks
            for s in range(exp.samples)
            if (name, p, t.opt, t.task_id, s) not in done
        ]
        if not todo:
            print(f"{name}: all {len(exp.prompts) * len(tasks) * exp.samples} done")
            continue
        chat = await ChatClient.connect(models[name], endpoints, wait)
        print(f"{name} ({chat.model}): {len(todo)} requests", flush=True)
        await run_model(name, chat, todo, versions, commit, out)


async def run_model(name, chat, todo, versions, commit, out) -> None:
    gate = asyncio.Semaphore(chat.config.concurrency)
    finished, failed, start = 0, 0, time.perf_counter()

    async def one(prompt, task, sample):
        nonlocal finished, failed
        async with gate:
            try:
                replies = await prompts.STRATEGIES[prompt].run(task, chat)
            except Exception as e:
                failed += 1
                print(f"  failed {prompt} {task.opt} task {task.task_id}: {e}", flush=True)
                return
        record = {
            "model": name,
            "prompt": prompt,
            "opt": task.opt,
            "task_id": task.task_id,
            "sample": sample,
            "model_id": chat.model,
            "params": chat.config.params,
            "prompt_version": versions[prompt],
            "commit": commit,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "output": replies[-1].text,
            "input_tokens": sum(r.input_tokens for r in replies),
            "output_tokens": sum(r.output_tokens for r in replies),
            "latency_s": round(sum(r.latency_s for r in replies), 2),
            "calls": [asdict(r) for r in replies],
        }
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        finished += 1
        if finished % 10 == 0 or finished + failed == len(todo):
            print(f"  {finished}/{len(todo)} done, {failed} failed, {time.perf_counter() - start:.0f}s", flush=True)

    await asyncio.gather(*(one(p, t, s) for p, t, s in todo))
    if failed:
        print(f"  {failed} requests failed; run again to retry them")
