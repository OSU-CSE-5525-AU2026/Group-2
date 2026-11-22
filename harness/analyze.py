import csv
import random
from collections import defaultdict
from math import comb
from statistics import mean

from .config import OPT_LEVELS, Experiment
from .records import key, read_jsonl

BOOTSTRAP = 2000


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    rng = random.Random(0)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def selected(exp: Experiment) -> list[dict]:
    return [
        e for e in read_jsonl(exp.results_dir / "evals.jsonl")
        if e["model"] in exp.models
        and e["prompt"] in exp.prompts
        and e["opt"] in exp.opt
        and (exp.tasks is None or e["task_id"] in exp.tasks)
        and e["sample"] < exp.samples
    ]


def summarize(exp: Experiment) -> list[dict]:
    evals = selected(exp)
    if not evals:
        raise SystemExit(f"no evals in {exp.results_dir}; run evaluate first")
    gens = {key(g): g for g in read_jsonl(exp.results_dir / "generations.jsonl")}

    groups = defaultdict(list)
    for e in evals:
        groups[e["model"], e["prompt"], e["opt"]].append((e, gens[key(e)]))

    rows = []
    for (model, prompt, opt), items in groups.items():
        by_task = defaultdict(list)
        for e, _ in items:
            by_task[e["task_id"]].append(e)
        pass_at_1 = [mean(x["passed"] for x in v) for v in by_task.values()]
        low, high = bootstrap_ci(pass_at_1)
        statuses = defaultdict(int)
        for e, _ in items:
            statuses[e["status"]] += 1
        rows.append({
            "model": model,
            "prompt": prompt,
            "opt": opt,
            "tasks": len(by_task),
            "samples": len(items),
            "compile_rate": round(mean(e["compiled"] for e, _ in items), 4),
            "pass_rate": round(mean(pass_at_1), 4),
            "ci_low": round(low, 4),
            "ci_high": round(high, 4),
            "truncated": sum(any(c["finish_reason"] == "length" for c in g["calls"]) for _, g in items),
            "avg_latency_s": round(mean(g["latency_s"] for _, g in items), 2),
            "input_tokens": sum(g["input_tokens"] for _, g in items),
            "output_tokens": sum(g["output_tokens"] for _, g in items),
            **{f"n_{s}": statuses[s] for s in ("passed", "test_fail", "runtime_error", "timeout", "compile_error", "no_code")},
        })
    return sorted(rows, key=lambda r: (
        OPT_LEVELS.index(r["opt"]), exp.models.index(r["model"]), exp.prompts.index(r["prompt"])
    ))


def compare(exp: Experiment, baseline: str) -> list[dict]:
    first = {}
    for e in selected(exp):
        if e["sample"] == 0:
            first[e["model"], e["prompt"], e["opt"], e["task_id"]] = e["passed"]
    rows = []
    for model, prompt, opt in sorted({k[:3] for k in first}):
        if prompt == baseline:
            continue
        pairs = [
            (first[k], first[(model, baseline, opt, k[3])])
            for k in first
            if k[:3] == (model, prompt, opt) and (model, baseline, opt, k[3]) in first
        ]
        if not pairs:
            continue
        gained = sum(p and not b for p, b in pairs)
        lost = sum(b and not p for p, b in pairs)
        rows.append({"model": model, "opt": opt, "prompt": prompt, "vs": baseline, "tasks": len(pairs),
                     "gained": gained, "lost": lost, "p_value": round(mcnemar_p(gained, lost), 4)})
    return rows


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def print_tables(rows: list[dict], comparisons: list[dict]) -> None:
    for opt in OPT_LEVELS:
        cells = [r for r in rows if r["opt"] == opt]
        if not cells:
            continue
        prompts = list(dict.fromkeys(r["prompt"] for r in cells))
        print(f"\n## Pass rate, {opt}\n")
        print("| Model | " + " | ".join(prompts) + " | Avg |")
        print("|---" * (len(prompts) + 2) + "|")
        for model in dict.fromkeys(r["model"] for r in cells):
            mine = {r["prompt"]: r for r in cells if r["model"] == model}
            row = [f"{pct(mine[p]['pass_rate'])} [{pct(mine[p]['ci_low'])}, {pct(mine[p]['ci_high'])}]"
                   if p in mine else "-" for p in prompts]
            avg = mean(r["pass_rate"] for r in mine.values())
            print(f"| {model} | " + " | ".join(row) + f" | {pct(avg)} |")

    print("\n## Detail\n")
    print("| Model | Prompt | Opt | Tasks | Compile | Pass | Truncated | Latency | Tokens in / out |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['prompt']} | {r['opt']} | {r['tasks']} | {pct(r['compile_rate'])} | "
              f"{pct(r['pass_rate'])} | {r['truncated']} | {r['avg_latency_s']}s | "
              f"{r['input_tokens']:,} / {r['output_tokens']:,} |")

    if comparisons:
        print(f"\n## Against {comparisons[0]['vs']} (McNemar, sample 0)\n")
        print("| Model | Opt | Prompt | Tasks | Gained | Lost | p |")
        print("|---|---|---|---|---|---|---|")
        for c in comparisons:
            print(f"| {c['model']} | {c['opt']} | {c['prompt']} | {c['tasks']} | {c['gained']} | {c['lost']} | {c['p_value']} |")


def analyze(exp: Experiment) -> None:
    rows = summarize(exp)
    comparisons = compare(exp, exp.prompts[0])
    for name, data in (("summary.csv", rows), ("comparisons.csv", comparisons)):
        if data:
            with (exp.results_dir / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    print_tables(rows, comparisons)
    print(f"\nwrote {exp.results_dir / 'summary.csv'}")
