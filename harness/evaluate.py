import os
import re
import resource
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Experiment
from .data import load_tasks
from .records import KEY_FIELDS, read_jsonl, write_jsonl

COMPILE_TIMEOUT = 30
RUN_TIMEOUT = 10
MEMORY_LIMIT = 2 * 1024**3
CODE_BLOCK = re.compile(r"```[ \t]*(?:c|cpp|c\+\+|C)?[ \t]*\n(.*?)```", re.S)
MAIN = re.compile(r"^[ \t]*(?:int|void)\s+main\s*\([^)]*\)\s*\{", re.M)


def extract_code(text: str) -> str | None:
    blocks = CODE_BLOCK.findall(text)
    return strip_main(blocks[-1]) if blocks else None


def strip_main(code: str) -> str:
    match = MAIN.search(code)
    if not match:
        return code
    depth, end = 0, len(code)
    for i in range(match.end() - 1, len(code)):
        depth += {"{": 1, "}": -1}.get(code[i], 0)
        if depth == 0:
            end = i + 1
            break
    return code[: match.start()] + code[end:]


def build_program(task, code: str) -> str:
    lines = task.c_func.split("\n") + task.c_test.split("\n")
    includes = [line for line in lines if "#include" in line]
    test_body = [line for line in task.c_test.split("\n") if "#include" not in line]
    return "\n".join(includes) + "\n" + code + "\n" + "\n".join(test_body)


def limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    resource.setrlimit(resource.RLIMIT_CPU, (RUN_TIMEOUT, RUN_TIMEOUT))


def check(task, code: str | None) -> tuple[str, str]:
    if code is None:
        return "no_code", ""
    with tempfile.TemporaryDirectory() as tmp:
        src, exe = Path(tmp) / "prog.c", Path(tmp) / "prog"
        src.write_text(build_program(task, code))
        try:
            build = subprocess.run(
                ["gcc", src, "-o", exe, "-lm"], capture_output=True, text=True, timeout=COMPILE_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            return "compile_error", "compile timeout"
        if build.returncode != 0:
            return "compile_error", build.stderr[-1000:]
        try:
            run = subprocess.run(
                [exe], cwd=tmp, capture_output=True, text=True, timeout=RUN_TIMEOUT, preexec_fn=limit_resources
            )
        except subprocess.TimeoutExpired:
            return "timeout", ""
    if run.returncode == 0:
        return "passed", ""
    if run.returncode == -6 and "Assertion" in run.stderr:
        return "test_fail", run.stderr[-500:]
    return "runtime_error", run.stderr[-500:] or f"exit {run.returncode}"


def gcc_version() -> str:
    return subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.split("\n")[0]


def evaluate(exp: Experiment) -> list[dict]:
    generations = read_jsonl(exp.results_dir / "generations.jsonl")
    if not generations:
        raise SystemExit(f"no generations in {exp.results_dir}; run generate first")
    tasks = {(t.opt, t.task_id): t for t in load_tasks(list({g["opt"] for g in generations}))}
    gcc = gcc_version()

    def one(g):
        code = extract_code(g["output"])
        status, error = check(tasks[g["opt"], g["task_id"]], code)
        return {
            **{f: g[f] for f in KEY_FIELDS},
            "status": status,
            "compiled": status not in ("no_code", "compile_error"),
            "passed": status == "passed",
            "error": error,
            "code": code,
            "gcc": gcc,
        }

    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        evals = list(pool.map(one, generations))
    write_jsonl(exp.results_dir / "evals.jsonl", evals)
    print(f"evaluated {len(evals)} generations -> {exp.results_dir / 'evals.jsonl'}")
    return evals
