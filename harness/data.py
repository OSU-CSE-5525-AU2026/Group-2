import json
from dataclasses import dataclass

from .config import ROOT

BENCHMARK = ROOT / "benchmark/LLM4Decompile/legacy-test/decompile-eval-executable-gcc-obj.json"


@dataclass(frozen=True)
class Task:
    task_id: int
    opt: str
    asm: str
    c_func: str
    c_test: str


def load_tasks(opt_levels: list[str], task_ids: list[int] | None = None) -> list[Task]:
    wanted = set(task_ids) if task_ids is not None else None
    return [
        Task(t["task_id"], t["type"], t["input_asm_prompt"], t["c_func"], t["c_test"])
        for t in json.loads(BENCHMARK.read_text())
        if t["type"] in opt_levels and (wanted is None or t["task_id"] in wanted)
    ]
