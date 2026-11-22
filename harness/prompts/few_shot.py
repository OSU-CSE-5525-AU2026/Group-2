import json
from functools import cache
from pathlib import Path

from .common import assistant, instruction

EXAMPLES = Path(__file__).parent / "few_shot" / "examples.json"


@cache
def examples() -> dict[str, list[dict]]:
    return json.loads(EXAMPLES.read_text())


async def run(task, chat):
    messages = []
    for ex in examples()[task.opt]:
        messages += [instruction(ex["asm"], task.opt), assistant(f"```c\n{ex['c']}\n```")]
    return [await chat([*messages, instruction(task.asm, task.opt)])]
