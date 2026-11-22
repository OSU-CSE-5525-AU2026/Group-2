import json
from pathlib import Path

KEY_FIELDS = ("model", "prompt", "opt", "task_id", "sample")


def key(record: dict) -> tuple:
    return tuple(record[f] for f in KEY_FIELDS)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
