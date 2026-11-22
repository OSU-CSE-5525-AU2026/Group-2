from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OPT_LEVELS = ("O0", "O1", "O2", "O3")


@dataclass
class ModelConfig:
    name: str
    base_url: str
    model: str | None = None
    api_key_env: str | None = None
    concurrency: int = 16
    params: dict = field(default_factory=dict)


@dataclass
class Experiment:
    name: str
    models: list[str]
    prompts: list[str]
    opt: list[str]
    tasks: list[int] | None
    samples: int

    @property
    def results_dir(self) -> Path:
        return ROOT / "results" / self.name


def load_models(path: Path = ROOT / "models.yaml") -> tuple[dict[str, ModelConfig], Path]:
    raw = yaml.safe_load(path.read_text())
    models = {name: ModelConfig(name=name, **spec) for name, spec in raw["models"].items()}
    return models, Path(raw["local_endpoints"])


def parse_tasks(value) -> list[int] | None:
    if value in (None, "all"):
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return sorted({i for v in value for i in parse_tasks(v)})
    ids = set()
    for part in str(value).split(","):
        start, _, end = part.strip().partition("-")
        ids.update(range(int(start), int(end or start) + 1))
    return sorted(ids)


def parse_opt(value) -> list[str]:
    levels = [value] if isinstance(value, str) else list(value)
    if "all" in levels:
        return list(OPT_LEVELS)
    unknown = set(levels) - set(OPT_LEVELS)
    if unknown:
        raise ValueError(f"unknown optimization levels {sorted(unknown)}; choose from {OPT_LEVELS} or all")
    return [o for o in OPT_LEVELS if o in levels]


def load_experiment(path: Path, **overrides) -> Experiment:
    raw = yaml.safe_load(Path(path).read_text())
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return Experiment(
        name=Path(path).stem,
        models=list(raw["models"]),
        prompts=list(raw["prompts"]),
        opt=parse_opt(raw.get("opt", "O0")),
        tasks=parse_tasks(raw.get("tasks", "all")),
        samples=int(raw.get("samples", 1)),
    )
