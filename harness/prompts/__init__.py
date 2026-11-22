import hashlib
import inspect

from . import common, few_shot, self_review, structured, zero_shot

STRATEGIES = {
    "zero_shot": zero_shot,
    "structured": structured,
    "self_review": self_review,
    "few_shot": few_shot,
}


def version(name: str) -> str:
    module = STRATEGIES[name]
    source = inspect.getsource(common) + inspect.getsource(module)
    if module is few_shot and few_shot.EXAMPLES.exists():
        source += few_shot.EXAMPLES.read_text()
    return hashlib.sha256(source.encode()).hexdigest()[:12]
