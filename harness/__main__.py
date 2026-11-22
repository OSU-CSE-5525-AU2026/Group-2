import argparse
import asyncio

from .analyze import analyze
from .config import load_experiment
from .evaluate import evaluate
from .fewshot import build
from .generate import generate


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m harness")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "generate", "evaluate", "analyze"):
        cmd = commands.add_parser(name)
        cmd.add_argument("experiment")
        cmd.add_argument("--opt", nargs="+", help="O0 O1 O2 O3 or all")
        cmd.add_argument("--tasks", help="all, 0-19, or 0,5,10-12")
        cmd.add_argument("--models", nargs="+")
        cmd.add_argument("--prompts", nargs="+")
        cmd.add_argument("--samples", type=int)
        if name in ("run", "generate"):
            cmd.add_argument("--wait", action="store_true", help="wait for local model servers to come up")
    commands.add_parser("build-examples")
    args = parser.parse_args()

    if args.command == "build-examples":
        build()
        return
    exp = load_experiment(
        args.experiment, opt=args.opt, tasks=args.tasks, models=args.models, prompts=args.prompts, samples=args.samples
    )
    if args.command in ("run", "generate"):
        asyncio.run(generate(exp, wait=args.wait))
    if args.command in ("run", "evaluate"):
        evaluate(exp)
    if args.command in ("run", "analyze"):
        analyze(exp)


if __name__ == "__main__":
    main()
