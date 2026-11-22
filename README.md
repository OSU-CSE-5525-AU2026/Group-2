# Group 2: LLM Decompilation Benchmark

Asks LLMs to decompile x86-64 assembly from [HumanEval-Decompile](https://github.com/albertan017/LLM4Decompile)
back into C, then compiles the answer and runs the benchmark's tests on it.

## Setup

Run everything on Cardinal, from the group folder:

```bash
ssh <you>@cardinal.osc.edu
cd /fs/ess/PAS3239/group2/Group-2
module load python/3.12 && source .venv/bin/activate
```

## Run

1. Start a model server. It runs on the debug queue and resubmits itself every hour until stopped.
   The name must match an entry in `models.yaml`:

   ```bash
   ../bv_llm_host/scripts/serve.sh qwen3-8b
   ```

2. Run an experiment. It waits for the server, queries the model, grades the answers and prints the tables:

   ```bash
   python -m harness run experiments/baseline.yaml --wait
   ```

   For long runs, submit it as a job instead: `sbatch slurm/eval.sbatch experiments/baseline.yaml`.

3. Stop the server: `../bv_llm_host/scripts/stop.sh qwen3-8b`. Check what is running with
   `../bv_llm_host/scripts/status.sh`.

Results go to `results/<experiment>/`: `generations.jsonl` (every model reply), `evals.jsonl`
(compile and test result per reply), `summary.csv` and `comparisons.csv`.

## Choosing what to run

An experiment file sets the defaults:

```yaml
# experiments/baseline.yaml
models: [qwen3-8b]                                        # names from models.yaml
prompts: [zero_shot, structured, self_review, few_shot]
opt: [O0]                                                 # O0 O1 O2 O3, or all
tasks: all                                                # or 0-19, or 0,5,10-12
samples: 1
```

Flags override them for one run:

```bash
python -m harness run experiments/baseline.yaml --opt O0 O2 --tasks 0-19 --prompts zero_shot few_shot
```

Stages can also run on their own: `generate` (query the model), `evaluate` (re-grade saved replies),
`analyze` (tables only).

## Models

`models.yaml` lists each model's endpoint and sampling settings. `base_url: local:<name>` means a
bv_llm_host server of that name. Any OpenAI-compatible API works too, for example Groq
(`export GROQ_API_KEY=...`, then use `gpt-oss-120b-groq`).

## Prompts

| Prompt | What the model gets |
|---|---|
| `zero_shot` | the assembly and an instruction to decompile it |
| `structured` | the same, plus steps to work through first: signature, locals, control flow, memory, calls |
| `self_review` | a zero-shot answer, then a second turn asking it to check and fix that answer |
| `few_shot` | three solved examples at the same optimization level first |

