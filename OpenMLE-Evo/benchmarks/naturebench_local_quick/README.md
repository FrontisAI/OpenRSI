# NatureBench local quick experiment

This example runs one NatureBench task in a local Python or Conda environment:

- `s42256-023-00611-x`
- Categorical Counterfactual Outcome Estimation

It uses the same OpenMLE-Evo NatureBench task builder, optimized AIRA-Evo
search, experience memory, three-factor parent selection,
`aggregate_improvement` fitness, and evaluator HTTP contract as the larger
benchmark profile. The default search preserves the formal experiment limits:
a four-hour effective model-plus-sandbox budget, a six-hour wall-clock limit,
and at most 160 nodes. It is quick to try because it runs only one task rather
than Lite-v2 or the full benchmark, not because it changes the per-task search
budget. `--smoke` reduces this to a single generated candidate for a pipeline
check.

The task was selected because it is compact, runs reliably on CPU and Apple
Silicon, produced a positive score in prior local experiments, and lets users
inspect a real multi-node trajectory. This example is a functionality demo,
not a substitute for reporting the full NatureBench benchmark.

See [Reading the local NatureBench trajectory](RESULTS.md) for an illustrative
end-to-end comparison, best-so-far visualization, operator ancestry, and
sub-dataset score attribution.

## Security boundary

Conda isolates Python dependencies, not files, networking, processes, or host
permissions. Model-generated code therefore runs with the current user's local
access. Use this mode only on a trusted development machine. Formal benchmark
runs should continue to use the documented Docker or SCM execution profiles.

## Setup

Clone NatureBench next to OpenRSI, then create the candidate runtime:

```bash
git clone https://github.com/FrontisAI/NatureBench.git
cd OpenRSI/OpenMLE-Evo
conda env create -f environments/naturebench-local.yml
cp .env.example .env
```

The OpenMLE-Evo controller dependencies still belong in the `.venv` described
by the root README. The Conda environment above is used only by generated
candidate programs and the NatureBench evaluator.

## Choose a model endpoint

OpenMLE-Evo uses the same OpenAI-compatible protocol for hosted API models and
self-hosted models. The NatureBench search and local candidate execution path
stay identical.

### Hosted API model

Provide the provider's `/v1` base URL, exposed model ID, and API key:

```bash
export PRIMARY_KEY='your-api-key'

.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo ../../NatureBench \
  --conda-env naturebench-local \
  --model-base-url https://model.example/v1 \
  --model-id served-model-name
```

Credentials are read only from `PRIMARY_KEY`; they are not written into Hydra
config snapshots.

Current NatureBench releases protect evaluator registration and timer controls
with a service control token. When this launcher starts the evaluator, it
creates a private token file under the experiment output and configures the
controller automatically; the token is not exposed to generated candidate
code. If reusing an evaluator that was started separately with a non-default
token path, pass the same path with `--eval-control-token-file PATH`. The
launcher remains compatible with older NatureBench checkouts that do not use
control tokens.

### Self-hosted SGLang model

Start SGLang before starting OpenMLE-Evo. The model path, GPU selection, tensor
parallelism, and context length belong to the SGLang command, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path /absolute/path/to/model \
  --served-model-name my-served-model \
  --host 127.0.0.1 \
  --port 30010 \
  --tp-size 2
```

When SGLang runs on the same machine, connect directly:

```bash
export PRIMARY_KEY=EMPTY

.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo ../../NatureBench \
  --conda-env naturebench-local \
  --model-base-url http://127.0.0.1:30010/v1 \
  --model-id my-served-model
```

When SGLang runs remotely and listens only on its loopback address, use the
model-only SSH tunnel:

```bash
export PRIMARY_KEY=EMPTY

.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo ../../NatureBench \
  --conda-env naturebench-local \
  --model-ssh-host user@model-host \
  --model-ssh-port 30010 \
  --model-id my-served-model
```

The runner needs the SSH host, SGLang API port, and served model ID. It does not
need the remote model-weight path. Candidate code, data, and evaluation remain
local; only model prompts traverse the tunnel.

## One-candidate smoke

Append `--smoke` to any command above. It keeps the same task but generates and
evaluates only one candidate:

```bash
.venv/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo ../../NatureBench \
  --conda-env naturebench-local \
  --model-base-url http://127.0.0.1:30010/v1 \
  --model-id my-served-model \
  --smoke
```

Existing task packages can be reused with `--data-dir PATH --skip-download`.
Advanced users can select another task with `--task TASK_ID` or append Hydra
overrides after `--`; those are intentionally outside this quick example.
