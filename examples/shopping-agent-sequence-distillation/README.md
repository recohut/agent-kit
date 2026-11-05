# Distilling Multi-Step Shopping Agents Into a 4B Model

Runnable companion example for the Recohut blog article on sequence-level
knowledge distillation and multi-step tool-use evaluation with `Qwen/Qwen3.5-4B`.
Read the complete article in [`article.md`](article.md) or on the [Recohut Blog](https://recohut.com/blog/shopping-agent-sequence-distillation).

## Overview

This example runs a two-stage sequence-distillation and evaluation workflow on
an NVIDIA GPU (16 GiB VRAM or higher, such as an NVIDIA T4 or L4):

1. **Trajectory filtering & QLoRA training**: Scores 12 published teacher
   trajectories (`GLM-5.3`) against the ShoppingBench product catalog using the
   native deterministic scorer, retains the 8 solved trajectories, and trains
   four 4-bit NF4 QLoRA updates (`r=4`, `lora_alpha=8`, `q_proj`/`v_proj`) on
   `Qwen/Qwen3.5-4B` with assistant-only loss masking and a 2,048-token cap.
2. **Multi-step ReAct evaluation**: Serves both the unmodified base model
   (`disable_adapter()`) and the distilled student adapter through a local
   OpenAI-compatible `/v1/chat/completions` loopback endpoint, running an
   8-step ReAct catalog search loop (512 max tokens per step, web search
   disabled) on a held-out product query.

## Setup

1. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the pinned ShoppingBench benchmark harness and catalog index under
   `${SEQKD_ROOT:-/content/seqkd-scored}/bench` using the upstream Apache-2.0
   [setup script](https://raw.githubusercontent.com/aws/sagemaker-hyperpod-recipes/520b248f47f488f60af52f40ee6172f606894e59/examples/sequence-distillation/scripts/setup_shopbench.sh):

   ```bash
   curl -fsSL https://raw.githubusercontent.com/aws/sagemaker-hyperpod-recipes/520b248f47f488f60af52f40ee6172f606894e59/examples/sequence-distillation/scripts/setup_shopbench.sh -o setup_shopbench.sh
   bash setup_shopbench.sh /content/seqkd-scored/bench
   ```

3. Open and run [`main.ipynb`](main.ipynb) top to bottom, or run the workflow
   from Python:

   ```python
   from workflow import evaluate, new_run, prepare, train

   run = new_run("seqkd")
   examples, case, solved = prepare(run)
   adapter, metrics = train(run, examples, solved)
   results = evaluate(run, case, adapter)
   print(results)
   ```

## Observed results

- **Teacher verification**: 8 of 12 candidate product trajectories passed the
  deterministic catalog scorer (`solved_teacher_tasks: 8`).
- **4-bit QLoRA training**: 4 updates completed with step losses `0.8264`,
  `0.8013`, `0.8838`, and `0.9406`, peaking at **9.30 GiB** of allocated CUDA
  memory.
- **Held-out tool-loop evaluation**: The base model terminated after 2 steps
  with `recommend: 0.0` (`asr: 0.0`), whereas the distilled student completed
  4 catalog tool-loop steps and passed with `recommend: 1.0` (`asr: 1.0`).

## Architecture figure

The editable Draw.io source is in [`assets/architecture.drawio`](assets/architecture.drawio)
alongside the exported [`assets/architecture.svg`](assets/architecture.svg) and
[`assets/architecture.png`](assets/architecture.png).
