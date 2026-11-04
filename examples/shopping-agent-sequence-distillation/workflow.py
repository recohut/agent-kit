"""Bounded CUDA sequence-distillation workflow for ShoppingBench product tasks."""

import gc
import gzip
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from tempfile import mkdtemp
from threading import Thread
from time import perf_counter
from urllib.request import urlretrieve

MODEL = "Qwen/Qwen3.5-4B"
RECIPE_COMMIT_ID = "520b248f47f488f60af52f40ee6172f606894e59"
RECIPE = (
    f"https://raw.githubusercontent.com/aws/sagemaker-hyperpod-recipes/"
    f"{RECIPE_COMMIT_ID}/examples/sequence-distillation"
)
ROOT = Path(os.environ.get("SEQKD_ROOT", "/content/seqkd-scored"))
BENCH = ROOT / "bench"
SB = BENCH / "ShoppingBench"
PYTHON = BENCH / "sbenv/bin/python"
TRACE = "teacher_trajectories/glm-5.3/rollout_teacher_s0_product.jsonl.gz"


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select an NVIDIA CUDA GPU before running.")
    return torch


def new_run(lane="seqkd"):
    require_cuda()
    ROOT.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix=f"{lane}-", dir=ROOT))


def _env():
    return {**os.environ, "JAVA_HOME": str(BENCH / "jdk"), "OPENAI_API_KEY": "unused"}


def _recipe_files():
    for name in (
        "scripts/split.py", "scripts/encode_rollouts.py", "scripts/score_rollouts.py",
        "scripts/shopbench_harness.py", "split/product.json", TRACE,
    ):
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            partial = path.with_suffix(path.suffix + ".partial")
            urlretrieve(f"{RECIPE}/{name}", partial)
            partial.replace(path)
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def prepare(run):
    if not PYTHON.is_file() or not any((SB / "indexes").glob("segments_*")):
        raise RuntimeError("Complete the pinned ShoppingBench setup first.")
    _recipe_files()
    from encode_rollouts import sft_rows
    from split import load_split, write_jsonl

    training, held_out = load_split(SB / "data", "product")
    if {row["query"] for row in training} & {row["query"] for row in held_out}:
        raise RuntimeError("The training and held-out split overlaps.")
    problems = run / "train-product.jsonl"
    write_jsonl(training, problems)
    rollout = run / "teacher-product.jsonl"
    with gzip.open(ROOT / TRACE, "rb") as source, rollout.open("wb") as dest:
        dest.writelines(islice(source, 12))
    scores = run / "teacher-scores.json"
    result = subprocess.run(
        [str(PYTHON), str(ROOT / "scripts/score_rollouts.py"), "--task", "product",
         "--rollouts", str(rollout), "--problems", str(problems), "--per-query", str(scores)],
        cwd=SB, env=_env(), capture_output=True, text=True, timeout=900,
    )
    (run / "teacher-scorer.log").write_text(result.stdout + result.stderr)
    if not scores.is_file():
        raise RuntimeError(f"Scorer failed. See {run / 'teacher-scorer.log'}.")
    solved = {q for q, score in json.loads(scores.read_text()).items() if score["solved"]}
    return sft_rows(rollout, solved, include_reasoning=True), held_out[0], len(solved)


def _render(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )


def _load_base(torch):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForImageTextToText.from_pretrained(
        MODEL, quantization_config=quantization, device_map={"": 0}, dtype=torch.float16,
    )


def train(run, examples, solved_count):
    torch = require_cuda()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    if torch.cuda.mem_get_info()[0] < 10 * 2**30:
        raise RuntimeError("Free at least 10 GiB of CUDA memory before starting.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = {}
    for query, example in examples:
        user, assistant = example["messages"]
        rendered = _render(tokenizer, user["content"])
        prompt = tokenizer.encode(rendered, add_special_tokens=False)
        target = assistant["content"]
        if rendered.rstrip().endswith("<think>"):
            target = target.removeprefix("<think>")
        tokens = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prompt) + len(tokens) <= 2048:
            rows.setdefault(query, (prompt, tokens))
        if len(rows) == 4:
            break
    if len(rows) != 4:
        raise RuntimeError("Fewer than four distinct scorer-approved tasks fit the context cap.")

    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    student = prepare_model_for_kbit_training(
        _load_base(torch), gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    student.config.use_cache = False
    student = get_peft_model(
        student, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                            task_type="CAUSAL_LM"),
    )
    student.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in student.parameters() if parameter.requires_grad), lr=1e-4,
    )
    losses = []
    for prompt, target in rows.values():
        ids = torch.tensor([prompt + target], device="cuda")
        labels = torch.tensor([[-100] * len(prompt) + target], device="cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = student(input_ids=ids, labels=labels).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    adapter = run / "scored-adapter"
    student.save_pretrained(adapter)
    metrics = {
        "solved_teacher_tasks": solved_count, "updates": len(losses), "losses": losses,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (run / "training.json").write_text(json.dumps(metrics, indent=2))
    del student, optimizer, loss
    gc.collect()
    torch.cuda.empty_cache()
    return adapter, metrics


def evaluate(run, case, adapter):
    torch = require_cuda()
    from flask import Flask, jsonify, request
    from peft import PeftModel
    from shopbench_harness import Harness
    from split import write_jsonl
    from transformers import AutoTokenizer
    from werkzeug.serving import make_server

    case_file = run / "held-out-product.jsonl"
    write_jsonl([case], case_file)
    runner = run / "one_rollout.py"
    runner.write_text(
        "import json, sys\n"
        "sys.path.insert(0, 'src/agent')\n"
        "import run_rollout as native\n"
        "native.MAX_STEPS = 8\n"
        "native.toolmap.pop('web_search', None)\n"
        "with open(sys.argv[1]) as source:\n"
        "    cfg = json.load(source)\n"
        "with open(cfg['synthesize_file']) as source:\n"
        "    query = json.loads(source.readline())['query']\n"
        "cfg['exclude_tools'] = ['web_search']\n"
        "native.react_loop(query, cfg)\n"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = PeftModel.from_pretrained(_load_base(torch), adapter).eval()
    app = Flask("seqkd-loopback")
    mode = "base"

    @app.post("/v1/chat/completions")
    def complete():
        body = request.get_json()
        text = "\n\n".join(message["content"] for message in body["messages"])
        encoded = tokenizer(
            _render(tokenizer, text), add_special_tokens=False, return_tensors="pt",
        ).to("cuda")
        length = encoded["input_ids"].shape[-1]
        if length > 8192:
            return jsonify(error={"message": "Prompt cap exceeded"}), 400
        with torch.inference_mode(), (model.disable_adapter() if mode == "base" else nullcontext()):
            output = model.generate(**encoded, max_new_tokens=512, do_sample=False)
        content = tokenizer.decode(output[0, length:], skip_special_tokens=True)
        return jsonify(
            id=f"seqkd-{mode}", object="chat.completion", created=0, model=MODEL,
            choices=[{"index": 0, "message": {"role": "assistant", "content": content},
                      "finish_reason": "stop"}],
        )

    harness = Harness(BENCH)
    server = make_server("127.0.0.1", 0, app)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    results = {}
    try:
        if not harness.search_ok():
            harness.start_search(workers=1)
        for mode in ("base", "student"):
            cfg, rollout = harness.write_rollout_cfg(
                run, mode, "product", case_file, f"http://127.0.0.1:{server.server_port}/v1",
                True, threads=1, max_tokens=512,
            )
            start = perf_counter()
            subprocess.run(
                [str(PYTHON), str(runner), str(cfg)], cwd=SB, env=_env(),
                check=True, timeout=420,
            )
            score = harness.score("product", rollout, case_file)
            if score["n"] != 1:
                raise RuntimeError(f"Incomplete {mode} evaluation: {score}")
            results[mode] = {
                **score, "elapsed_seconds": round(perf_counter() - start, 2),
                "steps": len(json.loads(rollout.read_text())),
            }
            (run / "evaluation.json").write_text(json.dumps(results, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()
        harness.stop_all()
    return results
