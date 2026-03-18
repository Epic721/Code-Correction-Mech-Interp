"""
eval_sandbox.py - generates code w/ the model and runs it against
mbpp test cases inside a sandboxed process. saves results to jsonl
incrementally so nothing is lost if we crash mid-run.

usage:
    python eval_sandbox.py --local        # quick test w/ gpt2
    python eval_sandbox.py                # full run on cluster w/ gemma
"""

import json
import logging
import re
import argparse
import multiprocessing as mp

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_config, setup_logging, PipelineConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------

def _guess_func_name(test_list):
    """tries to figure out what function name the test assertions expect"""
    if not test_list:
        return None
    m = re.search(r"assert\s+(\w+)\s*\(", test_list[0])
    return m.group(1) if m else None


def build_prompt(problem_text, func_name, tokenizer):
    """formats the problem into the model's chat template"""
    if func_name:
        body = (
            f"Write a Python function called `{func_name}` that solves "
            f"the following problem.\n\n{problem_text}\n\n"
            "Output ONLY the Python function. No explanations, no examples, "
            "no markdown fences."
        )
    else:
        body = (
            f"Write a Python function that solves the following problem.\n\n"
            f"{problem_text}\n\n"
            "Output ONLY the Python function. No explanations, no examples, "
            "no markdown fences."
        )

    messages = [{"role": "user", "content": body}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # fallback for models w/o chat templates (gpt2 etc)
        prefix = f"# {problem_text}\n"
        if func_name:
            prefix += f"def {func_name}("
        return prefix


# ---------------------------------------------------------------------------
# code extraction
# ---------------------------------------------------------------------------

def extract_code(raw_output):
    """
    pulls the python code out of the model's response.
    handles markdown fences, preamble text, etc
    """
    # models love wrapping stuff in markdown fences
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", raw_output, re.DOTALL)
    if fenced:
        # multiple blocks -> grab the longest (usually the actual solution)
        return max(fenced, key=len).strip()

    # no fences -- find where actual code starts
    lines = raw_output.split("\n")
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            code_start = i
            break

    return "\n".join(lines[code_start:]).rstrip()


# ---------------------------------------------------------------------------
# sandboxed code execution
# ---------------------------------------------------------------------------

def _sandbox_worker(script, conn):
    """
    runs in a child process -- exec the generated code + tests
    and send the result back through the pipe
    """
    try:
        exec(script, {})
        conn.send({"passed": True, "error": None})
    except Exception as e:
        conn.send({"passed": False, "error": f"{type(e).__name__}: {e}"})
    finally:
        conn.close()


def run_in_sandbox(code, test_list, setup_code="", timeout=10):
    """
    runs generated code + test assertions in an isolated process
    w/ a hard wall-clock timeout. infinite loops get killed.
    """
    parts = []
    if setup_code.strip():
        parts.append(setup_code.strip())
    parts.append(code)
    parts.append("\n".join(test_list))
    script = "\n\n".join(parts)

    parent_conn, child_conn = mp.Pipe(duplex=False)
    proc = mp.Process(target=_sandbox_worker, args=(script, child_conn))
    proc.start()
    child_conn.close()  # parent only reads from its end

    proc.join(timeout=timeout)

    if proc.is_alive():
        # still going = probably stuck in an infinite loop, nuke it
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            # terminate didnt work, pull the plug
            proc.kill()
            proc.join()
        parent_conn.close()
        return {"passed": False, "error": "timeout (possible infinite loop)"}

    # process finished -- grab result from the pipe
    if parent_conn.poll(timeout=1):
        result = parent_conn.recv()
    else:
        result = {
            "passed": False,
            "error": f"worker died without sending result (exit code {proc.exitcode})",
        }

    parent_conn.close()
    return result


# ---------------------------------------------------------------------------
# model loading + generation
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg):
    """loads the hf model + tokenizer w/ proper auth, dtype, and device"""
    log.info(f"loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, token=cfg.hf_token
    )

    log.info(f"loading model: {cfg.model_name} ({cfg.dtype} on {cfg.device})")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        token=cfg.hf_token,
        torch_dtype=cfg.torch_dtype,
        device_map="auto" if cfg.device == "cuda" else None,
    )

    if cfg.device != "cuda":
        model = model.to(cfg.device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    return model, tokenizer


def generate_code(model, tokenizer, prompt, cfg):
    """single forward pass, returns the decoded generation only"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    # greedy when temp=0, sample otherwise
    if cfg.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = cfg.temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    # only decode the newly generated tokens (skip the prompt)
    new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# main evaluation loop
# ---------------------------------------------------------------------------

def run_pipeline(cfg):
    """
    main loop -- loads mbpp, generates code for each problem,
    tests it in the sandbox, and dumps results to jsonl
    """
    model, tokenizer = load_model_and_tokenizer(cfg)

    log.info(f"loading mbpp (split='{cfg.mbpp_split}')")
    ds = load_dataset("google-research-datasets/mbpp", split=cfg.mbpp_split)
    if cfg.num_problems is not None:
        ds = ds.select(range(min(cfg.num_problems, len(ds))))
    log.info(f"{len(ds)} problems to evaluate")

    # check for previous results so we can resume if something crashed
    done_ids = set()
    out_path = cfg.generations_path
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["task_id"])
        log.info(f"found {len(done_ids)} existing results, resuming")

    n_pass, n_total = 0, 0

    with open(out_path, "a") as fout:
        for i, problem in enumerate(ds):
            tid = problem["task_id"]
            if tid in done_ids:
                continue

            func_name = _guess_func_name(problem["test_list"])
            prompt = build_prompt(problem["text"], func_name, tokenizer)
            raw_gen = generate_code(model, tokenizer, prompt, cfg)
            code = extract_code(raw_gen)

            setup = problem.get("test_setup_code", "") or ""
            result = run_in_sandbox(
                code, problem["test_list"],
                setup_code=setup, timeout=cfg.exec_timeout_sec,
            )

            record = {
                "task_id": tid,
                "text": problem["text"],
                "prompt": prompt,
                "raw_generation": raw_gen,
                "extracted_code": code,
                "test_list": problem["test_list"],
                "passed": result["passed"],
                "error": result["error"],
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()  # flush every write so we dont lose data on crashes

            n_total += 1
            if result["passed"]:
                n_pass += 1

            tag = "PASS" if result["passed"] else "FAIL"
            log.info(f"[{i + 1}/{len(ds)}] task {tid}: {tag}")

    # final summary
    if out_path.exists() and out_path.stat().st_size > 0:
        all_recs = [json.loads(ln) for ln in open(out_path)]
        total_pass = sum(r["passed"] for r in all_recs)
        log.info(
            f"done -- {total_pass}/{len(all_recs)} passed "
            f"({100 * total_pass / max(len(all_recs), 1):.1f}%)"
        )


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="generate + evaluate code against mbpp"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="swap in gpt2 + tiny subset for local testing",
    )
    parser.add_argument(
        "--num-problems", type=int, default=None,
        help="only run this many problems (useful for debugging)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = get_config(local=args.local)

    if args.num_problems is not None:
        cfg.num_problems = args.num_problems

    log.info(f"model={cfg.model_name}  device={cfg.device}  dtype={cfg.dtype}")
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
