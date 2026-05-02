"""
eval_sandbox.py - generates code w/ the model and runs it against
unit tests inside a sandboxed process. saves results to jsonl
incrementally so nothing is lost if we crash mid-run.

phase 2 update: now supports both mbpp and humaneval through a small
adapter abstraction. the multiprocessing sandbox is dataset-agnostic --
each adapter is responsible for assembling the test script its dataset
expects, and the sandbox just executes whatever string it gets.

usage:
    python eval_sandbox.py --local                          # smoke test on gpt2
    python eval_sandbox.py --dataset mbpp                   # full mbpp run on cluster
    python eval_sandbox.py --dataset humaneval              # full humaneval run on cluster
    python eval_sandbox.py --dataset humaneval --num-problems 20   # small subset
"""

import json
import logging
import re
import argparse
import multiprocessing as mp

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    get_config,
    setup_logging,
    log_config,
    PipelineConfig,
    SUPPORTED_DATASETS,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# dataset adapters
# ---------------------------------------------------------------------------
#
# each adapter knows how to (a) load its dataset, (b) build a chat-templated
# prompt for the model, (c) assemble a runnable test script for the sandbox,
# and (d) produce dataset-specific extra fields for the output record.
#
# the rest of the pipeline (sandbox, generation, model loading) stays
# completely dataset-agnostic. adding a new benchmark later (eg APPS) is
# just one more subclass.

class DatasetAdapter:
    """interface for plugging a code-gen dataset into the eval pipeline"""

    name: str = ""

    def load(self, cfg: PipelineConfig):
        raise NotImplementedError

    def task_id(self, problem) -> str | int:
        raise NotImplementedError

    def build_prompt(self, problem, tokenizer) -> str:
        raise NotImplementedError

    def assemble_test_script(self, problem, code: str) -> str:
        raise NotImplementedError

    def make_record_extras(self, problem) -> dict:
        """dataset-specific fields to attach to each output record"""
        return {}


class MBPPAdapter(DatasetAdapter):
    """
    mbpp = mostly basic python problems. ~500 problems in the test split,
    each w/ a natural-language description and a list of assert statements.
    """

    name = "mbpp"

    def load(self, cfg):
        log.info(f"loading mbpp (split='{cfg.mbpp_split}')")
        ds = load_dataset(
            "google-research-datasets/mbpp", split=cfg.mbpp_split
        )
        if cfg.num_problems is not None:
            ds = ds.select(range(min(cfg.num_problems, len(ds))))
        return ds

    def task_id(self, problem):
        return problem["task_id"]

    def _guess_func_name(self, problem):
        """
        mbpp doesnt explicitly tell us the function name, but the first
        assert statement usually does. eg `assert remove_Occ(...) == ...`
        """
        tests = problem.get("test_list") or []
        if not tests:
            return None
        m = re.search(r"assert\s+(\w+)\s*\(", tests[0])
        return m.group(1) if m else None

    def build_prompt(self, problem, tokenizer):
        # NOTE: this prompt string is intentionally byte-identical to the
        # milestone version. our cached activations were extracted at the
        # final token of these exact prompts, so changing this would break
        # the alignment between activations and generations.
        func_name = self._guess_func_name(problem)
        problem_text = problem["text"]

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
            # fallback for models without chat templates (gpt2 in local mode)
            prefix = f"# {problem_text}\n"
            if func_name:
                prefix += f"def {func_name}("
            return prefix

    def assemble_test_script(self, problem, code):
        # mbpp script layout:
        #   <test_setup_code>     (optional, often empty)
        #   <generated_code>      (defines the target function)
        #   <test_list joined>    (a bunch of `assert ...` statements)
        setup = (problem.get("test_setup_code") or "").strip()
        parts = []
        if setup:
            parts.append(setup)
        parts.append(code)
        parts.append("\n".join(problem["test_list"]))
        return "\n\n".join(parts)

    def make_record_extras(self, problem):
        return {
            "text": problem["text"],
            "test_list": problem["test_list"],
        }


class HumanEvalAdapter(DatasetAdapter):
    """
    humaneval = openai's 164-problem code completion benchmark. each
    problem ships w/ a function signature + docstring as the prompt, a
    canonical solution, an entry-point name, and a `check(candidate)`
    function in the test field that gets called against the generated
    function.
    """

    name = "humaneval"

    def load(self, cfg):
        log.info(f"loading humaneval (split='{cfg.humaneval_split}')")
        ds = load_dataset("openai_humaneval", split=cfg.humaneval_split)
        if cfg.num_problems is not None:
            ds = ds.select(range(min(cfg.num_problems, len(ds))))
        return ds

    def task_id(self, problem):
        return problem["task_id"]  # eg "HumanEval/0"

    def build_prompt(self, problem, tokenizer):
        # humaneval prompts are already a partial python file (signature +
        # docstring). we wrap them in the same chat template shape we use
        # for mbpp so the OOD generalization test is a fair apples-to-apples
        # comparison -- only the dataset content shifts, not the framing.
        prompt_block = problem["prompt"].rstrip()
        body = (
            "Implement the following Python function. The signature and "
            "docstring are given below.\n\n"
            f"```python\n{prompt_block}\n```\n\n"
            "Output ONLY the complete function definition (signature and "
            "implementation). No explanations, no examples, no markdown fences."
        )

        messages = [{"role": "user", "content": body}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # gpt2-style fallback: hand the model the partial file and let it
            # complete in raw completion mode
            return prompt_block + "\n"

    def assemble_test_script(self, problem, code):
        # humaneval script layout:
        #   <generated_code>           (must define the entry-point function)
        #   <problem.test>             (defines `check(candidate)`)
        #   check(<entry_point>)       (actually runs the test)
        #
        # if the model returns just the function body without the signature,
        # this will raise NameError for the entry point -- that's a real
        # model failure mode and shows up correctly in the labels
        parts = [
            code,
            problem["test"],
            f"check({problem['entry_point']})",
        ]
        return "\n\n".join(parts)

    def make_record_extras(self, problem):
        return {
            "text": problem["prompt"],
            "entry_point": problem["entry_point"],
            "test": problem["test"],
            "canonical_solution": problem.get("canonical_solution", ""),
        }


_ADAPTERS: dict[str, DatasetAdapter] = {
    MBPPAdapter.name: MBPPAdapter(),
    HumanEvalAdapter.name: HumanEvalAdapter(),
}


def get_adapter(name: str) -> DatasetAdapter:
    if name not in _ADAPTERS:
        raise ValueError(
            f"unknown dataset {name!r}; available: {list(_ADAPTERS)}"
        )
    return _ADAPTERS[name]


# ---------------------------------------------------------------------------
# code extraction
# ---------------------------------------------------------------------------

def extract_code(raw_output: str) -> str:
    """
    pulls the python code out of the model's response. handles markdown
    fences, preamble explanation text, etc. shared across datasets since
    both mbpp and humaneval responses look basically the same in practice.
    """
    # models love wrapping stuff in ```python``` fences regardless of how
    # politely we ask them not to
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", raw_output, re.DOTALL)
    if fenced:
        # multiple blocks -> grab the longest, usually the actual solution
        return max(fenced, key=len).strip()

    # no fences -- find where the actual code starts and chop the preamble
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

def _sandbox_worker(script: str, conn):
    """
    runs in a child process -- exec the assembled script and send the
    pass/fail result back through the pipe. an empty globals dict gives
    the worker a clean namespace each time.
    """
    try:
        exec(script, {})
        conn.send({"passed": True, "error": None})
    except BaseException as e:
        # BaseException catches KeyboardInterrupt + SystemExit too, in case
        # generated code does anything weird like sys.exit()
        conn.send({"passed": False, "error": f"{type(e).__name__}: {e}"})
    finally:
        conn.close()


def run_in_sandbox(script: str, timeout: int = 10) -> dict:
    """
    runs an assembled script string in an isolated process w/ a hard
    wall-clock timeout. infinite loops get killed.
    """
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

    # process finished -- try to grab the result from the pipe
    if parent_conn.poll(timeout=1):
        result = parent_conn.recv()
    else:
        # this happens if the worker segfaulted or got OOM-killed before it
        # could send anything back. its rare but worth surfacing clearly
        result = {
            "passed": False,
            "error": (
                f"worker died without sending result "
                f"(exit code {proc.exitcode})"
            ),
        }

    parent_conn.close()
    return result


# ---------------------------------------------------------------------------
# model loading + generation
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: PipelineConfig):
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


def generate_code(model, tokenizer, prompt: str, cfg: PipelineConfig) -> str:
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
# resume + summary helpers
# ---------------------------------------------------------------------------

def _read_done_ids(out_path) -> set:
    """
    read existing task_ids from the jsonl. tolerates a partially-written
    final line from a previous crash by skipping it w/ a warning rather
    than blowing up the whole resume path.
    """
    done_ids: set = set()
    if not out_path.exists():
        return done_ids

    with open(out_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done_ids.add(rec["task_id"])
            except (json.JSONDecodeError, KeyError):
                log.warning(
                    f"skipping malformed line {i} in {out_path} "
                    "(probably a partial write from a previous crash)"
                )
    return done_ids


def _check_legacy_layout(cfg: PipelineConfig):
    """
    if we're in mbpp mode and the user has milestone-era outputs sitting
    at outputs/generations.jsonl, log a hint about migrating them. we
    deliberately dont auto-move things -- last thing we want is to silently
    rearrange the user's files.
    """
    if cfg.dataset != "mbpp":
        return

    legacy = cfg.legacy_generations_path
    new = cfg.generations_path
    if legacy.exists() and not new.exists():
        log.warning("=" * 60)
        log.warning("legacy outputs detected at the milestone layout:")
        log.warning(f"  {legacy}")
        log.warning("phase 2 expects mbpp artifacts under outputs/mbpp/.")
        log.warning("to keep using your existing generations, run:")
        log.warning(f"  mv {legacy} {new}")
        log.warning(
            f"  mv {cfg.output_dir / 'activations'} "
            f"{cfg.dataset_path / 'activations'}"
        )
        log.warning(
            f"  mv {cfg.output_dir / 'probes'} "
            f"{cfg.dataset_path / 'probes'}"
        )
        log.warning("otherwise this run will start from scratch under outputs/mbpp/.")
        log.warning("=" * 60)


def _summarize(out_path):
    if not out_path.exists() or out_path.stat().st_size == 0:
        return
    with open(out_path, encoding="utf-8") as f:
        all_recs = [json.loads(ln) for ln in f if ln.strip()]
    if not all_recs:
        return
    total_pass = sum(r["passed"] for r in all_recs)
    log.info(
        f"done -- {total_pass}/{len(all_recs)} passed "
        f"({100 * total_pass / len(all_recs):.1f}%)"
    )


# ---------------------------------------------------------------------------
# main evaluation loop
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig):
    """
    main loop -- loads the chosen dataset, generates code for each problem,
    tests it in the sandbox, and dumps results to jsonl (incrementally,
    so a crash mid-run doesnt nuke previous progress).
    """
    adapter = get_adapter(cfg.dataset)
    log.info(f"using dataset adapter: {adapter.name}")
    _check_legacy_layout(cfg)

    model, tokenizer = load_model_and_tokenizer(cfg)
    ds = adapter.load(cfg)
    log.info(f"{len(ds)} problems to evaluate")

    out_path = cfg.generations_path
    done_ids = _read_done_ids(out_path)
    if done_ids:
        log.info(f"found {len(done_ids)} existing results, resuming")

    n_pass = n_total = 0

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, problem in enumerate(ds):
            tid = adapter.task_id(problem)
            if tid in done_ids:
                continue

            prompt = adapter.build_prompt(problem, tokenizer)
            raw_gen = generate_code(model, tokenizer, prompt, cfg)
            code = extract_code(raw_gen)

            script = adapter.assemble_test_script(problem, code)
            result = run_in_sandbox(script, timeout=cfg.exec_timeout_sec)

            record = {
                "dataset": cfg.dataset,
                "task_id": tid,
                "prompt": prompt,
                "raw_generation": raw_gen,
                "extracted_code": code,
                "passed": result["passed"],
                "error": result["error"],
                **adapter.make_record_extras(problem),
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()  # flush every write so we dont lose data on crashes

            n_total += 1
            if result["passed"]:
                n_pass += 1

            tag = "PASS" if result["passed"] else "FAIL"
            log.info(f"[{i + 1}/{len(ds)}] {tid}: {tag}")

    _summarize(out_path)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="generate + evaluate code on mbpp or humaneval"
    )
    parser.add_argument(
        "--dataset", default="mbpp", choices=list(SUPPORTED_DATASETS),
        help="which benchmark to evaluate on",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="swap in gpt2 + tiny subset for local cpu testing",
    )
    parser.add_argument(
        "--num-problems", type=int, default=None,
        help="cap on the number of problems (useful for debugging)",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--log-file", default=None,
        help="optional path to also tee logs into (handy for slurm runs)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level, log_file=args.log_file)
    cfg = get_config(local=args.local, dataset=args.dataset)

    if args.num_problems is not None:
        cfg.num_problems = args.num_problems

    log_config(cfg, log)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
