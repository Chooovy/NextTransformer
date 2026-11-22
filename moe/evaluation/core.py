"""CORE-22 for the MoE track: the dense protocol's pinned DCLM bundle and nanochat's scorer, unchanged.

Only the tokenizer (dolma2) and the context differ: prompts are left-truncated to the 4,096-token training
context (only SQuAD prompts are longer; every scored answer is kept, see nexttransformer/evaluation/core-lengths.json).
Centering and the aggregate are nanochat's: centered = (acc - b) / (1 - b), CORE = mean over the 22 tasks.
"""
import csv
import json
import random
from pathlib import Path
import time

import torch
import yaml

from nanochat.core_eval import evaluate_task
from nexttransformer.evaluation.metrics import verify_core
from moe.spec import PROTOCOL
from moe.tokenizer import Dolma2Tokenizer


class Truncated(torch.nn.Module):
    """nanochat's scorer reads `max_seq_len` to left-crop long prompts."""

    def __init__(self, model, max_tokens):
        super().__init__()
        self.model, self.max_seq_len, self.longest = model, max_tokens, 0

    def forward(self, idx):
        self.longest = max(self.longest, idx.size(1))
        return self.model(idx)


def tasks(resources):
    bundle = Path(resources) / "eval_bundle"
    config = yaml.safe_load((bundle / "core.yaml").read_text(encoding="utf-8"))
    with open(bundle / "eval_meta_data.csv", encoding="utf-8") as f:
        baselines = {row["Eval Task"]: float(row["Random baseline"]) for row in csv.DictReader(f)}
    return bundle, config["icl_tasks"], baselines


@torch.no_grad()
def evaluate(model, resources, device, *, max_per_task=-1, max_tokens=PROTOCOL.core_max_tokens):
    verify_core(resources)
    bundle, entries, baselines = tasks(resources)
    tokenizer = Dolma2Tokenizer()
    scorer = Truncated(model, max_tokens)
    results, centered = {}, {}
    started = time.monotonic()
    for task in entries:
        label = task["label"]
        meta = dict(task_type=task["icl_task_type"], dataset_uri=task["dataset_uri"],
                    num_fewshot=task["num_fewshot"][0], continuation_delimiter=task.get("continuation_delimiter", " "))
        with open(bundle / "eval_data" / meta["dataset_uri"], encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        accuracy = evaluate_task(scorer, tokenizer, data, device, meta)
        b = 0.01 * baselines[label]
        results[label], centered[label] = accuracy, (accuracy - b) / (1 - b)
    return dict(results=results, centered_results=centered, core_metric=sum(centered.values()) / len(centered),
                full_core=max_per_task <= 0, max_tokens=max_tokens, longest_input=scorer.longest,
                seconds=time.monotonic() - started)
