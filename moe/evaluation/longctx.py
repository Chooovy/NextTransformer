"""Long-context suite LC-5: synthetic retrieval and tracking tasks at 2,048 to 4,096 tokens.

Every example is generated deterministically from (task, length, index) and a pinned haystack, so all methods
see byte-identical prompts. Lengths are exact token counts of BOS + prompt + answer:

    scored lengths   2,048 / 3,072 / 4,096   (inside the 4,096-token training context; two of three above 2K)
    probe length     8,192                    (length extrapolation; reported, never part of MoE-CORE)

Tasks (100 examples per task and length):

    passkey          a 5-digit pass key hidden in repeated filler; answer the key                   exact match
    niah_single      one "special magic number" needle for a keyword in natural text                exact match
    niah_multikey    four needles with different keywords; answer the value of one                  exact match
    niah_choice      as niah_multikey, but choose among the four planted values (chance 1/4)        4-way choice
    var_tracking     VAR X = n ... VAR Y = VAR X, with two distractor chains; answer Y's value     exact match

Exact match is teacher-forced greedy match of every answer token (nanochat CORE's language-modeling rule).
The choice task picks the option with the lowest mean answer-token loss (CORE's multiple-choice rule) and is
centered by its 25% chance. Haystack text is the reserved dolma1_7-wiki-en validation stream, never trained on.

Separately, `long_document_bpb` scores validation documents of at least 4,097 tokens and reports bits per
byte by position bucket; the gain of positions 2,048-4,095 over 0-2,047 shows how much a model uses long
context. It is reported next to MoE-CORE, not inside it.
"""
from functools import lru_cache
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from moe.tokenizer import EOT, Dolma2Tokenizer

SCORED_LENGTHS = (2048, 3072, 4096)
PROBE_LENGTHS = (8192,)
EXAMPLES = 100
TASKS = ("passkey", "niah_single", "niah_multikey", "niah_choice", "var_tracking")
CHANCE = {"niah_choice": 0.25}
HAYSTACK_SOURCE = "dolma1_7-wiki-en"
FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")
KEYWORDS = ("apple", "harbor", "violet", "granite", "lantern", "meadow", "falcon", "copper", "willow", "saffron",
            "glacier", "orchid", "thunder", "pepper", "canyon", "marble", "sparrow", "cedar", "ember", "quartz",
            "lagoon", "tulip", "badger", "velvet", "comet", "juniper", "anchor", "maple", "basalt", "heron",
            "cobalt", "fig", "tundra", "walnut", "beacon", "lotus", "otter", "pewter", "summit", "zephyr")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Haystack:
    """Natural filler tokens: the reserved wiki validation stream with document separators as blank lines."""

    def __init__(self, tokens, tokenizer):
        newline = tokenizer.encode("\n\n")
        flat = np.asarray(tokens, dtype=np.int64)
        parts, start = [], 0
        for end in np.flatnonzero(flat == EOT).tolist() + [flat.size]:
            if end > start:
                parts.append(flat[start:end])
                parts.append(np.asarray(newline, dtype=np.int64))
            start = end + 1
        self.tokens = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def take(self, n, rng):
        if n > self.tokens.size:
            raise ValueError("haystack is shorter than the requested filler")
        start = rng.randrange(0, self.tokens.size - n + 1)
        return self.tokens[start:start + n].tolist()


def _filler(tokenizer, n):
    unit = tokenizer.encode(FILLER)
    return (unit * (n // len(unit) + 1))[:n]


def _number(rng, digits):
    return str(rng.randrange(10 ** (digits - 1), 10 ** digits))


def _variable(rng):
    return "".join(rng.choice(LETTERS) for _ in range(5))


def _insert(filler, needles, depths):
    """Insert token lists at fractional depths (ascending), keeping their order."""
    out, last = [], 0
    for needle, depth in sorted(zip(needles, depths), key=lambda p: p[1]):
        cut = int(round(depth * len(filler)))
        out += filler[last:cut] + needle
        last = cut
    return out + filler[last:]


def build(task, length, index, tokenizer, haystack):
    """One example: dict(prefix=token ids incl. BOS, answers=[token ids], gold=int)."""
    rng = random.Random(f"moe-lc5/{task}/{length}/{index}")
    enc = tokenizer.encode
    if task == "passkey":
        key = _number(rng, 5)
        needles = [enc(f" The pass key is {key}. Remember it. {key} is the pass key. ")]
        question = enc(" What is the pass key? The pass key is")
        answers, gold = [enc(f" {key}")], 0
        depths = [rng.uniform(0.05, 0.95)]
    elif task in ("niah_single", "niah_multikey", "niah_choice"):
        count = 1 if task == "niah_single" else 4
        keys = rng.sample(KEYWORDS, count)
        values = [_number(rng, 7) for _ in keys]
        needles = [enc(f"\nThe special magic number for {k} is: {v}.\n") for k, v in zip(keys, values)]
        target = rng.randrange(count)
        question = enc(f"\nWhat is the special magic number for {keys[target]} mentioned in the provided text? "
                       f"The special magic number for {keys[target]} is:")
        if task == "niah_choice":
            answers, gold = [enc(f" {v}") for v in values], target
        else:
            answers, gold = [enc(f" {values[target]}")], 0
        depths = sorted(rng.uniform(0.05, 0.95) for _ in keys)
    elif task == "var_tracking":
        names = set()
        while len(names) < 6:
            names.add(_variable(rng))
        names = sorted(names)
        rng.shuffle(names)
        chains = [(names[2 * i], names[2 * i + 1], _number(rng, 5)) for i in range(3)]
        needles, depths = [], []
        for first, second, value in chains:
            a, b = sorted(rng.uniform(0.05, 0.95) for _ in range(2))
            needles += [enc(f"\nVAR {first} = {value}\n"), enc(f"\nVAR {second} = VAR {first}\n")]
            depths += [a, b]
        target = chains[rng.randrange(3)]
        question = enc(f"\nQuestion: what number is VAR {target[1]} equal to? Answer: VAR {target[1]} =")
        answers, gold = [enc(f" {target[2]}")], 0
    else:
        raise ValueError(f"unknown long-context task {task}")
    budget = length - 1 - sum(map(len, needles)) - len(question) - max(map(len, answers))
    if budget <= 0:
        raise ValueError("needles and question do not fit the requested length")
    filler = _filler(tokenizer, budget) if task == "passkey" else haystack.take(budget, rng)
    prefix = [EOT] + _insert(filler, needles, depths) + question
    return dict(prefix=prefix, answers=answers, gold=gold)


def _score(model, example, device):
    """Exact match for one answer, lowest mean loss over several."""
    seqs = [example["prefix"] + a for a in example["answers"]]
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), EOT, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    ids = ids.to(device)
    logits = model(ids)
    start = len(example["prefix"])
    if len(seqs) == 1:
        stop = len(seqs[0])
        predicted = logits[0, start - 1:stop - 1].argmax(-1)
        return bool(torch.equal(predicted, ids[0, start:stop]))
    losses = []
    for i, s in enumerate(seqs):
        stop = len(s)
        losses.append(F.cross_entropy(logits[i, start - 1:stop - 1].float(), ids[i, start:stop]).item())
    return losses.index(min(losses)) == example["gold"]


def centered(task, accuracy):
    chance = CHANCE.get(task, 0.0)
    return (accuracy - chance) / (1 - chance)


@torch.no_grad()
def evaluate(model, haystack_tokens, device, *, lengths=SCORED_LENGTHS + PROBE_LENGTHS, examples=EXAMPLES,
             tasks=TASKS):
    """Accuracy per task and length, centered task scores over the scored lengths, and the LC-5 mean."""
    tokenizer = Dolma2Tokenizer()
    haystack = Haystack(haystack_tokens, tokenizer)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    capacity = getattr(model, "cos", torch.empty(0, 1 << 30)).size(1)
    results = {}
    for task in tasks:
        for length in lengths:
            if length > capacity:
                continue
            correct = torch.zeros(examples, device=device)
            for i in range(rank, examples, world):
                correct[i] = float(_score(model, build(task, length, i, tokenizer, haystack), device))
            if world > 1:
                dist.all_reduce(correct)
            results.setdefault(task, {})[str(length)] = correct.mean().item()
    tasks_centered = {t: sum(centered(t, by[str(n)]) for n in SCORED_LENGTHS) / len(SCORED_LENGTHS)
                      for t, by in results.items() if all(str(n) in by for n in SCORED_LENGTHS)}
    probe = {t: {n: by[n] for n in by if int(n) not in SCORED_LENGTHS} for t, by in results.items()}
    return dict(accuracy=results, centered=tasks_centered,
                lc_metric=sum(tasks_centered.values()) / len(tasks_centered) if tasks_centered else None,
                extrapolation=probe, scored_lengths=list(SCORED_LENGTHS), examples=examples)


def long_documents(tokens, length, limit):
    """The first `length` tokens (BOS included) of every validation document at least that long, in order."""
    flat = np.asarray(tokens)
    starts = np.flatnonzero(flat == EOT)
    ends = np.append(starts[1:], flat.size)
    docs = [flat[s:s + length] for s, e in zip(starts, ends) if e - s >= length]
    return docs[:limit]


@torch.no_grad()
def long_document_bpb(model, sources, token_bytes, device, *, length=4097, per_source=64,
                      buckets=((0, 2048), (2048, 4096))):
    """Bits per byte by position bucket on long validation documents (macro over sources that have them)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    token_bytes = token_bytes.to(device)
    out = {}
    for name, tokens in sources.items():
        docs = long_documents(tokens, length, per_source)
        sums = torch.zeros(len(buckets), 2, dtype=torch.float64, device=device)  # nats, bytes
        for i in range(rank, len(docs), world):
            row = torch.from_numpy(docs[i].astype(np.int64)).to(device)[None]
            loss = model(row[:, :-1], row[:, 1:], reduction="none").view(-1).double()
            nbytes = token_bytes[row[0, 1:]].double()
            for b, (lo, hi) in enumerate(buckets):
                sums[b, 0] += (loss[lo:hi] * (nbytes[lo:hi] > 0)).sum()
                sums[b, 1] += nbytes[lo:hi].sum()
        if world > 1:
            dist.all_reduce(sums)
        if len(docs):
            out[name] = dict(documents=len(docs), **{f"{lo}-{hi}": (sums[b, 0] / (np.log(2) * sums[b, 1])).item()
                                                       for b, (lo, hi) in enumerate(buckets)})
    keys = [f"{lo}-{hi}" for lo, hi in buckets]
    macro = {k: sum(v[k] for v in out.values()) / len(out) for k in keys} if out else {}
    return dict(sources=out, macro=macro, gain=(macro[keys[0]] - macro[keys[-1]]) if macro else None)


@lru_cache(maxsize=1)
def lengths_table():
    return dict(scored=list(SCORED_LENGTHS), probe=list(PROBE_LENGTHS), examples=EXAMPLES, tasks=list(TASKS))
