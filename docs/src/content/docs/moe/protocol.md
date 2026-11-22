---
title: MoE protocol
description: The fixed contract of the two MoE tracks (d768, d1024): shapes, budgets, compute, throughput and MoE-CORE.
---

The MoE track asks the NextTransformer question for sparse models: **when the training budget is fixed, how should
a Mixture-of-Experts language model spend its computation?** Every method trains on the same tokens in the same
order, at the same theoretical training FLOPs per token as the track's reference baseline, and is scored with the
same throughput harness and the same MoE-CORE suite. This page is normative; the machine-readable values are in
[`spec.py`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/spec.py), and every configuration, run and record carries the protocol and track fingerprints.

## Lifecycle

```text
 implement          calibrate              admit                 train                   evaluate               audit + export
 a plugin under ─▶  FLOPs/token within ─▶  config + cost   ─▶   all updates, seed 42 ─▶ val BPB every 500,  ─▶  ledger within 2%,
 moe/methods/       1% of the anchor       record match         --formal                CORE-22 + LC-5 x 4,    record + throughput
                                           the source                                   final BPB              → pull request
```

This is the same lifecycle as the dense protocol. Only the shapes, budgets, data and evaluation differ.

## Two tracks

Both tracks reproduce Marin's "May Recipe" compute-optimal Grug MoE cells: shape, context, batch and number of
updates ([Marin README](https://github.com/marin-community/marin/blob/cd0e932c780e0171c5612658faa0f180a851569e/experiments/grug/moe/README.md),
[issue #6074](https://github.com/marin-community/marin/issues/6074)).

| | **d768** | **d1024** |
|---|---|---|
| Marin cell | `moe_may_compute_opt_d768_ep1` | `moe_may_compute_opt_d1024_ep1` |
| Width / layers | 768 / 8 | 1024 / 11 |
| Attention | 6 query heads, 1 KV head, head dim 128 | 8 query heads, 2 KV heads, head dim 128 |
| Long (full-attention) layers | 3, 7 | 3, 7, 10 |
| Routed experts | 256 SwiGLU, width 384, top 4 | 256 SwiGLU, width 512, top 4 |
| Shared expert | 1 SwiGLU, width 768 | 1 SwiGLU, width 1024 |
| Parameters, total | 1,992,831,744 | 4,701,051,904 |
| Parameters, active per token | 209,203,968 | 341,072,896 |
| Active, excluding embedding and head | 55,063,296 | 135,552,000 |
| Context | 4,096 tokens | 4,096 tokens |
| Rows × tokens per update | 64 × 4,096 = 262,144 | 128 × 4,096 = 524,288 |
| Updates | 16,875 | 16,080 |
| **Training tokens** | **4,423,680,000** | **8,430,551,040** |
| **Training FLOPs per token** (the anchor) | **915,429,888** | **1,656,268,800** |
| Training FLOPs | 4.0496 × 10¹⁸ | 1.3963 × 10¹⁹ |
| Same budget in Marin's convention | 2.797 × 10¹⁸ (Marin: 2.81 × 10¹⁸) | 1.152 × 10¹⁹ (Marin: 1.16 × 10¹⁹) |
| Peak learning rate | 0.00193 | 0.00203 |
| CORE-22 + LC-5 events | 4,000 / 8,000 / 12,000 / 16,875 | 4,000 / 8,000 / 12,000 / 16,080 |
| Validation events | 35 (0, every 500, 16,875) | 34 (0, every 500, 16,080) |

Marin's convention counts 6 × active non-embedding parameters plus non-causal attention, and leaves out the head. It
reproduces Marin's published budgets to within 0.7%, which checks that the shapes match. This protocol counts more
precisely (next section), so its anchor is larger.

## Reference baseline (`method = "baseline"`)

Pre-norm blocks of RMSNorm → GQA attention → RMSNorm → MoE FFN, with an untied embedding and head.

- **Attention.** QK RMS norm and RoPE (θ = 10,000). Short layers use a 2,048-key sliding window. Long layers (every
  fourth layer and the last) attend causally over the full 4,096 tokens.
- **MoE FFN.** Every layer is an MoE layer: 256 routed SwiGLU experts (top 4) plus one always-on SwiGLU shared
  expert. Dispatch is dropless.
- **Router.** fp32 linear scores. Experts are selected on logits plus a non-trained balancing bias. The selected
  sigmoid gates are renormalized to sum to 2.5. After every update the bias moves by `0.001 × sign(mean load −
  expert load)`. There is no auxiliary loss and no z-loss.
- **Initialization.** N(0, (0.5/√d)²) for matrices, N(0, 1) for the embedding, ones for norm weights.

**Deviations from Marin's run-time code.** The baseline has no GatedNorm, no XSA and no attention output gate. It uses
full rather than half RoPE, and sign-update bias balancing instead of quantile balancing. It uses AdamW instead of
MuonH (see Optimizer). A method that wants any of these adds it as a plugin and pays for it under the compute control.

## What is controlled

| Control | Value |
|---|---|
| Data | [`xjxyys/dolma3_hq_300B_shuffled_clean`](https://huggingface.co/datasets/xjxyys/dolma3_hq_300B_shuffled_clean) at revision `128a6856…`. Training shards are 00000, 00001, … in order; shard 00255 is the development split; the six `validation/*.parquet` files are reserved for validation |
| Tokenizer | `allenai/dolma2-tokenizer` (revision `5292e5d6…`, `tokenizer.json` SHA-256 pinned): cl100k_base plus 22 added tokens, vocabulary 100,278, embedding and head padded to 100,352 rows |
| Packing | Every document gets a `<|endoftext|>` prefix. Documents are concatenated in file order. Row k is tokens [4096k, 4096k + 4097). No truncation, padding or document masking |
| Data order | Update u reads rows [uR, (u+1)R). Ranks take contiguous equal slices, and micro-batches split them in order. Any world size that divides R gives the same updates |
| Budget | The track's tokens, including warmup. Nothing else is trained |
| Randomness | Seed 42 |
| Compute | Training FLOPs per token within **1%** of the track anchor at admission, and charged FLOPs within **2%** cumulatively |
| Optimizer | AdamW (β = 0.9, 0.95; ε = 1e-8). Weight decay 0.1 on matrices, none on vectors and routing state. Gradient clip 1.0. Linear warmup over 1% of updates, then linear decay to 0. Peak LR from Marin's law `0.06602 · T^-0.395 · d^-0.150 · √B` |
| Precision | bf16 parameters and compute, fp32 gradient reduction and router (FSDP2) |
| Evaluation | Validation BPB at step 0, every 500 updates and the last update. CORE-22 + LC-5 at the four evaluation steps. At the end: full validation BPB and long-document BPB |

Parameters, memory and throughput are **outcomes**. A method may use more or fewer total parameters. It is measured
for speed, never admitted or rejected on speed.

## Compute accounting

Training FLOPs per input token = **measured matmul FLOPs** + **attention edges charged by rule**.

- **Matmul FLOPs are measured.** Each block and the head run one forward and backward pass at full size on CPU under
  `torch.utils.flop_counter.FlopCounterMode`, one block at a time. Dropless routing makes the matmul cost independent
  of which experts are chosen, so 256 random tokens measure it exactly. Embedding lookups and elementwise work are free.
- **Softmax attention is charged by rule.** It must go through `moe.model.attention`. The rule is
  3 × 4 × heads × head_dim × (key edges per query): causal over 4,096 tokens on long layers, and a 2,048-key window
  on short layers.
- For the baseline, the measurement equals the closed form exactly:
  3 × [attention projections + router (2dE) + k routed experts (6df) + shared expert (6ds) + head (2dV)] + the attention rule.

`python -m moe.compute calibrate --method <name> --track <track>` measures a method. It solves the method's declared
knob (`expert_width` by default; `shared_expert_width`, `layers` or `none` are also allowed) for the anchor. It then
writes `moe/configs/<track>/<name>.json` and the cost record `moe/evidence/costs/<track>/<name>.json`. The cost record
fingerprints `moe/{model,plugins,compute,spec}.py` and the method's own files, so a formal run refuses a stale record.

## Throughput standard

Throughput is reported with every result and measured by one harness, `python -m moe.throughput`:

- It runs the trainer's own update (FSDP2, bf16, grouped-GEMM experts) on the real training rows.
- It runs 10 warm-up updates, then 30 timed updates, and drops the fastest and slowest 10%.
- It measures the method and the baseline back to back on the same node.
- It reports seconds per update, tokens/s, tokens/s per device, MFU and peak memory.
- MFU = FLOPs per token × tokens/s ÷ (devices × 989.4 TFLOP/s dense bf16 peak of the reference class).

The published comparison is the **relative throughput** (method tokens/s ÷ baseline tokens/s on the same node, same
invocation). Absolute numbers are kept for reference.

| Planning reference (8 devices) | d768 | d1024 |
|---|---|---|
| Marin's measured cell (TPU v4-32, EP = 1) | 294,726 tok/s, 4.83 h | 219,720 tok/s, 11.72 h |
| Planning estimate at Marin's 8-device MoE MFU (10.3% / 13.9%) | ≈ 890,600 tok/s, 1.38 h, 11.0 device-hours | ≈ 664,300 tok/s, 3.53 h, 28.2 device-hours |
| Measured baseline (`moe.throughput`) | recorded in `references.json` after the first maintainer run | same |

## Evaluation

| Score | What it is |
|---|---|
| **Validation BPB** | Bits per byte on each of the six reserved Dolma 3 validation sources, macro-averaged. 128 rows per source at each event, all complete rows at the end |
| **CORE-22** | The dense protocol's pinned DCLM CORE bundle and nanochat scorer, unchanged. Centered accuracy, mean over 22 tasks. Prompts are left-truncated to 4,096 tokens (only SQuAD exceeds it) |
| **LC-5** | Five deterministic long-context tasks at **2,048 / 3,072 / 4,096** tokens, 100 examples each (details below). Each task's centered accuracy is averaged over the three lengths, then LC-5 = mean over the five tasks. 8,192 tokens is a length-extrapolation probe that is reported but not scored |
| **MoE-CORE** | The headline score: the mean centered accuracy over all **27 entries**, i.e. (22 × CORE-22 + 5 × LC-5) / 27 |
| **Long-document BPB** | BPB of validation documents of at least 4,097 tokens, at positions 0–2,047 and 2,048–4,095. The gain between them shows how much the model uses long context. Reported, not scored |

LC-5 tasks:

| Task | Prompt | Scoring |
|---|---|---|
| `passkey` | A 5-digit pass key hidden at a random depth in repeated filler | Greedy exact match |
| `niah_single` | One "special magic number" needle for a keyword, in natural text (reserved wiki validation stream) | Greedy exact match |
| `niah_multikey` | Four needles with different keywords; asks for one | Greedy exact match |
| `niah_choice` | As `niah_multikey`, but picks the lowest-loss option among the four planted values | 4-way choice, centered by 25% |
| `var_tracking` | `VAR X = n … VAR Y = VAR X` with two distractor chains; asks for Y's value | Greedy exact match |

Greedy exact match is teacher-forced: every answer token must be the argmax, which is CORE's language-modeling rule.
The choice task uses CORE's multiple-choice rule. Prompts are generated from `(task, length, index)` with a fixed
seed and a pinned haystack, so all methods score identical token sequences.

Quality context (larger budgets; not targets): Attnres_Feedback reports CORE-22 = 0.2450 for a d768 MoE (61M active,
47.9B tokens) and 0.2906 for d896 (102M active, 80.7B tokens). Seed-to-seed CORE noise is about 0.9 points. Every
reference number with its source is in [`references.json`](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/references.json).

## Formal and research modes

`python -m moe.train` runs in research mode by default. Research runs are never published. They may use
`--method/--track` without calibration, `--smoke-steps`, `--no-eval`, partial CORE and any PyTorch build.

`--formal` requires all of the following:

- An admitted configuration, whose cost record matches the current source files and PyTorch version (2.9.1).
- A verified data manifest.
- The full schedule and full evaluations.
- Formal mode in every resumed segment.

The audit (`python -m moe.audit`) checks every event, the ledger (all tokens; charged FLOPs within 2% of the anchor)
and the presence of a throughput measurement before `--export` writes the record.

## Versioning

Any change to a value above creates a new protocol version, and records of different versions are never ranked together.
The data stream may be extended (more shards, larger tracks) by adding tracks. Existing tracks are never changed.
