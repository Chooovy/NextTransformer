# NextTransformer-MoE

A compute-controlled comparison platform for **Mixture-of-Experts** language models, built on the NextTransformer
contract: same data, same order, same FLOPs per token, same evaluation; parameters and speed are outcomes. It has
two fixed tracks. Each reproduces one of Marin's compute-optimal Grug MoE cells: model, MuonH optimizer, schedule,
quantile-balanced routing, context, batch and updates.

| Track | Shape | Context | Tokens | FLOPs / token | Training FLOPs | Total / active params |
|---|---|---|---|---|---|---|
| **d768** | 8 × 768, GQA 6/1, 256 experts × 384 top-4 + shared 768 | 4,096 | 4.42B (16,875 × 262,144) | 9.369 × 10⁸ | 4.14 × 10¹⁸ | 2.00B / 213M |
| **d1024** | 11 × 1024, GQA 8/2, 256 experts × 512 top-4 + shared 1024 | 4,096 | 8.43B (16,080 × 524,288) | 1.695 × 10⁹ | 1.43 × 10¹⁹ | 4.71B / 347M |

The single score is **MoE-CORE**: the mean centered accuracy over 14 published tasks, 10 short and 4 long.
- **Short:** ARC-E/C, HellaSwag, PIQA, WinoGrande, CSQA, OBQA, LAMBADA, SQuAD, TriviaQA.
- **Long:** RULER S-NIAH-1 at 2K and 4K, FDA, SWDE.

Every result also reports validation BPB and its **relative throughput** against the baseline on the same node.

- Normative contract: [PROTOCOL.md](PROTOCOL.md) (and the [docs page](../docs/src/content/docs/moe/protocol.md))
- Machine-readable values: [spec.py](spec.py); reference numbers: [references.json](references.json)
- 中文说明: [README.zh-CN.md](README.zh-CN.md)

## Layout

```text
moe/
  spec.py           frozen protocol + the two tracks (fingerprinted)
  model.py          reference MoE LM (baseline): Marin's Grug MoE, ported line by line
  optim.py          Marin's MuonH / AdamH / Adam optimizer (FSDP2-aware)
  plugins.py        method plugins: moe/methods/<name>/{method.json, __init__.py}
  compute.py        cost ledger: measured matmul FLOPs + attention rule; calibrate/admit; `report`
  tokenizer.py      dolma2 tokenizer via tiktoken (id-for-id with allenai/dolma2-tokenizer)
  data.py           pinned Dolma 3 HQ download, one-time tokenization, fixed row order
  train.py          trainer (FSDP2, bf16, grouped GEMM), research / --formal
  throughput.py     standard throughput harness (method vs baseline, same node)
  evaluate.py       evaluate a checkpoint
  evaluation/       bpb.py (validation), short.py (10 DCLM-bundle tasks), ruler.py (RULER S-NIAH-1),
                    recall.py (FDA, SWDE), generate.py (greedy decoding), score.py (MoE-CORE)
  audit.py          audit a run, export its record, check published records
  configs/<track>/<method>.json          admitted configurations
  evidence/costs/<track>/<method>.json   cost records (the baseline's is the track anchor)
  evidence/audits/<track>/<method>.json  run audits
  results/<track>/<method>.json          published run records
  methods/                               contributed plugins
```

## Quick start

```bash
# 1. Data, tokenizer and evaluation data (downloads ~18 GB of shards and tokenizes the first 8.43B tokens once)
python -m moe.data prepare --root $RESOURCES

# 2. Shapes, parameters and FLOPs of both tracks
python -m moe.compute report

# 3. A short research run of the baseline (never published)
torchrun --nproc_per_node 8 -m moe.train --method baseline --track d768 \
    --resources $RESOURCES --output runs/moe-d768-smoke --micro-batch 8 --smoke-steps 50 --no-eval

# 4. Standard throughput: the baseline (and any method) back to back
torchrun --nproc_per_node 8 -m moe.throughput --track d768 --methods baseline \
    --resources $RESOURCES --micro-batch 8 --output throughput-d768.json
```

## Submitting a method

The flow is the dense track's ([plugins](../docs/src/content/docs/guides/plugins.md),
[submit a result](../docs/src/content/docs/guides/submit-a-result.md)), with `moe.*` entry points:

1. **Propose** it in an issue, then add a plugin `moe/methods/<name>/` with `method.json` and hooks (`router`,
   `ffn`, `mixer`, `block`, `model`, `init`, `check`; see [plugins.py](plugins.py)). Do not edit shared files
   (`moe/model.py`, `moe/train.py`, `moe/data.py`, `moe/compute.py`, `moe/evaluation/`, `moe/spec.py`).
2. **Calibrate** on each track you enter. The method must land within 1% of the anchor:
   ```bash
   python -m moe.compute calibrate --method <name> --track d768 --track d1024
   ```
   This writes `moe/configs/<track>/<name>.json` and `moe/evidence/costs/<track>/<name>.json`.
3. **Select** candidates with training loss or the development shard (00255) only. Validation and the MoE-CORE
   tasks are reserved for the formal run.
4. **Train formally** with the admitted configuration:
   ```bash
   torchrun --nproc_per_node 8 -m moe.train --config moe/configs/d768/<name>.json \
       --resources $RESOURCES --output runs/moe-d768-<name> --micro-batch 8 --formal
   ```
   Resume with `--resume runs/moe-d768-<name>/checkpoint_<step>`; every segment must be formal.
5. **Measure throughput** against the baseline on the same node:
   ```bash
   torchrun --nproc_per_node 8 -m moe.throughput --track d768 --methods baseline,<name> \
       --resources $RESOURCES --micro-batch 8 --output throughput-d768.json
   ```
6. **Audit and export**:
   ```bash
   python -m moe.audit runs/moe-d768-<name> --export --name "<display name>" --throughput throughput-d768.json
   python -m moe.audit --check
   ```
7. **Open one pull request** with the plugin, configs, cost records, tests (`tests/test_moe_<name>.py`), audit and
   record. Before it: `python -m pytest tests/test_moe_track.py -q` and `python -m nexttransformer.checks.changes --base origin/main`.

## Status

- Protocol, ledger, trainer, harness, evaluation and audit are implemented.
- CPU tests: `tests/test_moe_track.py`.
- Validated on one 8-device node (141 GiB, torch 2.9.1+cu128): equivalence of FlexAttention vs SDPA, grouped GEMM vs
  the loop and CUDA vs CPU; a 40-update run with a resume; and the full evaluation (short tasks, RULER, FDA, SWDE).
- Both baseline configurations are calibrated and admitted (`moe/configs/*/baseline.json`).
- Measured baseline throughput, 8 devices: **d768 686k tokens/s (MFU 8.1%, 1.8 h)** and **d1024 536k tokens/s
  (MFU 11.5%, 4.4 h)**, both at micro-batch 8. Details in `references.json`.
- **Pending:** the two formal baseline runs.
- **Scaling:** the 300B-token stream covers far larger tracks. A third size or longer budgets are added as new tracks;
  the two tracks here never change.
