---
title: MoE track
description: A compute-controlled comparison platform for Mixture-of-Experts language models, at two Marin-derived sizes.
---

The MoE track applies the NextTransformer contract to sparse models. Every method uses the same data, in the same
order, at the same training FLOPs per token, with the same evaluation. Parameters and speed are outcomes. There are
two fixed tracks. Each reproduces the shape, context, batch and number of updates of one of Marin's compute-optimal
Grug MoE cells.

| Track | Shape | Context | Tokens | FLOPs / token | Total / active params |
|---|---|---|---|---|---|
| **d768** | 8 × 768, GQA 6/1, 256 experts × 384 top-4 + shared 768 | 4,096 | 4.42B | 9.154 × 10⁸ | 1.99B / 209M |
| **d1024** | 11 × 1024, GQA 8/2, 256 experts × 512 top-4 + shared 1024 | 4,096 | 8.43B | 1.656 × 10⁹ | 4.70B / 341M |

- **Data.** Dolma 3 HQ (300B tokens, shuffled and cleaned), tokenized with the dolma2 tokenizer. Shards are read in a fixed order.
- **Score.** **MoE-CORE** is the mean centered accuracy over 27 entries. 22 are the CORE-22 tasks of the dense
  leaderboard. 5 are the long-context tasks of **LC-5**, run at 2,048 / 3,072 / 4,096 tokens: pass key, single-needle,
  multi-key, multi-key choice and variable tracking.
- **Throughput.** One harness, `moe.throughput`, measures every method back to back with the baseline on the same
  node. The relative throughput is published with the result.
- **Compute.** Matmul FLOPs are measured block by block and attention edges are charged by rule. A method is admitted
  within 1% of the track anchor and audited within 2% cumulatively.

The [protocol](../protocol/) gives every value. The
[MoE README](https://github.com/Interactive-Active-AI/NextTransformer/blob/main/moe/README.md#submitting-a-method)
gives the submission flow, which is the dense track's flow with `moe.*` entry points: plugin, calibrate, formal
training, throughput, audit and export, pull request.
