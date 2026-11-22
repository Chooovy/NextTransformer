# MoE method plugins

Each contributed MoE method is one directory here, `moe/methods/<name>/`, with a `method.json` declaration
and a Python package that implements its hooks (`router`, `ffn`, `mixer`, `block`, `model`, `init`, `check`).
The registry discovers it; a plugin never edits `moe/model.py`, the trainer, the data stream, the ledger or the
evaluation, so it cannot change another method.

The hook contract is in [`moe/plugins.py`](../plugins.py) and the end-to-end submission flow (calibrate,
train `--formal`, evaluate, audit, export, pull request) is in [`moe/README.md`](../README.md#submitting-a-method).
Test plugins in [`tests/moe_plugins/`](../../tests/moe_plugins) show a router and a whole-FFN replacement.
