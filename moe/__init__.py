"""NextTransformer-MoE: a compute-controlled comparison track for Mixture-of-Experts language models.

Two fixed tracks reproduce the shapes and budgets of Marin's compute-optimal Grug MoE cells (d768 and
d1024). Every method trains on the same tokens in the same order, at the same theoretical training FLOPs per
token as the track's reference baseline, and is scored by the same throughput harness and MoE-CORE suite.
The normative text is moe/PROTOCOL.md; the machine-readable values live in moe/spec.py.
"""
