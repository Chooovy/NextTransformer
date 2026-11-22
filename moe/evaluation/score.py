"""MoE-CORE: one centered score over the 22 CORE tasks and the 5 long-context tasks.

    MoE-CORE = mean over 27 entries of the centered accuracy
             = (22 * CORE-22 + 5 * LC-5) / 27

Each CORE entry is nanochat's centered accuracy; each LC-5 entry is the task's centered accuracy averaged over
the scored lengths 2,048 / 3,072 / 4,096. Every entry weighs the same, as in CORE. CORE-22 and LC-5 are always
published next to MoE-CORE, so the dense leaderboard's CORE stays directly comparable.
"""
CORE_TASKS = 22
LC_TASKS = 5


def moe_core(core, lc):
    if len(core["centered_results"]) != CORE_TASKS or len(lc["centered"]) != LC_TASKS:
        raise ValueError("MoE-CORE needs all 22 CORE tasks and all 5 long-context tasks")
    entries = [*core["centered_results"].values(), *lc["centered"].values()]
    return dict(moe_core=sum(entries) / len(entries), core22=core["core_metric"], lc5=lc["lc_metric"],
                entries=len(entries))
