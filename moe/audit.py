"""Audit a finished formal MoE run, export its record, and check published records.

    python -m moe.audit runs/moe-d768-<method>                         # writes moe/evidence/audits/<track>/<method>.json
    python -m moe.audit runs/moe-d768-<method> --export --name "<display name>" \\
        --throughput throughput-d768.json                              # writes moe/results/<track>/<method>.json
    python -m moe.audit --check                                         # validates every published record

The audit re-reads what the trainer wrote and checks it against the protocol: formal mode in every segment,
unchanged protocol/track/model fingerprints, every validation event, every CORE-22 + LC-5 event with the full
task sets, the final validation and long-document BPB, the ledger (all updates, all tokens, charged FLOPs within
2% of the track anchor) and a throughput measurement. It never records host names, paths or devices.
"""
import argparse
import json
from pathlib import Path
import sys

from moe import compute
from moe.evaluation.score import CORE_TASKS, LC_TASKS
from moe.spec import PROTOCOL, TRACKS

HERE = Path(__file__).resolve().parent
AUDITS = HERE / "evidence" / "audits"
RESULTS = HERE / "results"


def events(run_dir):
    """Last event of each (kind, step); a resumed run may repeat one."""
    out = {}
    for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines():
        e = json.loads(line)
        out[(e["event"], e["step"])] = e
    return out


def segments(run_dir):
    """run.json of every segment (the trainer keeps the latest; resumed segments archive earlier ones)."""
    paths = sorted(Path(run_dir).glob("run*.json"))
    return [json.loads(p.read_text()) for p in paths]


def audit(run_dir):
    run_dir = Path(run_dir)
    problems = []
    segs = segments(run_dir)
    if not segs:
        return ["run.json is missing"], {}
    run = segs[-1]
    track = TRACKS.get(run.get("track"))
    if track is None:
        return [f"unknown track {run.get('track')}"], {}
    for s in segs:
        if not s.get("formal"):
            problems.append("a training segment ran in research mode")
        if s.get("protocol_hash") != PROTOCOL.fingerprint:
            problems.append("protocol fingerprint differs from the current protocol")
        if s.get("track_hash") != track.fingerprint:
            problems.append("track fingerprint differs")
        if s.get("model_hash") != run.get("model_hash") or s.get("data_manifest_sha256") != run.get("data_manifest_sha256"):
            problems.append("segments disagree on model or data")
    ev = events(run_dir)
    for step in track.validation_steps:
        if ("val", step) not in ev and ("eval", step) not in ev:
            problems.append(f"missing validation event at step {step}")
    for step in track.core_steps:
        e = ev.get(("eval", step))
        if e is None:
            problems.append(f"missing CORE-22/LC-5 event at step {step}")
            continue
        core, lc = e.get("core", {}), e.get("lc", {})
        if not core.get("full_core") or len(core.get("centered_results", {})) != CORE_TASKS:
            problems.append(f"step {step}: CORE-22 is not complete")
        if len(lc.get("centered", {})) != LC_TASKS or lc.get("examples") != 100:
            problems.append(f"step {step}: LC-5 is not complete")
        if "score" not in e:
            problems.append(f"step {step}: MoE-CORE was not computed")
    final = ev.get(("final", track.updates))
    if final is None or "long_document_bpb" not in final:
        problems.append("missing final evaluation (full validation BPB, long-document BPB)")
    ledger = json.loads((run_dir / "ledger.json").read_text()) if (run_dir / "ledger.json").exists() else {}
    if ledger.get("updates") != track.updates or ledger.get("input_tokens") != track.total_tokens:
        problems.append("ledger does not cover the full token budget")
    try:
        anchor = compute.anchor(track.name) * track.total_tokens
        gap = ledger.get("charged_flops", 0) / anchor - 1
        if abs(gap) > PROTOCOL.cumulative_tolerance:
            problems.append(f"cumulative compute {gap:+.2%} is outside the 2% window")
    except FileNotFoundError:
        gap = None
        problems.append("no baseline anchor for the track")
    throughput = [e for (kind, _), e in ev.items() if kind == "throughput"]
    if not throughput:
        problems.append("no throughput event")
    summary = dict(method=run.get("method"), track=track.name, model_hash=run.get("model_hash"),
                   protocol_hash=PROTOCOL.fingerprint, track_hash=track.fingerprint,
                   data_manifest_sha256=run.get("data_manifest_sha256"), segments=len(segs),
                   world_sizes=sorted({s.get("world_size") for s in segs}), cumulative_gap=gap,
                   updates=ledger.get("updates"), input_tokens=ledger.get("input_tokens"),
                   problems=problems, passed=not problems)
    return problems, summary


def export(run_dir, name, throughput_file=None, note=None):
    problems, summary = audit(run_dir)
    if problems:
        raise ValueError("the audit failed:\n" + "\n".join(problems))
    run = segments(run_dir)[-1]
    track = TRACKS[run["track"]]
    ev = events(run_dir)
    val = [[step, (ev.get(("eval", step)) or ev[("val", step)])["val"]["macro_bpb"]] for step in track.validation_steps]
    evals = []
    for step in track.core_steps:
        e = ev[("eval", step)]
        evals.append(dict(step=step, moe_core=e["score"]["moe_core"], core22=e["core"]["core_metric"],
                          lc5=e["lc"]["lc_metric"], core_centered=e["core"]["centered_results"],
                          lc_centered=e["lc"]["centered"], lc_accuracy=e["lc"]["accuracy"]))
    final = ev[("final", track.updates)]
    own = [e for (kind, _), e in ev.items() if kind == "throughput"][-1]
    record = dict(protocol=PROTOCOL.version, protocol_hash=PROTOCOL.fingerprint, track=track.name,
                  method=run["method"], name=name, model=run["model"], model_hash=run["model_hash"],
                  parameters=run["parameters"], flops_per_token=run["flops_per_token"],
                  training_flops=run["flops_per_token"] * track.total_tokens, tokens=track.total_tokens,
                  validation=val, evaluations=evals, final=dict(
                      val_macro_bpb=final["val"]["macro_bpb"], val_sources=final["val"]["sources"],
                      moe_core=final.get("score", {}).get("moe_core", evals[-1]["moe_core"]),
                      long_document_bpb=final["long_document_bpb"]),
                  throughput=dict(training=dict(tokens_per_second=own["tokens_per_second"],
                                                tokens_per_second_per_device=own["tokens_per_second_per_device"],
                                                mfu=own.get("mfu"))),
                  audit=f"moe/evidence/audits/{track.name}/{run['method']}.json", note=note)
    if throughput_file:
        harness = json.loads(Path(throughput_file).read_text())
        mine = next(r for r in harness["results"] if r["method"] == run["method"])
        record["throughput"]["harness"] = {k: mine[k] for k in (
            "tokens_per_second", "tokens_per_second_per_device", "mfu", "relative_throughput", "peak_memory_gib",
            "world_size", "micro_batch")}
    path = RESULTS / track.name / f"{run['method']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def check():
    problems = []
    for path in sorted(RESULTS.glob("*/*.json")):
        r = json.loads(path.read_text())
        where = str(path.relative_to(HERE.parent))
        if r.get("protocol_hash") != PROTOCOL.fingerprint:
            problems.append(f"{where}: protocol fingerprint differs")
        if r.get("track") != path.parent.name or r.get("method") != path.stem:
            problems.append(f"{where}: file name disagrees with track/method")
        if [e["step"] for e in r.get("evaluations", [])] != list(TRACKS[r["track"]].core_steps):
            problems.append(f"{where}: evaluation schedule differs")
        audit_path = HERE.parent / r.get("audit", "")
        if not audit_path.is_file() or not json.loads(audit_path.read_text()).get("passed"):
            problems.append(f"{where}: no passing audit")
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", type=Path)
    p.add_argument("--export", action="store_true")
    p.add_argument("--name")
    p.add_argument("--throughput", type=Path, help="moe.throughput output measured on the same node")
    p.add_argument("--note")
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    if args.check:
        problems = check()
        print("\n".join(problems) if problems else f"{len(list(RESULTS.glob('*/*.json')))} MoE records are valid")
        raise SystemExit(1 if problems else 0)
    if args.run is None:
        p.error("pass a run directory or --check")
    problems, summary = audit(args.run)
    if summary:
        path = AUDITS / summary["track"] / f"{summary['method']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"audit written to {path.relative_to(HERE.parent)}")
    print("\n".join(problems) if problems else "audit passed")
    if problems:
        raise SystemExit(1)
    if args.export:
        if not args.name:
            p.error("--export needs --name")
        print(f"record written to {export(args.run, args.name, args.throughput, args.note).relative_to(HERE.parent)}")


if __name__ == "__main__":
    sys.exit(main())
