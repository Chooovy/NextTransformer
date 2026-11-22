"""The MoE comparison track: protocol, reference model, cost ledger, plugins, data order, evaluation, trainer."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from moe import compute, plugins, spec
from moe.data import Rows, micro_batches, update_rows
from moe.evaluation import score
from moe.model import Experts, MoEConfig, Router, build_model

ROOT = Path(__file__).resolve().parents[1]
TEST_PLUGINS = ROOT / "tests" / "moe_plugins"


@pytest.fixture
def test_plugins(monkeypatch):
    monkeypatch.setenv("NEXTT_MOE_METHOD_PATHS", str(TEST_PLUGINS))
    plugins.registry.cache_clear()
    plugins.module.cache_clear()
    yield
    plugins.registry.cache_clear()
    plugins.module.cache_clear()


def cl100k_available():
    try:
        import tiktoken
        tiktoken.get_encoding("cl100k_base")
        return True
    except Exception:
        return False


needs_tokenizer = pytest.mark.skipif(not cl100k_available(), reason="cl100k_base ranks are not cached/downloadable")


# --- protocol -------------------------------------------------------------------------------------------------

def test_tracks_follow_the_marin_cells():
    d768, d1024 = spec.TRACKS["d768"], spec.TRACKS["d1024"]
    assert (d768.layers, d768.heads, d768.kv_heads, d768.expert_width, d768.shared_expert_width) == (8, 6, 1, 384, 768)
    assert (d1024.layers, d1024.heads, d1024.kv_heads, d1024.expert_width, d1024.shared_expert_width) == (11, 8, 2, 512, 1024)
    assert d768.total_tokens == 16_875 * 64 * 4096 == 4_423_680_000
    assert d1024.total_tokens == 16_080 * 128 * 4096 == 8_430_551_040
    assert d768.validation_steps[0] == 0 and d768.validation_steps[-1] == 16_875 and len(d768.validation_steps) == 35
    assert d768.core_steps[-1] == d768.updates and d1024.core_steps[-1] == d1024.updates
    assert spec.long_layers(8) == (3, 7) and spec.long_layers(11) == (3, 7, 10)


def test_schedule_warms_up_and_decays_to_zero():
    t = spec.TRACKS["d768"]
    assert t.warmup_steps == 169
    assert spec.learning_rate(0, t) == pytest.approx(t.peak_lr / 169)
    assert spec.learning_rate(168, t) == pytest.approx(t.peak_lr)
    assert spec.learning_rate(t.updates, t) == pytest.approx(0.0)
    assert all(spec.learning_rate(s, t) >= spec.learning_rate(s + 1, t) for s in range(169, t.updates, 97))


def test_config_roundtrip_and_tamper_detection(tmp_path):
    config = MoEConfig.baseline("d768")
    path = spec.write_config(tmp_path / "c.json", config, {"status": "candidate"})
    loaded, calibration = spec.load_config(path)
    assert loaded == config and calibration == {"status": "candidate"}
    data = json.loads(path.read_text())
    data["protocol"]["sequence_len"] = 2048
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="protocol"):
        spec.load_config(path)
    data = json.loads(spec.write_config(tmp_path / "d.json", config, {}).read_text())
    data["track"]["updates"] = 1
    (tmp_path / "d.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="track"):
        spec.load_config(tmp_path / "d.json")


@pytest.mark.parametrize("track", ["d768", "d1024"])
def test_admitted_baseline_configs_match_the_protocol(track):
    path = spec.config_path("baseline", track)
    config, calibration = spec.load_config(path)
    assert config == MoEConfig.baseline(track)
    assert calibration["status"] == "admitted"
    record = json.loads((ROOT / calibration["cost_record"]).read_text())
    assert record["flops_per_token"] == pytest.approx(compute.analytic(config)["total"], rel=1e-9)
    assert record["source"]["sha256"] == compute.source_digest("baseline")["sha256"], \
        "the shared MoE core changed: rerun python -m moe.compute calibrate --method baseline --track " + track


# --- model and ledger -----------------------------------------------------------------------------------------

def debug_model(method="baseline", **overrides):
    config = plugins.config_for(method, "d768", debug=True)
    if overrides:
        from dataclasses import replace
        config = replace(config, **overrides)
    torch.manual_seed(0)
    return build_model(config, "cpu")


def test_debug_model_trains_one_step():
    model = debug_model()
    idx = torch.randint(0, 512, (2, 64))
    loss = model(idx, idx)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters())
    logits = model(idx)
    assert logits.shape == (2, 64, 512)


def test_router_gates_and_balancing():
    config = MoEConfig.baseline("d768").debug()
    router = Router(config, 0)
    torch.nn.init.normal_(router.weight)
    x = torch.randn(100, config.width)
    index, gates, aux = router.train()(x)
    assert index.shape == gates.shape == (100, config.top_k) and aux is None
    torch.testing.assert_close(gates.sum(-1), torch.full((100,), config.routing_renorm_sum))
    load = router.load.clone()
    router.balance(load)
    assert torch.all((router.bias > 0) == (load < load.mean())) or torch.all(router.bias[load == load.mean()] == 0)


def test_dropless_experts_match_a_dense_reference():
    config = MoEConfig.baseline("d768").debug()
    torch.manual_seed(1)
    experts = Experts(config, 0)
    for p in experts.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x = torch.randn(10, config.width)
    index = torch.stack([torch.randperm(config.routed_experts)[:config.top_k] for _ in range(10)])
    gates = torch.rand(10, config.top_k)
    reference = torch.zeros_like(x)
    for n in range(10):
        for j in range(config.top_k):
            e = index[n, j]
            g, u = (experts.w_in[e] @ x[n]).chunk(2)
            reference[n] += gates[n, j] * (experts.w_out[e] @ (torch.nn.functional.silu(g) * u))
    torch.testing.assert_close(experts(x, index, gates), reference)
    # The grouped-GEMM layout (padded groups, transposed weights) computes the same function.
    flat = index.flatten()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=config.routed_experts)

    def grouped_mm(a, b, offs):   # a reference implementation of torch._grouped_mm's contract
        out, start = [], 0
        for e, end in enumerate(offs.tolist()):
            out.append(a[start:end] @ b[e])
            start = end
        return torch.cat(out)

    original = getattr(torch, "_grouped_mm", None)
    torch._grouped_mm = grouped_mm
    try:
        ys = experts._grouped(x[order // config.top_k], counts, experts.w_in, experts.w_out)
    finally:
        if original is None:
            del torch._grouped_mm
        else:
            torch._grouped_mm = original
    loop = Experts.forward(experts, x, index, torch.ones_like(gates))
    combined = torch.zeros_like(x).index_add_(0, order // config.top_k, ys)
    torch.testing.assert_close(combined, loop)


def test_parameter_counts_agree_with_the_shapes():
    counts = build_model(MoEConfig.baseline("d768"), "meta").parameter_counts()
    d, L, E, f, s, V = 768, 8, 256, 384, 768, 100352
    attention = L * (d * 6 * 128 + 2 * d * 128 + 6 * 128 * d)
    experts = L * E * 3 * d * f
    other = L * (3 * d * s + E * d + 2 * d) + d + 2 * V * d
    assert counts["total"] == attention + experts + other
    assert counts["active"] == counts["total"] - experts + experts * 4 // E
    assert 55e6 < counts["active_non_embedding"] < 60e6   # Marin reports 58.6M incl. GatedNorm/gates


def test_attention_edges():
    assert compute.attention_edges(4, -1) == 10
    assert compute.attention_edges(4096, -1) == 4096 * 4097 // 2
    assert compute.attention_edges(4096, 2047) == 2048 * 2049 // 2 + 2048 * 2048


def test_measured_cost_equals_the_closed_form():
    config = MoEConfig.baseline("d768").debug()
    assert compute.measure(config, tokens=128)["flops_per_token"] == compute.analytic(config)["total"]


@pytest.mark.parametrize("track, flops, marin", [("d768", 915_429_888, 2.81e18), ("d1024", 1_656_268_800, 1.16e19)])
def test_track_budgets(track, flops, marin):
    row = compute.report()[track]
    assert row["flops_per_token"] == flops
    # Marin's own convention (6N_active + non-causal attention, no head) reproduces its reported budget.
    assert row["marin_convention_budget"] == pytest.approx(marin, rel=0.01)


# --- plugins --------------------------------------------------------------------------------------------------

def test_plugins_are_discovered_and_built(test_plugins):
    assert {"softmax_aux", "top1_wide"} <= set(plugins.methods())
    model = debug_model("softmax_aux")
    assert model.config.option("aux_coeff") == 0.1
    idx = torch.randint(0, 512, (2, 64))
    loss = model(idx, idx)
    assert model.last_aux is not None and model.last_aux > 0
    (loss + model.last_aux).backward()


def test_ffn_plugin_changes_measured_cost(test_plugins):
    base = MoEConfig.baseline("d768").debug()
    wide = plugins.config_for("top1_wide", "d768", debug=True)
    measured = compute.measure(wide, tokens=128)["flops_per_token"]
    assert measured < compute.measure(base, tokens=128)["flops_per_token"]


def test_plugin_declarations_are_validated(tmp_path, monkeypatch):
    bad = tmp_path / "Bad"
    bad.mkdir()
    (bad / "method.json").write_text(json.dumps({"name": "Bad"}))
    (bad / "__init__.py").write_text("")
    monkeypatch.setenv("NEXTT_MOE_METHOD_PATHS", str(tmp_path))
    plugins.registry.cache_clear()
    try:
        with pytest.raises(ValueError, match="lowercase"):
            plugins.registry()
    finally:
        plugins.registry.cache_clear()


# --- data order -----------------------------------------------------------------------------------------------

def test_rows_overlap_by_one_and_updates_ignore_the_layout(tmp_path):
    tokens = np.arange(10 * 16 + 1, dtype="<u4")
    tokens.tofile(tmp_path / "t.bin")
    rows = Rows(tmp_path / "t.bin", sequence_len=16)
    assert len(rows) == 10
    block = rows.get(2, 4)
    assert block[0, 0] == 32 and block[0, -1] == block[1, 0] == 48
    assert update_rows(1, 0, 1, 4) == (4, 8) and update_rows(1, 1, 2, 4) == (6, 8)
    whole = torch.cat([x for x, _ in micro_batches(rows, 1, 0, 1, 4, 4, torch.device("cpu"))])
    split = torch.cat([x for r in range(2) for x, _ in micro_batches(rows, 1, r, 2, 4, 1, torch.device("cpu"))])
    assert torch.equal(whole, split)


# --- evaluation -----------------------------------------------------------------------------------------------

@needs_tokenizer
def test_dolma2_tokenizer_matches_known_ids():
    from moe.tokenizer import EOT, Dolma2Tokenizer
    tok = Dolma2Tokenizer()
    assert tok.vocab_size == 100_278 and tok.get_bos_token_id() == EOT == 100_257
    assert tok.encode("call |||PHONE_NUMBER||| now") == [6797, 220, 100261, 1457]
    assert tok.encode(["a", "b"], prepend=EOT) == [[EOT, 64], [EOT, 65]]
    path = os.environ.get("DOLMA2_TOKENIZER_JSON")
    if path:
        from tokenizers import Tokenizer
        hf = Tokenizer.from_file(path)
        text = "def f(x):\n\treturn x**2  # 1234567 |||EMAIL_ADDRESS||| <|endoftext|>"
        assert hf.encode(text, add_special_tokens=False).ids == tok.encode(text)


@needs_tokenizer
@pytest.mark.parametrize("length", [2048, 3072, 4096, 8192])
def test_long_context_examples_have_exact_lengths(length):
    from moe.evaluation import longctx
    from moe.tokenizer import Dolma2Tokenizer
    tok = Dolma2Tokenizer()
    haystack = longctx.Haystack(np.array(tok.encode("Natural text about many things. " * 4000) + [100257] * 3),
                                tok)
    for task in longctx.TASKS:
        a = longctx.build(task, length, 3, tok, haystack)
        b = longctx.build(task, length, 3, tok, haystack)
        assert a == b
        longest = max(len(a["prefix"]) + len(x) for x in a["answers"])
        assert longest == length
        assert len(a["answers"]) == (4 if task == "niah_choice" else 1)


@needs_tokenizer
def test_long_context_scoring_runs_on_a_debug_model():
    from moe.evaluation import longctx
    model = debug_model(vocab_size=100_352).eval()   # full vocabulary, tiny everything else
    tokens = np.random.default_rng(0).integers(0, 100_000, 50_000)
    result = longctx.evaluate(model, tokens, torch.device("cpu"), lengths=(128,), examples=2,
                              tasks=("passkey", "niah_choice"))
    assert set(result["accuracy"]) == {"passkey", "niah_choice"}
    assert all(0 <= v["128"] <= 1 for v in result["accuracy"].values())


def test_moe_core_weights_every_entry_equally():
    core = dict(centered_results={f"t{i}": 0.2 for i in range(22)}, core_metric=0.2)
    lc = dict(centered={f"l{i}": 0.5 for i in range(5)}, lc_metric=0.5)
    assert score.moe_core(core, lc)["moe_core"] == pytest.approx((22 * 0.2 + 5 * 0.5) / 27)
    with pytest.raises(ValueError):
        score.moe_core(dict(centered_results={}, core_metric=0), lc)


# --- trainer and audit ----------------------------------------------------------------------------------------

def fake_resources(root, rows=64, S=64):
    out = root / "moe"
    (out / "val").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rng.integers(0, 512, rows * S + 1, dtype=np.uint32).astype("<u4").tofile(out / "train.bin")
    for source in spec.VALIDATION_SOURCES:
        rng.integers(0, 512, 8 * S + 1, dtype=np.uint32).astype("<u4").tofile(out / "val" / f"{source}.bin")
    (out / "manifest.json").write_text(json.dumps(dict(data_revision=spec.DATA_REVISION,
                                                       tokenizer_revision=spec.TOKENIZER_REVISION,
                                                       train_tokens=rows * S + 1, outputs={})))
    return root


def test_trainer_smoke_and_checkpoint_roundtrip(tmp_path):
    from moe import train
    resources = fake_resources(tmp_path / "res")
    out = tmp_path / "run"
    train.main(["--method", "baseline", "--track", "d768", "--debug", "--resources", str(resources),
                "--output", str(out), "--micro-batch", "2", "--smoke-steps", "3", "--no-eval"])
    events = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    kinds = [(e["event"], e["step"]) for e in events]
    assert ("val", 0) in kinds and ("val", 3) in kinds and ("end", 3) in kinds
    assert json.loads((out / "ledger.json").read_text())["updates"] == 3
    assert not json.loads((out / "run.json").read_text())["formal"]
    model = debug_model()
    opt = train.optimizer(model, spec.TRACKS["d768"])
    model(torch.randint(0, 512, (1, 64)), torch.randint(0, 512, (1, 64))).backward()
    opt.step()
    train.save(model, opt, 7, tmp_path / "ck")
    fresh = debug_model()
    fresh_opt = train.optimizer(fresh, spec.TRACKS["d768"])
    assert train.load(fresh, fresh_opt, tmp_path / "ck" / "checkpoint_00007") == 7
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values()):
        torch.testing.assert_close(a, b)


def test_formal_mode_refuses_research_shortcuts(tmp_path):
    from moe import train
    with pytest.raises(SystemExit):
        train.main(["--method", "baseline", "--track", "d768", "--debug", "--formal",
                    "--resources", str(tmp_path), "--output", str(tmp_path / "o")])


def test_audit_rejects_an_incomplete_run(tmp_path):
    from moe import audit
    (tmp_path / "run.json").write_text(json.dumps(dict(track="d768", formal=False, method="baseline")))
    (tmp_path / "metrics.jsonl").write_text(json.dumps(dict(event="val", step=0, val={"macro_bpb": 2.0})) + "\n")
    problems, summary = audit.audit(tmp_path)
    assert not summary["passed"]
    assert any("research mode" in p for p in problems) and any("step 500" in p for p in problems)
    assert audit.check() == []
