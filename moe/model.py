"""Reference MoE language model of the two tracks, in plain PyTorch.

The baseline (`method="baseline"`) is the shape of Marin's compute-optimal Grug MoE cell:

    pre-norm blocks of [RMSNorm -> GQA attention -> RMSNorm -> MoE FFN], untied embedding and head
    attention   GQA, head_dim 128, QK RMS norm, RoPE (theta 10,000); a 2,048-token sliding window except on
                long layers (every fourth layer and the last), which attend to the full 4,096-token context
    MoE FFN     256 routed SwiGLU experts (top 4) plus one always-on SwiGLU shared expert of width d;
                every layer is an MoE layer; dropless dispatch
    router      fp32 linear scores, sigmoid gates; experts are selected on gate logits plus a non-trained
                balancing bias, and the selected gates are renormalized to sum to 2.5 (Marin's
                routing_renorm_sum). The bias moves by `balance_rate * sign(mean load - expert load)` after
                every update (auxiliary-loss-free balancing). No auxiliary loss and no z-loss.

Deviations from Marin's run-time code, all recorded in moe/PROTOCOL.md: no GatedNorm, no XSA, no attention
output gate, full rather than half RoPE, and a sign-update balancing bias instead of quantile balancing. A
method that wants any of them adds it as a plugin and pays for it under the compute control.

Parameter names are stable: checkpoints store tensors by name.
"""
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math

import torch
from torch import nn
import torch.nn.functional as F

from moe.spec import PROTOCOL, long_layers, track as get_track

GROUPED_MM = hasattr(torch, "_grouped_mm")


@dataclass(frozen=True)
class MoEConfig:
    track: str
    method: str = "baseline"
    width: int = 768
    layers: int = 8
    heads: int = 6
    kv_heads: int = 1
    head_dim: int = 128
    routed_experts: int = 256
    top_k: int = 4
    expert_width: int = 384
    shared_expert_width: int = 768
    vocab_size: int = PROTOCOL.padded_vocab_size
    sequence_len: int = PROTOCOL.sequence_len
    sliding_window: int = PROTOCOL.sliding_window
    long_layer_every: int = PROTOCOL.long_layer_every
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    routing_renorm_sum: float = 2.5
    balance_rate: float = 1e-3
    init_scale: float = 0.5
    options: tuple = field(default=())   # sorted (name, value) pairs owned by a plugin
    scale: str = "formal"                # "formal" or "debug" (tiny CPU shape used by tests)

    def __post_init__(self):
        if self.width % 64 or self.heads % self.kv_heads or min(self.layers, self.heads, self.kv_heads) < 1:
            raise ValueError("width must be a multiple of 64 and heads a multiple of kv_heads")
        if not 1 <= self.top_k <= self.routed_experts or self.expert_width <= 0 or self.shared_expert_width < 0:
            raise ValueError("invalid expert configuration")
        if self.scale not in ("formal", "debug"):
            raise ValueError("scale is formal or debug")
        object.__setattr__(self, "options", tuple(sorted((str(k), v) for k, v in dict(self.options).items())))

    @classmethod
    def baseline(cls, track_name, method="baseline", **overrides):
        t = get_track(track_name)
        shape = dict(width=t.width, layers=t.layers, heads=t.heads, kv_heads=t.kv_heads, head_dim=t.head_dim,
                     routed_experts=t.routed_experts, top_k=t.top_k, expert_width=t.expert_width,
                     shared_expert_width=t.shared_expert_width)
        return cls(track=track_name, method=method, **{**shape, **overrides})

    def debug(self):
        """A tiny CPU shape with the same structure: 4 layers so that one is long and one is short."""
        return replace(self, width=64, layers=4, heads=4, kv_heads=2, head_dim=16, routed_experts=8, top_k=2,
                       expert_width=32, shared_expert_width=64 if self.shared_expert_width else 0, vocab_size=512,
                       sequence_len=64, sliding_window=16, scale="debug")

    def option(self, name, default=None):
        return dict(self.options).get(name, default)

    @property
    def long_layers(self):
        return long_layers(self.layers, self.long_layer_every)

    def window(self, layer):
        """Left context in keys before the current one; -1 means full causal attention."""
        return -1 if layer in self.long_layers else self.sliding_window - 1

    def serialized(self):
        data = asdict(self)
        data["options"] = {k: v for k, v in self.options}
        return data

    @classmethod
    def from_serialized(cls, data):
        data = dict(data)
        data["options"] = tuple(sorted(dict(data.get("options", {})).items()))
        return cls(**data)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.serialized(), sort_keys=True).encode()).hexdigest()


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight.to(x.dtype), self.eps)


def rotary(config, length, device):
    half = config.head_dim // 2
    freq = config.rope_theta ** (-torch.arange(half, dtype=torch.float32, device=device) / half)
    angle = torch.outer(torch.arange(length, dtype=torch.float32, device=device), freq)
    return angle.cos()[None, :, None, :], angle.sin()[None, :, None, :]


def apply_rotary(x, cos_sin):
    cos, sin = (t[:, :x.size(1)].to(x.dtype) for t in cos_sin)
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)


def attention(q, k, v, window):
    """Causal softmax attention, (B, T, H, D) layout, GQA by head groups.

    Every softmax attention of the track goes through this function: the cost ledger charges its causal or
    windowed edges by rule (moe.compute.attention_edges) instead of measuring the kernel.
    """
    from nanochat.flash_attention import flash_attn_func
    return flash_attn_func(q, k, v, causal=True, window_size=(window, 0))


class Attention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config, self.layer = config, layer
        d, hd = config.width, config.head_dim
        self.q = nn.Linear(d, config.heads * hd, bias=False)
        self.k = nn.Linear(d, config.kv_heads * hd, bias=False)
        self.v = nn.Linear(d, config.kv_heads * hd, bias=False)
        self.o = nn.Linear(config.heads * hd, d, bias=False)

    def forward(self, x, cos_sin):
        B, T, _ = x.shape
        c = self.config
        q = self.q(x).view(B, T, c.heads, c.head_dim)
        k = self.k(x).view(B, T, c.kv_heads, c.head_dim)
        v = self.v(x).view(B, T, c.kv_heads, c.head_dim)
        q = apply_rotary(F.rms_norm(q, (c.head_dim,), eps=c.norm_eps), cos_sin)
        k = apply_rotary(F.rms_norm(k, (c.head_dim,), eps=c.norm_eps), cos_sin)
        y = attention(q, k, v, c.window(self.layer))
        return self.o(y.reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Router(nn.Module):
    """Sigmoid gates, top-k on gate logits plus a balancing bias, renormalized selected gates."""

    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.routed_experts, config.width))
        self.register_buffer("bias", torch.zeros(config.routed_experts), persistent=True)
        self.register_buffer("load", torch.zeros(config.routed_experts), persistent=False)

    def forward(self, x):
        logits = F.linear(x.float(), self.weight.float())
        _, index = torch.topk(logits + self.bias, self.config.top_k, dim=-1, sorted=False)
        gates = torch.sigmoid(logits).gather(-1, index)
        gates = gates * (self.config.routing_renorm_sum / gates.sum(-1, keepdim=True).clamp_min(1e-9))
        if self.training:
            self.load += torch.bincount(index.flatten(), minlength=self.config.routed_experts).float()
        return index, gates, None

    @torch.no_grad()
    def balance(self, load):
        """Move the bias toward equal load; `load` is the global token count per expert since the last update."""
        self.bias += self.config.balance_rate * torch.sign(load.mean() - load)
        self.last_load = load.clone()


class Experts(nn.Module):
    """Routed SwiGLU experts with dropless dispatch: grouped GEMM on CUDA, a per-expert loop elsewhere.

    Weights are stored output-major, (E, out, in), like nn.Linear; the grouped GEMM takes their transposed views
    and per-expert row groups padded to multiples of 16 (the layout torchtitan uses with torch._grouped_mm).
    """
    ALIGN = 16

    def __init__(self, config, layer):
        super().__init__()
        E, d, f = config.routed_experts, config.width, config.expert_width
        self.w_in = nn.Parameter(torch.empty(E, 2 * f, d))   # [gate | up]
        self.w_out = nn.Parameter(torch.empty(E, d, f))

    def forward(self, x, index, gates):
        N, d = x.shape
        k = index.size(1)
        flat = index.flatten()
        order = torch.argsort(flat, stable=True)
        token = order // k
        counts = torch.bincount(flat, minlength=self.w_in.size(0))
        xs = x[token]
        w_in, w_out = self.w_in.to(x.dtype), self.w_out.to(x.dtype)
        if GROUPED_MM and x.is_cuda and x.dtype == torch.bfloat16:
            ys = self._grouped(xs, counts, w_in, w_out)
        else:
            # unbind once: indexing w_in[e] per expert would build a full-size gradient per expert
            parts = []
            for rows, wi, wo in zip(torch.split(xs, counts.tolist()), w_in.unbind(0), w_out.unbind(0)):
                if rows.size(0):
                    g, u = F.linear(rows, wi).chunk(2, dim=-1)
                    parts.append(F.linear(F.silu(g) * u, wo))
            ys = torch.cat(parts) if parts else xs.new_zeros(0, d)
        ys = ys * gates.flatten()[order].to(ys.dtype)[:, None]
        return x.new_zeros(N, d).index_add_(0, token, ys)

    def _grouped(self, xs, counts, w_in, w_out):
        a = self.ALIGN
        padded = (counts + a - 1) // a * a
        ends = torch.cumsum(padded, 0)
        starts = ends - padded
        within = torch.arange(xs.size(0), device=xs.device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        slot = torch.repeat_interleave(starts, counts) + within
        buffer = xs.new_zeros(int(ends[-1]) if ends.numel() else 0, xs.size(1)).index_copy(0, slot, xs)
        offsets = ends.to(torch.int32)
        h = torch._grouped_mm(buffer, w_in.transpose(-2, -1), offs=offsets)
        g, u = h.chunk(2, dim=-1)
        return torch._grouped_mm(F.silu(g) * u, w_out.transpose(-2, -1), offs=offsets)[slot]


class MoE(nn.Module):
    def __init__(self, config, layer, router=None, experts=None):
        super().__init__()
        self.router = router or Router(config, layer)
        self.experts = experts or Experts(config, layer)
        self.shared = SwiGLU(config.width, config.shared_expert_width) if config.shared_expert_width else None
        self.last_aux = None

    def forward(self, x):
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        index, gates, aux = self.router(flat)
        self.last_aux = aux
        y = self.experts(flat, index, gates)
        if self.shared is not None:
            y = y + self.shared(flat)
        return y.view(B, T, d)


class Block(nn.Module):
    def __init__(self, config, layer, hooks):
        super().__init__()
        self.attn_norm = RMSNorm(config.width, config.norm_eps)
        self.attn = hooks.mixer(config, layer) if hooks.mixer else Attention(config, layer)
        self.ffn_norm = RMSNorm(config.width, config.norm_eps)
        if hooks.ffn:
            self.ffn = hooks.ffn(config, layer)
        else:
            router = hooks.router(config, layer) if hooks.router else None
            self.ffn = MoE(config, layer, router=router)

    def forward(self, x, cos_sin):
        x = x + self.attn(self.attn_norm(x), cos_sin)
        return x + self.ffn(self.ffn_norm(x))


class MoELM(nn.Module):
    def __init__(self, config, hooks=None):
        super().__init__()
        from moe.plugins import hooks_for
        hooks = hooks or hooks_for(config.method)
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(Block(config, i, hooks) for i in range(config.layers))
        if hooks.block:
            self.blocks = nn.ModuleList(hooks.block(b, config, i) for i, b in enumerate(self.blocks))
        self.norm = RMSNorm(config.width, config.norm_eps)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.max_seq_len = config.sequence_len
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)
        self._init_hook = hooks.init

    @torch.no_grad()
    def reset_buffers(self):
        """Rotary tables (twice the training context, for length-extrapolation probes) and routing state."""
        device = self.embed.weight.device
        self.cos, self.sin = rotary(self.config, 2 * self.config.sequence_len, device)
        for router in self.routers():
            router.bias.zero_()
            router.load.zero_()
        return self

    @torch.no_grad()
    def initialize(self):
        """Reference initialization, then the plugin's own `init(model)`."""
        self.reset_buffers()
        std = self.config.init_scale / math.sqrt(self.config.width)
        for name, p in self.named_parameters():
            if name.endswith("norm.weight") or p.ndim == 1:
                nn.init.ones_(p) if name.endswith("weight") else nn.init.zeros_(p)
            elif name == "embed.weight":
                nn.init.normal_(p, std=1.0)
            else:
                nn.init.normal_(p, std=std)
        if self._init_hook:
            self._init_hook(self)
        return self

    def get_device(self):
        return self.embed.weight.device

    def routers(self):
        return [m for m in self.modules() if isinstance(m, Router)]

    def forward(self, idx, targets=None, reduction="mean"):
        T = idx.size(1)
        if T > self.cos.size(1):
            raise ValueError(f"input of {T} tokens exceeds the rotary table ({self.cos.size(1)})")
        x = self.embed(idx)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        for block in self.blocks:
            x = block(x, cos_sin)
        logits = self.head(self.norm(x)).float()
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1,
                               reduction=reduction)
        aux = [m.last_aux for m in self.modules() if getattr(m, "last_aux", None) is not None]
        self.last_aux = sum(aux) if aux else None
        return loss

    def parameter_counts(self):
        """Total parameters and the parameters one token activates (routed experts counted top_k times)."""
        c = self.config
        total = sum(p.numel() for p in self.parameters())
        routed = sum(p.numel() for m in self.modules() if isinstance(m, Experts) for p in m.parameters())
        embedding = self.embed.weight.numel()
        active = total - routed + routed * c.top_k // c.routed_experts
        return dict(total=total, active=active, embedding=embedding, head=self.head.weight.numel(),
                    active_non_embedding=active - embedding - self.head.weight.numel(),
                    total_non_embedding=total - embedding - self.head.weight.numel())


def build_model(config, device="cpu", initialize=True):
    """Build on the meta device, then materialize on `device` and run the reference initialization."""
    from moe.plugins import model_class
    cls = model_class(config.method) or MoELM
    with torch.device("meta"):
        model = cls(config)
    if device == "meta":
        return model
    model.to_empty(device=device)
    return model.initialize() if initialize else model.reset_buffers()
