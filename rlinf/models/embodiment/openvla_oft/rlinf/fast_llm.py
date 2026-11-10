# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bitwise-exact fused inference path of the (bidirectional, SDPA) Llama-2 language model of OpenVLA-OFT.

Used only by the rollout (``predict_action_batch``: no autograd). It runs the same modules' weights through the same
GEMMs and the same SDPA call (identical shapes, strides and dtypes) and replaces only the memory-bound elementwise
glue between them by Triton kernels that reproduce PyTorch's arithmetic bit for bit:

* RMSNorm (optionally fused with the preceding residual add): the sum of squares is accumulated in exactly the order
  of ATen's ``reduce_kernel`` for a contiguous fp32 row of 4096 (block 32x16, one warp per row, float4 loads with 4
  per-thread accumulators combined in order, then the warp shuffle-down tree with offsets 16, 8, 4, 2, 1); the mean
  is the sum times 2**-12; then ``+eps``, ``rsqrt``, ``x * r`` (fp32), ``-> bf16``, ``weight * x`` (bf16 product).
* RoPE: ``bf16(bf16(q*cos) + bf16(rotate_half(q)*sin))`` written directly in the contiguous (B, H, L, D) layout
  that the reference produces with ``.contiguous()`` before SDPA; v is copied to the same layout.
* SiLU-gated MLP product: ``bf16(bf16(g / (1 + exp(-g))) * u)`` with IEEE division and the precise ``expf``.
* cos/sin of the rotary embedding are computed once per forward by the layer-0 module (every layer computes the same
  values from the same inputs).

No FMA contraction (``enable_fp_fusion=False``). Every kernel is verified bitwise against the PyTorch ops (exhaustively
where the input space is small: all bf16 values for SiLU, all fp32 values for rsqrt) by the tests in ws/runs.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_HIDDEN = 4096
_NT = 32  # threads (lanes) per row in ATen's reduce config for this shape
_VEC = 4  # float4 input vectorization in ATen's reduce kernel
_ITERS = _HIDDEN // (_NT * _VEC)


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    r_ptr,
    h_ptr,
    w_ptr,
    out_ptr,
    eps,
    n_rows,
    HAS_RES: tl.constexpr,
    ROWS: tl.constexpr,
    NT: tl.constexpr,
    ITERS: tl.constexpr,
):
    VEC: tl.constexpr = 4
    CH: tl.constexpr = NT * VEC
    D: tl.constexpr = ITERS * CH
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    cols = tl.arange(0, CH)
    base = rows[:, None].to(tl.int64) * D + cols[None, :]
    m2 = rmask[:, None]
    acc = tl.zeros((ROWS, CH), tl.float32)
    for k in tl.static_range(ITERS):
        off = base + k * CH
        xv = tl.load(x_ptr + off, mask=m2, other=0.0).to(tl.float32)
        if HAS_RES:
            rv = tl.load(r_ptr + off, mask=m2, other=0.0).to(tl.float32)
            hb = (rv + xv).to(tl.bfloat16)
            tl.store(h_ptr + off, hb, mask=m2)
            hv = hb.to(tl.float32)
        else:
            hv = xv
        acc = acc + hv * hv
    # per-thread accumulators j = 0..3 combined in order: ((a0 + a1) + a2) + a3
    a = tl.reshape(acc, (ROWS * NT, 2, 2))
    e, o = tl.split(a)  # j0 = 0 / 1 -> (ROWS*NT, 2) over j1
    a0, a2 = tl.split(e)
    a1, a3 = tl.split(o)
    v = ((a0 + a1) + a2) + a3
    # warp shuffle-down tree, offsets 16, 8, 4, 2, 1: lane i + 16 first (halves)
    v = tl.reshape(v, (ROWS, 2, NT // 2))
    v = tl.sum(v, axis=1)
    v = tl.reshape(v, (ROWS, 2, NT // 4))
    v = tl.sum(v, axis=1)
    v = tl.reshape(v, (ROWS, 2, NT // 8))
    v = tl.sum(v, axis=1)
    v = tl.reshape(v, (ROWS, 2, NT // 16))
    v = tl.sum(v, axis=1)
    v = tl.reshape(v, (ROWS, 2, 1))
    v = tl.sum(v, axis=1)  # (ROWS, 1)
    mean = v * (1.0 / D)
    rstd = libdevice.rsqrt(mean + eps)
    for k in tl.static_range(ITERS):
        off = base + k * CH
        if HAS_RES:
            hv = tl.load(h_ptr + off, mask=m2, other=0.0).to(tl.float32)
        else:
            hv = tl.load(x_ptr + off, mask=m2, other=0.0).to(tl.float32)
        y = (hv * rstd).to(tl.bfloat16).to(tl.float32)
        wk = tl.load(w_ptr + k * CH + cols)[None, :].to(tl.float32)
        tl.store(out_ptr + off, (wk * y).to(tl.bfloat16), mask=m2)


@triton.jit
def _rope_qkv_kernel(
    q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr, vo_ptr, L, H: tl.constexpr, D: tl.constexpr
):
    pid = tl.program_id(0)  # token index b * L + l
    b = pid // L
    l = pid - b * L
    HALF: tl.constexpr = D // 2
    h = tl.arange(0, H)[:, None]
    d = tl.arange(0, D)[None, :]
    row = pid.to(tl.int64) * (H * D)
    in_off = row + h * D + d
    rot_d = tl.where(d < HALF, d + HALF, d - HALF)
    in_rot = row + h * D + rot_d
    c = tl.load(cos_ptr + pid.to(tl.int64) * D + d).to(tl.float32)
    s = tl.load(sin_ptr + pid.to(tl.int64) * D + d).to(tl.float32)
    out_off = ((b * H + h).to(tl.int64) * L + l) * D + d
    # q
    x = tl.load(q_ptr + in_off).to(tl.float32)
    xr = tl.load(q_ptr + in_rot).to(tl.float32)
    xr = tl.where(d < HALF, -xr, xr)
    t1 = (x * c).to(tl.bfloat16).to(tl.float32)
    t2 = (xr * s).to(tl.bfloat16).to(tl.float32)
    tl.store(qo_ptr + out_off, (t1 + t2).to(tl.bfloat16))
    # k
    x = tl.load(k_ptr + in_off).to(tl.float32)
    xr = tl.load(k_ptr + in_rot).to(tl.float32)
    xr = tl.where(d < HALF, -xr, xr)
    t1 = (x * c).to(tl.bfloat16).to(tl.float32)
    t2 = (xr * s).to(tl.bfloat16).to(tl.float32)
    tl.store(ko_ptr + out_off, (t1 + t2).to(tl.bfloat16))
    # v: layout change only
    tl.store(vo_ptr + out_off, tl.load(v_ptr + in_off))


@triton.jit
def _silu_mul_kernel(g_ptr, u_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.math.div_rn(g, 1.0 + libdevice.exp(-g))
    s = s.to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + offs, (s * u).to(tl.bfloat16), mask=m)


@triton.jit
def _rsqrt_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    """Test helper: the rsqrt used in _rmsnorm_kernel."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, libdevice.rsqrt(tl.load(x_ptr + offs, mask=m, other=1.0)), mask=m)


_ROWS = 4


def rmsnorm(x, weight, eps, residual=None):
    """(h, out): h = residual + x (bf16) if residual is given else x; out = LlamaRMSNorm(weight, eps)(h)."""
    n_rows = x.numel() // _HIDDEN
    out = torch.empty_like(x)
    if residual is not None:
        h = torch.empty_like(x)
        _rmsnorm_kernel[(triton.cdiv(n_rows, _ROWS),)](
            x, residual, h, weight, out, float(eps), n_rows,
            HAS_RES=True, ROWS=_ROWS, NT=_NT, ITERS=_ITERS, num_warps=4, enable_fp_fusion=False,
        )
    else:
        h = x
        _rmsnorm_kernel[(triton.cdiv(n_rows, _ROWS),)](
            x, x, x, weight, out, float(eps), n_rows,
            HAS_RES=False, ROWS=_ROWS, NT=_NT, ITERS=_ITERS, num_warps=4, enable_fp_fusion=False,
        )
    return h, out


def rope_qkv(q, k, v, cos, sin, num_heads, head_dim):
    """q, k, v: (B, L, H*D) contiguous; cos, sin: (B, L, D). Returns contiguous (B, H, L, D) q_embed, k_embed, v."""
    B, L, _ = q.shape
    shape = (B, num_heads, L, head_dim)
    qo = torch.empty(shape, dtype=q.dtype, device=q.device)
    ko = torch.empty(shape, dtype=q.dtype, device=q.device)
    vo = torch.empty(shape, dtype=q.dtype, device=q.device)
    _rope_qkv_kernel[(B * L,)](
        q, k, v, cos, sin, qo, ko, vo, L, H=num_heads, D=head_dim, num_warps=8, enable_fp_fusion=False
    )
    return qo, ko, vo


_SILU_BLOCK = 2048


def silu_mul(g, u):
    out = torch.empty_like(g)
    n = g.numel()
    _silu_mul_kernel[(triton.cdiv(n, _SILU_BLOCK),)](
        g, u, out, n, BLOCK=_SILU_BLOCK, num_warps=8, enable_fp_fusion=False
    )
    return out


def rsqrt_test(x):
    out = torch.empty_like(x)
    n = x.numel()
    _rsqrt_kernel[(triton.cdiv(n, 4096),)](x, out, n, BLOCK=4096, num_warps=8, enable_fp_fusion=False)
    return out


def supported(lm, inputs_embeds):
    """Whether the fused path reproduces this language model's forward exactly (else use the HF forward)."""
    return (
        inputs_embeds.is_cuda
        and inputs_embeds.dtype == torch.bfloat16
        and inputs_embeds.dim() == 3
        and inputs_embeds.is_contiguous()
        and inputs_embeds.shape[0] * inputs_embeds.shape[1] >= 16
        and not torch.is_grad_enabled()
        and config_supported(lm)
    )


def config_supported(lm):
    """The model-configuration part of ``supported``."""
    cfg = lm.config
    m = lm.model
    try:
        attn0 = m.layers[0].self_attn
        ok = (
            getattr(cfg, "_attn_implementation", None) == "sdpa"
            and getattr(cfg, "pretraining_tp", 1) == 1
            and getattr(cfg, "rope_scaling", None) is None
            and cfg.hidden_size == _HIDDEN
            and cfg.num_key_value_heads == cfg.num_attention_heads
            and cfg.hidden_act == "silu"
            and attn0.head_dim * attn0.num_heads == cfg.hidden_size
            and attn0.head_dim % 2 == 0
            and (attn0.num_heads & (attn0.num_heads - 1)) == 0
            and (attn0.head_dim & (attn0.head_dim - 1)) == 0
            and (not lm.training or cfg.attention_dropout == 0.0)
            and type(attn0).__name__ == "LlamaSdpaAttention"
            and type(attn0.rotary_emb).__name__ == "LlamaRotaryEmbedding"
            and getattr(attn0, "past_key_value", None) is None
            and not m.gradient_checkpointing
        )
    except AttributeError:
        return False
    return bool(ok)


@torch.no_grad()
def lm_forward(lm, inputs_embeds, position_ids, key_padding_mask):
    """LlamaForCausalLM forward (bidirectional SDPA fork) of the OFT rollout, bitwise equal to
    ``lm(inputs_embeds=..., attention_mask=mask, position_ids=..., output_hidden_states=True)``.

    key_padding_mask: None when the 2D attention mask has no zero (the reference then passes no mask to SDPA),
    else the (B, L) bool tensor on the device (True = attend).
    Returns (logits fp32 (B, L, V), last hidden state (B, L, D) after the final norm).
    """
    m = lm.model
    layers = m.layers
    B, L, Dm = inputs_embeds.shape
    attn0 = layers[0].self_attn
    H, Dh = attn0.num_heads, attn0.head_dim
    cos, sin = attn0.rotary_emb(inputs_embeds, position_ids)
    cos = cos.contiguous()
    sin = sin.contiguous()

    if key_padding_mask is not None:
        # the reference's per-layer mask: the last row of the 4D causal mask (0 where attended, finfo.min where
        # padded), expanded over the query axis
        dtype = inputs_embeds.dtype
        last_row = torch.zeros((B, 1, L), dtype=dtype, device=inputs_embeds.device)
        last_row.masked_fill_(~key_padding_mask[:, None, :], torch.finfo(dtype).min)
        attn_mask = last_row.unsqueeze(2).expand(-1, -1, L, -1)
    else:
        attn_mask = None

    h = inputs_embeds
    l0 = layers[0].input_layernorm
    _, x = rmsnorm(h, l0.weight, l0.variance_epsilon)
    n = len(layers)
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        q = attn.q_proj(x)
        k = attn.k_proj(x)
        v = attn.v_proj(x)
        q, k, v = rope_qkv(q, k, v, cos, sin, H, Dh)
        a = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        a = a.transpose(1, 2).contiguous().view(B, L, Dm)
        a = attn.o_proj(a)
        pn = layer.post_attention_layernorm
        h, x = rmsnorm(a, pn.weight, pn.variance_epsilon, residual=h)
        mlp = layer.mlp
        y = silu_mul(mlp.gate_proj(x), mlp.up_proj(x))
        y = mlp.down_proj(y)
        nn_ = layers[i + 1].input_layernorm if i + 1 < n else m.norm
        h, x = rmsnorm(y, nn_.weight, nn_.variance_epsilon, residual=h)
    logits = lm.lm_head(x).float()
    return logits, x
