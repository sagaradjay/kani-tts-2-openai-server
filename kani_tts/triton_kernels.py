"""
Fused Triton kernels for optimized LFM2 decode step.

Replaces multi-op PyTorch sequences with single fused kernels,
reducing kernel launch overhead across 16 layers per decode step.
"""

import torch
import triton
import triton.language as tl


# ==================== Fused RMSNorm ====================
# Replaces 5 PyTorch ops: to(f32) → pow(2) → mean → rsqrt → mul → to(bf16)
# Used 33 times per step (16 layers × 2 norms + 1 final norm)

@triton.jit
def _rms_norm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    y = (x * rrms) * w

    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Fused RMSNorm in a single Triton kernel.

    Args:
        x: [..., N] bf16 input
        weight: [N] bf16 norm weight
        eps: epsilon
    Returns:
        [..., N] bf16 normalized output
    """
    orig_shape = x.shape
    N = orig_shape[-1]
    x_2d = x.reshape(-1, N).contiguous()
    num_rows = x_2d.shape[0]
    y = torch.empty_like(x_2d)

    BLOCK_N = triton.next_power_of_2(N)
    _rms_norm_kernel[(num_rows,)](
        x_2d, weight, y,
        N=N, eps=eps, BLOCK_N=BLOCK_N,
    )
    return y.reshape(orig_shape)


# ==================== Fused SiLU × Multiply ====================
# Replaces: F.silu(gate) + elementwise mul
# Used 16 times per step (one per layer)

@triton.jit
def _silu_mul_kernel(
    X_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    gate = tl.load(X_ptr + row * 2 * N + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(X_ptr + row * 2 * N + N + offs, mask=mask, other=0.0).to(tl.float32)

    y = (gate * tl.sigmoid(gate)) * up

    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


def fused_silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU(gate) * up from concatenated [gate | up] tensor.

    Args:
        gate_up: [..., 2 * intermediate_size] bf16
    Returns:
        [..., intermediate_size] bf16
    """
    orig_shape = gate_up.shape
    N = orig_shape[-1] // 2
    x_2d = gate_up.reshape(-1, orig_shape[-1]).contiguous()
    num_rows = x_2d.shape[0]
    y = torch.empty(num_rows, N, dtype=gate_up.dtype, device=gate_up.device)

    BLOCK_N = triton.next_power_of_2(N)
    _silu_mul_kernel[(num_rows,)](
        x_2d, y,
        N=N, BLOCK_N=BLOCK_N,
    )
    return y.reshape(*orig_shape[:-1], N)


# ==================== Fused RoPE ====================
# Replaces: slice → cat(-x2,x1) → mul(cos) → mul(sin) → add (×2 for Q,K)
# Used 6 times per step (one per attention layer)

@triton.jit
def _rope_kernel(
    X_ptr, COS_ptr, SIN_ptr, Y_ptr,
    head_dim: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    head = tl.program_id(0)
    offs = tl.arange(0, head_dim)

    x = tl.load(X_ptr + head * head_dim + offs).to(tl.float32)
    cos = tl.load(COS_ptr + offs).to(tl.float32)
    sin = tl.load(SIN_ptr + offs).to(tl.float32)

    # rotate_half: pair element i with (i+half)%dim, negate first half
    paired = tl.where(offs < HALF_DIM, offs + HALF_DIM, offs - HALF_DIM)
    sign = tl.where(offs < HALF_DIM, -1.0, 1.0)
    x_paired = tl.load(X_ptr + head * head_dim + paired).to(tl.float32)

    y = x * cos + sign * x_paired * sin
    tl.store(Y_ptr + head * head_dim + offs, y.to(tl.bfloat16))


def fused_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply RoPE to Q and K in fused Triton kernels.

    Args:
        q: [1, num_q_heads, 1, head_dim] bf16
        k: [1, num_k_heads, 1, head_dim] bf16
        cos: [head_dim] or any shape broadcastable, bf16
        sin: [head_dim] or any shape broadcastable, bf16
    Returns:
        (q_rotated, k_rotated) same shapes
    """
    num_q = q.shape[1]
    num_k = k.shape[1]
    hd = q.shape[-1]
    half = hd // 2

    q_flat = q.reshape(num_q, hd).contiguous()
    k_flat = k.reshape(num_k, hd).contiguous()
    cos_flat = cos.reshape(hd).contiguous()
    sin_flat = sin.reshape(hd).contiguous()

    q_out = torch.empty_like(q_flat)
    k_out = torch.empty_like(k_flat)

    _rope_kernel[(num_q,)](q_flat, cos_flat, sin_flat, q_out,
                           head_dim=hd, HALF_DIM=half)
    _rope_kernel[(num_k,)](k_flat, cos_flat, sin_flat, k_out,
                           head_dim=hd, HALF_DIM=half)

    return q_out.reshape_as(q), k_out.reshape_as(k)
