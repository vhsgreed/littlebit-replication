"""Methods under test.

LittleBit / LittleBit-2 initialisation is run through Samsung's OWN LittleBitLinear class
(SamsungLabs/LittleBit, pinned commit in PLAN.md), imported from a local clone, so the method
under test is the authors' code, not our reimplementation. The baseline (Tiny-Rank FP16) is a
truncated SVD at the largest rank that fits the same bit budget.
"""
import os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SAMSUNG = Path(os.environ.get('LITTLEBIT_SRC', Path(__file__).resolve().parent.parent / 'vendor' / 'LittleBit'))
sys.path.insert(0, str(SAMSUNG))
import quantization.utils  # noqa: E402,F401  (authors' entry order; avoids their circular import)
from quantization.modules.littlebit import LittleBitLinear  # noqa: E402
from quantization.functions.binary import STEBinary  # noqa: E402

os.environ.setdefault('LOCAL_RANK', '0')


def littlebit(W, bpp, use_itq, seed, exact=False, residual=True, itq_n_iter=50):
    """Authors' initialisation. Returns a dict with the reconstruction and registered stats.

    exact=True (arm B): torch.svd_lowrank is replaced by the exact truncated SVD during the run
    (registered deviation; PLAN.md Sec. 2). exact=False (arm A): released recipe verbatim.
    """
    out_f, in_f = W.shape
    torch.manual_seed(seed)
    lin = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(W))
    lin.__class__ = LittleBitLinear
    orig = torch.svd_lowrank
    if exact:
        torch.svd_lowrank = _exact_lowrank
    try:
        lin.__quant_convert__(do_train=True, quant_func=STEBinary, eff_bit=bpp, residual=residual,
                              use_itq=use_itq, itq_n_iter=itq_n_iter, min_split_dim=8)
    finally:
        torch.svd_lowrank = orig
    with torch.no_grad():
        W_hat = reconstruct(lin)
        lam_mean, lam_max = lambda_stats(lin)
        err = rel_err(W, W_hat)
        main_err = rel_err(W, reconstruct(lin, main_only=True))
    return dict(rel_err=err, main_rel_err=main_err, eff_bits=float(lin.eff_bit_actual),
                split_dim=int(lin.split_dim), lam_mean=lam_mean, lam_max=lam_max)


_SVD_CACHE = {}


def _exact_lowrank(A, q=6, niter=2, M=None):
    """Exact truncated SVD. Value-keyed cache: the authors' code recomputes the same SVD of W
    on every call (per budget/arm/seed); identical input -> identical factors, so reuse is
    bit-identical and only saves wall time (PLAN.md change log 2026-10-19)."""
    key = (tuple(A.shape), round(float(A.sum()), 3), round(float(A[0, 0]), 6),
           round(float(A[-1, -1]), 6))
    hit = _SVD_CACHE.get(key)
    if hit is None:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        hit = (U, S, Vh)
        if len(_SVD_CACHE) > 8:
            _SVD_CACHE.clear()
        _SVD_CACHE[key] = hit
    U, S, Vh = hit
    return U[:, :q], S[:q], Vh[:q].t().contiguous()


def _part(lin, sfx=''):
    U, V = lin.quantize(getattr(lin, 'U' + sfx)), lin.quantize(getattr(lin, 'V' + sfx))
    u1, u2 = getattr(lin, 'u1' + sfx), getattr(lin, 'u2' + sfx)
    v1, v2 = getattr(lin, 'v1' + sfx), getattr(lin, 'v2' + sfx)
    # Same algebra as the authors' residual-init line: (U * (u1^T u2)) @ (V * (v1^T v2))
    return (U * (u1.t() @ u2)) @ (V * (v1.t() @ v2))


def reconstruct(lin, main_only=False):
    """Dense W_hat of a LittleBitLinear, cross-checked against its own forward() in tests."""
    W_hat = _part(lin)
    if lin.residual and not main_only:
        W_hat = W_hat + _part(lin, '_R')
    return W_hat.float().numpy()


def lambda_hat(W, s, main_rel_err, r):
    """Paper's Lambda = E_quant / int_0^r sigma^2, from the MAIN path error at rank r=split_dim.

    E_quant = ||W - W_hat_main||^2_F - sum_{k>r} sigma_k^2; int_0^r sigma^2 = sum_{k<=r} sigma_k^2.
    """
    err_sq = main_rel_err ** 2 * float((s ** 2).sum())
    tail = float((s[r:] ** 2).sum())
    head = float((s[:r] ** 2).sum())
    return float((err_sq - tail) / head) if head > 0 else float('nan')


def lambda_stats(lin):
    """Paper Sec. 4.3 distortion per row x of X=[U; V^T]: lambda = 1 - (sum|x|)^2/(r*||x||^2).

    Returns (mean, max) over rows. For arm B-no-ITQ this is 'before rotation'; LB2-B 'after'.
    """
    X = torch.cat([lin.U, lin.V.t()], dim=0).float().numpy()
    r = X.shape[1]
    s1 = np.abs(X).sum(axis=1)
    s2 = (X ** 2).sum(axis=1)
    lam = 1.0 - (s1 ** 2) / (r * np.maximum(s2, 1e-30))
    return float(lam.mean()), float(lam.max())


def fp16_rank(a, b, bpp):
    """Largest rank r with 16*r*(a+b) <= bpp*a*b, at least 1."""
    return max(1, int(bpp * a * b // (16 * (a + b))))


def tiny_rank_fp16(W, bpp, svd=None):
    """Truncated SVD with factors rounded to FP16. Returns (W_hat, eff_bits, rank)."""
    out_f, in_f = W.shape
    r = fp16_rank(out_f, in_f, bpp)
    U, s, Vt = svd if svd is not None else np.linalg.svd(W, full_matrices=False)
    A = (U[:, :r] * np.sqrt(s[:r])).astype(np.float16).astype(np.float32)
    B = (np.sqrt(s[:r])[:, None] * Vt[:r]).astype(np.float16).astype(np.float32)
    return A @ B, 16 * r * (out_f + in_f) / (out_f * in_f), r


def _quant_int(X, bits, group=128, clips=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55)):
    """Symmetric round-to-nearest, one FP16 scale per row per group of `group` columns,
    per-group clip ratio picked from `clips` by minimum squared error."""
    qmax = 2 ** (bits - 1) - 1
    out = np.empty_like(X)
    for g0 in range(0, X.shape[1], group):
        G = X[:, g0:g0 + group]
        amax = np.abs(G).max(axis=1, keepdims=True) + 1e-12
        best, best_e = None, None
        for c in clips:
            sc = (amax * c / qmax).astype(np.float16).astype(np.float32)
            Q = np.clip(np.round(G / sc), -qmax - 1, qmax) * sc
            e = ((G - Q) ** 2).sum(axis=1, keepdims=True)
            if best is None:
                best, best_e = Q, e
            else:
                m = e < best_e
                best = np.where(m, Q, best)
                best_e = np.where(m, e, best_e)
        out[:, g0:g0 + group] = best
    return out


def int_bits(a, b, r, bits, group=128):
    return (bits * r * (a + b) + 16 * (a + b) * -(-r // group)) / (a * b)


def lowrank_int(W, bpp, bits, svd=None, group=128):
    """Low-rank factors quantised to `bits` (RTN + clip search), largest rank within budget."""
    out_f, in_f = W.shape
    r = 1
    while int_bits(out_f, in_f, r + 1, bits, group) <= bpp:
        r += 1
    U, s, Vt = svd if svd is not None else np.linalg.svd(W, full_matrices=False)
    A = _quant_int((U[:, :r] * np.sqrt(s[:r])).astype(np.float32), bits, group)
    B = _quant_int((np.sqrt(s[:r])[:, None] * Vt[:r]).T.astype(np.float32), bits, group).T
    return A @ B, int_bits(out_f, in_f, r, bits, group), r


def rel_err(W, W_hat):
    return float(np.linalg.norm(W - W_hat) / np.linalg.norm(W))


def gamma(s, lo=0.0, hi=1.0):
    """Spectral decay rate: slope of log sigma_k vs log k, k over the full spectrum.

    The paper says gammas 'are calculated by log linear regression of real weights' and gives no
    range, so the full spectrum is used (registered in PLAN.md; range is a declared choice).
    """
    k = np.arange(1, len(s) + 1)
    n = len(s)
    sel = slice(int(lo * n), max(int(hi * n), 2))
    keep = s[sel] > 0
    x, y = np.log(k[sel][keep]), np.log(s[sel][keep])
    return float(-np.polyfit(x, y, 1)[0])
