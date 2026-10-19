"""H0: reproduce the paper's synthetic break-even experiment (LittleBit-2 Sec. 5.1, Fig. 6 top).

W = Q1 diag(k^-gamma) Q2^T, Q1/Q2 Haar-random orthogonal, 4096x4096, at 1.0 bpp, gamma grid
0.10..0.80 step 0.025. Break-even = first gamma at which the method's error exceeds Tiny-Rank
FP16 (linear interpolation of the error difference between grid points). One matrix per gamma,
generator seed 20261019 + grid index; LittleBit seed 0. Both arms A (released recipe) and B
(exact truncated SVD) are run; the H0 bar is on arm A (PLAN.md).

usage: python -m lbrep.synthetic [--n 4096] [--out out/h0_synthetic.json]
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from . import methods

ROOT = Path(__file__).resolve().parent.parent
GRID = [round(0.10 + 0.025 * i, 3) for i in range(29)]


def haar(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def crossing(gs, diff):
    """First gamma where diff (method_err - fp16_err) turns positive; None if never/always."""
    for i in range(1, len(gs)):
        if diff[i - 1] <= 0 < diff[i]:
            return gs[i - 1] + (gs[i] - gs[i - 1]) * (-diff[i - 1]) / (diff[i] - diff[i - 1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4096)
    ap.add_argument('--out', default=str(ROOT / 'out' / 'h0_synthetic.json'))
    args = ap.parse_args()
    torch.set_num_threads(8)
    n = args.n
    rows = []
    for i, g in enumerate(GRID):
        rng = np.random.default_rng(20261019 + i)
        s = np.arange(1, n + 1, dtype=np.float64) ** (-g)
        Q1, Q2 = haar(n, rng), haar(n, rng)
        W = ((Q1 * s) @ Q2.T).astype(np.float32)
        svd = (Q1, s, Q2.T)
        row = dict(gamma=g)
        for arm, exact in (('A', False), ('B', True)):
            lb = methods.littlebit(W, 1.0, False, 0, exact=exact)
            lb2 = methods.littlebit(W, 1.0, True, 0, exact=exact)
            row[f'lb_{arm}'], row[f'lb2_{arm}'] = lb['rel_err'], lb2['rel_err']
            row['eff_bits'] = lb['eff_bits']
        row['fp16'] = methods.rel_err(W, methods.tiny_rank_fp16(W, row['eff_bits'], svd)[0])
        rows.append(row)
        print(f"gamma {g:.3f} lbA {row['lb_A']:.4f} lb2A {row['lb2_A']:.4f} "
              f"lbB {row['lb_B']:.4f} lb2B {row['lb2_B']:.4f} fp16 {row['fp16']:.4f}", flush=True)
    gs = [r['gamma'] for r in rows]
    res = dict(n=n, rows=rows)
    for k in ('lb_A', 'lb2_A', 'lb_B', 'lb2_B'):
        res[f'breakeven_{k}'] = crossing(gs, [r[k] - r['fp16'] for r in rows])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print({k: res[k] for k in res if k.startswith('breakeven')})


if __name__ == '__main__':
    main()
