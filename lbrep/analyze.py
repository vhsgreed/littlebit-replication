"""Scorer for the pre-registered bars (PLAN.md Sec. 3). One look; every clause implemented here.

usage: python -m lbrep.analyze [--out OUTDIR] [--selftest]
Emits OUTDIR/analysis.json and OUTDIR/analysis.out. --selftest runs the synthetic-null checks
(known-answer null and strong-effect rows) and writes OUTDIR/null_smoke.json.
"""
import argparse, hashlib, json, math
from pathlib import Path

import numpy as np

from . import units

ROOT = Path(__file__).resolve().parent.parent
BASELINES = ['fp16', 'int8', 'int4', 'int3', 'int2']
PAPER_GAMMA_LB, PAPER_GAMMA_LB2, TOL = 0.36, 0.51, 0.03


def wilson_lb(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def mad_seed(rows, model, bpp):
    """Pooled seed-to-seed MAD of LB2-B across the model's variance-subset matrices."""
    vals = []
    for r in rows:
        if r['model'] == model and r.get('var_subset'):
            errs = list(r['cells'][str(bpp)]['lb2_B'].values())
            if len(errs) > 1:
                m = np.median(errs)
                vals.append(float(np.median(np.abs(np.asarray(errs) - m))))
    if not vals:
        for r in rows:
            if r.get('var_subset'):
                errs = list(r['cells'][str(bpp)]['lb2_B'].values())
                if len(errs) > 1:
                    m = np.median(errs)
                    vals.append(float(np.median(np.abs(np.asarray(errs) - m))))
    return max(float(np.median(vals)), 1e-9) if vals else 1e-9


def strongest(c):
    return min(BASELINES, key=lambda b: c[b]['err'])


def outcome(c, mad):
    """(win, tie, loss) of LB2-B seed 0 vs the strongest baseline under the registered tie rule."""
    best = min(c[b]['err'] for b in BASELINES)
    d = best - c['lb2_B']['0']
    if abs(d) <= 0.5 * mad:
        return 0, 1, 0
    return (1, 0, 0) if d > 0 else (0, 0, 1)


def rates(rows, sel, bpp, model):
    w = t = l = 0
    mad = mad_seed(rows, model, bpp)
    for r in rows:
        if not sel(r):
            continue
        w_, t_, l_ = outcome(r['cells'][str(bpp)], mad)
        w, t, l = w + w_, t + t_, l + l_
    n = w + t + l
    wr = (w + 0.5 * t) / n if n else float('nan')
    return dict(n=n, wins=w, ties=t, losses=l, win_rate=wr, wilson_lb=wilson_lb(w + 0.5 * t, n))


def spearman(x, y):
    def ranks(v):
        order = np.argsort(v)
        rk = np.empty(len(v), float)
        rk[order] = np.arange(len(v), dtype=float)
        return rk
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def h3c_threshold(rows, model, bpp='1.0'):
    """Threshold on gamma fitted on even-hash half, tested on odd-hash half (margin sign)."""
    fit, test = [], []
    for r in rows:
        if r['model'] != model or r.get('pilot'):
            continue
        half = int(hashlib.sha256(r['name'].encode()).hexdigest(), 16) % 2
        c = r['cells'][bpp]
        best = min(c[b]['err'] for b in BASELINES)
        (fit if half == 0 else test).append((r['gamma_full'], 1 if best - c['lb2_B']['0'] > 0 else 0))
    if len(fit) < 10 or len(test) < 10:
        return dict(acc=float('nan'), n_fit=len(fit), n_test=len(test))
    gammas = sorted({g for g, _ in fit})
    cands = [(gammas[0] - 1e-6) if gammas else 0.0] + \
            [(a + b) / 2 for a, b in zip(gammas, gammas[1:])] + [gammas[-1] + 1e-6]
    def bal_acc(th):
        out = []
        for label in (0, 1):
            pts = [(g, y) for g, y in fit if y == label]
            if pts:
                out.append(np.mean([(1 if g < th else 0) == y for g, y in pts]))
        return float(np.mean(out)) if out else 0.0
    th = max(cands, key=bal_acc)
    # heavy-tailed (low gamma) predicts a win: y_hat = 1 iff gamma < th
    acc = float(np.mean([(1 if g < th else 0) == y for g, y in test]))
    return dict(acc=acc, n_fit=len(fit), n_test=len(test), threshold=th)


def score(rows, h0=None):
    rows = [r for r in rows if not r.get('pilot')]
    models = [m for m, _, _ in units.MODELS if any(r['model'] == m for r in rows)]
    res = dict(models_scored=models, n_matrices=len(rows), hypotheses={})

    # H0
    if h0:
        res['hypotheses']['H0'] = {
            'lb_A': h0.get('breakeven_lb_A'), 'paper': PAPER_GAMMA_LB,
            'lb2_A': h0.get('breakeven_lb2_A'), 'paper_lb2': PAPER_GAMMA_LB2,
            'verdict': 'PASS' if h0.get('breakeven_lb_A') is not None
                       and h0.get('breakeven_lb2_A') is not None
                       and abs(h0['breakeven_lb_A'] - PAPER_GAMMA_LB) <= TOL
                       and abs(h0['breakeven_lb2_A'] - PAPER_GAMMA_LB2) <= TOL else 'FAIL'}

    # H1: win-rate vs strongest per model x budget
    cells = {}
    for m in models:
        for bpp in units.BPPS:
            r = rates(rows, lambda x: x['model'] == m, bpp, m)
            ok = r['n'] >= 50 and r['win_rate'] >= 0.75 and r['wilson_lb'] > 0.60
            incon = r['n'] >= 50 and 0.65 <= r['win_rate'] < 0.75 and r['wilson_lb'] <= 0.60
            cells[f'{m}@{bpp}'] = dict(r, verdict='PASS' if ok else ('INCONCLUSIVE' if incon else 'FAIL'))
    passes = sum(1 for c in cells.values() if c['verdict'] == 'PASS')
    res['hypotheses']['H1'] = dict(cells=cells, n_cells=len(cells), n_pass=passes,
                                   verdict='PASS' if passes == len(cells) else
                                           ('PARTIAL' if passes >= len(cells) / 2 else 'FAIL'))

    # H2: ITQ gain vs budget per model (Spearman)
    h2 = {}
    for m in models:
        gains = []
        for bpp in units.BPPS:
            g = [r['cells'][str(bpp)]['lb_B']['0'] - r['cells'][str(bpp)]['lb2_B']['0']
                 for r in rows if r['model'] == m]
            gains.append(float(np.median(g)) if g else float('nan'))
        rho = spearman(units.BPPS, gains) if not any(np.isnan(gains)) else float('nan')
        h2[m] = dict(gains=dict(zip(map(str, units.BPPS), gains)), rho=rho)
    strong = sum(1 for v in h2.values() if v['rho'] >= 0.8)
    opposite = sum(1 for v in h2.values() if v['rho'] <= -0.8)
    res['hypotheses']['H2'] = dict(models=h2, n_rho_ge_08=strong, n_rho_le_neg08=opposite,
                                   verdict='PASS' if strong >= 4 else
                                           ('FAIL' if opposite >= 4 else 'INCONCLUSIVE'))

    # H3a/b/c
    def gsel(op):
        return lambda r: op(r['gamma_full'])
    h3a = rates(rows, gsel(lambda g: g < 0.36), '1.0', None)
    h3b = rates(rows, gsel(lambda g: g >= 0.36), '1.0', None)
    h3c_res = {m: h3c_threshold(rows, m) for m in models}
    a_ok = h3a['n'] >= 20 and h3a['win_rate'] >= 0.80
    b_ok = h3b['n'] >= 10 and (h3b['losses'] + 0.5 * h3b['ties']) / max(h3b['n'], 1) >= 0.75
    c_ok = sum(1 for v in h3c_res.values() if v['acc'] >= 0.75) >= 4
    res['hypotheses']['H3'] = dict(h3a=h3a, h3b=h3b, h3c=h3c_res,
                                   verdict='PASS' if a_ok and b_ok and c_ok else
                                           ('INCONCLUSIVE' if a_ok and b_ok else 'FAIL'))

    # H4a/b/c (budget 1.0)
    h4 = {}
    for m in models:
        lam_b, lmax_b, lmax_a, lmean_a = [], [], [], []
        for r in rows:
            if r['model'] != m:
                continue
            c = r['cells']['1.0']
            lam_b.append(c['lambda_hat'])
            lmax_b.append(c['lam_before_max'])
            lmax_a.append(c['lam_after_max'])
            lmean_a.append(c['lam_after_mean'])
        h4[m] = dict(lambda_hat=float(np.median(lam_b)), lam_max_before=float(np.median(lmax_b)),
                     lam_max_after=float(np.median(lmax_a)), lam_mean_after=float(np.median(lmean_a)))
    a_ok = sum(1 for v in h4.values() if 0.30 <= v['lambda_hat'] <= 0.50) >= 4
    b_ok = sum(1 for v in h4.values() if v['lam_max_after'] <= v['lam_max_before'] / 2) >= 4
    c_ok = sum(1 for v in h4.values() if 0.30 <= v['lam_mean_after'] <= 0.50) >= 4
    res['hypotheses']['H4'] = dict(models=h4,
                                   verdict='PASS' if a_ok and b_ok and c_ok else
                                           ('INCONCLUSIVE' if (a_ok + b_ok + c_ok) == 2 else 'FAIL'))
    return res


def render(res):
    out = ['== LittleBit pre-registered study: one-look scoring ==',
           f"models scored: {', '.join(res['models_scored'])} | matrices: {res['n_matrices']}"]
    for h, v in res['hypotheses'].items():
        out.append(f"\n{h}: {v['verdict']}")
        if h == 'H0':
            out.append(f"  break-even lb_A {v['lb_A']} vs paper {v['paper']} | lb2_A {v['lb2_A']} vs {v['paper_lb2']}")
        if h == 'H1':
            for k, c in v['cells'].items():
                out.append(f"  {k}: wr {c['win_rate']:.3f} (W{c['wins']}/T{c['ties']}/L{c['losses']}, wilson-lb {c['wilson_lb']:.3f}) {c['verdict']}")
        if h == 'H2':
            for m, d in v['models'].items():
                out.append(f"  {m}: rho {d['rho']:+.3f} gains {d['gains']}")
        if h == 'H3':
            out.append(f"  H3a (gamma<0.36, 1.0bpp): wr {v['h3a']['win_rate']:.3f} n={v['h3a']['n']}")
            out.append(f"  H3b (gamma>=0.36): loss-rate {(v['h3b']['losses'] + 0.5*v['h3b']['ties'])/max(v['h3b']['n'],1):.3f} n={v['h3b']['n']}")
            for m, d in v['h3c'].items():
                out.append(f"  H3c {m}: held-out acc {d['acc']:.3f} (fit {d['n_fit']}/test {d['n_test']})")
        if h == 'H4':
            for m, d in v['models'].items():
                out.append(f"  {m}: lambda_hat {d['lambda_hat']:.3f} | lam_max {d['lam_max_before']:.3f}->{d['lam_max_after']:.3f} | lam_mean_after {d['lam_mean_after']:.3f}")
    return '\n'.join(out)


def make_rows(mode, n_models=6, n=60, seed=0):
    """Synthetic rows for the null smoke.

    Modes: 'null' (no effect: method = best baseline + noise), 'strong' (every registered bar
    satisfied: wins everywhere, ITQ gain grows with budget, lambda in band), 'h3' (outcome
    coupled to gamma: win below 0.36, lose at/above it — tests the H3 clauses specifically).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for mi, (m, _, _) in enumerate(units.MODELS[:n_models]):
        for i in range(n):
            g = rng.uniform(0.15, 0.6)
            cells = {}
            for bpp in units.BPPS:
                base = {b: dict(err=float(rng.uniform(0.6, 0.95)), bits=bpp, rank=10)
                        for b in BASELINES}
                best = min(v['err'] for v in base.values())
                if mode == 'null':
                    e = best + rng.normal(0, 0.002)
                elif mode == 'strong':
                    e = best * 0.55 + rng.normal(0, 0.002)
                else:  # h3
                    e = best - 0.02 if g < 0.36 else best + 0.02
                lb = e + (0.1 * bpp if mode != 'null' else 0.05 * e)
                cells[str(bpp)] = dict(base, lb2_B={'0': e, '1': e + rng.normal(0, 0.001)},
                                       lb_B={'0': lb}, lambda_hat=0.36,
                                       lam_before_max=0.9, lam_after_max=0.4, lam_after_mean=0.36)
            rows.append(dict(model=m, name=f'layers.{i}.q_proj.weight', gamma_full=g,
                             pilot=False, var_subset=(i % 5 == 0), cells=cells))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'out'))
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    if args.selftest:
        checks = {}
        null = score(make_rows('null'))
        strong = score(make_rows('strong'))
        h3w = score(make_rows('h3'))
        checks['H1_null_not_pass'] = null['hypotheses']['H1']['verdict'] != 'PASS'
        checks['H2_null_not_pass'] = null['hypotheses']['H2']['verdict'] != 'PASS'
        checks['H3_null_not_pass'] = null['hypotheses']['H3']['verdict'] != 'PASS'
        checks['H1_strong_pass'] = strong['hypotheses']['H1']['verdict'] == 'PASS'
        checks['H2_strong_pass'] = strong['hypotheses']['H2']['verdict'] == 'PASS'
        checks['H4_strong_pass'] = strong['hypotheses']['H4']['verdict'] == 'PASS'
        checks['H3_gamma_world_pass'] = h3w['hypotheses']['H3']['verdict'] == 'PASS'
        checks['all'] = all(checks.values())
        (out / 'null_smoke.json').write_text(json.dumps(checks, indent=1))
        print(json.dumps(checks, indent=1))
        return
    rows = []
    for f in sorted(out.glob('*.jsonl')):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    h0 = json.loads((out / 'h0_synthetic.json').read_text()) if (out / 'h0_synthetic.json').exists() else None
    res = score(rows, h0)
    txt = render(res)
    (out / 'analysis.json').write_text(json.dumps(res, indent=1))
    (out / 'analysis.out').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
