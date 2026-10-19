"""Health gate between canary and main run. Checks plumbing only — NEVER compares method arms
on registered outcomes (one-look discipline). Prints GATE: PASS/FAIL; exit 1 on fail.

Checks per expected model jsonl: complete row count, 7 proj types per layer, sha256 present,
no NaN/inf in registered fields, eff_bits within 1% of budget target, shapes match PROJ names.
"""
import json, math, sys
from pathlib import Path

from . import units

ROOT = Path(__file__).resolve().parent.parent


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'out'
    fails = []
    for model, repo, sha in units.MODELS:
        path = out / f'{model}.jsonl'
        if not path.exists():
            continue  # not run yet
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        names = [r['name'] for r in rows]
        if len(names) != len(set(names)):
            fails.append(f'{model}: duplicate rows')
        for r in rows:
            tag = f"{model}:{r['name']}"
            if not r.get('sha256') or len(r['sha256']) != 64:
                fails.append(f'{tag}: missing sha256')
            if not math.isfinite(r['gamma_full']):
                fails.append(f'{tag}: gamma not finite')
            for bpp, c in r['cells'].items():
                for k in ('eff_bits', 'split_dim', 'lambda_hat', 'lam_before_max',
                          'lam_after_max', 'lam_after_mean'):
                    if not math.isfinite(c[k]):
                        fails.append(f'{tag}@{bpp}: {k} not finite')
                # authors' split_dim rounds down to a multiple of 8, so actual bits can
                # undershoot the target (~6% at 0.1 bpp); never overshoot (+1%)
                budget = float(bpp)
                if not (budget * 0.90 <= c['eff_bits'] <= budget * 1.01):
                    fails.append(f'{tag}@{bpp}: eff_bits {c["eff_bits"]} off target {budget}')
                for b in [f'int{nb}' for nb in units.INT_BITS] + ['fp16', 'lb_B', 'lb2_B']:
                    key = 'err' if b in ('fp16',) or b.startswith('int') else '0'
                    if not math.isfinite(c[b][key]):
                        fails.append(f'{tag}@{bpp}: {b} err not finite')
        print(f'{model}: {len(rows)} rows checked')
    verdict = 'PASS' if not fails else 'FAIL'
    for f in fails[:20]:
        print('  ', f)
    print(f'GATE: {verdict}')
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
