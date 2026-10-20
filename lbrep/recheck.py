"""Post-run hardware guard (PLAN.md change log 2026-10-20).

Recompute 20 matrices drawn with seed 20261020 from the finished rows and require every
registered error field to match within 1e-4 relative. Catches silent corruption (bad RAM)
in the run itself. Prints RECHECK: PASS/FAIL; exit 1 on fail.
"""
import json, random, sys
from pathlib import Path

from . import units
from .run import one

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-4


def errs(rec):
    """Flatten every registered error value into {key: float}."""
    out = {}
    for bpp, c in rec['cells'].items():
        for k, v in c.items():
            if k.startswith('t_'):
                continue
            if isinstance(v, dict) and 'err' in v:
                out[f'{bpp}/{k}'] = v['err']
            elif isinstance(v, dict):
                for s, e in v.items():
                    out[f'{bpp}/{k}/{s}'] = e
            elif isinstance(v, (int, float)):
                out[f'{bpp}/{k}'] = v
    return out


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'out'
    rows = []
    for f in sorted(out.glob('*.jsonl')):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: (r['model'], r['name']))
    pick = random.Random(20261020).sample(rows, min(20, len(rows)))
    repos = {m: (repo, sha) for m, repo, sha in units.MODELS}
    fails = []
    for r in pick:
        repo, sha = repos[r['model']]
        var = units.in_var_subset(r['model'], r['name'])
        new, _ = one(r['model'], repo, sha, r['layer'], r['type'], r['name'],
                     units.VAR_SEEDS if var else [], arm_a=bool(var))
        if new['sha256'] != r['sha256']:
            fails.append(f"{r['model']}:{r['name']} sha256 differs")
        a, b = errs(r), errs(new)
        worst = 0.0
        for k in a:
            d = abs(a[k] - b.get(k, float('nan'))) / max(abs(a[k]), 1e-12)
            worst = max(worst, d) if d == d else float('inf')
            if not d <= TOL:
                fails.append(f"{r['model']}:{r['name']} {k} {a[k]} vs {b.get(k)}")
        print(f"{r['model']} {r['name']} worst rel diff {worst:.2e}", flush=True)
    for f in fails[:20]:
        print('  ', f)
    print(f"RECHECK: {'PASS' if not fails else 'FAIL'} ({len(pick)} matrices)")
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
