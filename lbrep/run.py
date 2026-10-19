"""Confirmatory runner. One JSON line per weight matrix in out/<model>.jsonl; resumable.

Per matrix: exact SVD (spectrum + baselines), primary arm B (authors' code + exact truncated
SVD) at 4 budgets with seeds, arm A (released recipe) on the registered 10% variance subset,
every baseline at arm B's ACTUAL effective bits.

usage: python -m lbrep.run [model_key ...] [--only TENSOR_NAME] [--out DIR]
"""
import argparse, json, os, platform, sys, time
from pathlib import Path

import numpy as np
import torch

from . import fetch, methods, units

ROOT = Path(__file__).resolve().parent.parent


def one(model, repo, sha, layer, ptype, name, seeds_extra, arm_a):
    t0 = time.time()
    W, digest = fetch.tensor(repo, sha, name)
    t_fetch = time.time() - t0
    a, b = W.shape
    t0 = time.time()
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    svd = (U, s, Vt)
    t_svd = time.time() - t0
    rec = dict(model=model, layer=layer, type=ptype, name=name, shape=[a, b], sha256=digest,
               pilot=units.is_pilot(model, layer), var_subset=bool(seeds_extra),
               gamma_full=methods.gamma(s), gamma_top50=methods.gamma(s, 0.0, 0.5),
               gamma_top10=methods.gamma(s, 0.0, 0.1), cells={})
    tail = float((s[int(min(a, b) // 2):] ** 2).sum())
    for bpp in units.BPPS:
        c = {}
        t = time.time()
        r0 = methods.littlebit(W, bpp, False, 0, exact=True)
        c['t_lb_B'] = time.time() - t
        c['lb_B'] = {'0': r0['rel_err']}
        c.update(lam_before_mean=r0['lam_mean'], lam_before_max=r0['lam_max'])
        c['eff_bits'], c['split_dim'] = r0['eff_bits'], r0['split_dim']
        t = time.time()
        r1 = methods.littlebit(W, bpp, True, 0, exact=True)
        c['t_lb2_B'] = time.time() - t
        c['lb2_B'] = {'0': r1['rel_err']}
        c.update(lam_after_mean=r1['lam_mean'], lam_after_max=r1['lam_max'])
        c['lambda_hat'] = methods.lambda_hat(W, s, r1['main_rel_err'], r1['split_dim'])
        for seed in seeds_extra:
            c['lb2_B'][str(seed)] = methods.littlebit(W, bpp, True, seed, exact=True)['rel_err']
        if arm_a:
            c['lb_A'] = {'0': methods.littlebit(W, bpp, False, 0, exact=False)['rel_err']}
            c['lb2_A'] = {'0': methods.littlebit(W, bpp, True, 0, exact=False)['rel_err']}
        budget = c['eff_bits']
        Wh, ebf, r = methods.tiny_rank_fp16(W, budget, svd)
        c['fp16'] = dict(err=methods.rel_err(W, Wh), bits=ebf, rank=r)
        for nb in units.INT_BITS:
            Wh, ebi, rk = methods.lowrank_int(W, budget, nb, svd)
            c[f'int{nb}'] = dict(err=methods.rel_err(W, Wh), bits=ebi, rank=rk)
        rec['cells'][str(bpp)] = c
    rec['t_fetch'], rec['t_svd'] = t_fetch, t_svd
    rec['t_total'] = time.time() - t0 + t_fetch
    return rec, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='*')
    ap.add_argument('--only', action='append')
    ap.add_argument('--out', default=str(ROOT / 'out'))
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('LB_THREADS', '8')))
    out = Path(args.out)
    (out / 'spectra').mkdir(parents=True, exist_ok=True)
    env = dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
               cpu=platform.processor() or platform.machine(), threads=torch.get_num_threads())
    for model, repo, sha in units.MODELS:
        if args.models and model not in args.models:
            continue
        path = out / f'{model}.jsonl'
        done = set()
        if path.exists():
            done = {json.loads(l)['name'] for l in path.read_text().splitlines() if l.strip()}
        mats = units.matrices(fetch.index(repo, sha))
        if args.only:
            mats = [m for m in mats if m[2] in args.only]
        todo = [m for m in mats if m[2] not in done]
        print(f'{model}: {len(mats)} matrices, {len(todo)} to do', flush=True)
        for layer, ptype, name in todo:
            var = units.in_var_subset(model, name)
            rec, s = one(model, repo, sha, layer, ptype, name,
                         units.VAR_SEEDS if var else [], arm_a=bool(var))
            rec['env'] = env
            np.save(out / 'spectra' / f'{model}__{name}.npy', s.astype(np.float32))
            with open(path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f'{model} {name} {rec["t_total"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
