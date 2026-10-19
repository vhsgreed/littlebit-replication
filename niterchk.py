# Pilot-only engineering check: randomized-SVD accuracy penalty vs niter (pilot matrix, excluded)
import torch, time
from lbrep import fetch, methods
W,_ = fetch.tensor('NousResearch/Llama-2-7b-hf','8efe6c9b93655b934e27bd9981e3ec13e55aee9d','model.layers.15.mlp.down_proj.weight')
orig = torch.svd_lowrank
for nit in [2, 4, 8, 16]:
    torch.svd_lowrank = lambda A, q=6, niter=2, M=None, _n=nit: orig(A, q=q, niter=_n)
    for bpp in [1.0, 0.1]:
        t=time.time(); Wh,_,_ = methods.littlebit(W,bpp,True,seed=0)
        print(f'niter {nit} bpp {bpp} lb2 {methods.rel_err(W,Wh):.5f} {time.time()-t:.0f}s', flush=True)
