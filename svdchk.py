# Engineering check on a PILOT matrix (excluded from scoring): does the authors' randomized
# torch.svd_lowrank (niter=2) cost LittleBit accuracy vs an exact truncated SVD?
import torch, numpy as np, time
from lbrep import fetch, methods
W,_ = fetch.tensor('NousResearch/Llama-2-7b-hf','8efe6c9b93655b934e27bd9981e3ec13e55aee9d','model.layers.2.self_attn.v_proj.weight')
orig = torch.svd_lowrank
def exact(A, q=6, niter=2, M=None):
    U,S,Vh = torch.linalg.svd(A, full_matrices=False); return U[:,:q],S[:q],Vh[:q].t()
for bpp in [1.0,0.1]:
    for itq in [False,True]:
        e=[]
        for f in (orig, exact):
            torch.svd_lowrank=f; Wh,eb,_=methods.littlebit(W,bpp,itq,seed=0); e.append(methods.rel_err(W,Wh))
        torch.svd_lowrank=orig
        print(f'bpp {bpp} itq {itq}: lowrank {e[0]:.5f} exact {e[1]:.5f} diff {e[0]-e[1]:+.5f}')
