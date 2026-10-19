import time, numpy as np, torch
from lbrep import fetch, methods
for repo,sha,name in [('NousResearch/Meta-Llama-3-8B','315b20096dc791d381d514deb5f8bd9c8d6d3061','model.layers.10.mlp.down_proj.weight'),('NousResearch/Meta-Llama-3-8B','315b20096dc791d381d514deb5f8bd9c8d6d3061','model.layers.10.self_attn.k_proj.weight')]:
    t=time.time(); W,_=fetch.tensor(repo,sha,name); tf=time.time()-t
    t=time.time(); svd=np.linalg.svd(W,full_matrices=False); ts=time.time()-t
    tl={}
    for bpp in [1.0,0.55,0.3,0.1]:
        for itq in [False,True]:
            t=time.time(); Wh,eb,sd=methods.littlebit(W,bpp,itq,seed=0); tl[(bpp,itq)]=time.time()-t
    t=time.time()
    for bpp in [1.0,0.55,0.3,0.1]:
        for b in (16,8,4,3,2):
            (methods.tiny_rank_fp16(W,bpp,svd) if b==16 else methods.lowrank_int(W,bpp,b,svd))
    tb=time.time()-t
    per_seed=sum(tl.values())
    print(name, W.shape, f'fetch {tf:.1f}s svd {ts:.1f}s littlebit-all-cells/seed {per_seed:.1f}s baselines {tb:.1f}s', {k:round(v,1) for k,v in tl.items()})
