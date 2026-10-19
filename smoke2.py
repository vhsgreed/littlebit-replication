import time, numpy as np, torch
from lbrep import fetch, methods
W,_ = fetch.tensor('NousResearch/Llama-2-7b-hf','main','model.layers.15.self_attn.q_proj.weight')
t=time.time(); svd=np.linalg.svd(W, full_matrices=False); print('svd f32', round(time.time()-t,1),'s')
s=svd[1]; print('gamma full', round(methods.gamma(s),3), 'gamma 1-50%', round(methods.gamma(s,0,0.5),3))
for bpp in [1.0,0.55,0.3,0.1]:
    row=[]
    for name,f in [('fp16',lambda: methods.tiny_rank_fp16(W,bpp,svd)),('int8',lambda: methods.lowrank_int(W,bpp,8,svd)),('int4',lambda: methods.lowrank_int(W,bpp,4,svd)),('int3',lambda: methods.lowrank_int(W,bpp,3,svd)),('int2',lambda: methods.lowrank_int(W,bpp,2,svd))]:
        t=time.time(); Wh,eb,r=f(); row.append(f'{name} r{r} {eb:.3f}b {methods.rel_err(W,Wh):.4f} ({time.time()-t:.1f}s)')
    print(bpp,' | '.join(row))
