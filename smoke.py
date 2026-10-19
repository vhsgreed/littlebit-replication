import time, numpy as np, torch
from lbrep import fetch, methods
torch.set_num_threads(8)
W, h = fetch.tensor('NousResearch/Llama-2-7b-hf', 'main', 'model.layers.15.self_attn.q_proj.weight')
print(W.shape, W.dtype, h[:12])
t=time.time(); svd=np.linalg.svd(W.astype(np.float64), full_matrices=False); print('svd', round(time.time()-t,1),'s')
for bpp in [1.0, 0.1]:
    for itq in [False, True]:
        t=time.time(); Wh, eb, sd = methods.littlebit(W, bpp, itq, seed=0)
        print(f'bpp {bpp} itq {itq}: eff_bits {eb:.4f} split {sd} err {methods.rel_err(W,Wh):.4f}  {time.time()-t:.1f}s')
    Wf, ebf, r = methods.tiny_rank_fp16(W, bpp, svd)
    print(f'bpp {bpp} FP16 rank {r} eff_bits {ebf:.4f} err {methods.rel_err(W,Wf):.4f}')
# cross-check reconstruct() against the module forward()
import torch.nn as nn
lin = nn.Linear(4096,4096,bias=False); lin.weight.data.copy_(torch.from_numpy(W)); lin.__class__=methods.LittleBitLinear
lin.__quant_convert__(do_train=True, quant_func=methods.STEBinary, eff_bit=0.55, residual=True, use_itq=True, itq_n_iter=50)
x = torch.randn(7,4096)
with torch.no_grad(): y = lin(x); y2 = x @ torch.from_numpy(methods.reconstruct(lin)).t()
print('forward vs reconstruct max rel diff', float((y-y2).abs().max()/y.abs().max()))
