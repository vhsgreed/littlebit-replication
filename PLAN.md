# PLAN.md — Pre-registered study: LittleBit's sub-1-bit break-even on real weights

**Question.** LittleBit-2 (Lee, Banseok; Kim, Dongkyu; you, Youngcheon; Kim, Young-Min; Samsung; ICML 2026, arXiv:2603.00042v2) claims that low-rank binary factorization beats Tiny-Rank FP16 at matched sub-1-bit budgets whenever the weight spectrum is heavy-tailed (Proposition 4.1, spectral break-even), and that Joint-ITQ extends the break-even to gamma ~ 0.51. The break-even was validated on **synthetic** power-law matrices (4096x4096) and calibrated against Llama-2 7B gammas only. Gaps this study closes: (a) reconstruction win/loss rates on **real** weights from model families the paper never analyzed (Llama-3, Qwen3, Mistral, OLMo-2, Gemma-3); (b) baselines restricted to FP16 factors — no integer-quantized (int8/4/3/2) low-rank baselines; (c) memory claims against naive FP16 baselines. Independent reproduction: yes. The method under test runs through the authors' own `LittleBitLinear` class (SamsungLabs/LittleBit @ 42d658b0c79f76450b34b6a3547462c7cdc3e1a0, vendored).

**Freeze.** Tag `freeze-v1`, pushed to github.com/vhsgreed/littlebit-replication before any scored datum was generated. One look at scoring. Every later change is a dated change-log entry; bars are never edited.

## 1. Data (all public, pinned)

| key | repo @ commit | layers | matrices | var-subset |
|---|---|---|---|---|
| llama2-7b | NousResearch/Llama-2-7b-hf @ 8efe6c9b93655b934e27bd9981e3ec13e55aee9d | 32 | 224 | 28 |
| llama3-8b | NousResearch/Meta-Llama-3-8B @ 315b20096dc791d381d514deb5f8bd9c8d6d3061 | 32 | 224 | 20 |
| qwen3-8b | Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218 | 36 | 252 | 25 |
| mistral-7b-v0.3 | mistralai/Mistral-7B-v0.3 @ caa1feb0e54d415e2df31207e5f4e273e33509b1 | 32 | 224 | 21 |
| olmo2-7b | allenai/OLMo-2-1124-7B @ 7df9a82518afdecae4e8c026b27adccc8c1f0032 | 32 | 224 | 22 |
| gemma3-4b | unsloth/gemma-3-4b-pt @ 52aba93981c6ad7712b030eb6dd496ece1d279d6 | 34 | 238 | 23 |

Units: **every** `layers.<i>.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}.weight` of the language model (1,386 matrices total). Excluded, registered: embeddings, norms, lm_head (not LittleBit's compression unit), Gemma-3 vision tower (not LLM weights). Model selection was made before any of these weights were measured: Llama-2 7B (anchor, paper's calibration model), Llama-3 8B (paper's evaluation model), and four model families with no published LittleBit analysis (Qwen3, Mistral, OLMo-2, Gemma-3), all unpinned-gate open weights. Weights fetched by HTTP range requests; each matrix's raw bytes are sha256-recorded in its JSONL row.

**Pilot disclosure (excluded from every count below):** llama2-7b layers 2 and 15 (all 7 types) — outcome values were seen before this freeze (engineering checks: `svdchk.py`, `niterchk.py`, `smoke.py`, `smoke2.py` on layers 2/15 q_proj, v_proj, down_proj). Whole layers excluded so no seen matrix enters scoring. Timings also pilot.

## 2. Methods

Per matrix W (float32 from storage), per budget in {1.0, 0.55, 0.3, 0.1} effective bits:

- **LB-B (primary, LittleBit)** and **LB2-B (primary, LittleBit-2)**: authors' `LittleBitLinear.__quant_convert__` (residual=True, eff_bit=budget, min_split_dim=8, itq_n_iter=50; LB2 = use_itq=True), with `torch.svd_lowrank` replaced by the exact truncated SVD. Registered deviation from released code: the paper's method section says "starting from a truncated SVD", and the randomized niter=2 default measurably handicaps the method (pilot: +0.007..+0.009 rel err at niter=2 vs exact, +0.01 at niter=2 vs niter=16). Primary tests give the paper's architecture its theory-faithful, strongest form.
- **LB-A / LB2-A (secondary sensitivity)**: released recipe exactly (`torch.svd_lowrank`, niter=2, seed 0). Run on the variance subset (10% of matrices) and on all of H0. Connects to the numbers the paper's code produces.
- **Baselines (all at the cell's ACTUAL effective bits from the authors' `_compute_eff_bits` formula, residual=True):**
  - `fp16`: Tiny-Rank FP16 — truncated SVD, factors rounded to FP16, largest rank fitting the budget. The paper's own baseline.
  - `int8/int4/int3/int2`: Low-Rank Int-b — same truncated-SVD factors, symmetric round-to-nearest to b bits, one FP16 scale per row per 128-column group (scales counted), per-group clip ratio searched over {1.0, 0.95, ..., 0.55} by minimum squared error. Stronger-than-FP16 baselines at the same stored size.
  - `strongest` per cell = minimum rel err over the 5 baselines (registered reference; never chosen per bootstrap).
- **Error:** rel err = ||W - W_hat||_F / ||W||_F. Seeds: `torch.manual_seed(seed)` before each LittleBit run; seed 0 for everything, seeds 1-4 on the variance subset (LB2-B only; LB-B with exact SVD is deterministic).

Budgets are target effective bits; all bars are evaluated at **actual** effective bits (authors' formula), recorded per cell. Ranks/dims per cell are recorded (fp16 rank, int ranks, split_dim).

## 3. Hypotheses and bars (the only scored claims)

Scoring unit = matrix. Win/loss/tie per matrix per budget against `strongest`:
**Tie rule:** tie if |err_method - err_baseline| <= 0.5 * MAD_seed(model, budget), where MAD_seed is the pooled seed-to-seed median absolute deviation of LB2-B across the model's variance-subset matrices (pooled over the model's subset; if a subset is empty, pooled over all models at that budget). Ties count as 0.5 win, 0.5 loss in rates and are reported separately. Win-rate = (wins + 0.5*ties)/n.

**H0 — Reproduce their synthetic break-even (LittleBit-2 Sec. 5.1, Fig. 6 top).**
Method: W = Q1 diag(k^-gamma) Q2^T, Q1/Q2 Haar-orthogonal (numpy QR, sign-fixed), n=4096, gamma grid 0.10..0.80 step 0.025 (29 points), one matrix per gamma, generator seed 20261019+index, both arms A and B, budget 1.0. Break-even gamma* = first gamma where method err exceeds fp16 err (linear interpolation between grid points).
Bar: LB-A gamma* within +/-0.03 of the paper's 0.36; LB2-A gamma* within +/-0.03 of 0.51. Arm B reported alongside (expected >= arm A).
Pass = their numbers reproduce from their description. Fail = outside tolerance. **Fail is ambiguous between contradiction and an under-specified generator** ("random matrices with singular values decaying according to a power law" — matrix construction not specified in the paper): a Fail is reported as "not reproducible from the paper's description" with the generator gap named, never as "the paper is wrong".

**H1 — Binary beats the strongest low-rank baseline at matched size on real weights (primary claim).**
For each model x budget cell (6 x 4 = 24 cells; min 50 scored matrices per cell): win-rate of LB2-B vs `strongest` >= 0.75, AND Wilson 95% CI lower bound > 0.60. Also reported per budget pooled over models and for LB-B (no bar).
Pass = the paper's architecture advantage holds against strongest low-rank baselines in that cell. Fail = it does not. Inconclusive = n < 50 (not possible by design) or Wilson lower bound <= 0.60 with point estimate in [0.65, 0.75].
Reported per cell regardless: the full win/tie/loss table vs each baseline separately.

**H2 — Joint-ITQ's advantage shrinks as the budget shrinks (their "extreme regime" framing).**
Per model (6): Spearman rho of per-budget gain G(b) = median over matrices of [err(LB-B) - err(LB2-B)] at budgets {0.1, 0.3, 0.55, 1.0} vs the budget value (n=4 per model).
Bar: rho >= +0.8 in >= 4 of 6 models (gain largest at 1.0 bpp, smallest at 0.1 bpp). FAIL if rho <= -0.8 in >= 4 of 6 (the opposite shape: alignment matters MOST at extreme budgets). Otherwise INCONCLUSIVE. Values are 4-point Spearman; the exact table is reported so readers can judge.
Pass = their claim that alignment matters most at standard budgets, least at extreme ones. Fail = gain is flat or larger at low budgets (which would support the opposite framing).

**H3 — The spectral break-even predicts wins/losses on real weights.**
gamma (gamma_full) = OLS slope of log(sigma_k) vs log(k) over the full exact singular spectrum (positive values). Registered sensitivity gammas (no bars): top-50%, top-10% fits.
- H3a: among scored matrices with gamma < 0.36, at budget 1.0, win-rate of LB2-B vs `strongest` >= 0.80 (min 20 matrices pooled across models).
- H3b: among scored matrices with gamma >= 0.36, at budget 1.0, loss-rate >= 0.75 (min 10 matrices).
- H3c: per model, split scored matrices into fit/test halves by sha256(name) parity (fit on even, test on odd); on fit half pick the gamma threshold maximizing balanced accuracy of margin sign; report held-out accuracy at budget 1.0. Bar: held-out accuracy >= 0.75 in >= 4 of 6 models. (Chance = 0.5.)
Pass = Proposition 4.1's regime split predicts real-weight behavior beyond the paper's Llama-2 calibration. Fail = gamma alone does not separate winners from losers. Inconclusive = H3a and H3b pass but H3c fails (regime split works, threshold not transferable) — reported as such.

**H4 — The distortion coefficient Lambda and the rotation's geometric claim (paper Sec. 4.3-4.4).**
Per matrix per budget (LB2-B): define for each row x of X = [U; V^T] (stacked binary factors before sign) lambda(x) = 1 - (sum|x_j|)^2 / (r * ||x||^2) (matches the paper's E[Lambda] = 1 - 2/pi arithmetic). Lambda_hat = (||W - W_hat||^2_F - sum_{k>r} sigma_k^2) / sum_{k<=r} sigma_k^2 at r = split_dim (the paper's E_quant / int_0^r sigma^2).
- H4a: pooled median Lambda_hat (budget 1.0) in [0.30, 0.50] in >= 4 of 6 models (their 0.36 post-rotation claim). **W_hat here = the MAIN binary path only** (residual path excluded from this one quantity, since Proposition 4.1 is stated for a single factorization); split_dim = the module's registered split_dim.
- H4b: pooled median max-row lambda after rotation <= (before rotation)/2 at budget 1.0 in >= 4 of 6 models (their "suppresses the maximum distortion" claim).
- H4c: pooled median mean-row lambda after rotation in [0.30, 0.50] at budget 1.0 in >= 4 of 6 models.
Before/after = LB-B vs LB2-B factors (same init, ITQ off/on), arm B.
Pass = their geometric claims hold on real weights from unseen families. Fail = not within band / not halved. Inconclusive = band-straddling medians in >= 3 models.

**H5 — Memory accounting (DESCRIPTIVE, no verdict).** For each model: total bits/param and total size at budgets 1.0 and 0.1 under three accountings: (i) authors' `_compute_eff_bits` formula (non-emb), (ii) all-in (every stored byte incl. scales, embeddings, norms, routers, packed bits), (iii) naive FP16 whole model. Plus BitNet b1.58-style 1.58-bit non-emb reference. Published table only; the review article's claim checks cite this table.

**Reading of every outcome (written before the data):**
- H1 pass at >= 3 budgets + H0 pass + H3a pass = "the break-even story survives independent reproduction and extends to unseen families and stronger baselines".
- H1 pass at 1.0/0.55/0.3 but fail at 0.1 = "the advantage holds at standard sub-1-bit budgets and reverses at 0.1 bpp; the 'extreme regime stability' framing is baseline-relative (FP16 only)". This is the shape the pilot suggests; if confirmed it is the headline nuance of the article.
- H1 fail at >= 2 budgets = "the architecture's reconstruction advantage does not generalize to strongest low-rank baselines at matched size"; the article must then center the mechanism analysis (H2-H4) and drop any reproduction-strength claim.
- H0 fail + H3 fail = "Proposition 4.1 is not independently reproducible from the paper's description"; mechanism findings still reported.
- A clean FAIL is a finding. No bar is ever amended after the first scored datum.

## 4. Scoring

- One look: `python -m lbrep.analyze` reads out/*.jsonl, emits out/analysis.json + out/analysis.out. Every clause above is implemented in analyze.py BEFORE the run and smoke-tested on synthetic data with known answers (pure-noise null and strong-effect null) — results in out/null_smoke.json, committed with the freeze.
- Canary: llama2-7b + llama3-8b run first end-to-end (2/6 models). Bars are already frozen; the canary validates plumbing and gives the first read. No bar changes after; bugs, if found, are fixed with a change-log entry before the remaining 4 models run and the canary rows are re-generated if the fix touches their numbers.
- Run order (registered, for the hard deadline case): llama2-7b, llama3-8b, qwen3-8b, mistral-7b-v0.3, olmo2-7b, gemma3-4b. A partial model is never scored; its rows are marked pilot.
- Hard deadline for compute: **2026-10-25 23:59 CET**; if incomplete, score the registered complete subset (models in run order whose jsonl is complete) and mark the rest NOT SCORED. Expected compute: ~42h (pilot-measured: down_proj 3.1 min, q/o 0.7 min, v/k 0.13 min per matrix per full cell set at 8 threads) + ~1.5h synthetic.

## 5. Replicability (standing rule)

Reproducible = same repo+data, same numbers. Replicable = fresh generation from public inputs, any machine, same finding. Registered criteria for a successful external replication:
1. Regenerate every matrix from the pinned HF commits (sha256 of raw tensor bytes must match the JSONL rows).
2. Rerun `make all`; per-cell rel-err agreement within 1e-4 relative (BLAS/thread-count differences) and identical winner labels in >= 99% of cells.
3. Every hypothesis verdict identical (including Inconclusive). Partial agreement = not a replication; report which clause broke.
Known non-determinism sources, registered: CPU SVD rounding across BLAS builds (covered by the 1e-4 tolerance), `torch.svd_lowrank` in arm A (seeded), thread count (fixed at 8 in the run config, recorded per row).

## 6. Change log

- **2026-10-19** Registration written. Pilot disclosed (Sec. 1). Code frozen at tag `freeze-v1` pushed to github.com/vhsgreed/littlebit-replication. No scored datum generated before this push (out/pilot/ is excluded pilot).
