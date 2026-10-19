# littlebit-replication

Pre-registered weight-level reproduction and extension of **LittleBit / LittleBit-2** (Samsung,
NeurIPS 2025 / ICML 2026, arXiv:2603.00042v2): does low-rank binary factorization beat
Tiny-Rank FP16 *and* int-quantized low-rank baselines at matched sub-1-bit budgets on real
weights from model families the paper never analyzed?

The registration (hypotheses, bars, tie rules, pilot disclosure, change log) is [PLAN.md](PLAN.md).
Tag `freeze-v1` was pushed before any scored datum was generated. Scoring is one-look:
`python -m lbrep.analyze` reads `out/*.jsonl` and writes `out/analysis.{json,out}`.

## Reproduce / replicate

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy safetensors requests torch --index-url https://download.pytorch.org/whl/cpu transformers
make all        # fetch weights (range requests, pinned HF commits) + run + analyze
```

Inputs: six public Hugging Face repos pinned to commit shas (PLAN.md Sec. 1); each matrix's raw
bytes are sha256-checked against the recorded digest. The method under test is the authors' own
`LittleBitLinear` class, vendored at commit 42d658b0c79f76450b34b6a3547462c7cdc3e1a0
(SamsungLabs/LittleBit, CC BY-NC 4.0). Replication criteria are registered in PLAN.md Sec. 5.

Expected compute: ~42h single-machine CPU for all 1,386 matrices (8 threads) + ~1.5h for the
synthetic break-even replication.
