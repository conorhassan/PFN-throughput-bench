# Custom Triton Kernels — Training Throughput Benchmark

## Model Spec
- 12 transformer layers, H=8, D=64, d_model=512, d_ff=1365
- Rectangular attention: (M+N) queries x M keys
- K=10 GMM head MLPs: 512→1365→3 (logit_weight, mu, log_sigma)
- No positional encoding, no bias, no dropout
- Target GPUs: H100 (primary, FP8+BF16) + A100 (fallback, BF16)

## Definitions
- **tokens/sec**: `B * (M + N) / step_time_sec` — all comparisons use identical shapes and dtype
- **step_time**: wall-clock time for one forward + backward + optimizer step (CUDA-synchronized)

## Project Structure
```
custom-triton-kernels/
├── pyproject.toml
├── src/
│   ├── model.py              # TransformerNP model (12 layers)
│   ├── attention_sdpa.py     # Baseline rectangular SDPA wrapper
│   ├── triton_fwd.py         # Phase 2: Triton rectangular attention forward
│   ├── triton_bwd.py         # Phase 3: Triton rectangular attention backward
│   ├── triton_fused_attn.py  # Phase 4: Recompute-fused backward kernel
│   ├── triton_fp8_attn.py    # Phase 5: FP8 attention (H100)
│   ├── triton_fused_mlp.py   # Phase 6: Fused MLP kernel
│   ├── grouped_gemm.py       # Phase 7: Grouped GEMM for K=10 heads
│   ├── triton_gmm_logprob.py # Phase 8: Fused GMM log-prob
│   ├── checkpoint.py         # Phase 9: Blockwise checkpointing toggle
│   └── cuda_graphs.py        # Phase 10: CUDA graph capture utilities
├── bench/
│   ├── harness.py            # Benchmark harness (timing, memory, logging)
│   ├── run_benchmark.py      # End-to-end benchmark matrix (Phase 12)
│   └── shapes.py             # (M,N) shape configs: small/mid/large
└── tests/
    ├── test_attention_fwd.py
    ├── test_attention_bwd.py
    ├── test_fused_attn.py
    ├── test_fp8_attn.py
    ├── test_fused_mlp.py
    ├── test_grouped_gemm.py
    ├── test_gmm_logprob.py
    └── test_e2e.py
```

## Incremental Build Phases

### Phase 1 — Baseline + Harness
**Files:** `src/model.py`, `src/attention_sdpa.py`, `bench/harness.py`, `bench/shapes.py`, `bench/run_benchmark.py`
- Implement the full 12-layer TransformerNP model in PyTorch
- Rectangular attention via slicing: `K, V = K[:, :, :M, :], V[:, :, :M, :]` so Q is (B,H,M+N,D) attending to K,V of (B,H,M,D). This keeps SDPA on the Flash Attention backend (no boolean masks)
- **SDPA backend sanity check:** use `torch.backends.cuda.sdp_kernel()` context manager to force/log Flash backend; assert Flash is selected, warn if fallback to MemEff/Math
- GMM head + log-prob loss in pure PyTorch
- Benchmark harness: step time, tokens/sec (`B*(M+N)/step_time`), peak memory, per-stage timing
- Shapes: small (64,500), mid (10k,1k), large (200k,5k)
- **Memory feasibility test:** run forward+backward on large shape (200k,5k) B=1 to measure peak memory; decide checkpointing strategy early
- **Validation:** model trains, loss decreases, baseline numbers recorded

### Phase 2 — Triton Rectangular Attention Forward
**Files:** `src/triton_fwd.py`, `tests/test_attention_fwd.py`
- Online softmax tiled kernel: maintain running max `m` and sum `l` per query row
- Tiling with `tl.make_block_ptr` / `tl.advance`, `tl.swizzle2d` for L2 reuse
- FP32 accumulation, BF16 output
- Autotune (BM, BN, BK, warps, stages) — Phase 11 configs baked in here
- **Validation:** max abs error vs SDPA < 1e-2 (BF16), no NaNs, correct shapes

### Phase 3 — Triton Rectangular Attention Backward
**Files:** `src/triton_bwd.py`, `tests/test_attention_bwd.py`
- Flash-style backward recomputing attention from Q/K/V + saved `m`, `l` (shape `[B, H, M+N]` per query row, stored during forward)
- Two-pass dK/dV reduction (per-block partials → reduction kernel) to avoid atomics
- **Validation:** dQ/dK/dV match SDPA gradients within tolerance on small shapes

### Phase 4 — Recompute-Fused Backward Kernel
**Files:** `src/triton_fused_attn.py`, `tests/test_fused_attn.py`
- Backward kernel that recomputes forward attention tiles on-the-fly (from Q/K/V + saved `m`/`l`) and simultaneously accumulates dQ/dK/dV — all in a single kernel launch
- This is NOT a literal forward+backward fusion (dO is unavailable during forward). It fuses the recomputation of attention weights with gradient computation in the backward pass, keeping Q/K tiles in SRAM/registers for both
- dK/dV partials accumulated per tile block, reduced in a second pass
- **Validation:** numerics match separate fwd+bwd kernels from Phases 2-3, step time improvement measured, no register spill

### Phase 5 — FP8 Path (H100 only)
**Files:** `src/triton_fp8_attn.py`, `tests/test_fp8_attn.py`
- E4M3 Q/K/V with per-block scaling, FP32 accumulation, BF16 output
- Safe softmax adjustment to prevent repeated-max degeneracy
- Auto-detect H100; skip on A100
- **Validation:** no NaNs on medium shapes, loss decreases, accuracy delta vs BF16 acceptable

### Phase 6 — Fused MLP
**Files:** `src/triton_fused_mlp.py`, `tests/test_fused_mlp.py`
- Fuse GEMM + SiLU + GEMM in a single Triton kernel (BF16)
- **Validation:** output matches unfused MLP, per-block latency improvement

### Phase 7 — Grouped GEMM for K=10 Heads
**Files:** `src/grouped_gemm.py`, `tests/test_grouped_gemm.py`
- Batched/grouped GEMM: shared input x → 10 head-specific weight sets → [B, L, K, 3]
- **Validation:** output matches loop-based head MLPs, microbench speedup

### Phase 8 — Fused GMM Log-Prob
**Files:** `src/triton_gmm_logprob.py`, `tests/test_gmm_logprob.py`
- Single kernel computes full GMM log-prob:
  - `log_w = log_softmax(logit_weight)` across K components
  - `log_N_k = -0.5 * ((y - mu_k) / exp(log_sigma_k))^2 - log_sigma_k - 0.5*log(2π)`
  - `log_p = logsumexp_k(log_w_k + log_N_k)` (stable: subtract max before exp)
  - `loss = -mean(log_p)`
- Avoid materializing intermediate per-head outputs to HBM
- **Validation:** matches PyTorch reference, no overflow/underflow in exp/logsumexp

### Phase 9 — Blockwise Checkpointing
**Files:** `src/checkpoint.py`
- `torch.utils.checkpoint` wrapping transformer blocks with `use_reentrant=False` (avoids extra autograd overhead and works correctly with non-deterministic ops)
- Toggle on/off; only keep if batch size increase offsets recompute cost
- **Validation:** compare throughput with/without

### Phase 10 — CUDA Graphs
**Files:** `src/cuda_graphs.py`
- Graph-capture fixed-shape training step for final throughput numbers
- **Validation:** identical outputs vs non-graph run, stable timing

### Phase 11 — Autotuning (woven into Phases 2-8)
- Autotune configs for H100 (BF16 + FP8) and A100 (BF16)
- Parameters: BM, BN, BK, num_warps, num_stages

### Phase 12 — End-to-End Benchmark Matrix
**Files:** `bench/run_benchmark.py`
- Compare: SDPA baseline, custom BF16 separate, custom BF16 fused, custom FP8 fused (H100), with/without checkpointing, with/without CUDA graphs
- All comparisons use identical shapes and dtype
- Report: step time, tokens/sec (`B*(M+N)/step_time`), peak memory, speedup vs SDPA

## Verification
After each phase:
1. Run that phase's test file (`pytest tests/test_*.py`)
2. Run benchmark harness to compare against baseline
3. Check for NaNs, shape correctness, numerical tolerance

Final validation: `python bench/run_benchmark.py` produces the full comparison matrix.
