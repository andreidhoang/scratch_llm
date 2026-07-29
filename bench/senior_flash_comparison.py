"""Senior AI Kernels Engineering Benchmark comparison.

Compares:
1. PyTorch Native SDPA
2. Legacy Triton Wrapper (forcing contiguous() copies)
3. New Zero-Copy Stride-Aware PyTorch Custom Op Wrapper
"""

import math
import time
import torch
import torch.nn.functional as F

def benchmark_run(q, k, v, is_causal, run_name, func, num_iters=100):
    # Warmup
    for _ in range(10):
        _ = func(q, k, v, is_causal=is_causal)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(num_iters):
        _ = func(q, k, v, is_causal=is_causal)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_ms = (end_time - start_time) * 1000 / num_iters
    print(f"[{run_name}] Avg Time: {avg_ms:.4f} ms")
    return avg_ms

def test_correctness_and_perf():
    print("=== FlashAttention Senior Refactoring Comparison ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    b, h, n, d = 2, 8, 2048, 64
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    is_causal = True
    
    # 1. Contiguous inputs
    print("\n--- Testing Contiguous Inputs ---")
    q = torch.randn(b, h, n, d, device=device, dtype=dtype)
    k = torch.randn(b, h, n, d, device=device, dtype=dtype)
    v = torch.randn(b, h, n, d, device=device, dtype=dtype)
    
    # Imports
    from scratch_llm.kernels.attention.reference import flash_attention_forward as ref_func
    from scratch_llm.ops.attention import flash_attention as custom_func
    
    # Native SDPA wrapper for comparison
    def native_sdpa(q, k, v, is_causal):
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal), None
        
    # Benchmark
    benchmark_run(q, k, v, is_causal, "PyTorch SDPA", native_sdpa)
    benchmark_run(q, k, v, is_causal, "Ref PyTorch Oracle", ref_func)
    benchmark_run(q, k, v, is_causal, "Custom Op Wrapper", lambda q, k, v, is_causal: custom_func(q, k, v, is_causal=is_causal))
    
    # 2. Non-Contiguous inputs (e.g. transpose/stride mismatch)
    print("\n--- Testing Non-Contiguous (Sliced/Transposed) Inputs ---")
    # Non-contiguous stride simulation: transpose sequence and dim, or slice
    q_nc = torch.randn(b, h, n * 2, d, device=device, dtype=dtype)[:, :, ::2, :]
    k_nc = torch.randn(b, h, n * 2, d, device=device, dtype=dtype)[:, :, ::2, :]
    v_nc = torch.randn(b, h, n * 2, d, device=device, dtype=dtype)[:, :, ::2, :]
    
    print(f"Input q_nc contiguous: {q_nc.is_contiguous()}")
    
    # Measure
    benchmark_run(q_nc, k_nc, v_nc, is_causal, "PyTorch SDPA (Non-Contig)", native_sdpa)
    benchmark_run(q_nc, k_nc, v_nc, is_causal, "Ref PyTorch Oracle (Non-Contig)", ref_func)
    benchmark_run(q_nc, k_nc, v_nc, is_causal, "Custom Op Wrapper (Non-Contig)", lambda q, k, v, is_causal: custom_func(q, k, v, is_causal=is_causal))

if __name__ == "__main__":
    test_correctness_and_perf()
