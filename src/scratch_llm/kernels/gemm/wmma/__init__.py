"""WMMA fragment GEMM (raw CUDA C++ via torch.utils.cpp_extension.load_inline).

Not Triton — the whole point of this rung is the explicit ``nvcuda::wmma`` fragment machinery
(``load_matrix_sync → mma_sync → store_matrix_sync``) that ``tl.dot`` abstracts away.
"""
