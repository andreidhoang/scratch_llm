"""Bench scripts: bench/kernels/ (kernel ladder) + top-level serving/system benches.

The bench/ tree is NOT a installable package — these ``__init__.py`` files exist only so
``python -m bench.kernels.run`` resolves as a module. Each bench script is still meant to be
run directly (``python bench/kernels/<family>/<x>.py``) or via the runner.
"""
