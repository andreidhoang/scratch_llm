"""CPU smoke gate — proves the package imports and the tree is green-able.

Intentionally trivial: real coverage arrives with the first load-bearing module
(utils/monitors.py). This exists so `pytest` collects >0 tests (an empty
collection exits 5, which the CI gate reads as failure).
"""

import reasoning_llm


def test_package_imports_with_version() -> None:
    assert reasoning_llm.__version__ == "0.1.0"


def test_all_layer_subpackages_importable() -> None:
    import importlib

    for sub in ("algos", "rewards", "envs", "rollout", "scaling", "data", "utils"):
        assert importlib.import_module(f"reasoning_llm.{sub}") is not None
