"""``kernels/linear_attn/`` — the family seam is a RE-EXPORT, asserted rather than described.

``kernels/linear_attn/reference.py`` states its whole invariant in prose: "This module adds
**zero mathematics**", "there is no second oracle to reconcile". Prose invariants rot — that is
the stated reason ``test_kernel_dispatch_boundary.py`` exists at all ("Hand-enforced, that rule
silently rots ... These tests make it CI-checkable"). This file applies the same standard to the
one property the family is FOR.

The threat being closed, named in reference.py itself: a future contributor "invents a weaker
reference to get green" — replacing the import with a local recurrence, or wrapping it in a
try/except that swallows the sealed oracle's ``NotImplementedError``. Every other gate in the
repo would stay green through that; this one would not.

Identity only — nothing here CALLS the oracle, so the sealed ``NotImplementedError`` (the
human's L0.1 work) is never reached and this file cannot be used to route around it.
"""

from __future__ import annotations

import scratch_llm.kernels.linear_attn as la
import scratch_llm.kernels.linear_attn.dispatch as la_dispatch
import scratch_llm.kernels.linear_attn.reference as la_reference
import scratch_llm.mastery.paths as mastery_paths
import scratch_llm.mastery.reference as mastery_reference


def test_family_oracle_IS_the_mastery_oracle() -> None:
    """Object identity, not merely equal behaviour: one recurrence, not four."""
    for mod in (la, la_dispatch, la_reference):
        assert mod.recurrent_reference is mastery_reference.recurrent_reference, mod.__name__
        assert mod.chunked_wy is mastery_paths.chunked_wy, mod.__name__


def test_family_reference_defines_no_mathematics_of_its_own() -> None:
    """The module's only executable content is imports — no function or class definitions."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(la_reference))
    defined = [
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    assert not defined, f"linear_attn/reference.py must re-export only; it defines {defined}"


def test_no_gpu_backend_is_claimed_before_one_exists() -> None:
    """``_LAZY_GPU`` is empty by design — a family may not advertise a backend it has not
    graduated (oracle-first test + measured roofline row + dispatch entry)."""
    assert la_dispatch._LAZY_GPU == {}, (
        "a backend was registered; give it a test and a roofline row"
    )
