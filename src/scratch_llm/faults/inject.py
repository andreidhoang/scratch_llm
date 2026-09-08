"""The fault injector — three faults, deliberately induced, each at the place the real one lands.

A postmortem you cannot re-run is a story. These three are re-runnable:

**Rank death** (:class:`RankKill`) — ``os._exit`` from inside the step loop. Not ``sys.exit``, not
an exception: those unwind, run ``atexit``, and let the process group tear down politely, which is
exactly what a real rank death does NOT do. ``os._exit`` leaves the other ranks blocked in a
collective with a half-open socket, which is the failure the survivors have to notice.

**NaN batch** (:class:`NaNBatch`) — a corrupted sample. Token ids are integers, so a batch tensor
cannot literally hold a NaN; the corruption becomes visible at the first floating-point tensor the
batch touches, which is the embedding output. That is also where a real one shows up (a bad float
feature, an overflowing sample weight, a denormal that flushes wrong on one card), so the hook
lands there rather than on the loss — poisoning the loss directly would skip the whole backward
and prove nothing about whether gradients were kept clean.

**Bit flip** (:func:`flip_bit_`) — one bit of one element, chosen by position, via a same-width
integer view of the storage. Nothing is scaled, rounded, or "made small": the flip is exactly the
single-event upset a cosmic ray or a marginal SM produces, and the bit index is the knob the
detector's sensitivity is measured against.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

# Same-width SIGNED integer view for each element size. Signed is fine and is what torch offers:
# XOR is a bit operation, and two's complement makes the top bit of the signed word the sign bit
# of the float — flipping it with the (negative) mask below is bit-identical to flipping bit 31.
_INT_VIEW: dict[int, torch.dtype] = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}

# (mantissa bits, exponent bits) per float dtype — the fields the detectors' sensitivity is
# expressed in. bfloat16 is fp32's exponent with fp16's storage, which is why its mantissa is 7.
_FIELDS: dict[torch.dtype, tuple[int, int]] = {
    torch.float64: (52, 11),
    torch.float32: (23, 8),
    torch.float16: (10, 5),
    torch.bfloat16: (7, 8),
}


def mantissa_bits(dtype: torch.dtype) -> int:
    """Stored mantissa width. ``torch.finfo`` exposes ``bits`` and ``eps`` but not this, and every
    sensitivity bound in this package is expressed in mantissa bits, so it lives here."""
    return _FIELDS[dtype][0]


def describe_bit(dtype: torch.dtype, bit: int) -> str:
    """Name the IEEE-754 field bit ``bit`` lives in, e.g. ``"mantissa bit 0 (lsb)"``.

    The whole sensitivity argument is field-relative: a sign flip changes a gradient by 2x its own
    magnitude, the top exponent bit changes it by 2**128, and mantissa bit 0 changes it by one ulp.
    A detector that only catches the first two catches only the failures you would have noticed.
    """
    width = torch.finfo(dtype).bits if dtype.is_floating_point else torch.iinfo(dtype).bits
    if not 0 <= bit < width:
        raise ValueError(f"bit {bit} out of range for {dtype} ({width} bits)")
    if not dtype.is_floating_point:
        return f"bit {bit}"
    mantissa, exponent = _FIELDS[dtype]
    if bit == width - 1:
        return "sign bit"
    if bit >= mantissa:
        return f"exponent bit {bit - mantissa} (x2**{1 << (bit - mantissa)})"
    return f"mantissa bit {bit}{' (lsb)' if bit == 0 else ''}"


@dataclass(frozen=True)
class BitFlip:
    """What was flipped and what it did — the row a postmortem quotes."""

    where: str
    flat_index: int
    bit: int
    field: str
    before: float
    after: float

    def __str__(self) -> str:
        return (
            f"{self.where}[{self.flat_index}] {self.field}: "
            f"{self.before!r} -> {self.after!r} (rel {self.relative_change:.3e})"
        )

    @property
    def relative_change(self) -> float:
        """|after - before| / |before| — the perturbation the detector has to see."""
        if self.before == 0.0:
            return float("inf") if self.after != 0.0 else 0.0
        return abs(self.after - self.before) / abs(self.before)


def flip_bit_(t: Tensor, flat_index: int, bit: int, *, where: str = "tensor") -> BitFlip:
    """XOR bit ``bit`` of element ``flat_index`` of ``t``, in place. Returns what changed.

    Contiguity is REQUIRED, not silently fixed: ``.contiguous()`` on a strided tensor returns a
    copy, and the flip would land on the copy while the caller kept the pristine original — a fault
    injector that silently injects nothing is worse than no injector.

    Flipping the same bit twice restores the original word exactly (XOR is an involution); the test
    suite uses that as the injector's own correctness check.
    """
    if not t.is_contiguous():
        raise ValueError(
            f"flip_bit_ needs a contiguous tensor ({where} is not) — .contiguous() would copy and "
            "the flip would land on the copy, injecting nothing"
        )
    if not 0 <= flat_index < t.numel():
        raise IndexError(f"flat_index {flat_index} out of range for {where} ({t.numel()} elements)")
    width = t.element_size() * 8
    if not 0 <= bit < width:
        raise ValueError(f"bit {bit} out of range for {t.dtype} ({width} bits)")
    raw = t.detach()  # shares storage; .view(int) refuses on a leaf that requires grad
    words = raw.view(_INT_VIEW[t.element_size()]).reshape(-1)
    # 1 << (width-1) overflows the signed word; its two's-complement value is -(1 << (width-1)),
    # whose bit pattern is exactly the top bit set.
    mask = -(1 << bit) if bit == width - 1 else (1 << bit)
    before = float(raw.reshape(-1)[flat_index].item())
    words[flat_index] = words[flat_index] ^ torch.tensor(
        mask, dtype=words.dtype, device=words.device
    )
    return BitFlip(
        where=where,
        flat_index=flat_index,
        bit=bit,
        field=describe_bit(t.dtype, bit),
        before=before,
        after=float(raw.reshape(-1)[flat_index].item()),
    )


def flip_grad_bit_(model: nn.Module, param_name: str, flat_index: int, bit: int) -> BitFlip:
    """Flip one bit of one element of ``model.<param_name>.grad``.

    Raises if the parameter has no gradient — a bit flip on a gradient that does not exist yet is
    the injector lying about having injected, and every downstream "detector stayed silent" would
    then be meaningless.
    """
    params = dict(model.named_parameters())
    if param_name not in params:
        raise KeyError(f"{param_name!r} is not a parameter of {type(model).__name__}")
    grad = params[param_name].grad
    if grad is None:
        raise ValueError(f"{param_name!r} has no .grad — call backward() before injecting")
    return flip_bit_(grad, flat_index, bit, where=f"{param_name}.grad")


def largest_grad_index(model: nn.Module, param_name: str) -> int:
    """Flat index of the largest-|.| element of that parameter's gradient.

    The bit-flip sensitivity of a *norm* comparison scales with (g_i / ||g||)**2, so where you flip
    changes the answer by orders of magnitude. Picking the largest element makes the boundary
    measurement the detector's BEST case, and the test says so — the worst case is 1/P of it.
    """
    params = dict(model.named_parameters())
    grad = params[param_name].grad
    if grad is None:
        raise ValueError(f"{param_name!r} has no .grad — call backward() first")
    return int(grad.detach().reshape(-1).abs().argmax().item())


@dataclass(frozen=True)
class RankKill:
    """Kill rank ``rank`` at step ``step`` the way hardware does: ``os._exit``, no unwinding.

    The survivors are left blocked in the next collective on a socket that will never answer. Under
    ``torch.multiprocessing.spawn`` the parent sees ``ProcessExitedException`` with ``exit_code``
    and ``error_index``; under ``torchrun`` the agent sees the worker fail and tears the group down.
    Either way the *launcher* is the thing that notices — no in-process detector can, which is why
    this fault's detection is a resume check and not a monitor.
    """

    rank: int
    step: int
    exit_code: int = 137  # 128 + SIGKILL, what an OOM-killed or hardware-reset rank reports

    def maybe_kill(self, *, step: int, rank: int) -> None:
        """Called at the top of every step; exits hard when this is the targeted (rank, step)."""
        if rank != self.rank or step != self.step:
            return
        print(f"RankKill: rank {rank} dying at step {step} (exit {self.exit_code})", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(self.exit_code)


class NaNBatch:
    """Poison the batch: on the targeted steps the module's output becomes all-NaN.

    Attach to the token embedding — the first floating-point tensor a batch of token ids reaches.
    ``current_step`` is set by the loop before each forward; the hook is otherwise a no-op, so a
    clean run with the injector attached is bit-identical to one without it (tested).

    The corruption is ADDED to the output, never substituted for it. Returning
    ``torch.full_like(out, nan)`` looks equivalent and is not: it is a fresh leaf, so it severs the
    autograd graph, and the poisoned rank finishes ``backward()`` with ``token_emb.weight.grad is
    None`` while every other rank has one. The ranks then disagree about how many all-reduces to
    issue and the job hangs forever in a collective — measured on this box, and the exact shape of
    the classic NCCL hang. Adding keeps ``+``'s identity backward, so the NaN propagates into the
    gradients the way a real one does, which is what the detector is supposed to catch.
    """

    def __init__(self, steps: Iterable[int], value: float = float("nan")) -> None:
        self.steps = frozenset(steps)
        self.value = value
        self.current_step = -1
        self.fired_at: list[int] = []

    def attach(self, module: nn.Module) -> RemovableHandle:
        """Register the forward hook; the caller owns the returned handle."""

        def hook(_m: nn.Module, _inp: tuple[object, ...], out: Tensor) -> Tensor:
            if self.current_step not in self.steps:
                return out
            self.fired_at.append(self.current_step)
            return out + torch.full_like(out, self.value)

        return module.register_forward_hook(hook)


__all__ = [
    "BitFlip",
    "NaNBatch",
    "RankKill",
    "describe_bit",
    "flip_bit_",
    "flip_grad_bit_",
    "largest_grad_index",
    "mantissa_bits",
]
