# Runbook — A2 multi-GPU NCCL bench (DDP ladder · ZeRO-1 · FSDP · all-reduce gloo-vs-nccl)

> **Exec-spec node:** `docs/EXECUTION_SPEC_CS336_FINISH.md` → **W10**. Rental-deferred: written now,
> run when a 2–8×GPU node is rented. **Self-contained** — a fresh agent on a freshly-rented pod runs
> this top-to-bottom with no other context.
>
> **Status when unrun:** the four A2 distributed modules are **proven *correct*** on 2-rank CPU/gloo
> (in-repo tests, green) but **not yet *measured*** at real GPU/NCCL scale. This runbook closes that
> gap and validates the W4 comms-algebra *predictions* against measured wire time. Until it runs, the
> numbers below are `[INFERENCE]` (closed-form predictions); a completed run promotes them to `[FACT]`.

---

## 1. Purpose — what this measures, and which scratch_llm code it exercises

Four benchmarks, each exercising one real module at a scale the CPU tests cannot reach:

| # | Benchmark | Module exercised (real path) | What it measures |
|---|---|---|---|
| **B1** | DDP mode ladder | `src/scratch_llm/utils/ddp.py` — `DDP(module, mode)`, modes `naive`/`flat`/`overlap`, `finish_gradient_synchronization()` | per-step wall time per mode + **% of step in exposed comms**; proves `overlap` hides the all-reduce under backward and `flat` beats `naive` on launch latency |
| **B2** | ZeRO-1 memory + throughput | `src/scratch_llm/utils/zero1.py` — `ShardedOptimizer(params, AdamW, …)`, `.state_numel_on_rank()` | real per-rank **optimizer-state memory** (`torch.cuda.max_memory_allocated`) vs DDP; throughput parity (ZeRO-1 is comms-free memory saving) |
| **B3** | FSDP memory + throughput | `src/scratch_llm/utils/fsdp.py` — `FSDP(module)`, `.finish_gradient_synchronization()`, `.resident_param_numel()`/`.resident_param_bytes()` | resident **param memory between steps** (≈ full/W) vs DDP; throughput cost of the 1.5× wire bytes |
| **B4** | all-reduce 1 MB–1 GB, gloo vs nccl | `src/scratch_llm/utils/comms_calc.py` — `ring_allreduce_bytes`, `ring_allreduce_time`, `ddp_step_bytes`, `fsdp_step_bytes`, `zero1_step_bytes` | achieved **bus bandwidth** of `dist.all_reduce` across sizes/backends; the small-size latency floor that *is* the reason DDP flattens grads |

**The through-line (why all four are one deliverable).** `comms_calc.py` is a pure closed-form cost
model (`docs/design/A2_COMMS_ALGEBRA.md` is generated from it, `tests/test_comms_calc.py` locks it).
Its central identity — ring all-reduce moves `2·(W−1)/W·S` bytes per device, DDP=2 legs, FSDP=3 legs
(1.5×), ZeRO-1=DDP — is **analytical**. B1–B4 put a real NCCL wire under those formulas and check the
predictions hold. That is the W4→W10 validation loop the exec spec calls for.

**These modules are backend-agnostic.** They call `dist.all_reduce` / `all_gather` /
`reduce_scatter_tensor` / `broadcast` with no backend assumption — the in-repo tests drive them on
`gloo`/CPU; here we drive the *same code* on `nccl`/CUDA. Nothing in `utils/` changes.

**Predict-before-run discipline (FOP-2/3).** Every benchmark prints its `comms_calc` prediction
*first* (§5), you write the number down, *then* you measure. A run with no pre-registered prediction
is not a result.

---

## 2. GPU tier — what to rent and why

Two tiers. **Tier A is the default** (cheap, proves every mechanism at W=2). **Tier B** adds the
world-size sweep and real NVLink numbers — rent it only if you want the scaling-vs-W curve.

| | **Tier A — 2× RTX 4090 (default)** | **Tier B — 8× A100 SXM4 80 GB (NVLink)** |
|---|---|---|
| Why | Cheapest node that runs the full ladder + memory ratios + nccl-vs-gloo. Proves *correctness at scale* and the naive→flat→overlap story. | The `W ∈ {2,4,8}` sweep, **real NVLink bus bandwidth** (~250–350 GB/s achieved), and the comms-bound crossover check. |
| Interconnect | **PCIe** (4090 has no NVLink; P2P often routed through host → modest ~6–25 GB/s). NCCL bus bw will be PCIe-limited — *this is honest and expected*; note it. | **NVLink** (600 GB/s theoretical; ~250–350 GB/s achieved on ring all-reduce). |
| World sizes | W=2 only | W=2, 4, 8 (sweep) |
| Runs | B1, B2, B3, B4 at W=2 | all of the above at W=2/4/8 + B4 NVLink curve |
| Rough $ | ~$0.6–1.0/hr for the pair → **~$1–2** for a ≤90-min session | ~$8–13/hr → **~$8–14** for a ~60-min session |

> **Why not H100/B200 here.** This benchmark is about *scaling behaviour and comms mechanics*, not
> peak FLOP/s — a 2× consumer node shows the whole ladder for ~$2. Reserve H100/B200 dollars for the
> A5 RL runs (`A5_qwen_math_rl.md`) where compute peak actually matters.

### 2.1 vastai search + create (per the `vastai` skill — always `--raw`)

One-time auth (skip if `vastai show user` already works):

```bash
vastai set api-key <YOUR_API_KEY>                    # https://console.vast.ai/manage-keys/
vastai create ssh-key ~/.ssh/id_ed25519.pub          # register BEFORE create
vastai show user                                     # verify auth + credit balance
```

**Tier A — 2× RTX 4090:**

```bash
# Search: 2 GPUs on ONE machine, verified, reliable, direct SSH, fast net for model/data pulls.
vastai search offers \
  'gpu_name=RTX_4090 num_gpus=2 reliability>0.98 direct_port_count>=1 inet_down>=200 \
   disk_space>=60 rentable=true verified=true' \
  -o 'dph_total' --raw | jq -r '.[0] | "\(.id)  $\(.dph_total)/hr  \(.num_gpus)x\(.gpu_name)  rel=\(.reliability)"'
# take the offer id from that line:
OFFER=<offer_id>
vastai create instance "$OFFER" \
  --image pytorch/pytorch:@vastai-automatic-tag --disk 60 --ssh --direct --raw
# → {"success": true, "new_contract": <INSTANCE_ID>}
```

**Tier B — 8× A100 SXM (NVLink):**

```bash
# num_gpus=8 on one host; SXM (not PCIe A100) for NVLink; big disk for the venv + torch.
vastai search offers \
  'gpu_name=A100_SXM4 num_gpus=8 reliability>0.98 direct_port_count>=1 inet_down>=500 \
   disk_space>=120 rentable=true verified=true' \
  -o 'dph_total' --raw | jq -r '.[0] | "\(.id)  $\(.dph_total)/hr  \(.num_gpus)x\(.gpu_name)  rel=\(.reliability)"'
OFFER=<offer_id>
vastai create instance "$OFFER" \
  --image pytorch/pytorch:@vastai-automatic-tag --disk 120 --ssh --direct --raw
```

Poll until running, then get the SSH URL:

```bash
INSTANCE=<new_contract>
# wait for actual_status == running (guard against exited/offline — they never reach running):
while true; do
  st=$(vastai show instance "$INSTANCE" --raw | jq -r '.actual_status')
  echo "status: $st"; [ "$st" = running ] && break
  case "$st" in exited|offline|unknown) echo "BAD status $st — destroy + retry a different offer"; break;; esac
  sleep 10
done
vastai ssh-url "$INSTANCE"        # ssh://root@HOST:PORT — connect with: ssh -p PORT root@HOST
```

> **Cost starts now.** Storage bills from *create*; GPU bills from *running*. Destroy the moment
> B1–B4 are captured (§9).

---

## 3. Pod bootstrap — reconstitute the repo + env on fresh metal

Two paths. **3a** is one command if you have the repo's launch tooling and a GitHub token; **3b** is
the manual clone+bootstrap. Either lands a working `.venv` with torch+CUDA.

### 3a. Provision via the repo's launcher (from your laptop)

`deploy/01_launch.sh` already does search→create→wait-SSH→`deploy/provision.sh` (clone the private
repo + `uv pip install -e .[gpu,dev]`). If you rented manually above, just push the repo onto the pod:

```bash
# from your laptop, with gh authed (token is passed to the pod for one command, never written to disk):
scp -P PORT deploy/provision.sh root@HOST:/root/provision.sh
ssh -p PORT root@HOST "REPO=andreidhoang/scratch_llm GITHUB_TOKEN=$(gh auth token) bash /root/provision.sh"
```

### 3b. Manual (on the pod)

```bash
# on the pod:
apt-get update -qq && apt-get install -y -qq git build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
git clone https://<GITHUB_TOKEN>@github.com/andreidhoang/scratch_llm.git /workspace/scratch_llm
cd /workspace/scratch_llm
bash scripts/bootstrap-pod.sh          # uv venv + [dev] deps + torch, re-links hooks, verifies torch+CUDA
source .venv/bin/activate
```

**Torch/arch note (READ THIS).** `scripts/bootstrap-pod.sh` force-installs `torch … --index-url
.../whl/cu130` because the *standing* box is sm120 Blackwell (needs CUDA ≥ 12.8). On the **rented**
node:

- **A100 (sm80)** / **RTX 4090 (sm89)**: the `cu130` wheel supports these arches fine — bootstrap
  as-is works. If you hit any wheel/arch friction, drop the `--index-url` and let uv pick the default
  build: `uv pip install --python .venv/bin/python torch` (default index tracks a current CUDA).
- Verify before benching (this is the go/no-go gate):

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| devices", torch.cuda.device_count(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "need >=2 visible CUDA devices"
PY
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvidia-smi topo -m          # confirm interconnect: NV# = NVLink (Tier B), PHB/PXB/SYS = PCIe (Tier A)
```

Sanity that the modules import and their CPU invariants still hold on this box (fast, ~30 s):

```bash
pytest -m "not gpu" tests/test_ddp.py tests/test_zero1.py tests/test_fsdp.py tests/test_comms_calc.py -q
```

**No dataset or model download needed.** B1–B4 use a from-scratch `TransformerLM` on random token
ids — the comms cost is set by *parameter count*, not data. Everything runs from the repo alone.

---

## 4. The benchmark harness — write these to a throwaway dir on the pod

> **Hard constraint (do NOT violate).** These are **throwaway** scripts. Write them **outside the
> committed tree** at `/root/mgbench/` and **never `git add` them**. Do not edit anything under
> `src/`, `tests/`, `pyproject.toml`, `performance/`, or `serving/`. The only committed change this
> node produces is an **additive** results block in `bench/RESULTS.md` (§8), made on the standing box.

```bash
mkdir -p /root/mgbench && cd /root/mgbench
```

All scripts launch under `torchrun` (which sets `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/
`MASTER_PORT`); each `init_process_group`s and pins `cuda:LOCAL_RANK`.

### 4.0 Shared config + init (`common.py`)

```python
# /root/mgbench/common.py — shared model config, dist init, CUDA-event timing.
import os, torch, torch.distributed as dist
from scratch_llm.model import ModelConfig, TransformerLM, cross_entropy

# Default model: ~369M params, fp32 weights ~1.48 GB. Tiny for a 24/80 GB card on purpose —
# leaves room for optimizer state + activations and makes the memory RATIOS clean. To make the
# FSDP/ZeRO memory story starker on a big card, bump to d_model=2048,n_layers=32 (~1.6B).
CFG = ModelConfig(vocab_size=32000, d_model=1024, n_layers=24, n_heads=16, context_length=1024)
SEQ = 1024
MICRO_BS = 8            # per-GPU micro-batch

def setup():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    torch.manual_seed(0)          # identical init on every rank (DDP/FSDP also broadcast rank0)
    return dist.get_rank(), dist.get_world_size(), local

def make_model(device):
    return TransformerLM(CFG).to(device)

def batch(device):
    x = torch.randint(0, CFG.vocab_size, (MICRO_BS, SEQ), device=device)
    y = torch.randint(0, CFG.vocab_size, (MICRO_BS, SEQ), device=device)
    return x, y

def loss_of(model, x, y):
    return cross_entropy(model(x), y)

def timed(fn, iters=20, warmup=5):
    """Median ms of fn() over iters, CUDA-event timed, barrier-fenced. fn does one full step."""
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); dist.barrier()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        dist.barrier(); s.record()
        fn()
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times)//2]
```

### 4.1 B1 — DDP mode ladder (`b1_ddp.py`)

Methodology: measure the full step per mode, and a **comms-free compute floor** (raw unwrapped model,
no collectives). `exposed_comms = step − compute_floor`; `%comms = exposed/step`. Story to confirm:
`naive` has the highest %comms (one collective *per param* → launch-latency bound), `flat` lower (one
collective), `overlap` lowest (all-reduce fired from grad hooks, hidden under the still-running
backward).

```python
# /root/mgbench/b1_ddp.py  —  torchrun --standalone --nproc_per_node=N b1_ddp.py
import copy, torch
from torch.distributed import all_reduce, ReduceOp
from scratch_llm.utils.ddp import DDP
from scratch_llm.optim import AdamW
from common import setup, make_model, batch, loss_of, timed

rank, world, local = setup()
dev = torch.device("cuda", local)
base = make_model(dev)
x, y = batch(dev)

# --- comms-free compute floor: raw model, no DDP, no collective ---
raw = copy.deepcopy(base); opt = AdamW(raw.parameters(), lr=1e-4)
def compute_step():
    opt.zero_grad(set_to_none=True); loss_of(raw, x, y).backward(); opt.step()
compute_ms = timed(compute_step)

rows = {}
for mode in ("naive", "flat", "overlap"):
    ddp = DDP(copy.deepcopy(base), mode=mode)
    opt = AdamW(ddp.module.parameters(), lr=1e-4)
    def step():
        opt.zero_grad(set_to_none=True)
        loss_of(ddp, x, y).backward()
        ddp.finish_gradient_synchronization()   # naive/flat: all-reduce here; overlap: wait handles
        opt.step()
    step_ms = timed(step)
    rows[mode] = step_ms

if rank == 0:
    print(f"[B1] world={world}  compute_floor={compute_ms:.2f} ms/step")
    for mode, step_ms in rows.items():
        exposed = max(step_ms - compute_ms, 0.0)
        print(f"[B1] {mode:8s} step={step_ms:7.2f} ms  exposed_comms={exposed:7.2f} ms  %comms={100*exposed/step_ms:5.1f}%")
```

Launch (N = number of GPUs, 2 for Tier A; 2/4/8 for Tier B):

```bash
torchrun --standalone --nproc_per_node=2 b1_ddp.py
# Tier B sweep:
for N in 2 4 8; do echo "== W=$N =="; torchrun --standalone --nproc_per_node=$N b1_ddp.py; done
```

### 4.2 B2 — ZeRO-1 optimizer-state memory + throughput (`b2_zero1.py`)

Confirms `ShardedOptimizer` shards Adam moment state to ~1/W per rank at **no throughput cost** (same
wire bytes as DDP — the module's central claim). Memory measured with `torch.cuda.max_memory_allocated`.

```python
# /root/mgbench/b2_zero1.py  —  torchrun --standalone --nproc_per_node=N b2_zero1.py
import copy, torch, torch.distributed as dist
from scratch_llm.utils.ddp import DDP
from scratch_llm.utils.zero1 import ShardedOptimizer, optimizer_state_numel
from scratch_llm.optim import AdamW
from common import setup, make_model, batch, loss_of, timed

rank, world, local = setup()
dev = torch.device("cuda", local)
base = make_model(dev)
x, y = batch(dev)

def run(make_opt, label):
    ddp = DDP(copy.deepcopy(base), mode="overlap")
    opt = make_opt(ddp.module)
    def step():
        opt.zero_grad(set_to_none=True); loss_of(ddp, x, y).backward()
        ddp.finish_gradient_synchronization(); opt.step()
    step()  # materialize Adam state
    torch.cuda.reset_peak_memory_stats()
    ms = timed(step)
    peak = torch.cuda.max_memory_allocated() / 1e9
    state_numel = opt.state_numel_on_rank() if hasattr(opt, "state_numel_on_rank") else optimizer_state_numel(opt)
    if rank == 0:
        print(f"[B2] {label:8s} step={ms:7.2f} ms  peak_mem={peak:6.2f} GB  opt_state/rank={state_numel*8/1e9:6.3f} GB")

run(lambda m: AdamW(m.parameters(), lr=1e-4), "DDP")
run(lambda m: ShardedOptimizer(m.parameters(), AdamW, lr=1e-4), "ZeRO-1")
```

```bash
torchrun --standalone --nproc_per_node=2 b2_zero1.py
```

### 4.3 B3 — FSDP resident-param memory + throughput (`b3_fsdp.py`)

Confirms `FSDP` holds only ~full/W params **between steps** (`resident_param_bytes()`), for the price
of 1.5× DDP wire bytes. Note the minimal container's documented behaviour: peak **during** fwd/bwd =
full + shards (it keeps the gathered weights resident through backward — see the module docstring).

```python
# /root/mgbench/b3_fsdp.py  —  torchrun --standalone --nproc_per_node=N b3_fsdp.py
import copy, torch
from scratch_llm.utils.ddp import DDP
from scratch_llm.utils.fsdp import FSDP
from scratch_llm.optim import AdamW
from common import setup, make_model, batch, loss_of, timed

rank, world, local = setup()
dev = torch.device("cuda", local)
base = make_model(dev)
full_numel = sum(p.numel() for p in base.parameters())
x, y = batch(dev)

# DDP reference
ddp = DDP(copy.deepcopy(base), mode="overlap"); opt = AdamW(ddp.module.parameters(), lr=1e-4)
def ddp_step():
    opt.zero_grad(set_to_none=True); loss_of(ddp, x, y).backward()
    ddp.finish_gradient_synchronization(); opt.step()
ddp_step(); torch.cuda.reset_peak_memory_stats(); ddp_ms = timed(ddp_step)
ddp_peak = torch.cuda.max_memory_allocated()/1e9

# FSDP
fsdp = FSDP(copy.deepcopy(base)); fopt = AdamW(fsdp.parameters(), lr=1e-4)
def fsdp_step():
    fopt.zero_grad(set_to_none=True); loss_of(fsdp, x, y).backward()
    fsdp.finish_gradient_synchronization(); fopt.step()
fsdp_step()
resident_between = fsdp.resident_param_bytes()/1e9   # ≈ full/W
torch.cuda.reset_peak_memory_stats(); fsdp_ms = timed(fsdp_step)
fsdp_peak = torch.cuda.max_memory_allocated()/1e9

if rank == 0:
    print(f"[B3] world={world} full_params={full_numel/1e6:.1f}M ({full_numel*4/1e9:.2f} GB fp32)")
    print(f"[B3] DDP  step={ddp_ms:7.2f} ms  peak_mem={ddp_peak:6.2f} GB")
    print(f"[B3] FSDP step={fsdp_ms:7.2f} ms  peak_mem={fsdp_peak:6.2f} GB  resident_params/rank={resident_between:.3f} GB (≈ full/{world})")
```

```bash
torchrun --standalone --nproc_per_node=2 b3_fsdp.py
```

### 4.4 B4 — all-reduce 1 MB–1 GB, nccl vs gloo (`b4_allreduce.py`)

The bandwidth microbench that validates `ring_allreduce_bytes`/`ring_allreduce_time`. Per size we
report **algorithm BW** = `S/time` and **bus BW** = `2·(W−1)/W·S / time` (the per-device egress the
ring formula predicts). Small sizes are latency-bound (the α term) — that floor is *why* DDP's `flat`
mode buckets grads. Run once per backend (nccl on CUDA tensors, gloo on CPU tensors).

```python
# /root/mgbench/b4_allreduce.py  —  torchrun --standalone --nproc_per_node=N b4_allreduce.py --backend {nccl|gloo}
import os, sys, time, torch, torch.distributed as dist
from scratch_llm.utils.comms_calc import ring_allreduce_bytes

backend = "nccl"
if "--backend" in sys.argv: backend = sys.argv[sys.argv.index("--backend")+1]
local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local)
dist.init_process_group(backend)
rank, world = dist.get_rank(), dist.get_world_size()
dev = torch.device("cuda", local) if backend == "nccl" else torch.device("cpu")

SIZES = [1<<20, 1<<22, 1<<24, 1<<26, 1<<28, 1<<30]   # 1MB … 1GB
for nbytes in SIZES:
    n = nbytes // 4
    t = torch.ones(n, dtype=torch.float32, device=dev)
    for _ in range(5): dist.all_reduce(t)             # warmup
    if backend == "nccl": torch.cuda.synchronize()
    dist.barrier(); t0 = time.perf_counter()
    reps = 20
    for _ in range(reps): dist.all_reduce(t)
    if backend == "nccl": torch.cuda.synchronize()
    dist.barrier(); dt = (time.perf_counter() - t0) / reps
    if rank == 0:
        alg = nbytes / dt / 1e9
        bus = ring_allreduce_bytes(nbytes, world) / dt / 1e9   # per-device egress / time
        print(f"[B4:{backend}] W={world} S={nbytes/1e6:8.1f}MB  t={dt*1e3:9.3f}ms  algBW={alg:7.2f} GB/s  busBW={bus:7.2f} GB/s")
dist.destroy_process_group()
```

```bash
torchrun --standalone --nproc_per_node=2 b4_allreduce.py --backend nccl
torchrun --standalone --nproc_per_node=2 b4_allreduce.py --backend gloo
# Tier B: also sweep W and get the NVLink curve
for N in 2 4 8; do torchrun --standalone --nproc_per_node=$N b4_allreduce.py --backend nccl; done
```

---

## 5. Predict-before-run — write these down FIRST

Run the prediction cell, **record the outputs**, *then* run §4. All numbers come straight from
`comms_calc` (no free parameters) — the benchmark's job is to falsify or confirm them.

```bash
cd /workspace/scratch_llm && python - <<'PY'
import torch
from scratch_llm.model import ModelConfig, TransformerLM
from scratch_llm.utils.comms_calc import (ddp_step_bytes, fsdp_step_bytes, zero1_step_bytes,
    ring_allreduce_bytes, ring_allreduce_time)
CFG = ModelConfig(vocab_size=32000, d_model=1024, n_layers=24, n_heads=16, context_length=1024)
P = sum(p.numel() for p in TransformerLM(CFG).parameters())
print(f"params={P/1e6:.1f}M  fp32 weights={P*4/1e9:.2f} GB  Adam state (8B/param)={P*8/1e9:.2f} GB")
for W in (2,4,8):
    d,f,z = (fn(P, torch.float32, W)/1e9 for fn in (ddp_step_bytes, fsdp_step_bytes, zero1_step_bytes))
    print(f"W={W}: DDP wire={d:.3f}GB  FSDP wire={f:.3f}GB (={f/d:.2f}x)  ZeRO-1 wire={z:.3f}GB  | opt-state/rank ZeRO&FSDP={P*8/1e9/W:.3f}GB vs DDP {P*8/1e9:.3f}GB")
for S,l in ((1e6,'1MB'),(16e6,'16MB'),(64e6,'64MB'),(256e6,'256MB'),(1e9,'1GB')):
    print(f"all-reduce S={l:>5} egress/dev(W=8)={ring_allreduce_bytes(S,8)/1e6:7.1f}MB  "
          f"t@NVLink250={ring_allreduce_time(S,8,250e9,latency=8e-6)*1e3:7.3f}ms  "
          f"t@PCIe12={ring_allreduce_time(S,8,12e9,latency=8e-6)*1e3:7.3f}ms  "
          f"t@gloo2={ring_allreduce_time(S,8,2e9,latency=8e-6)*1e3:7.3f}ms")
PY
```

### 5.1 Pre-registered predictions (the 369M default model, fp32 grads)

**Wire bytes / step (from `comms_calc`, exact):**

| W | DDP wire | FSDP wire | ZeRO-1 wire | Adam-state/rank (ZeRO/FSDP) | Adam-state/rank (DDP) |
|---|---|---|---|---|---|
| 2 | 1.477 GB | 2.215 GB (1.50×) | 1.477 GB | 1.477 GB | 2.953 GB |
| 4 | 2.215 GB | 3.322 GB (1.50×) | 2.215 GB | 0.738 GB | 2.953 GB |
| 8 | 2.584 GB | 3.876 GB (1.50×) | 2.584 GB | 0.369 GB | 2.953 GB |

**All-reduce predicted time (α–β model, α=8 µs, W=8):**

| S | egress/dev | NVLink @250 GB/s | PCIe @12 GB/s | gloo @2 GB/s |
|---|---|---|---|---|
| 1 MB | 1.75 MB | 0.12 ms | 0.26 ms | 0.99 ms |
| 16 MB | 28 MB | 0.22 ms | 2.4 ms | 14.1 ms |
| 64 MB | 112 MB | 0.56 ms | 9.4 ms | 56.1 ms |
| 256 MB | 448 MB | 1.90 ms | 37.4 ms | 224 ms |
| 1 GB | 1750 MB | 7.11 ms | 146 ms | 875 ms |

*(NVLink/PCIe/gloo bandwidths are order-of-magnitude placeholders — measure the actual link BW from
B4's large-size `busBW` column and re-plug it; the point is the shape + the ratios, not the exact ms.)*

### 5.2 Qualitative predictions per benchmark

- **B1:** `%comms(naive) > %comms(flat) > %comms(overlap)`. `overlap` exposed-comms should be small
  vs `naive` (all-reduce hidden under backward). On **Tier A / PCIe**, comms may stay partly exposed
  even for `overlap` (grads ~1.5 GB over a slow PCIe link don't fully hide under a 369M backward) —
  that is a legitimate **documented negative**, not a bug. On **Tier B / NVLink**, `overlap` should
  approach the compute floor.
- **B2:** ZeRO-1 `peak_mem` and `opt_state/rank` drop ~1/W vs DDP; **throughput within noise of DDP**
  (ZeRO-1 wire bytes == DDP wire bytes — comms-free memory saving). At W=2, opt-state/rank halves
  (2.95→1.48 GB).
- **B3:** FSDP `resident_params/rank` ≈ full/W (0.74 GB at W=2 for the 369M model); FSDP step time
  ≥ DDP (pays the 1.5× wire) — the gap widens as comms-bound. `peak_mem` savings are modest for the
  minimal container (it keeps full weights resident through backward — expected per docstring).
- **B4:** large-size `busBW` should be **flat across W** (the ring bound is per-device-bytes-bounded)
  and approach the node's link peak; **nccl ≫ gloo** (≥5× on NVLink, still clearly faster on PCIe);
  small sizes (1–16 MB) are **latency-bound** — algBW far below peak, `nccl` latency ~tens of µs,
  `gloo` ~hundreds of µs to ms. This latency floor is the `comms_calc` α term made real.

### 5.3 Kill / investigate criteria (pre-committed)

| Signal | Verdict |
|---|---|
| B1 `%comms(overlap) ≥ %comms(naive)` on **NVLink** | overlap not hiding — hooks not firing / not async / backend sync. Investigate `register_post_accumulate_grad_hook` path. |
| B2 ZeRO-1 `opt_state/rank` **not** ≈ DDP/W (beyond largest-param slack) | `partition_by_numel` bug — greedy assignment not covering/balancing. |
| B2 ZeRO-1 throughput **>20 % slower** than DDP | the per-param param-republish `broadcast` isn't hiding — expected small on NVLink; if large, that's the finding to log. |
| B3 FSDP `resident_params/rank` **not** ≈ full/W | shard accounting / `.data`-swap bug — the module's own test asserts this on CPU; a GPU miss = device-path regression. |
| B4 `busBW > measured p2p link peak` | measurement bug (impossible physically) — fix timing before trusting any row. |
| B4 nccl not **≥2×** gloo at 1 GB | NCCL not using the fast link — check `nvidia-smi topo -m`, `NCCL_P2P_DISABLE`, `NCCL_DEBUG=INFO`. |

---

## 6. Validate the W4 comms-algebra against measured (the point of the exercise)

After B1–B4, fill this table — it is the W4→W10 closure the exec spec asks for:

| `comms_calc` prediction | How to check against measured | Pass condition |
|---|---|---|
| `ring_allreduce_bytes = 2(W−1)/W·S` | B4 `busBW` flat across W at large S | busBW(W=2)≈busBW(W=4)≈busBW(W=8) within ~15 % |
| `fsdp_step_bytes = 1.5·ddp_step_bytes` | B3 FSDP-vs-DDP step-time gap in the comms-bound regime; B4 link BW × the two byte counts | FSDP exposed-comms ≈ 1.5× DDP exposed-comms |
| `zero1_step_bytes = ddp_step_bytes` | B2 ZeRO-1 vs DDP throughput | within measurement noise (~±10 %) |
| α-term (latency) dominates small S | B4 small-S algBW ≪ large-S algBW; `flat`<`naive` in B1 | monotone: algBW rises with S; `flat` beats `naive` |
| ZeRO-1/FSDP state = full/W | B2/B3 measured `opt_state`/`resident_params` per rank | ≈ full/W within padding + largest-param slack |

If a prediction **misses**, the honest output is a `[FACT]` measured row + a one-line root cause
(e.g. "PCIe P2P disabled → NCCL routed through host, busBW 5 GB/s not the PCIe4 25 GB/s peak; ring
formula still holds, the *link* is the limiter"). A documented miss with a cause is a result; a
silent green is not.

---

## 7. Full run sequence (copy-paste order)

```bash
# on the pod, after §3 bootstrap + go/no-go gate:
cd /workspace/scratch_llm && source .venv/bin/activate
python - <<'PY'  # §5 predictions — SAVE THIS OUTPUT
# (paste the §5 prediction cell here)
PY
mkdir -p /root/mgbench && cd /root/mgbench
# (write common.py, b1_ddp.py, b2_zero1.py, b3_fsdp.py, b4_allreduce.py from §4)
export PYTHONPATH=/root/mgbench:$PYTHONPATH          # so scripts can `import common`
export NCCL_DEBUG=WARN                               # set INFO if B4 nccl looks wrong

N=2                                                  # =GPU count; Tier B: loop 2 4 8
torchrun --standalone --nproc_per_node=$N b1_ddp.py     | tee /root/mgbench/b1_W$N.txt
torchrun --standalone --nproc_per_node=$N b2_zero1.py   | tee /root/mgbench/b2_W$N.txt
torchrun --standalone --nproc_per_node=$N b3_fsdp.py    | tee /root/mgbench/b3_W$N.txt
torchrun --standalone --nproc_per_node=$N b4_allreduce.py --backend nccl | tee /root/mgbench/b4_nccl_W$N.txt
torchrun --standalone --nproc_per_node=$N b4_allreduce.py --backend gloo | tee /root/mgbench/b4_gloo_W$N.txt
cat /root/mgbench/b*_W$N.txt                          # collect for transcription
```

Pull the logs back to the standing box for the write-up (or just copy the printed numbers):

```bash
# from the standing box / laptop:
vastai copy <INSTANCE>:/root/mgbench/ local:./_mgbench_results/
```

---

## 8. Where results go — `bench/RESULTS.md` (main-track section, additive)

Record predicted-vs-measured in **`bench/RESULTS.md`** under **"## Main track (CS336 A2→A5) — analysis
results"** as a new subsection `### W10 · A2 multi-GPU NCCL bench`. **Do this on the standing checkout**
(the committed repo), as an **append-only** edit — never edit a logged row, and this runbook itself
commits nothing (the standing-box agent does the additive edit + green-CI commit). Use the ledger's
column shape:

```
| date | node / artifact | hardware | metric | predicted (pre-registered) | measured | bound | note |
```

Log at minimum: B1 `%comms` per mode, B2/B3 memory ratios + throughput deltas, B4 nccl-vs-gloo busBW
at 1 GB and the small-size latency floor, and the §6 validation verdicts. Mark each row `[FACT]`
(measured under `cuda.synchronize` + barrier + warmup) and note the interconnect (NVLink vs PCIe) —
the honesty rail: report "% of *this* node," never imply datacenter numbers. Then tick **W10**'s
`A2_multigpu_nccl_bench.md` line in `docs/EXECUTION_SPEC_CS336_FINISH.md` and add a MASTERY_DEBT row
if the run surfaced anything.

---

## 9. Teardown — stop the meter

```bash
vastai destroy instance <INSTANCE> -y     # stops ALL billing; disk wiped. Do this the moment §7 is captured.
vastai show instances --raw | jq -r '.[] | "\(.id) \(.actual_status)"'   # confirm nothing is still running
```

---

## 10. Cost estimate

| Tier | Node | $/hr (typ.) | Session | **Total** |
|---|---|---|---|---|
| **A (default)** | 2× RTX 4090 (PCIe) | ~$0.6–1.0 | ≤90 min (bootstrap + B1–B4 @ W=2) | **~$1–2** |
| **B (scaling)** | 8× A100 SXM 80 GB (NVLink) | ~$8–13 | ~60 min (B1–B4 × W∈{2,4,8}) | **~$8–14** |

Bootstrap (venv + torch download) is ~10–15 min of that — the benches themselves are minutes. Budget
**~$2 for the default run, ~$15 if you also take the 8× scaling curve.** Destroy immediately after
capture (§9); storage bills even when stopped, so `destroy`, don't `stop`, once the numbers are in
`bench/RESULTS.md`.

---

### Appendix — quick failure triage

- `torchrun: command not found` → `python -m torch.distributed.run …` (same flags), or activate the venv.
- NCCL hang at init → set `NCCL_DEBUG=INFO`; on some hosts `NCCL_P2P_DISABLE=1` unblocks (then busBW
  reflects the fallback path — note it). Check `nvidia-smi topo -m` for the real interconnect.
- `RuntimeError: CUDA error: no kernel image` → torch wheel arch mismatch (§3 torch note): reinstall
  torch from the default index for this node's GPU.
- gloo run tries to use CUDA → confirm `b4_allreduce.py --backend gloo` puts tensors on **cpu** (it does).
- OOM on a smaller card → drop `MICRO_BS` or `d_model`/`n_layers` in `common.py`; the comms *ratios*
  are size-independent (they depend on param count only via the same formula on both sides).
```
