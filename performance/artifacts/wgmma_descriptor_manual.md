# WGMMA hand-decode reference — `wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16`

> **A3 PTX artifact** (deliverable §8.2). A paper exercise (no H100 required) that decodes one
> Hopper warpgroup-MMA instruction token-by-token and hand-encodes the 64-bit SMEM matrix descriptor
> it consumes. Written for a frontier-lab kernel peer. Purpose: when we drop to inline PTX in Phase 2
> Rung 3 (`performance/A3_tensor_cores.md` §3 Rung 3.1), the descriptor world and the qualifier soup
> are already legible — no hieroglyphics.
>
> **Claim discipline** (FOP-4): `[FACT]` = verified against a primary source (PTX ISA text, CUTLASS
> source) or against compiler output I generated on this box; `[INFERENCE]` = reconstructed geometry
> I have not byte-checked against silicon; `[VERIFY-PHASE2]` = to be diffed against `nvcc`/CUTLASS on
> the H100.
>
> **Ground-truth note.** Every qualifier string, operand-vector width, and the fence protocol below
> was **emitted by `nvcc -arch=sm_90a -ptx` (CUDA 13.0, PTX ISA 9.0) on the standing sm_120 box**
> (PTX *emission* is host-side and arch-checked only for text, so no Hopper GPU is needed to produce
> it — only to *run* it). The probe and its output are reproduced in §5. So the decode is not from
> memory; it is checked against the compiler. The one part still `[INFERENCE]` is the *numeric* value
> of the LBO/SBO fields in §2's worked example — those are byte-checked in Phase 2 against
> `cute::make_gmma_desc`.

Primary sources: **NVIDIA PTX ISA** §9.7.16 "Asynchronous Warpgroup Level Matrix Multiply-Accumulate
Instructions" — subsections `wgmma.mma_async` (§9.7.16.5.x), "Matrix Descriptor" (§9.7.16.4.1),
"Shared Memory Matrix Layout" / swizzling (§9.7.16.4.2–4.3), `wgmma.fence` / `wgmma.commit_group` /
`wgmma.wait_group` (§9.7.16.5.2–5.4). **CUTLASS**: `GmmaDescriptor` union in
`include/cute/arch/mma_sm90_desc.hpp` (older tree: `include/cute/atom/mma_traits_sm90_gmma.hpp`);
`make_gmma_desc` in `include/cute/atom/mma_traits_sm90_gmma.hpp`. **Colfax Research**, "WGMMA on
Hopper" (the `make_smem_desc` helper). (PTX ISA subsection numbers drift by release — locate by title.)

---

## 0. TL;DR (the whole card on one screen)

```
wgmma . mma_async . sync . aligned . m64n64k16 . f32 . f16 . f16
  |        |          |       |          |         |     |    |
  |        |          |       |          |         |     |    +-- B operand elem type = f16
  |        |          |       |          |         |     +------- A operand elem type = f16
  |        |          |       |          |         +------------- D (accumulator) + C type = f32
  |        |          |       |          +----------------------- tile M=64(fixed) N=64 K=16(=32B, f16)
  |        |          |       +---------------------------------- all 128 threads converged & active
  |        |          +------------------------------------------ warpgroup-collective (128 threads)
  |        +----------------------------------------------------- async: D valid only after wait_group
  +-------------------------------------------------------------- warpgroup MMA (4 warps cooperate)

operands (compiler-confirmed, §5):  D = {32 x .f32}   A = SMEM-desc OR {4 x .b32}   B = SMEM-desc (only)
per-thread accumulator regs = 64*64/128 = 32 ; warpgroup total = 4096 FP32 regs (one full 64x64 D tile)
FMAs retired by this one instruction = 64*64*16 = 65,536

64-bit SMEM descriptor field map (bit ranges, CUTLASS GmmaDescriptor):
  [0 :14)  start_address   (byte addr of tile >> 4 ; 16B-granular)
  [14:16)  --unused--
  [16:30)  leading_byte_offset  LBO (bytes >> 4)
  [30:32)  --unused--
  [32:46)  stride_byte_offset   SBO (bytes >> 4)
  [46:49)  --unused--
  [49:52)  base_offset (3 bits ; swizzle phase, usually 0)
  [52:62)  --unused--
  [62:64)  layout_type / swizzle : 0=none 1=128B 2=64B 3=32B

worked example (128x64 f16 tile, 128B swizzle, base@SMEM 0x2000):
  start>>4 = 0x2000>>4 = 0x200   LBO=16B ->1   SBO=1024B ->64   swizzle=1
  => 64-bit descriptor = 0x4000004000010200
```

---

## 1. Full qualifier decode

The general syntax of the SMEM-operand form (PTX ISA §9.7.16.5.1) is:

```
wgmma.mma_async.sync.aligned.shape.dtype.atype.btype  d, a-desc, b-desc,
      scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b ;
```

Dot-separated qualifiers, left to right. `[FACT]` for every token — the string below is exactly what
`nvcc` emitted (§5).

### `wgmma` — warpgroup matrix-multiply-accumulate
The instruction class. A **warpgroup** is **4 contiguous warps = 128 threads** (warp 0..3 of a warp
tuple, `warpid % 4`). All 128 threads issue this *one* instruction and *cooperate* on a single MMA
whose operand tiles are far larger than any one thread (or one warp) could hold. This is the
issue-thread-count step of the throughline: `mma.sync` was warp(32)-collective; `wgmma` is
warpgroup(128)-collective; `tcgen05` will be single-thread(1). Widening the collective is how each
generation raised FMAs-per-issue (here **65,536 FMAs** = 64·64·16 for one instruction). `[FACT]`

### `.mma_async` — asynchronous multiply-accumulate
The op is `D = A·B + D` (or `A·B`, see `scale-d`), issued **asynchronously**. The instruction
*returns immediately*; the tensor core reads A/B and writes the accumulator registers **in the
background**, on its own timeline. The consequence you must respect: **the D registers are NOT valid
when the next instruction executes** — they become valid only after a matching
`wgmma.wait_group` retires the group (§1.9). This is the `sync → async` spine landing: `mma.sync`
blocked all 32 threads until the MMA retired; `wgmma.mma_async` decouples issue from completion, which
is *precisely* what lets a producer TMA copy overlap the consumer MMA in a software pipeline. `[FACT]`

### `.sync` — warpgroup barrier semantics
The instruction is **collective over the warpgroup**: it has an implied synchronization such that all
128 threads must reach it. It is *not* the same as `bar.sync` (which blocks); here `.sync` asserts the
warpgroup-granular contract (issue is a warpgroup event). Combined with `.async`, the mental model is:
"the 128 threads collectively *launch* the MMA and move on; the *completion* is reclaimed later by
`wait_group`." `[FACT]`

### `.aligned` — converged, all-threads-active
All 128 threads of the warpgroup must execute this instruction with **converged control flow** (same
PC, none predicated off, none exited). It is an *assertion* the programmer/compiler makes; violating
it (issuing wgmma inside a divergent branch, or with a dead lane) is **undefined behavior**, not a
diagnosed error. This is why warp-specialized kernels keep the *entire consumer warpgroup* on the
mainloop path and never let an early-exit thin it. `[FACT]`

### `.m64n64k16` — the tile shape M×N×K
- **M = 64, always.** `[FACT]` Every `wgmma.mma_async` has M fixed at 64. The output D tile is 64 rows
  tall; that is why the accumulator is exactly `64·N/128` regs/thread and why the warpgroup (128 = 2·64)
  is the natural owner.
- **N = 64 here; N ∈ {8, 16, 24, …, 256}, multiple of 8.** `[FACT]` N is the free knob. Larger N =
  more FMAs amortised per descriptor fetch (Rung 3.2 walks N: 64 → 128 → 256, book 318 → 433 TFLOPS).
  N=256 is the largest atom (128 accum regs/thread — near the 255-reg ceiling, which is *why* 256 is
  the max).
- **K = 16 here; K is set by the operand dtype, not free.** `[FACT]` For 16-bit operands (f16/bf16)
  K=16 (= 32 bytes per row of the K-strip). For 8-bit (f8/int8) K=32; for 1-bit K=256. The rule is
  "K is whatever makes the K-strip a fixed byte width the core matrix tiles cleanly." So `m64n64k16`
  is the f16/bf16 shape; the FP8 Rung-4 kernel is `m64n64k32`.

FMAs per instruction = M·N·K = 64·64·16 = **65,536**. `[FACT]`

### `.f32` — accumulator (D) and addend (C) type
D lives in **registers** as **FP32**, and the accumulation is done in FP32 regardless of the f16
inputs. `[FACT]` This is the load-bearing mixed-precision fact (§5 numerics of the curriculum): inputs
are f16 for bandwidth/throughput, the reduction is FP32 for numerical stability across a long K. The
alternative `.f16` accumulator exists but bleeds accuracy as K grows — the curriculum proves it by
watching the error curve. On Hopper, `wgmma` supports `.f32` (and `.f16`) accumulators for f16 inputs;
FP8 inputs accumulate in `.f32` only.

### `.f16.f16` — A operand type, then B operand type
`atype` = **f16** (A), `btype` = **f16** (B), in that order. `[FACT]` If you fed bf16 it would read
`.bf16.bf16`; mixed A/B dtypes are only legal in the combinations the ISA table lists (e.g. some
f8 e4m3/e5m2 mixes). Here both are plain IEEE half.

### The trailing operands (not part of the `.`-qualifier string, but part of the instruction)
From the emitted `... d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;`:
- **`scale-d`** (imm 0/1): 0 ⇒ `D = A·B` (overwrite, ignore incoming D); 1 ⇒ `D = A·B + D`
  (accumulate). `[FACT]` In a K-loop you set 0 on the first K-tile and 1 thereafter — that is how the
  accumulator is zero-initialised *for free* without touching the registers.
- **`imm-scale-a`, `imm-scale-b`** (imm ±1): negate the operand (+1 keep, −1 negate). `[FACT]`
- **`imm-trans-a`, `imm-trans-b`** (imm 0/1): transpose the SMEM operand as it is read (the descriptor
  "walk direction"). `[FACT]` **These exist only for SMEM operands.** In the A-in-registers form the
  `imm-trans-a` slot disappears (RF operands can't be transposed on read) — see §1.8, confirmed by the
  compiler dropping it to 4 trailing immediates.

### 1.8 Where the operands may live `[FACT]`
| Operand | Legal homes | Passed as |
|---|---|---|
| **A** | SMEM **or** registers | 64-bit SMEM descriptor **or** `{4 × .b32}` fragment (8 halves/thread) |
| **B** | **SMEM only** — never registers | 64-bit SMEM descriptor |
| **D** (+ C) | **registers only** | `{32 × .f32}` vector (for m64n64) |

B *must* come from SMEM: the whole point of `wgmma` is that the tensor core walks B directly out of
shared memory via the descriptor, so B never occupies register file. A is allowed in RF for kernels
that keep the left operand resident (e.g. A produced by a previous op), but the canonical GEMM keeps
both in SMEM and passes two descriptors. The compiler confirmed both forms (§5): the SS form takes
`%rd1, %rd2` (two descriptors); the RS form takes `{%r1,%r2,%r3,%r4}, %rd3` (A fragment + B descriptor).

### 1.9 The accumulator register budget `[FACT]`
Per thread: `M·N / 128 = 64·64 / 128 = 32` FP32 registers. The compiler allocated exactly
`%f65 … %f96` = **32 `.f32` registers** in the operand vector (§5). Across the warpgroup: `32 × 128 =
4096` FP32 registers = one full 64×64 D tile (4096 elements), one element per (thread, reg) slot. This
is the RF pressure that caps N: `m64n256` needs 128 regs/thread — right up against the 255-register
architectural limit, which is *why* N≤256.

### 1.10 The fence / commit / wait protocol (the async glue) `[FACT]`
`wgmma.mma_async` is useless without three companion instructions (all emitted verbatim by the
compiler, §5):

```
wgmma.fence.sync.aligned ;            // (A) order prior register writes before the async reads
   ... write/zero the D regs, load A-frag regs if RF ...
for k in K-tiles:
    wgmma.mma_async ... scale-d=(k>0) ;   // (B) issue; accumulate after the first tile
wgmma.commit_group.sync.aligned ;     // (C) seal all wgmma since last commit into ONE group
wgmma.wait_group.sync.aligned N ;     // (D) block until <= N groups remain in flight
   ... now the D regs are valid to read ...
```

- **`wgmma.fence.sync.aligned`** — a *register* ordering fence. It guarantees that writes the thread
  made to the accumulator regs (and to A-fragment regs, if RF) *before* the fence are visible to the
  async wgmma that reads them *after*. Needed (1) once before the first wgmma of a warpgroup, and (2)
  again any time non-wgmma code touches the accumulator regs between groups. Omit it and the tensor
  core may read stale registers — a silent-garbage bug. `[FACT]`
- **`wgmma.commit_group.sync.aligned`** — bundles every `wgmma.mma_async` issued since the last commit
  into a single **wgmma-group** and pushes it onto the async pipeline. `[FACT]`
- **`wgmma.wait_group.sync.aligned N`** — blocks the warpgroup until **at most N** committed groups are
  still in flight. `wait_group 0` ⇒ wait for all (D fully valid). **N>0 is the overlap knob**: leave N
  groups running while you start the next tile's work — this is exactly how FlashAttention-3 keeps the
  previous MMA in flight while it does the softmax. After `wait_group` returns, and only then, the D
  registers of the completed groups may be read. `[FACT]`

---

## 2. The 64-bit SMEM matrix descriptor — bit-field by bit-field, with a worked encoding

A `wgmma` SMEM operand is not a pointer — it is a **64-bit descriptor** that tells the tensor core how
to *walk* a tile out of shared memory: where it starts, how far to step between the sub-tiles it reads
("core matrices"), and which swizzle to un-permute. `[FACT]` PTX ISA §9.7.16.4.1.

### 2.1 The field map `[FACT]` (from CUTLASS `GmmaDescriptor`)

CUTLASS's `union GmmaDescriptor` (in `cute/arch/mma_sm90_desc.hpp`) is the authoritative bit layout; the
bitfield, transcribed:

```cpp
union GmmaDescriptor {
  uint64_t desc_;
  struct {
    uint16_t start_address_      : 14, : 2;  // bits [ 0,14) ; low 4 addr bits dropped (>>4)
    uint16_t leading_byte_offset_: 14, : 2;  // bits [16,30) ; LBO, >>4
    uint16_t stride_byte_offset_ : 14, : 2;  // bits [32,46) ; SBO, >>4
    uint8_t  : 1, base_offset_   :  3, : 4;  // bits [49,52) ; base_offset (swizzle phase)
    uint8_t  : 6, layout_type_   :  2;        // bits [62,64) ; swizzle mode
  } bitfield;
};
```

| Bits | Field | Meaning | Encoding |
|---|---|---|---|
| `[0,14)` | **start_address** | byte address of the tile *within the CTA's SMEM window* (0-based, from `cvta.to.shared`) | value `= addr >> 4` — SMEM is 16-byte-granular here, low 4 bits are always 0 and dropped `[FACT]` |
| `[14,16)` | unused | — | 0 |
| `[16,30)` | **leading_byte_offset (LBO)** | byte stride between core matrices along the **leading** dim (K for a K-major operand) | value `= LBO_bytes >> 4` `[FACT]` |
| `[30,32)` | unused | — | 0 |
| `[32,46)` | **stride_byte_offset (SBO)** | byte stride between core matrices along the **stride** dim (M for A / N for B) | value `= SBO_bytes >> 4` `[FACT]` |
| `[46,49)` | unused | — | 0 |
| `[49,52)` | **base_offset** | 3-bit swizzle *phase* — a within-atom rotation seed; **0** when the tile is atom-aligned (base is a multiple of the swizzle atom) | raw 3-bit `[FACT]` |
| `[52,62)` | unused | — | 0 |
| `[62,64)` | **layout_type (swizzle)** | 0 = none/linear, 1 = 128B, 2 = 64B, 3 = 32B | raw 2-bit `[FACT]`, CUTLASS enum `LayoutType{INTERLEAVE=0,B128=1,B64=2,B32=3}` |

**The `>>4` (16-byte unit) encoding** `[FACT]`: `start_address`, `LBO`, `SBO` are all stored as
`value >> 4` — i.e. in units of 16 bytes. That is why every field is 14 bits yet can reach a 18-bit
byte range (14 + 4). It also *requires* every tile base and every offset to be **16-byte aligned**; a
non-aligned base is a silent corruption (§4 adversarial case — "must fail loudly"). CUTLASS spells the
encode as `matrix_descriptor_encode(x) = (x & 0x3FFFF) >> 4` (mask 18 bits, drop low 4 → 14 bits).

### 2.2 Core matrices — why LBO/SBO exist at all `[FACT] / [INFERENCE for exact geometry]`

The tensor core does not read the tile as one blob; it reads a grid of **core matrices**. For 16-bit
operands a core matrix is an **8-row × 8-column** sub-tile of elements = **8 rows × 16 bytes**
(8 halves · 2 B). `[FACT: 8×8 unit; INFERENCE: the exact byte mapping below]` The MMA's K=16 strip is
therefore **2 core matrices** deep (16/8), and M=64 is **8 core matrices** tall (64/8). A (64×16) =
8 (M) × 2 (K) = 16 core matrices.

- **LBO** answers: *"from one core matrix to the next along the leading (K) dimension, how many bytes?"*
- **SBO** answers: *"from one 8-row core-matrix group to the next along the stride (M) dimension, how
  many bytes?"*

The tensor core computes each core matrix's base as
`start + i_M·SBO + i_K·LBO` (units of 16 B after decode), then applies the swizzle within. Two numbers
(LBO, SBO) fully describe a *rectangular, regularly-strided* tiling of core matrices — which is why the
descriptor needs only those two, not a full per-dimension stride list.

### 2.3 WORKED hand-encoding — a 128×64 f16 SMEM tile, 128B swizzle

**Setup.** Pick a concrete operand tile in SMEM:
- shape **128 rows (M/stride) × 64 cols (K/leading)** of **f16**, **K-major** (K contiguous within a row);
- **64 cols · 2 B = 128 bytes per row** — this equals *exactly one 128B swizzle atom width*, the reason
  128B swizzle is the natural choice for a 64-wide f16 K-strip;
- **128B swizzle** (`Swizzle<3,4,3>`, §3);
- tile base at SMEM byte offset **0x2000** (8192, 16-byte aligned ✓), atom-aligned so `base_offset = 0`.

**Field-by-field.**

1. **start_address.** `0x2000 >> 4 = 0x200 = 512`. Fits 14 bits (max 16383). `[FACT]`

2. **LBO (leading, along K).** The two K-adjacent core matrices are core-mat-0 = K[0:8] (row bytes
   0..15) and core-mat-1 = K[8:16] (row bytes 16..31). Their first-element byte offsets differ by
   **16 bytes**. `[INFERENCE]` So `LBO_bytes = 16`, encoded `16 >> 4 = 1`. Intuition: LBO = the byte
   width of one core matrix's K-strip = **8 halves = 16 B**.

3. **SBO (stride, along M).** The 8-row core-matrix group at M[0:8] starts at row 0 (byte 0); the next
   group M[8:16] starts at row 8. Each row is 128 bytes, so row 8 is at `8 · 128 = 1024` bytes.
   `[INFERENCE]` `SBO_bytes = 1024`, encoded `1024 >> 4 = 64`. Intuition: SBO = **(8 rows) × (128 B per
   row) = 1024 B = the full byte size of one swizzle atom** — this is the "8 × (swizzle-width)-byte
   atom" the assignment points at: 8 rows tall, `swizzle`=128 bytes wide.

4. **base_offset.** Tile base is a multiple of the 1024-byte atom (0x2000 = 8 atoms), so no intra-atom
   phase ⇒ `base_offset = 0`. `[INFERENCE]`

5. **layout_type (swizzle).** 128B ⇒ **1**. `[FACT]`

**Assemble the 64 bits** (place each field, then OR):

```
 field            value   shift   contribution (64-bit hex)
 start_address    0x200    <<0    0x0000_0000_0000_0200
 LBO              0x001    <<16   0x0000_0000_0001_0000
 SBO              0x040    <<32   0x0000_0040_0000_0000
 base_offset      0x0      <<49   0x0000_0000_0000_0000
 swizzle          0x1      <<62   0x4000_0000_0000_0000
 ------------------------------------------------------- OR
 DESCRIPTOR                       0x4000_0040_0001_0200
```

**`descriptor = 0x4000004000010200`**  `[INFERENCE — to be byte-checked in Phase 2]`

Sanity re-decode (round-trip): `& 0x3FFF = 0x200 → <<4 = 0x2000` base ✓; `(>>16)&0x3FFF = 1 → 16 B`
LBO ✓; `(>>32)&0x3FFF = 0x40 = 64 → 1024 B` SBO ✓; `(>>62)&0x3 = 1 → 128B swizzle` ✓;
`(>>49)&0x7 = 0` base_offset ✓.

**Cross-check against Colfax `make_smem_desc`** `[FACT]`: Colfax's Hopper WGMMA tutorial hand-builds
the descriptor for exactly this shape (a 64-wide f16 tile, 128B swizzle) as
`encode(addr) | (encode(16)<<16) | (encode(1024)<<32) | (1ull<<62)` — i.e. **LBO=16, SBO=1024,
swizzle=1**, identical to the values derived above. That is the independent confirmation that the
geometry reasoning is right; only the concrete `start_address` differs (theirs is the runtime SMEM
pointer, mine is 0x2000).

### 2.4 The B operand differs only in which axis is "stride"
For **B (K-major, i.e. contiguous along K, shape K×N)** the same two numbers apply but "stride" now
means **N**: SBO steps between 8-column N-groups, LBO steps between K core matrices. The descriptor
*format* is identical; what changes is the caller's choice of LBO/SBO and the `imm-trans-b` bit if B is
stored N-major and must be walked transposed. `[FACT]` The classic silent bug (§5 of the assignment):
feeding a K-major descriptor to an N-major tile → the core reads a transposed tile → "plausible garbage."

---

## 3. Swizzling — why, how 128B maps, and the descriptor bits

### 3.1 Why XOR-swizzle at all `[FACT]`
SMEM is **32 banks × 4 bytes**, word-interleaved: byte address `a` lives in bank `(a >> 2) & 31`. A
warp (or the tensor core's read engine) that touches 32 addresses all sharing a bank ⇒ up to **32-way
bank conflict**, serialized. Our f16 tile has **128-byte rows = exactly 32 banks (128/4)**, so a given
column sits on the *same bank in every row*. The tensor core reads a core matrix as an 8-row column
slice → 8 accesses to one bank = **8-way conflict**, every read. `ldmatrix` has the same pathology.
Unswizzled, the MMA is bank-conflict-bound, not math-bound.

### 3.2 How 128B swizzle maps addresses `[FACT]`
The fix is to **XOR a function of the row index into the column index**, so the 8 rows of a column
slice scatter across 8 *distinct* bank groups. CuTe expresses it as `Swizzle<B,M,S>`; **128B swizzle =
`Swizzle<3,4,3>`**:
- **B = 3** → `2^3 = 8` rows participate (one core-matrix group height);
- **M = 4** → the swizzle granule is `2^4 = 16 bytes` = one core matrix's K-width (= 8 halves) — the
  smallest unit permuted;
- **S = 3** → `2^3 = 8` granules span the `8 · 16 B = 128-byte` swizzle atom width.

Operationally, for an address decomposed as `[ row_bits | col_granule_bits | offset_in_granule ]`, the
permutation is
`col_granule' = col_granule XOR row_bits`  (the low B bits of the row XOR the S bits of the
column-granule). Row 0 is unrotated; row 1's 128-byte line is rotated by one 16-byte granule; …; row 7
by seven — so the 8 rows of any column land on 8 different banks. **Result: 0 bank conflicts**
(`ncu shared_ld/st_bank_conflict ≈ 0`, the Rung-2 DoD). 64B swizzle (`Swizzle<2,4,3>`... i.e. B=2) and
32B swizzle scatter fewer rows and suit narrower K-strips.

### 3.3 The atom ↔ descriptor-bit relationship `[FACT]`
Three things must agree or the tensor core reads permuted garbage *silently*:
1. the **swizzle the data was written with** (the TMA `CU_TENSOR_MAP_SWIZZLE_128B`, or the `cp.async`
   + CuTe swizzle atom used to fill SMEM);
2. the descriptor's **`layout_type` bits [62,64)** (here `1` = 128B) — these tell the hardware which
   inverse permutation to apply as it walks the tile;
3. the **swizzle-atom width == the tile's leading-dim byte width** (128 B here) so the atom tiles the
   row exactly. A 64-half row demands the 128B atom; a 32-half row demands the 64B atom; mismatch =
   the atom straddles rows and the mapping is wrong.

The `base_offset` field [49,52) carries the *phase* when the tile base is not atom-aligned (a sub-atom
starting offset) — for atom-aligned bases it is 0, as in §2.3. **Descriptor swizzle bits do not create
the swizzle; they declare it** so the walk can undo it. This is why Rung 3's caveat is "pick a GMMA
canonical swizzle atom that matches your TMA swizzle" — the two are one contract split across the TMA
descriptor and the wgmma descriptor.

---

## 4. Phase-2 pairing note — the `nvcc` invocation and what to diff

When we rent the H100 (Phase 2, `performance/PERF_PLAN.md` §"Phase 2", 7th–8th items), compile the
same probe *for real* and diff against this hand-decode.

**Compile to PTX** (host-side; already runs on any box, incl. the sm_120 standing card — see §5):
```
nvcc -arch=sm_90a -ptx  wgmma_probe.cu -o wgmma_probe.ptx
```
**Compile to SASS** (needs the H100 toolchain target; inspect the machine instruction):
```
nvcc -arch=sm_90a -cubin wgmma_probe.cu -o wgmma_probe.cubin
cuobjdump -sass wgmma_probe.cubin        # look for HGMMA.64x64x16.F32 and the LDSM feeders
```
> `-arch=sm_90a` — the trailing **`a`** ("architecture-specific / accelerated") is **mandatory**: plain
> `sm_90` does **not** expose `wgmma`/`wgmma.mma_async`, TMA (`cp.async.bulk.tensor`), or `setmaxnreg`.
> `[FACT]` A `sm_90` build silently lacks the instruction. Same trap on Blackwell: `sm_100a`, not
> `sm_100`.

**What to diff, line by line:**
1. **Qualifier string** — the emitted `wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16` must match
   §1 verbatim. (Already confirmed on the standing box, §5.)
2. **Accumulator vector width** — count the `{%f…}` operands: must be **32** for m64n64 (§1.9). Change
   the probe to `m64n128k16` and confirm it becomes **64**; `m64n256k16` → **128**.
3. **A/B operand forms** — SS form shows two 64-bit descriptor regs `%rd, %rd`; RS form shows
   `{%r,%r,%r,%r}, %rd` (4× A-frag + 1 B-desc) and **one fewer trailing immediate** (no `imm-trans-a`).
   Confirmed §5.
4. **The 64-bit descriptor value** — the real diff of interest. Compile a tiny kernel that calls
   `cute::make_gmma_desc<GMMA::Major::K, half_t>(...)` (or the Colfax `make_smem_desc`) on a known
   SMEM pointer, print `desc` with `printf("%016llx")`, and compare field-by-field against §2.3:
   `start = ptr>>4`, `LBO=1`, `SBO=64`, `swizzle=1`. This turns the §2.3 `[INFERENCE]` values into
   `[FACT]`. Expect the *low* `start_address` bits to differ (runtime pointer) — the **LBO/SBO/swizzle
   fields are the assertion**.
5. **Fence protocol** — confirm the compiler brackets the mainloop with
   `wgmma.fence` … `commit_group` … `wait_group N`, and check what **N** it chooses under
   pipelining (N>0 ⇒ the overlap is live). Confirmed present §5.

---

## 5. Compiler ground truth (generated on the standing box)

`nvcc -arch=sm_90a -ptx` (CUDA **13.0**, PTX ISA **9.0**) on the sm_120 box — PTX text emission is
host-side and needs no Hopper GPU. The probe (self-contained; save as `wgmma_probe.cu` and recompile
in Phase 2) forces both operand forms — SS = A,B both descriptors; RS = A in 4× `.b32`, B descriptor:

```cuda
#include <cuda_fp16.h>
#include <cstdint>
// SS form: 32 FP32 accumulators, A-desc + B-desc, 5 trailing immediates.
__device__ void wgmma_ss(float d[32], uint64_t a, uint64_t b) {
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
    "%32, %33, 1, 1, 1, 0, 0;\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
      "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
      "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
      "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
    : "l"(a), "l"(b));
}
// RS form: A in {4 x .b32} fragment, B-desc, only 4 trailing immediates (no imm-trans-a).
__device__ void wgmma_rs(float d[32], const uint32_t a[4], uint64_t b) {
  asm volatile(
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
    "{%32,%33,%34,%35}, %36, 1, 1, 1, 0;\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
      "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
      "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
      "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "l"(b));
}
__global__ void k(float* out, const uint32_t* a, const uint64_t* desc) {
  float d[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = out[i];
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  wgmma_ss(d, desc[0], desc[1]);
  wgmma_rs(d, a, desc[1]);
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  #pragma unroll
  for (int i = 0; i < 32; ++i) out[i] = d[i];
}
```

Emitted (header + the relevant lines):

```
.version 9.0
.target sm_90a
.address_size 64
   ...
wgmma.fence.sync.aligned;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
   ...
// SS form: A-desc %rd1, B-desc %rd2 ; 32 FP32 accumulators %f65..%f96 ; 5 trailing imm
wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
   {%f65,%f66,%f67,%f68,%f69,%f70,%f71,%f72,%f73,%f74,%f75,%f76,%f77,%f78,%f79,%f80,
    %f81,%f82,%f83,%f84,%f85,%f86,%f87,%f88,%f89,%f90,%f91,%f92,%f93,%f94,%f95,%f96},
   %rd1, %rd2, 1, 1, 1, 0, 0;
// RS form: A-frag {%r1,%r2,%r3,%r4}, B-desc %rd3 ; note ONLY 4 trailing imm (no imm-trans-a)
wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
   {%f65,...,%f96}, {%r1,%r2,%r3,%r4}, %rd3, 1, 1, 1, 0;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
```

Verified counts: **32** `.f32` accumulator regs (`%f65…%f96`); **4** `.b32` A-frag regs in the RS
form; **2** 64-bit descriptor operands in the SS form; SS carries **5** trailing immediates
(`scale-d, sa, sb, ta, tb`), RS carries **4** (the `imm-trans-a` slot vanishes for the RF operand).
Every §1 qualifier and the §1.10 fence/commit/wait protocol appear verbatim. `[FACT]`

---

## 6. What is FACT vs INFERENCE in this artifact (audit)

- `[FACT]` (compiler- or source-verified): the entire qualifier decode (§1); operand homes and forms
  (§1.8); the 32-reg accumulator budget (§1.9, counted in the PTX); the fence/commit/wait protocol
  (§1.10, §5); the descriptor **bit-field map and the `>>4` 16-byte encoding** (§2.1, CUTLASS
  `GmmaDescriptor`); the swizzle enum values (§2.1); the swizzle *why* and the 32-bank arithmetic
  (§3.1); the `sm_90a` "a"-suffix requirement (§4); all of §5.
- `[INFERENCE]` (geometry reconstructed, byte-check pending): the exact **LBO=16 B / SBO=1024 B**
  numeric derivation from the 8×8 core-matrix tiling (§2.2–2.3) and hence the assembled descriptor
  value **0x4000004000010200** — though these match Colfax's independently-written `make_smem_desc`
  constants for the same tile (LBO=16, SBO=1024, swizzle=1), which is strong corroboration. The
  `base_offset=0` claim assumes an atom-aligned base.
- **Phase-2 close-out:** §4 step 4 (print `cute::make_gmma_desc` and diff the 64-bit value) is the one
  test that promotes the `[INFERENCE]` block to `[FACT]`. Until then, treat 0x4000004000010200 as a
  derived-and-corroborated value, not a silicon-checked one.
