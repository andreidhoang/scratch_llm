# Bài 0 — Claude Code vận hành thế nào trên repo này (first principles, tiếng Việt)

> **Series 0 · Meta** · pin theo commit **`9233dc4`**
> Đọc thật: `CLAUDE.md`, `.claude/settings.json`, `.claude/hooks/*.sh`, `docs/CONTEXT_ENGINEERING.md`
> (field manual gốc, tiếng Anh — bài này là bản dẫn nhập tiếng Việt từ first principles).
>
> **Bài này trả lời:** vì sao một session Claude "não trắng" vẫn biết chính xác việc tiếp theo phải
> làm, và vì sao mỗi mảnh của bộ harness `.claude/` tồn tại. Không phải mô tả tính năng — mà **suy
> từ một tiên đề duy nhất** ra toàn bộ thiết kế.

---

## 0. Tiên đề gốc — LLM là một hàm KHÔNG TRẠNG THÁI

Bản chất mô hình chỉ là:

```
f(context) → token tiếp theo
```

Nó **không có bộ nhớ** giữa hai lần gọi. Không có "hôm qua tôi đã làm X". Thứ **duy nhất** quyết
định đầu ra là **những gì đang nằm trong cửa sổ context** tại đúng thời điểm đó. Mọi thứ còn lại
trong bài này đều là **hệ quả suy ra** từ câu này — không có gì là "tính năng cho vui".

Feynman-check: nếu bạn tin điều ngược lại ("AI nhớ dự án"), bạn sẽ thiết kế sai — bạn sẽ dặn dò
bằng lời và mong nó nhớ. Repo này thiết kế **ngược lại**: giả định AI **quên sạch** mỗi session,
nên trạng thái phải sống **trên đĩa**, và kỷ luật phải sống **trong code chạy được**, không trong
lời văn.

---

## 1. Hệ quả 1 — Context hữu hạn và bị "mục" ⇒ phải phân tầng tri thức

Cửa sổ context có giới hạn token. Nạp càng nhiều thì (a) tốn tiền cho **mọi** turn, (b) tín hiệu
loãng đi giữa rác — hiện tượng `docs/CONTEXT_ENGINEERING.md` gọi là **context rot** (§0.2). Suy ra:
không thể nhét tất cả tri thức vào chỗ always-on. Phải chia theo **4 đòn bẩy** (§1 của field manual):

| Đòn bẩy | Định nghĩa | Giá phải trả | Hiện thân trong repo |
|---|---|---|---|
| **1 · Always-on** | nạp MỌI turn | đắt nhất → phải nhỏ + load-bearing | `CLAUDE.md` (root) |
| **2 · On-demand** | đọc khi cần, qua con trỏ | ~0 khi không dùng | `docs/`, `performance/`, spec notes, `docs/learning/` |
| **3 · Subagents** | giao việc đọc/tìm lớn cho agent con | giữ kết luận, vứt "rác đọc" | `.claude/agents/*.md` |
| **4 · Hooks** | kỷ luật bằng CODE, không bằng lời | chạy tự động | `.claude/hooks/*.sh` |

**Vì sao mảnh này quan trọng:** đây là lý do `CLAUDE.md` chỉ chứa **hiến pháp + con trỏ**, không
chứa nội dung sâu. Bảng "Where things live" trong `CLAUDE.md` là một danh bạ Lever-2: mỗi dòng trỏ
tới một doc, và doc đó **trỏ ngược lại** — nên vào từ đâu cũng tới được phần còn lại, và **không có
sự thật nào bị chép hai bản** (một nguồn sự thật cho mỗi fact; mọi nơi khác link tới nó).

---

## 2. Hệ quả 2 — AI quên sạch mỗi session ⇒ "bộ nhớ thật" là git + docs, KHÔNG phải model

Đây là câu trả lời sâu nhất cho "làm sao session sau biết làm gì tiếp". Vì `f` không trạng thái,
**trạng thái dự án phải sống ngoài model**, ở nơi bất biến và push được lên GitHub:

```
 git commits ........... trạng thái BẤT BIẾN — cái gì đã ship (chuỗi phase R3b→R4.1)
 bench/RESULTS.md ...... số đo BỀN — GPU trả rồi số vẫn còn (FOP-4: measured ≠ implemented)
 performance/PERF_PLAN.md  CON TRỎ NODE — nguyên văn "việc tiếp theo" + DoD + kill criteria
 docs/STATUS.md ........ toàn cảnh: mặt trận nào active, cái gì parked
 docs/learning/ ........ tri thức ĐÃ DẠY (đóng băng lúc dạy, không viết trước build)
```

`CLAUDE.md` ghi chính triết lý này thành luật (§"Carry context forward, durably"):
> *"The next session must orient from what you left, not from what anyone remembers."*

**Vì sao mảnh này quan trọng:** nó lật ngược trực giác. Bạn KHÔNG "dặn AI nhớ"; bạn **ghi trạng
thái ra đĩa** và để session sau **đọc lại**. Việc của bạn (và của tôi cuối mỗi rung) là cập nhật 5
file trên — đó là lý do mỗi rung đóng lại đều kèm: log `bench/RESULTS.md` → advance node
`PERF_PLAN.md` → sync `STATUS.md` → commit → push.

---

## 3. Vòng đời một session — chuỗi file agent đi qua (thật, có thứ tự)

### 3.1 Tầng tự động — 0 giây, không ai làm gì

```
Claude Code mở tại /workspace/scratch_llm
│
├─► CLAUDE.md ─────────── NẠP TỰ ĐỘNG mỗi turn (Lever 1): FOP 7 điều · giao thức
│                          "Orient before you build" 4 bước · bảng con trỏ "Where things live"
│
└─► .claude/hooks/session-start.sh ── CHẠY TỰ ĐỘNG
        (wiring: .claude/settings.json → "SessionStart" → command)
        stdout của hook (exit 0) được CỘNG THẲNG vào context. Nó chỉ in trạng thái ĐỘNG mà
        file tĩnh không thể biết (mỗi dòng phải "đáng tiền token"):
          · branch · số file uncommitted · venv ready/missing
          · git log --oneline -5  ("latest commits — analyze before proceeding")
          · grep -m1 '^Phase:' performance/PERF_PLAN.md
            → "current node → 1a — A1 R0–R3 + R4.1 DONE → R4.2 (chunked prefill) next"
```

Cơ chế then chốt (first principle của hook): **hook là cầu Lever 4 → Lever 1** — code chạy để
*bơm* tri thức động vào chỗ always-on. `CLAUDE.md` tĩnh không thể biết "commit mới nhất là gì";
`session-start.sh` chạy `git log` lúc khởi động và tiêm kết quả vào. Đó là lý do nó tồn tại.

### 3.2 Tầng thủ tục — giao thức 4 bước BẮT BUỘC (CLAUDE.md §"Orient before you build")

Sau khi hook bơm slice khởi đầu, agent PHẢI làm, đúng thứ tự (đây là FOP áp dụng thành *workflow*):

```
BƯỚC 1 · Đọc trạng thái, ĐỪNG đoán
   git log --oneline -15 ─────────► cái gì VỪA ship
   performance/PERF_PLAN.md ──────► ★ "Current Node" + "Next action" — nguyên văn việc tiếp theo:
   │   R4.2 chunked prefill: interleave prefill chunks between decode steps;
   │   DoD = TTFT/ITL curve vs chunk size, no output divergence (oracle = R3b greedy);
   │   motivation MEASURED: ITL p99 ~29ms admission spikes; then R4.3 → R4.4
   bench/RESULTS.md ──────────────► ledger [FACT] (đo được ≠ mới code)
   docs/STATUS.md ────────────────► active front = perf curriculum; A2-CS336 parked

BƯỚC 2 · Tái dựng mạch (nói lại bằng lời của mình cái gì đã đóng, đo-được vs mới-code)
   performance/PERF_ENGINEERING_SPEC.md §A1 ─► gate + falsifiable predictions của R4.2
   performance/A1_transformer_inference.md §4.2 ─► nội dung assignment
   performance/notes/A1_R3_*.md, A1_R41_*.md ──► HAI bài mẫu pre-registration (khuôn để bắt chước)

BƯỚC 3 · Pre-register TRƯỚC khi code (FOP-2 spec-with-falsifiers, FOP-3 predict-before-run)
   viết performance/notes/A1_R42_chunked_prefill.md + rows "— (pending)" vào bench/RESULTS.md

BƯỚC 4 · Build test-first → đo → log → advance node → commit (green gate chặn nếu đỏ)
```

Tắt nhanh: gõ **`/standup`** hoặc **`/next`** (`.claude/commands/`) = chạy đúng nghi thức trên
thành một lệnh. Nếu là session **dạy học** thì cửa vào là `docs/learning/INDEX.md` (Bài kế đang chờ).

---

## 4. Hệ quả 3 — Model có thể quên lời dặn ⇒ kỷ luật phải ở tầng HOOK, không ở prose

Một câu dặn trong `CLAUDE.md` ("chỉ commit khi CI xanh") là **Lever 1** — model *có thể* lỡ. Một
hook là **Lever 4** — nó *không thể* lỡ, vì nó là code chạy ngoài model. Bốn hook thật trong repo
(`.claude/settings.json` wiring → `.claude/hooks/`):

| Hook | Sự kiện (matcher) | Làm gì | First principle |
|---|---|---|---|
| `session-start.sh` | SessionStart | bơm trạng thái động vào context | Lever 4 **nuôi** Lever 1 (§3.1) |
| `green-ci-gate.sh` | PreToolUse · `Bash` chứa `git commit` | ruff+pyright+`pytest -m "not gpu"`; đỏ → **exit 2 chặn commit** | "red commit phải BẤT KHẢ THI, không chỉ là bị khuyên" (dòng 3 của hook) |
| `kernel-write-guard.sh` | PreToolUse · `Edit`/`Write` | chặn agent GHI thân kernel Mode-3 (con người tự dựng); test/harness/oracle KHÔNG chặn | ranh giới Mode-3 (FOP-6) thành backstop vật lý |
| `lint-on-edit.sh` | PostToolUse · `Edit`/`Write` | lint nhẹ sau mỗi sửa (non-blocking) | phản hồi sớm, không chặn |

**Chi tiết sâu đáng học** (đọc `green-ci-gate.sh:8–16`): matcher `if: Bash(git commit*)` trong
`settings.json` **không được harness này honor** — nên hook **tự enforce scope trong shell** (case
`*"git commit"*`) và cho mọi Bash khác đi thẳng. Bài học first-principles: **đừng tin wiring khai
báo là đủ; hook tự bảo vệ ý định của nó** (fail-safe). Và nó ưu tiên `.venv/bin` (dòng 25) vì
system Python thiếu torch/regex sẽ làm một commit hợp lệ trượt oan — chính là cái bug đã xảy ra và
được ghi ở `docs/STATUS.md` 2026-06-29.

Ngoài hook, `.claude/settings.json` còn có **`permissions.allow`**: whitelist các lệnh chạy không
cần hỏi (`Bash(uv *)`, `Bash(pytest *)`, `Bash(ruff *)`, `Bash(git status *)`…). Đây là Lever 4
giảm ma sát — việc an toàn, hay lặp thì cho qua; việc phá hủy vẫn phải xin phép.

---

## 5. Hệ quả 4 — Vòng lặp agentic: tool là giác quan, tri thức sinh từ QUAN SÁT

Claude Code không "chạy một phát". Nó **lặp**:

```
   ┌───────────────────────────────────────────────┐
   │  nạp context → SUY NGHĨ → gọi tool             │
   │       ▲                       │                │
   │       │   kết quả tool ◄──────┘                │
   │       └── (Read=mắt · Bash/Edit=tay ·          │
   │            pytest/nvidia-smi=xúc giác chạm sự thật) │
   └───────────────────────────────────────────────┘
```

**Vì sao mảnh này quan trọng:** tri thức đúng **sinh ra từ quan sát tool, không từ trí nhớ có
sẵn**. Ví dụ thật, đo được, từ phase vừa xong: khi build kernel paged, dynamo in ra
`cache.py_lengths[26] > py_lengths[16]` → dòng log đó quay vào context → agent suy ra "guard đang
nướng theo THỨ TỰ list python" → sửa `view_len` thành int do scheduler nuôi → recompile storm biến
mất (`unique_graphs` 47→2). Không một bước nào dựa vào "tôi nhớ Triton hoạt động thế nào" — tất cả
dựa vào **đọc kết quả tool và suy luận**. Đó là lý do FOP-4 khắt khe: *"Implemented ≠ measured —
chỉ một run đo/profiled mới là kết quả."*

---

## 6. Phép thử Feynman cuối — hệ hội tụ bất kể ai ngồi ghế lái

Xóa sạch trí nhớ tôi ngay bây giờ. Một Claude **hoàn toàn mới** mở repo này sẽ, trong ~60 giây đầu:

1. Thấy `session-start.sh` in ra `current node → … R4.2 (chunked prefill) next`.
2. `CLAUDE.md` (auto-load) ép nó chạy giao thức 4 bước → đọc `PERF_PLAN.md` thấy **nguyên văn**
   việc phải làm + DoD + kill criteria + động lực đã đo (ITL p99 ~29ms).
3. Thấy hai bài mẫu `performance/notes/A1_R3_*.md`, `A1_R41_*.md` để bắt chước cấu trúc
   pre-registration.
4. Nếu định commit ẩu → `green-ci-gate.sh` chặn (exit 2).

⇒ Hệ **hội tụ về cùng một hành động tiếp theo bất kể ai — hay model nào — ngồi ghế lái.** Đó chính
là định nghĩa của *context engineering* làm đúng: trạng thái ngoài não, kỷ luật trong code, con trỏ
một chiều-tới-sự-thật. Con người chỉ cần ra quyết định **taste** (chọn node EV cao nhất, duyệt
fork Mode-2); phần "biết đang ở đâu, làm gì tiếp" là thuộc tính của **hệ thống**, không của trí nhớ.

---

## 7. Cổng teach-back

1. Vì sao repo này **giả định AI quên sạch mỗi session** lại dẫn tới thiết kế "trạng thái sống trên
   đĩa, kỷ luật sống trong hook"? Suy từ tiên đề §0.
2. Phân biệt: câu "chỉ commit khi xanh" trong `CLAUDE.md` vs hook `green-ci-gate.sh` — cùng mục
   tiêu, khác **đòn bẩy** nào, và vì sao cái sau mới *đảm bảo* được?
3. **Modify-and-predict:** giả sử ai đó xóa dòng `grep -m1 '^Phase:' performance/PERF_PLAN.md` khỏi
   `session-start.sh`. Session sau CÓ còn tìm được node tiếp theo không, bằng đường nào, và chậm đi
   ở đâu? (Gợi ý: giao thức 4 bước vẫn còn.)

<details>
<summary>Phác đáp án</summary>

1. Vì `f(context)→token` không trạng thái: nếu tin AI nhớ, bạn dặn bằng lời và nó lỡ. Chấp nhận nó
   quên ⇒ trạng thái phải bất biến & đọc-lại-được (git+docs), và kỷ luật phải là code chạy ngoài
   model (hook) chứ không phải lời văn model có thể bỏ qua.
2. `CLAUDE.md` = Lever 1 (prose, always-on, model *có thể* quên). Hook = Lever 4 (code, PreToolUse,
   exit 2). Cái sau *đảm bảo* vì nó chạy **ngoài** vòng suy nghĩ của model — model không có quyền
   phủ quyết một tool call bị hook chặn.
3. CÓ — giao thức 4 bước (Bước 1) vẫn ép đọc `PERF_PLAN.md` "Current Node". Mất dòng grep chỉ mất
   **slice tự-động-tiêm** lúc khởi động ⇒ agent không thấy node ngay 3 giây đầu, phải chủ động mở
   file (chậm hơn, và rủi ro hơn nếu agent lười orient). Đó đúng là lý do hook tồn tại: biến "nên
   đọc" thành "đã có sẵn trước mắt" — dư thừa có chủ đích với giao thức, để an toàn kép.
</details>

---

*Liên quan:* `docs/CONTEXT_ENGINEERING.md` (field manual đầy đủ, tiếng Anh, mọi file trong harness
+ đòn bẩy biện minh cho nó) · `docs/OPERATING_RHYTHM.md` (nhịp ngày `/standup`→deep-work→`/eod`) ·
`docs/learning/INDEX.md` (các bài mastery component).
