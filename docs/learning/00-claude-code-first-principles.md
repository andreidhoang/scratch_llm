# Bài 0 — Claude Code vận hành thế nào trên repo này (first principles)

> **Bài nền, đọc một lần** · mọi trích dẫn pin theo commit **`9233dc4`**
> Đối tượng giải phẫu: chính cái harness đang chạy bạn đọc — `CLAUDE.md`, `.claude/settings.json`,
> `.claude/hooks/*` (4 hooks), `.claude/agents/*` (6 agents), `.claude/commands/*` (10 lệnh),
> và chuỗi con trỏ `performance/PERF_PLAN.md` → `bench/RESULTS.md` → `docs/STATUS.md`.
> Bản field-manual đầy đủ (tiếng Anh, góc nhìn thiết kế): `docs/CONTEXT_ENGINEERING.md`.
> Bài này là bản Feynman: **suy từng mảnh từ nguyên lý gốc — vì sao nó PHẢI tồn tại**.

---

## 0. Nguyên lý gốc: LLM là một hàm KHÔNG TRẠNG THÁI

```
        f(context) ──► token tiếp theo
```

Trọng số model **đóng băng** — không có gì "được học" giữa hai lần gọi. Một "session" thực chất
là một **bản ghi (transcript) lớn dần**: mỗi lượt, harness ghép lại toàn bộ những gì đã xảy ra và
đưa cho model đọc lại từ đầu. "Trí nhớ" của AI = *những gì được nạp lại vào context* — không hơn.

Hệ quả nghiêm ngặt, và là tiên đề của mọi thứ bên dưới:

> **Nếu một tri thức không nằm trong context của lượt này, thì với model, tri thức đó KHÔNG TỒN
> TẠI.** Session mới = não trắng. Muốn AI "tiếp tục công việc", tri thức điều hướng phải nằm ở
> nơi chắc chắn được nạp lại: **trên đĩa, trong repo, đã push**.

## 1. Nguyên lý 1: context hữu hạn và bị "mục" — kinh tế học token

Cửa sổ context có giới hạn, và tín hiệu **loãng dần khi nội dung phình ra** (context rot —
`CONTEXT_ENGINEERING.md` §0.2). Suy ra bài toán kinh tế:

- Nội dung **always-on** (nạp mỗi lượt) trả giá ở **mọi lượt** — kể cả lượt không cần nó.
- Nội dung **on-demand** (đọc khi cần) trả giá **một lần, đúng lúc**.
- Vậy: luật bất biến thì always-on nhưng phải NHỎ; tri thức sâu thì on-demand qua **con trỏ**.

Hai nguyên lý này (0: không trạng thái, 1: context đắt) đẻ ra toàn bộ kiến trúc còn lại.

## 2. Cỗ máy bên trong: vòng lặp agentic

Claude Code không "chạy một phát ăn ngay" — nó **lặp**:

```
┌─────