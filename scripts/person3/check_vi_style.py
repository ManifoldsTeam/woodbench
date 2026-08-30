"""
Kiem tra chat luong hanh van ban tieng Viet.

Luat lay tu hai nguon:
  * scientific-writing.md (goc repo): cam dau gach ngang lam dau cau; van
    phong khach quan; suc tich; nhat quan thuat ngu.
  * .claude/skills/academic-paper/references/writing_quality_check.md:
    muc B (dau cau), C (mo bai vong vo), D (mau cau truc), E (bien thien
    do dai cau).

Khac voi check_vi_prose.py (dem tu, truy nguyen con so), tep nay chi soi
HANH VAN. Chay ca hai truoc khi coi ban thao la xong.

Usage:
    python3 scripts/person3/check_vi_style.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VI = ROOT / "paper_draft_person3/sections_vi"

# Muc A cua writing_quality_check.md, chuyen sang tuong duong tieng Viet.
# Khong cam tuyet doi: moi lan gap phai tu hoi day co phai tu chinh xac nhat
# hay chi la tu mac dinh.
FLAGGED = {
    "toàn diện": "đầy đủ, chi tiết, trên diện rộng",
    "then chốt": "chính, quyết định",
    "mấu chốt": "chính, quyết định",
    "cốt lõi": "trung tâm, chính",
    "đa chiều": "phức tạp, nhiều mặt",
    "sâu sắc": "chi tiết, cụ thể",
    "vượt trội": "cao hơn, tốt hơn (kem so lieu)",
    "đột phá": "mới, chưa từng có",
    "tiên tiến": "gần đây, hiện đại",
    "tận dụng": "dùng, áp dụng",
    "thúc đẩy": "làm tăng, hỗ trợ",
    "bức tranh": "kết quả, tình hình",
    "đáng kể": "nêu rõ độ lớn thay vì tính từ",
    "rõ ràng là": "bỏ, để số liệu tự nói",
    "thú vị": "bỏ, tính từ biểu cảm",
    "ấn tượng": "bỏ, tính từ biểu cảm",
}

# Muc C: mo dau vong vo / cau meta mo ta bai dang lam gi.
THROAT = [
    "Cần lưu ý rằng", "Đáng chú ý là", "Điều đáng nói là",
    "Có thể nói rằng", "Trong bối cảnh đó", "Nhìn chung thì",
    "Mục này sẽ trình bày", "Phần tiếp theo sẽ", "Chúng ta hãy",
    "Không thể phủ nhận rằng", "Trước hết cần khẳng định",
]

# Gach ngang dung LAM DAU CAU. scientific-writing.md mien tru gach noi
# trong so lieu va khoang gia tri, nen bo qua "40.3--44.0" va tu ghep ky
# thuat kieu "train--test". Hang trong tabular cung bo qua: dau "---" o o
# trong la ky hieu khong-ap-dung, khong phai dau cau.
DASH = re.compile(r"---|(?<!-)--(?!-)|[\u2013\u2014]|(?<= )-(?= )")
NUM_RANGE = re.compile(r"\d\s*--\s*\d")
COMPOUND = re.compile(r"[A-Za-z]--[A-Za-z]")


def dash_hits(text: str) -> list[int]:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("%"):
            continue
        clean = re.sub(r"\$[^$]*\$", " ", line)
        clean = re.sub(r"\\texttt\{[^}]*\}", " ", clean)
        if "&" in clean and clean.rstrip().endswith("\\\\"):
            continue                       # hang bang
        clean = NUM_RANGE.sub(" ", clean)
        clean = COMPOUND.sub(" ", clean)
        if DASH.search(clean):
            hits.append(i)
    return hits


def body(text: str) -> str:
    t = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))
    return re.sub(r"\\(input|includegraphics|label|ref)(\[[^\]]*\])?\{[^}]*\}", " ", t)


def prose(text: str) -> str:
    t = body(text)
    t = re.sub(r"\\begin\{(tabular|center)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"\\(section|subsection|paragraph|textbf|textit|emph|texttt)\*?\{([^}]*)\}", r"\2", t)
    t = re.sub(r"\\[a-zA-Z@]+\*?", " ", t)
    return re.sub(r"[{}\\&~^_%]", " ", t)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", " ".join(prose(text).split()))
    return [s for s in parts if len(s.split()) >= 3]


def paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", prose(text)) if len(p.split()) >= 20]


def check(name: str, text: str) -> int:
    issues = 0
    words = len(prose(text).split())

    # B1. dau gach ngang lam dau cau
    hits = dash_hits(text)
    if hits:
        print(f"    [B1] {len(hits)} dong con dau gach ngang lam dau cau: dong {hits[:6]}")
        issues += 1

    # B2. dau cham phay: <= 2 tren 1000 tu
    semis = prose(text).count(";")
    cap = max(2, round(2 * words / 1000))
    if semis > cap:
        print(f"    [B2] {semis} dau cham phay / {words} tu (nguong {cap}). Tach thanh cau rieng.")
        issues += 1

    # C. mo dau vong vo
    for phrase in THROAT:
        if phrase.lower() in prose(text).lower():
            print(f"    [C]  mo dau vong vo: \"{phrase}\"")
            issues += 1

    # A. tu bi danh dau
    for term, alt in FLAGGED.items():
        c = prose(text).lower().count(term)
        if c:
            print(f"    [A]  \"{term}\" x{c}  -> can nhac: {alt}")
            issues += 1

    # D. do dai doan qua deu
    ps = [len(p.split()) for p in paragraphs(text)]
    if len(ps) >= 4:
        spread = (max(ps) - min(ps)) / (sum(ps) / len(ps))
        if spread < 0.6:
            print(f"    [D]  {len(ps)} doan co do dai qua deu ({min(ps)}-{max(ps)} tu). Can bien thien.")
            issues += 1

    # E. bien thien do dai cau: 5 cau lien tiep cung khoang
    ss = [len(s.split()) for s in sentences(text)]
    run = 1
    for i in range(1, len(ss)):
        run = run + 1 if abs(ss[i] - ss[i - 1]) <= 5 else 1
        if run >= 5:
            print(f"    [E]  5+ cau lien tiep do dai gan bang nhau (quanh cau {i+1}, ~{ss[i]} tu)")
            issues += 1
            run = 1

    if not issues:
        print(f"    sach ({words} tu, {len(ps)} doan, {len(ss)} cau)")
    return issues


def main() -> None:
    total = 0
    for f in sorted(VI.glob("*.tex")):
        print(f"  {f.name}")
        total += check(f.name, f.read_text(encoding="utf-8"))
    print(f"\nTONG: {total} van de hanh van")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
