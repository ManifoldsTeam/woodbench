"""
Kiem tra ban tieng Viet theo yeu cau bien tap.

Hai phep kiem tra. Cac luat HANH VAN (dau gach ngang, dau cham phay, tu
bi danh dau, nhip cau, do dai doan) nam o scripts/person3/check_vi_style.py
de tranh hai cai dat lech nhau. Chay ca hai truoc khi coi ban thao la xong.

Hai phep kiem tra o day:
  1. Dem tu than bai (bo comment, bo lenh LaTeX, bo cong thuc toan) va
     doi chieu voi chi tieu so tu cua tung muc.
  2. Moi con so trong ban VI phai truy nguoc duoc ve mot nguon da co:
     ban EN tuong ung, cac bang/hinh da sinh tu registry, hoac
     artifact/person3/per_class/family_confusion.json.
     Day la bat bien thay the cho phep so trung khop 1-1 cu. Sau khi ban VI
     duoc viet lai ngan hon theo chi tieu so tu, hai ban khong con doi ung
     tung cau, nhung ban VI van khong duoc phep sinh ra con so tu hu khong.

Usage:
    python3 scripts/person3/check_vi_prose.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VI = ROOT / "paper_draft_person3/sections_vi"
EN = ROOT / "paper_draft_person3/sections"

# Nguoi dung da bo yeu cau gioi han so tu, nen day chi la so lieu tham
# khao chu khong con la nguong lam FAIL.
REPORT = ["00_abstract.tex", "04_results.tex", "05_discussion.tex", "06_conclusion.tex"]

NUM = re.compile(r"\d+(?:\.\d+)?")


def body(text: str) -> str:
    """Bo dong comment va bo duong dan tep (khong phai so lieu cua bai)."""
    t = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))
    return re.sub(r"\\(input|includegraphics|label|ref)(\[[^\]]*\])?\{[^}]*\}", " ", t)


def allowed_numbers() -> Counter:
    """Kho con so hop le: ban EN, cac bang/hinh da sinh, va family_confusion.json."""
    pool: Counter = Counter()
    for f in EN.glob("*.tex"):
        pool.update(NUM.findall(body(f.read_text(encoding="utf-8"))))
    for sub in ("tables", "figures", "tables_vi", "figures_vi"):
        for f in (ROOT / "artifact/person3" / sub).glob("*.tex"):
            pool.update(NUM.findall(body(f.read_text(encoding="utf-8"))))
    fam = ROOT / "artifact/person3/per_class/family_confusion.json"
    if fam.exists():
        pool.update(NUM.findall(fam.read_text(encoding="utf-8")))
    return pool


def prose(text: str) -> str:
    """Con lai chu de doc: bo toan, bo lenh, giu doi so van ban."""
    t = body(text)
    t = re.sub(r"\$[^$]*\$", " ", t)                    # cong thuc toan
    t = re.sub(r"\\(label|ref|texttt)\{[^}]*\}", " ", t)  # nhan, tham chieu
    t = re.sub(r"\\(begin|end)\{[^}]*\}", " ", t)       # moi truong
    t = re.sub(r"\\[a-zA-Z@]+\*?", " ", t)              # lenh con lai
    t = re.sub(r"[{}\\&~^_%]", " ", t)
    return t


def count_words(text: str) -> int:
    return len([w for w in prose(text).split() if any(c.isalnum() for c in w)])



def main() -> None:
    fails = 0

    print("== 1. So tu than bai (tham khao, khong phai nguong) ==")
    for name in REPORT:
        n = count_words((VI / name).read_text(encoding="utf-8"))
        print(f"       {name:20s} {n:5d} tu")

    print("\n== 2. Con so ban VI phai truy nguoc duoc ==")
    pool = set(allowed_numbers())
    for f in sorted(VI.glob("*.tex")):
        vi_n = Counter(NUM.findall(body(f.read_text(encoding="utf-8"))))
        extra = sorted(set(vi_n) - pool)
        if extra:
            fails += 1
            print(f"  FAIL {f.name}: con so khong co nguon -> {extra}")
        else:
            print(f"  OK   {f.name:20s} {sum(vi_n.values()):3d} con so, deu truy nguoc duoc")

    print(f"\nTONG: {fails} loi")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
