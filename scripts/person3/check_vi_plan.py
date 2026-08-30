"""
Doi chieu ban tieng Viet voi checklist cua Nguoi 3 trong ke hoach du an.

Hai muc kiem duoc tu dong hoa:

  1. "Moi nhan xet dinh luong tro toi bang/hinh va nguon registry."
     Do o cap TIEU MUC: moi tieu muc co chua so lieu phai dan it nhat mot
     bang/hinh qua \\ref, hoac neu ro nguon registry. Do o cap tung doan van
     thi qua gat, vi mot doan nối tiep doan truoc thuong khong lap lai
     tham chieu. Phep truy nguyen tung con so nam o check_vi_prose.py.

  2. "Chi viet sections/00_abstract.tex sau cung; khong dua ket qua chua
     xuat hien trong Results." Moi con so cua Muc 0 phai co mat trong
     Muc 4. Day la rang buoc chat hon phep truy nguyen chung o
     check_vi_prose.py, vi no chi cho phep dung Muc 4 lam nguon.

  3. "Viet sections/05_discussion.tex" voi sau chu de bat buoc.
     Kiem tra du sau nhan tieu muc, va liet ke cac tieu muc them ngoai
     ke hoach de nguoi dung tu quyet dinh giu hay bo.

Usage:
    python3 scripts/person3/check_vi_plan.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VI = ROOT / "paper_draft_person3/sections_vi"

# Sau chu de bat buoc cua Muc 5 theo ke hoach.
REQUIRED_DISCUSSION = {
    "vi:sec:discussion:ranking":   "xep hang backbone va bien dong giua seed",
    "vi:sec:discussion:rotation":  "rotation khong cai thien dong deu",
    "vi:sec:discussion:oneseed":   "gioi han ablation mot seed",
    "vi:sec:discussion:originkey": "anh huong cua origin-key construction",
    "vi:sec:discussion:external":  "chua co external / field validation",
    "vi:sec:discussion:scope":     "Task A / resolution neu quyet dinh bo",
}

# So khong phai "nhan xet dinh luong": nam, commit, phien ban thu vien.
CONTEXT_ONLY = re.compile(
    r"20\d\d-\d\d-\d\d|20\d\d|\b[0-9a-f]{6,}\b|v?\d+\.\d+\.\d+|"
    r"cu\d+|T\d{6}Z|\\texttt\{[^}]*\}"
)
NUM = re.compile(r"\d+(?:\.\d+)?")


def body(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))


def check_quantitative(text: str) -> list[str]:
    """Tra ve danh sach tieu muc co so lieu nhung khong dan bang/hinh."""
    bad = []
    chunks = re.split(r"(\\subsection\{[^}]*\})", body(text))
    # chunks: [truoc tieu muc dau, tieu de 1, than 1, tieu de 2, than 2, ...]
    pairs = [("(dau muc)", chunks[0])] if chunks[0].strip() else []
    pairs += [(re.sub(r"\\subsection\{|\}$", "", chunks[i]), chunks[i + 1])
              for i in range(1, len(chunks) - 1, 2)]
    for title, chunk in pairs:
        stripped = CONTEXT_ONLY.sub(" ", chunk)
        if not NUM.search(stripped):
            continue
        has_ref = re.search(r"\\ref\{(tab|fig):", chunk)
        has_registry = "registry" in chunk.lower()
        if not (has_ref or has_registry):
            bad.append(title)
    return bad


def main() -> None:
    fails = 0

    print("== 1. Tieu muc co so lieu phai dan bang/hinh hoac nguon registry ==")
    for f in sorted(VI.glob("*.tex")):
        bad = check_quantitative(f.read_text(encoding="utf-8"))
        print(f"  {'OK  ' if not bad else 'FAIL'} {f.name:20s} "
              f"{len(bad)} tieu muc thieu" + (f": {bad}" if bad else ""))
        fails += len(bad)

    print("\n== 2. Muc 0 khong duoc co ket qua ngoai Muc 4 ==")
    abs_n = Counter(NUM.findall(body((VI / "00_abstract.tex").read_text(encoding="utf-8"))))
    res_n = set(NUM.findall(body((VI / "04_results.tex").read_text(encoding="utf-8"))))
    extra = sorted(set(abs_n) - res_n)
    if extra:
        fails += 1
        print(f"  FAIL con so co trong Muc 0 nhung khong co trong Muc 4: {extra}")
    else:
        print(f"  OK   {sum(abs_n.values())} con so cua Muc 0 deu xuat hien trong Muc 4")

    print("\n== 3. Muc 5 phai co du sau chu de theo ke hoach ==")
    disc = (VI / "05_discussion.tex").read_text(encoding="utf-8")
    labels = set(re.findall(r"\\label\{(vi:sec:discussion:[a-z]+)\}", disc))
    for lab, desc in REQUIRED_DISCUSSION.items():
        ok = lab in labels
        fails += not ok
        print(f"  {'OK  ' if ok else 'THIEU'} {desc}")
    extra = sorted(labels - set(REQUIRED_DISCUSSION) - {"vi:sec:discussion"})
    if extra:
        print("  (them ngoai ke hoach, giu hay bo do nguoi dung quyet dinh):")
        for lab in extra:
            print(f"      {lab}")

    print(f"\nTONG: {fails} loi")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
