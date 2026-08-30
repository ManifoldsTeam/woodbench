"""
Tinh co cau nham lan theo HO tu registry bat bien, tach loi NOI HO va
loi NGOAI HO.

Ly do ton tai: phan Ban luan cua ban tieng Viet phat bieu ve nham lan noi
ho (vi du cua Cupressaceae va cua Pinaceae). Nhung con so do khong co san
trong bat ky bang nao da sinh, nen chung duoc tinh o day va ghi ra file de
truy nguoc duoc, thay vi go tay vao ban thao.

Nguon : confusion_matrices.npz cua registry, chi lay 15 lan chay CHINH
        (khoa co '__main__' va ket thuc bang '__test'), dung dung tap
        lan chay ma Bang 1 va Bang 3 dung.
Ra    : artifact/person3/per_class/family_confusion.json

Usage:
    .venv/bin/python scripts/person3/make_family_confusion.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "docs/reports/experiments/phase_1b/final_b_new_recomputed/20260819T043733Z"
OUT = ROOT / "artifact/person3/per_class/family_confusion.json"

# So do NAM ho dung xuyen suot Muc Ket qua (khac so do bon ho cua nhiem vu
# cap ho trong ke hoach du an; xem phan ban luan ve pham vi).
FAMILY = {
    "Araucaria araucana": "Araucariaceae",
    "Callitris columellaris": "Cupressaceae",
    "Calocedrus rupestris": "Cupressaceae",
    "Chamaecyparis obtusa": "Cupressaceae",
    "Cryptomeria japonica": "Cupressaceae",
    "Cunninghamia lanceolata": "Cupressaceae",
    "Fokienia hodginsii": "Cupressaceae",
    "Thujopsis dolabrata": "Cupressaceae",
    "Xanthocyparis vietnamensis": "Cupressaceae",
    "Picea glauca": "Pinaceae",
    "Pinus cernua": "Pinaceae",
    "Pinus dalatensis": "Pinaceae",
    "Pinus krempfii": "Pinaceae",
    "Pinus radiata": "Pinaceae",
    "Pseudotsuga brevifolia": "Pinaceae",
    "Podocarpus pilgeri": "Podocarpaceae",
    "Taxus wallichiana": "Taxaceae",
}
ORDER = ["Araucariaceae", "Cupressaceae", "Pinaceae", "Podocarpaceae", "Taxaceae"]


def class_names() -> list[str]:
    idx = {}
    with (REG / "per_class_metrics.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx[int(row["class_index"])] = row["class_name"]
    return [idx[i] for i in range(len(idx))]


def main() -> None:
    names = class_names()
    assert set(names) == set(FAMILY), "danh sach loai lech voi so do ho"

    data = np.load(REG / "confusion_matrices.npz", allow_pickle=True)
    keys = sorted(k for k in data.files if "__main__" in k and k.endswith("__test"))
    assert len(keys) == 15, f"can dung 15 lan chay chinh, dang co {len(keys)}"

    total = sum(data[k].astype(int) for k in keys)
    n_pred = int(total.sum())
    n_err = int(n_pred - np.trace(total))

    intra, cross, support = Counter(), Counter(), Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    for i, true_name in enumerate(names):
        support[FAMILY[true_name]] += int(total[i].sum())
        for j, pred_name in enumerate(names):
            if i == j or total[i, j] == 0:
                continue
            count = int(total[i, j])
            pairs[(true_name, pred_name)] += count
            bucket = intra if FAMILY[true_name] == FAMILY[pred_name] else cross
            bucket[FAMILY[true_name]] += count

    families = {}
    for fam in ORDER:
        err = intra[fam] + cross[fam]
        families[fam] = {
            "n_predictions": support[fam],
            "errors_within_family": intra[fam],
            "errors_across_family": cross[fam],
            "errors_total": err,
            "error_rate_percent": round(100 * err / support[fam], 2),
        }

    out = {
        "source_registry": str(REG.relative_to(ROOT)),
        "runs": keys,
        "n_runs": len(keys),
        "n_predictions": n_pred,
        "n_errors": n_err,
        "family_scheme": "five-family",
        "families": families,
        "top_confusions": [
            {
                "true": a,
                "predicted": b,
                "count": c,
                "within_family": FAMILY[a] == FAMILY[b],
                "family_true": FAMILY[a],
                "family_predicted": FAMILY[b],
            }
            for (a, b), c in pairs.most_common(10)
        ],
        "within_family_pairs": {
            fam: [
                {"true": a, "predicted": b, "count": c}
                for (a, b), c in sorted(pairs.items(), key=lambda kv: -kv[1])
                if FAMILY[a] == fam and FAMILY[b] == fam
            ]
            for fam in ORDER
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[OK] {OUT.relative_to(ROOT)}")
    print(f"     {len(keys)} lan chay chinh, {n_pred} du doan, {n_err} loi")
    for fam in ORDER:
        f = families[fam]
        print(f"     {fam:<16} noi ho {f['errors_within_family']:>3}"
              f"  ngoai ho {f['errors_across_family']:>3}"
              f"  ty le {f['error_rate_percent']:>5.2f}%")


if __name__ == "__main__":
    main()
