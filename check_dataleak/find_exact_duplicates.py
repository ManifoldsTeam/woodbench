#!/usr/bin/env python3
import hashlib
import csv
from pathlib import Path
from collections import defaultdict

def get_md5(path: Path) -> str:
    """Tính mã băm MD5 để xác định nội dung file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    data_root = Path("data") # Trỏ đến thư mục data của bạn
    print(f"Đang quét toàn bộ ảnh trong: {data_root.resolve()} ...")
    
    hash_to_paths = defaultdict(list)
    
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for path in data_root.rglob(ext):
            if ".cache" in path.parts:
                continue
            if path.is_file():
                h = get_md5(path)
                hash_to_paths[h].append(path)

    duplicates = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}

    if not duplicates:
        print("Tuyệt vời! Không phát hiện ảnh nào bị trùng lặp.")
        return

    out_csv = "duplicate_report.csv"
    with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Mã_MD5", "Số_lượng_trùng", "Đường_dẫn_chính_xác"])
        
        for h, paths in duplicates.items():
            paths_str = ", ".join(str(p) for p in paths)
            writer.writerow([h, len(paths), paths_str])

    print(f"\nPhát hiện {len(duplicates)} nhóm ảnh bị trùng lặp!")
    print(f"Đã xuất danh sách chi tiết ra file: {out_csv}")
    print("Bạn có thể mở file này bằng Excel hoặc VS Code để xem vị trí chính xác của từng ảnh.")

if __name__ == "__main__":
    main()