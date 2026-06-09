#!/usr/bin/env python3
import hashlib
import shutil
from pathlib import Path

def get_md5(path: Path) -> str:
    """Tính mã băm MD5 để xác định nội dung file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    project_root = Path(__file__).resolve().parent.parent
    
    data_root = project_root / "data"
    output_root = project_root / "clean_data"
    
    if not data_root.exists():
        print(f"Lỗi: Không tìm thấy thư mục {data_root}")
        return

    print(f"Bắt đầu tạo dataset sạch tại: {output_root.resolve()}")
    print("Đang quét và xử lý ảnh (có thể mất vài phút)...\n")
    
    seen_hashes = set()
    copied_count = 0
    duplicate_skipped_count = 0
    
    for file_path in data_root.rglob("*"):
        if not file_path.is_file():
            continue
        if ".cache" in file_path.parts:
            continue
        if file_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        class_name = file_path.parent.name
        
        file_hash = get_md5(file_path)
        if file_hash in seen_hashes:
            duplicate_skipped_count += 1
            continue
            
        seen_hashes.add(file_hash)
        
        class_dir = output_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = class_dir / file_path.name
        
        counter = 1
        while dest_path.exists():
            dest_path = class_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
            counter += 1
            
        shutil.copy2(file_path, dest_path)
        copied_count += 1

    print("=== HOÀN THÀNH ===")
    print(f"Tổng số ảnh duy nhất đã được gom: {copied_count}")
    print(f"Số lượng ảnh trùng lặp bị loại bỏ: {duplicate_skipped_count}")
    print(f"Dataset mới đã sẵn sàng tại: {output_root}")

if __name__ == "__main__":
    main()