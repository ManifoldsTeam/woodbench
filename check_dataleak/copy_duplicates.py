import csv
import shutil
from pathlib import Path

def main():
    script_dir = Path(__file__).parent.resolve()
    csv_path = script_dir / "duplicate_report.csv"
    output_root = Path("duplicate_data")
    
    if not csv_path.exists():
        print(f"Lỗi: Không tìm thấy file tại {csv_path}")
        return

    print(f"Đang đọc dữ liệu từ {csv_path}...")
    total_files_copied = 0
    group_count = 0

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            md5_hash = row["Mã_MD5"]
            paths_str = row["Đường_dẫn_chính_xác"]
            
            file_paths = [p.strip().strip('"') for p in paths_str.split(", ") if p.strip()]
            
            if not file_paths:
                continue
                
            group_count += 1
            md5_dir = output_root / md5_hash

            for path_str in file_paths:
                src_path = Path(path_str)
                
                if src_path.exists() and src_path.is_file():
                    dest_path = md5_dir / src_path
                    
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(src_path, dest_path)
                    total_files_copied += 1
                else:
                    print(f"Cảnh báo: Không tìm thấy file gốc tại đường dẫn: {src_path}")

    print("\n=== HOÀN THÀNH ===")
    print(f"Đã xử lý: {group_count} nhóm ảnh bị trùng.")
    print(f"Tổng số file đã sao chép: {total_files_copied} file.")
    print(f"Tất cả dữ liệu đã được lưu vào thư mục: {output_root.resolve()}")

if __name__ == "__main__":
    main()