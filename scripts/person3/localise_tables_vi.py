"""
Sinh ban tieng Viet cua cac bang/hinh tu ban tieng Anh da co.

NGUYEN TAC: chi thay phan CHU (caption, header cot, nhan hang). MOI CON SO
duoc sao chep nguyen ven tu file goc -- script khong tinh toan lai gi ca, nen
khong the lam lech so lieu. Buoc kiem tra o cuoi doi chieu tap con so cua hai
ban va bao loi neu khac nhau.

Usage:
    python3 scripts/person3/localise_tables_vi.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_TAB, DST_TAB = ROOT / "artifact/person3/tables", ROOT / "artifact/person3/tables_vi"
SRC_FIG, DST_FIG = ROOT / "artifact/person3/figures", ROOT / "artifact/person3/figures_vi"

# caption tieng Viet co dau, khoa theo \label cua tung bang/hinh.
# LUU Y: moi con so trong caption duoc giu nguyen y het ban tieng Anh --
# ke ca dau thap phan la dau CHAM. Do la dieu kien de phep kiem tra trung
# khop tap con so o cuoi file con hieu luc.
CAPTIONS = {
    "tab:task_b": (
        "Hiệu năng phân loại ở cấp loài trên tập test Task~B (17 loài, 532 ảnh, "
        "phân hoạch gom theo khóa nguồn sao cho các ảnh cùng một khóa nguồn nằm "
        "trong cùng một tập). Giá trị là trung bình $\\pm$ độ lệch chuẩn trên 3 "
        "seed huấn luyện (42, 123, 456); độ phân tán được báo cáo là độ lệch "
        "chuẩn trên ba lần chạy, không phải ước lượng khoảng. Giá trị tốt nhất "
        "mỗi cột in đậm."
    ),
    "tab:ablation": (
        "Ảnh hưởng của từng ablation lên hiệu năng test Task~B, đối chiếu với cấu "
        "hình chính ở \\emph{cùng seed và cùng backbone}. Hai cột trái cho điểm "
        "tuyệt đối của lần chạy chính với seed 42; mỗi cột $\\Delta$ là ablation "
        "trừ đi lần chạy chính cùng backbone, tính bằng điểm phần trăm. Mỗi "
        "ablation chỉ huấn luyện với một seed (42), nên không ước lượng được độ "
        "phân tán cho các lần chạy này và các chênh lệch dưới đây là ước lượng "
        "điểm từ một lần chạy chứ không phải hiệu ứng đã lấy trung bình. Để đối "
        "chiếu, độ lệch chuẩn giữa các seed của các lần chạy chính ở "
        "Bảng~\\ref{tab:task_b} có độ lớn tương đương với phần lớn các chênh "
        "lệch này."
    ),
    "tab:perspecies_full": (
        "Điểm F1 theo từng loài (\\%) trên tập test Task~B (17 loài, 532 ảnh), "
        "cho từng backbone. Giá trị là trung bình $\\pm$ độ lệch chuẩn trên 3 seed "
        "huấn luyện chính (42, 123, 456); độ phân tán là độ lệch chuẩn trên ba "
        "lần chạy, không phải ước lượng khoảng. Các loài được nhóm theo họ. $n$ "
        "là số ảnh test của mỗi loài."
    ),
    "tab:efficiency": (
        "Độ phức tạp của năm kiến trúc ở độ phân giải đầu vào $224\\times224$ dùng "
        "xuyên suốt, với đầu ra 17 lớp. Số tham số, số phép nhân cộng (MACs) và "
        "kích thước mô hình là thuộc tính của kiến trúc: chúng được đo trên mô "
        "hình khởi tạo ngẫu nhiên và không phụ thuộc vào bất kỳ checkpoint đã "
        "huấn luyện nào. Báo cáo MACs thay vì FLOPs; script đo ghi "
        "$\\text{FLOPs} = 2\\times\\text{MACs}$, và các giá trị dưới đây là cột đó "
        "chia đôi, để so sánh trực tiếp được với quy ước dùng trong tài liệu về "
        "kiến trúc. Độ trễ suy luận không được báo cáo: nó đo trên phần cứng khác "
        "với phần cứng huấn luyện và đánh giá, nên không so sánh được với bất kỳ "
        "con số nào khác trong bài."
    ),
    "fig:confusion": (
        "Ma trận nhầm lẫn của ConvNeXt-Tiny trên tập test Task~B. Các count được "
        "cộng theo từng phần tử qua ba seed chính (42, 123, 456), cho "
        "$532\\times3 = 1{,}596$ dự đoán; mỗi seed đóng góp như nhau. Màu nền ô "
        "là tỷ lệ phần trăm đã chuẩn hóa theo hàng của số ảnh thuộc lớp thật, với "
        "mẫu số là tổng hàng của các count đã cộng; số ghi trong mỗi ô là count "
        "thô đã cộng. Các loài được sắp theo họ và các đường kẻ đen đánh dấu ranh "
        "giới họ. Ô ngoài đường chéo được viền đỏ. ConvNeXt-Tiny được chọn theo "
        "tiêu chí cố định trước (macro-F1 trung bình cao nhất trên ba seed ở "
        "Bảng~\\ref{tab:task_b}), chứ không phải bằng cách xem ma trận này. "
        "Ba ma trận riêng từng seed được công bố kèm bản gộp để kiểm tra lại được "
        "phép cộng."
    ),
    "fig:perclass": (
        "Điểm F1 theo từng loài trên tập test Task~B, mỗi panel một kiến trúc. "
        "Mỗi chấm là trung bình trên ba seed chính (42, 123, 456) và mỗi thanh là "
        "độ lệch chuẩn trên các seed đó; thanh này là thước đo phân tán trên ba "
        "lần chạy, không phải ước lượng khoảng. Các loài được sắp theo họ, các "
        "khối họ tách nhau bằng đường kẻ ngang và được ghi nhãn ở lề trái. Trục "
        "ngang mã hóa bằng vị trí nên không bắt đầu từ không; các giá trị dưới "
        "90\\,\\% được ghi nhãn trực tiếp. Các kiến trúc được tách thành panel "
        "riêng thay vì phân biệt bằng màu, để danh tính được mang bởi vị trí và "
        "tiêu đề panel."
    ),
    "fig:gradcam": (
        "Bản đồ Grad-CAM cho ConvNeXt-Tiny, Swin-Tiny và MobileNetV3-Large trên "
        "năm ảnh test Task~B, mỗi họ một ảnh: \\textit{Araucaria araucana}, "
        "\\textit{Calocedrus rupestris}, \\textit{Pinus radiata}, "
        "\\textit{Podocarpus pilgeri} và \\textit{Taxus wallichiana}. Ảnh được "
        "chọn theo quy tắc cố định trước: trong số những ảnh cả ba mô hình đều "
        "phân loại đúng, lấy ảnh gần nhất với mức độ tin cậy trung vị của "
        "ConvNeXt-Tiny trong họ của nó, phá hòa bằng đường dẫn ảnh. Không ảnh nào "
        "được chọn vì heatmap trông đẹp. Mỗi bản đồ đều lấy từ checkpoint seed 42 "
        "của các lần chạy ở Bảng~\\ref{tab:task_b}. Các vùng quy gán trải rộng và "
        "ba kiến trúc làm nổi bật những vùng khác nhau trên cùng một ảnh; Grad-CAM "
        "định vị nơi gradient tựa vào và tự nó không xác định được đặc trưng nào "
        "đang được sử dụng."
    ),
    "fig:tsne": (
        "Phép nhúng t-SNE hai chiều của đặc trưng ConvNeXt-Tiny cho cả 532 ảnh "
        "test Task~B, lấy từ lần chạy seed 42 ở Bảng~\\ref{tab:task_b}. Đặc trưng "
        "là kích hoạt 768 chiều trước lớp logit, được chuẩn hóa, giảm xuống 50 "
        "chiều bằng PCA rồi xuống hai chiều bằng t-SNE với perplexity 30, khởi tạo "
        "PCA và seed ngẫu nhiên cố định. t-SNE chỉ bảo toàn láng giềng cục bộ: "
        "khoảng cách và diện tích cụm trong hình không phải metric, và phép nhúng "
        "được trình bày như một cách nhìn định tính về biểu diễn chứ không phải "
        "bằng chứng về tính tách được."
    ),
}

# header cot / nhan trong bang
WORDS = [
    (r"\textbf{Architecture}", r"\textbf{Kiến trúc}"),
    (r"\textbf{Top-1 Accuracy (\%)}", r"\textbf{Độ chính xác Top-1 (\%)}"),
    (r"\textbf{Macro-F1 (\%)}", r"\textbf{Macro-F1 (\%)}"),
    (r"\textbf{Species}", r"\textbf{Loài}"),
    (r"\textbf{Params (M)}", r"\textbf{Tham số (M)}"),
    (r"\textbf{MACs (G)}", r"\textbf{MACs (G)}"),
    (r"\textbf{Model size (MB)}", r"\textbf{Kích thước (MB)}"),
    (r"\textbf{Main (seed 42)}", r"\textbf{Chính (seed 42)}"),
    (r"\textbf{Discrete $90^\circ$ rotation}", r"\textbf{Xoay rời rạc $90^\circ$}"),
    (r"\textbf{Label smoothing $0.0$}", r"\textbf{Label smoothing $0.0$}"),
    (r"\textbf{Label smoothing $0.2$}", r"\textbf{Label smoothing $0.2$}"),
    (r"Acc.\ (\%)", r"Acc.\ (\%)"),
    (r"$\Delta$Acc.", r"$\Delta$Acc."),
]


def replace_caption(tex: str, new_caption: str) -> str:
    """Thay noi dung \\caption{...} bang cach dem ngoac de tim dung dau dong."""
    i = tex.index(r"\caption{")
    j = i + len(r"\caption{")
    depth = 1
    while depth:
        if tex[j] == "{":
            depth += 1
        elif tex[j] == "}":
            depth -= 1
        j += 1
    return tex[:i] + r"\caption{" + new_caption + "}" + tex[j:]


def localise(tex: str) -> str:
    label = re.search(r"\\label\{([^}]+)\}", tex).group(1)
    if label in CAPTIONS:
        tex = replace_caption(tex, CAPTIONS[label])
    for en, vi in WORDS:
        tex = tex.replace(en, vi)
    # GIU NGUYEN ten nhan: ban EN va VI duoc bien dich thanh hai tai lieu
    # rieng biet, nen khong co xung dot nhan; giu nguyen de cac \ref trong
    # sections_vi/ (vd \ref{tab:task_b}) van tro dung dich.
    return tex


def numbers(tex: str) -> Counter:
    body = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))
    return Counter(re.findall(r"\d+\.?\d*", body))


def main() -> None:
    fails = 0
    for src, dst in ((SRC_TAB, DST_TAB), (SRC_FIG, DST_FIG)):
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.tex")):
            en = f.read_text(encoding="utf-8")
            vi = localise(en)
            # KIEM TRA: tap con so phai giong het ban goc
            a, b = numbers(en), numbers(vi)
            if a != b:
                print(f"[FAIL] {f.name}: con so lech {(a-b) or ''} {(b-a) or ''}")
                fails += 1
                continue
            (dst / f.name).write_text(vi, encoding="utf-8")
            print(f"[OK] {dst.name}/{f.name}  ({sum(a.values())} con so giu nguyen)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
