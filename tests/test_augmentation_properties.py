"""Property-based tests cho augmentation pipeline.

Su dung Hypothesis de kiem tra cac tinh chat bat bien (invariants)
cua pipeline augmentation Task A.
"""

import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# Import truc tiep module directory_copier de tranh circular import qua src/__init__.py
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# Load module truc tiep khong qua src package init
_spec = importlib.util.spec_from_file_location(
    "directory_copier",
    _project_root / "src" / "data" / "directory_copier.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
DirectoryCopier = _module.DirectoryCopier


# --- Strategies ---

# Strategy sinh ten family hop le
family_names_st = st.sampled_from(
    ["Cupressaceae", "Pinaceae", "Podocarpaceae", "Taxaceae"]
)

# Strategy sinh ten species hop le (ten khoa hoc don gian)
species_names_st = st.from_regex(r"[A-Z][a-z]{3,10} [a-z]{3,10}", fullmatch=True)

# Strategy sinh noi dung file anh gia lap (bytes ngau nhien)
file_content_st = st.binary(min_size=100, max_size=1000)


@st.composite
def source_directory_with_symlinks(draw):
    """Strategy tao cau truc thu muc nguon voi mix of symlinks va real files.

    Tao mot thu muc tam co cau truc:
        {split}/{family}/{species}/file.png

    Mot so file la real files, mot so la symlinks tro den real files.
    """
    # So luong families va species
    num_families = draw(st.integers(min_value=1, max_value=3))
    families = draw(
        st.lists(
            family_names_st,
            min_size=num_families,
            max_size=num_families,
            unique=True,
        )
    )

    split = draw(st.sampled_from(["train", "val", "test"]))

    # Cau hinh so file va ti le symlinks
    num_files_per_species = draw(st.integers(min_value=1, max_value=5))
    symlink_ratio = draw(st.floats(min_value=0.0, max_value=1.0))

    # Tao thu muc tam
    tmp_dir = tempfile.mkdtemp()
    src_root = Path(tmp_dir) / "source"
    src_root.mkdir()

    # Tao thu muc chua file goc de symlinks tro den
    raw_dir = Path(tmp_dir) / "raw"
    raw_dir.mkdir()

    created_files = []

    for family in families:
        # Moi family co 1 species
        species_name = draw(species_names_st)
        species_dir = src_root / split / family / species_name
        species_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_files_per_species):
            filename = f"img_{i:03d}.png"
            content = draw(file_content_st)

            # Quyet dinh file nay la real file hay symlink
            is_symlink = draw(
                st.floats(min_value=0.0, max_value=1.0)
            ) < symlink_ratio

            if is_symlink:
                # Tao real file trong raw_dir, sau do tao symlink
                raw_file = raw_dir / family / species_name / filename
                raw_file.parent.mkdir(parents=True, exist_ok=True)
                raw_file.write_bytes(content)

                link_path = species_dir / filename
                link_path.symlink_to(raw_file)
            else:
                # Tao real file truc tiep
                real_file = species_dir / filename
                real_file.write_bytes(content)

            created_files.append(species_dir / filename)

    return tmp_dir, src_root, split, families, created_files


# --- Property Tests ---


# Feature: task-a-augmentation, Property 1: No symlinks in output
class TestNoSymlinksInOutput:
    """Property 1: Tat ca file trong output phai la real files, khong co symlinks.

    Validates: Requirements 1.2
    """

    @given(data=source_directory_with_symlinks())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_output_contains_no_symlinks(self, data):
        """Voi bat ky cau truc thu muc nguon nao (co mix symlinks va real files),
        sau khi copy, output khong duoc chua bat ky symlink nao.

        **Validates: Requirements 1.2**
        """
        tmp_dir, src_root, split, families, _ = data

        try:
            dst_root = Path(tmp_dir) / "output"
            dst_root.mkdir(parents=True, exist_ok=True)

            copier = DirectoryCopier()
            copier.copy_split(
                src_root=src_root,
                dst_root=dst_root,
                split=split,
                families=families,
            )

            # Kiem tra tat ca file trong output khong phai symlink
            for root, _dirs, files in os.walk(dst_root):
                for filename in files:
                    file_path = Path(root) / filename
                    assert not os.path.islink(
                        file_path
                    ), f"Tim thay symlink trong output: {file_path}"
                    # Dam bao file ton tai va la file thuc
                    assert file_path.is_file(), (
                        f"File khong ton tai hoac khong phai file thuc: {file_path}"
                    )
        finally:
            # Don dep thu muc tam
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)


# Feature: task-a-augmentation, Property 2: Non-augmented files are byte-identical
class TestByteIdenticalCopies:
    """Property 2: File copy phai byte-identical voi source (sau khi resolve symlinks).

    Validates: Requirements 1.3, 1.4
    """

    @given(data=source_directory_with_symlinks())
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None,
    )
    def test_copied_files_are_byte_identical_to_resolved_source(self, data):
        """Voi bat ky cau truc thu muc nguon nao (co mix symlinks va real files),
        sau khi copy, moi file output phai co noi dung byte-identical voi file
        source tuong ung (da resolve symlinks).

        **Validates: Requirements 1.3, 1.4**
        """
        tmp_dir, src_root, split, families, _ = data

        try:
            dst_root = Path(tmp_dir) / "output"
            dst_root.mkdir(parents=True, exist_ok=True)

            copier = DirectoryCopier()
            copier.copy_split(
                src_root=src_root,
                dst_root=dst_root,
                split=split,
                families=families,
            )

            # Duyet qua tat ca file trong output, so sanh voi source
            for root, _dirs, files in os.walk(dst_root):
                for filename in files:
                    dst_file = Path(root) / filename

                    # Xac dinh duong dan source tuong ung
                    # dst_file co dang: output/{split}/{family}/{species}/file.png
                    # source tuong ung: source/{split}/{family}/{species}/file.png
                    rel_path = dst_file.relative_to(dst_root)
                    src_file = src_root / rel_path

                    # Resolve symlinks de lay noi dung thuc
                    resolved_src = src_file.resolve()

                    assert resolved_src.exists(), (
                        f"Khong tim thay source file: {resolved_src}"
                    )

                    # So sanh noi dung byte-by-byte
                    src_bytes = resolved_src.read_bytes()
                    dst_bytes = dst_file.read_bytes()

                    assert dst_bytes == src_bytes, (
                        f"File output khong byte-identical voi source.\n"
                        f"  Output: {dst_file}\n"
                        f"  Source (resolved): {resolved_src}\n"
                        f"  Output size: {len(dst_bytes)}, Source size: {len(src_bytes)}"
                    )
        finally:
            # Don dep thu muc tam
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
