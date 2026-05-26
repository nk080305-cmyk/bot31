import os

from bot import debug_export


def test_debug_export_writes_artifacts_with_context(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_DEBUG", "1")
    monkeypatch.setattr(debug_export, "DEBUG_DIR", str(tmp_path))

    token = debug_export.set_context("case-1", "20260526_100000Z")
    try:
        text_path = debug_export.write_text("ocr.txt", "hello")
        json_path = debug_export.write_json("parsed_fields.json", {"a": 1})
    finally:
        debug_export.reset_context(token)

    assert text_path is not None
    assert json_path is not None
    assert os.path.basename(text_path) == "case-1_20260526_100000Z_ocr.txt"
    assert (tmp_path / "case-1_20260526_100000Z_ocr.txt").read_text(encoding="utf-8") == "hello"
    assert (tmp_path / "case-1_20260526_100000Z_parsed_fields.json").exists()


def test_debug_export_prunes_old_case_groups(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_DEBUG", "1")
    monkeypatch.setattr(debug_export, "DEBUG_DIR", str(tmp_path))
    monkeypatch.setattr(debug_export, "OCR_DEBUG_MAX_CASES", 1)

    debug_export.write_text("ocr.txt", "first", case_id="case-old", timestamp="20260526_090000Z")
    debug_export.write_text("ocr.txt", "new", case_id="case-new", timestamp="20260526_100000Z")

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["case-new_20260526_100000Z_ocr.txt"]
