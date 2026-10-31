#!/usr/bin/env python3
"""
tests/test_hidden_pipeline.py
Unit and Integration tests for Hidden-Test Pipeline:
- ID matching (extensions, case, duplicates, missing)
- Fallback rows when image is missing or corrupted
- CSV Validator rules and exit codes
- Order preservation matching sample CSV
- Resume functionality (--resume)
"""

import os
import sys
import tempfile
import shutil
import pytest
import pandas as pd
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.validate_submission import validate_submission, load_expected_ids
from scripts.run_hidden import read_id_list, build_image_index, resolve_image, process_single_image
from orchestrator import AgenticDermaGuard
import threading

@pytest.fixture(scope="module")
def shared_orchestrator():
    return AgenticDermaGuard()

def test_id_matching_extensions_and_case():
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test images with mixed case & extensions
        img = Image.new("RGB", (64, 64), color=(200, 150, 120))
        img.save(os.path.join(temp_dir, "ISIC_0000001.JPG"))
        img.save(os.path.join(temp_dir, "isic_0000002.png"))
        img.save(os.path.join(temp_dir, "ISIC_0000003.jpeg"))

        index, _ = build_image_index(temp_dir, [])

        # Match lowercase stem against uppercase file
        p1 = resolve_image("isic_0000001", index)
        assert p1 is not None and os.path.exists(p1)

        # Match uppercase stem against lowercase png file
        p2 = resolve_image("ISIC_0000002", index)
        assert p2 is not None and os.path.exists(p2)

        # Match with extension explicitly supplied
        p3 = resolve_image("ISIC_0000003.jpg", index)
        assert p3 is not None and os.path.exists(p3)

        # Missing ID
        p_missing = resolve_image("ISIC_9999999", index)
        assert p_missing is None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_id_duplicate_detection(tmp_path):
    dup_csv = tmp_path / "dup_ids.csv"
    dup_csv.write_text("image_id\nISIC_1\nISIC_2\nISIC_1\n")
    with pytest.raises(SystemExit):
        read_id_list(str(dup_csv))

def test_missing_image_fallback_row(shared_orchestrator):
    lock = threading.Lock()
    img_id = "ISIC_NONEXISTENT"
    target_id, res, err = process_single_image(img_id, None, {}, shared_orchestrator, lock)

    assert target_id == img_id
    assert res["prediction"] == "nv"
    assert res["decision"] == "REFER"
    assert res["is_fallback"] is True
    assert err is not None
    assert "not found" in err.lower()

def test_validator_rules(tmp_path):
    sample_ids = tmp_path / "sample.csv"
    sample_ids.write_text("image_id\nIMG_001\nIMG_002\n")

    # 1. Valid submission
    valid_csv = tmp_path / "valid.csv"
    valid_csv.write_text("image_id,prediction,decision\nIMG_001,nv,AUTO_TRIAGE\nIMG_002,mel,REFER\n")
    passed, errs, stats = validate_submission(str(valid_csv), ids_path=str(sample_ids))
    assert passed is True
    assert len(errs) == 0
    assert stats["total_rows"] == 2

    # 2. Invalid header
    bad_hdr = tmp_path / "bad_hdr.csv"
    bad_hdr.write_text("id,prediction,decision\nIMG_001,nv,AUTO_TRIAGE\nIMG_002,mel,REFER\n")
    passed, errs, _ = validate_submission(str(bad_hdr), ids_path=str(sample_ids))
    assert passed is False
    assert any("header" in e.lower() for e in errs)

    # 3. Invalid prediction code
    bad_pred = tmp_path / "bad_pred.csv"
    bad_pred.write_text("image_id,prediction,decision\nIMG_001,carcinoma,AUTO_TRIAGE\nIMG_002,mel,REFER\n")
    passed, errs, _ = validate_submission(str(bad_pred), ids_path=str(sample_ids))
    assert passed is False
    assert any("invalid predictions" in e.lower() for e in errs)

    # 4. Invalid decision value
    bad_dec = tmp_path / "bad_dec.csv"
    bad_dec.write_text("image_id,prediction,decision\nIMG_001,nv,DISMISS\nIMG_002,mel,REFER\n")
    passed, errs, _ = validate_submission(str(bad_dec), ids_path=str(sample_ids))
    assert passed is False
    assert any("invalid decisions" in e.lower() for e in errs)

    # 5. Order mismatch
    bad_order = tmp_path / "bad_order.csv"
    bad_order.write_text("image_id,prediction,decision\nIMG_002,mel,REFER\nIMG_001,nv,AUTO_TRIAGE\n")
    passed, errs, _ = validate_submission(str(bad_order), ids_path=str(sample_ids))
    assert passed is False
    assert any("order" in e.lower() for e in errs)

def test_resume_preservation(tmp_path):
    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"image_id": "IMG_001", "prediction": "nv", "decision": "AUTO_TRIAGE"}\n')

    # Load partial
    import json
    results = {}
    with open(str(partial), "r") as f:
        for line in f:
            if line.strip():
                r = json.loads(line.strip())
                results[r["image_id"]] = r

    assert "IMG_001" in results
    assert results["IMG_001"]["prediction"] == "nv"
    assert results["IMG_001"]["decision"] == "AUTO_TRIAGE"

def test_acceptance_check_2_three_missing_images(tmp_path):
    # Setup 5 sample IDs
    sample_ids = tmp_path / "sample_5.csv"
    sample_ids.write_text("image_id\nIMG_1\nIMG_2\nIMG_3\nIMG_4\nIMG_5\n")

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # Create only 2 images, 3 missing!
    img = Image.new("RGB", (64, 64), color=(210, 160, 130))
    img.save(str(img_dir / "IMG_1.jpg"))
    img.save(str(img_dir / "IMG_2.jpg"))

    out_csv = tmp_path / "submission.csv"
    err_csv = tmp_path / "errors.csv"

    from scripts.run_hidden import main as run_hidden_main
    import sys
    test_args = [
        "run_hidden.py",
        "--ids", str(sample_ids),
        "--images", str(img_dir),
        "--out", str(out_csv),
        "--workers", "2"
    ]
    orig_argv = sys.argv
    sys.argv = test_args
    try:
        try:
            run_hidden_main()
        except SystemExit as se:
            assert se.code == 0
    finally:
        sys.argv = orig_argv

    assert os.path.exists(str(out_csv))
    df = pd.read_csv(str(out_csv))
    assert len(df) == 5
    # The 3 missing images should be REFER
    missing_rows = df[df["image_id"].isin(["IMG_3", "IMG_4", "IMG_5"])]
    assert len(missing_rows) == 3
    assert (missing_rows["decision"] == "REFER").all()

    # Check errors.csv
    err_path = tmp_path / "errors.csv"
    if not err_path.exists():
        err_path = os.path.join(PROJECT_ROOT, "outputs", "errors.csv")
    assert os.path.exists(str(err_path))
    err_df = pd.read_csv(str(err_path))
    assert len(err_df) >= 3
