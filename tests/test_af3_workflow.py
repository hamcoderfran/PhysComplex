"""Tests for streamlined AF3 workflow utilities."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


def test_af3_confidence_from_bundled_zip():
    from physrna_filter.structure.af3_confidence import (
        af3_iptm,
        read_af3_confidence,
        sequences_from_af3_zip,
    )

    zips = [p for p in Path("af3_predictions").glob("*.zip") if p.stat().st_size > 128]
    if not zips:
        return
    z = next((p for p in zips if "p1" in p.name.lower()), zips[0])
    conf = read_af3_confidence(z)
    assert "iptm" in conf
    assert 0.0 <= af3_iptm(z) <= 1.0
    name, protein, rna = sequences_from_af3_zip(z)
    assert name
    assert protein and len(protein) > 10
    assert rna and len(rna) >= 3


def test_rbp_filter_excludes_wrong_jobs():
    from physrna_filter.data.candidate_manifest import (
        _is_wrong_partner_control,
        _rbp_matches,
        discover_af3_jobs,
    )

    assert _is_wrong_partner_control("N6_Lin28_with_MS2_RNA")
    assert not _is_wrong_partner_control("P6_Lin28_prelet7")
    assert _rbp_matches("LIN28A", "LIN28A", "P6_Lin28_prelet7")
    assert _rbp_matches("LIN28A", None, "P6_Lin28_prelet7")
    assert not _rbp_matches("LIN28A", "U1A", "P1_U1A_hairpin")

    if not Path("af3_predictions").is_dir():
        return
    lin28 = discover_af3_jobs("af3_predictions", rbp_filter="LIN28A")
    names = {r["job_name"] for r in lin28}
    assert all("lin28" in n.lower() for n in names)
    assert not any("with_" in n.lower() for n in names)


def test_generate_swap_panel():
    from physrna_filter.data.candidate_manifest import generate_swap_panel

    cands = [
        {"id": "C1", "job_name": "a", "zip_file": "a.zip", "rbp_name": "LIN28A",
         "rna_sequence": "AAAA", "partner_group": "lin28"},
        {"id": "C2", "job_name": "b", "zip_file": "b.zip", "rbp_name": "LIN28A",
         "rna_sequence": "UUUU", "partner_group": "lin28"},
    ]
    panel = generate_swap_panel(cands, max_negatives_per_positive=1)
    assert any(e["label"] == "negative" for e in panel)
    assert any(e.get("positive_id") == "C1" for e in panel)


def test_html_report_from_csv(tmp_path):
    from physrna_filter.validation.report_af3 import build_html_report, write_html_report

    csv = tmp_path / "t.csv"
    pd.DataFrame([
        {"rank": 1, "job_name": "test", "af3_iptm": 0.8, "composite_score": -0.1,
         "combined_verdict": "WARN", "bio_verdict": "PASS"},
    ]).to_csv(csv, index=False)
    html = write_html_report(csv, tmp_path / "r.html")
    text = html.read_text(encoding="utf-8")
    assert "PhysRNA" in text
    assert "test" in text


def test_unified_score():
    from physrna_filter.validation.screen_af3 import _unified_score

    assert _unified_score({"af3_iptm": 0.9, "composite_score": -0.2}) > _unified_score(
        {"af3_iptm": 0.5, "composite_score": 2.0}
    )


def test_ensure_public_checkpoint():
    from physrna_filter.validation.download_gt_checkpoint import ensure_public_checkpoint

    path = ensure_public_checkpoint()
    assert path.is_file()
    assert path.stat().st_size > 10_000


def test_pabp_ugua_still_fails():
    from physrna_filter.analysis.biological_plausibility import assess_biological_plausibility

    r = assess_biological_plausibility(rna_sequence="UUGUAUAU", rbp_name="PABPC1")
    assert r.verdict == "FAIL"
