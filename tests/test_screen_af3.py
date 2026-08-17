"""Tests for AF3 batch screening."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd


def test_collect_structures_from_dir(tmp_path):
    from physrna_filter.structure.af3_io import collect_structure_inputs

    (tmp_path / "a.pdb").write_text("ATOM")
    (tmp_path / "b.cif").write_text("data")
    (tmp_path / "skip.txt").write_text("x")
    files = collect_structure_inputs([str(tmp_path)])
    names = {f.name for f in files}
    assert names == {"a.pdb", "b.cif"}


def test_screen_af3_writes_csv(tmp_path):
    from physrna_filter.validation import screen_af3

    pdb = tmp_path / "test.pdb"
    pdb.write_text("ATOM")

    fake_result = {
        "combined_verdict": "PASS",
        "confidence": 0.9,
        "rmsd_score": 1.5,
        "rmsd_verdict": "PASS",
        "geom_score": 20.0,
        "geom_verdict": "PASS",
        "contact_energy": -5.0,
        "contact_verdict": "PASS",
        "clash_n_severe": 0,
        "clash_verdict": "PASS",
        "gt_score": -1.0,
        "gt_score_raw": -10.0,
        "gt_score_norm": -1.0,
        "gt_score_per_nt": -0.5,
        "gt_verdict": "PASS",
        "gt_physics_only": False,
        "bio_verdict": "UNKNOWN",
        "n_prot_rna_edges": 12,
        "interface_residues": [("A", 1)],
    }

    out = tmp_path / "out.csv"
    with patch.object(screen_af3, "run_pipeline", return_value=fake_result):
        df = screen_af3.screen_af3_structures(
            inputs=[str(pdb)],
            output_csv=str(out),
            quiet=True,
        )
    assert len(df) == 1
    assert out.exists()
    loaded = pd.read_csv(out)
    assert loaded.iloc[0]["combined_verdict"] == "PASS"
