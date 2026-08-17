"""Tests for literature-mined training data."""

from __future__ import annotations


def test_fetch_literature_loads_entries():
    from physrna_filter.data.fetch_literature import fetch_literature

    df = fetch_literature()
    assert len(df) >= 1
    assert (df["source"] == "literature").all()
    assert "F259A" in df["mutation"].values


def test_fetch_training_data_includes_literature(monkeypatch):
    import pandas as pd
    from physrna_filter.data import fetch_training_data as ftd

    pronab = pd.DataFrame({
        "pdb_id": ["1URN"],
        "mutation": ["Y13A"],
        "ddg": [2.8],
        "method": ["ITC"],
        "type": ["RNA-Protein"],
        "wt_aa": ["Y"],
        "position": [13],
        "mut_aa": ["A"],
    })
    lit = pd.DataFrame({
        "pdb_id": ["7Q33"],
        "mutation": ["F259A"],
        "ddg": [0.9],
        "method": ["ITC"],
        "type": ["RNA-Protein"],
        "source": ["literature"],
        "wt_aa": ["F"],
        "position": [259],
        "mut_aa": ["A"],
    })

    monkeypatch.setattr(ftd, "fetch_pronab", lambda **kw: pronab)
    monkeypatch.setattr(ftd, "fetch_nabe", lambda **kw: pd.DataFrame())
    monkeypatch.setattr(ftd, "fetch_literature", lambda: lit)

    merged = ftd.fetch_training_data(include_nabe=False, include_literature=True)
    assert len(merged) == 2
    assert set(merged["source"]) == {"pronab", "literature"}
