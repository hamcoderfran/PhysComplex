"""Tests for Nabe / merged training data fetchers."""

from __future__ import annotations


def test_normalize_mutation_string():
    from physrna_filter.data.fetch_nabe import normalize_mutation_string

    assert normalize_mutation_string("D 210A") == "D210A"
    assert normalize_mutation_string("y13a") == "Y13A"


def test_parse_mutation_with_space():
    from physrna_filter.data.fetch_pronab import parse_mutation_string

    assert parse_mutation_string("D 210A") == ("D", 210, "A")


def test_fetch_training_data_dedup(monkeypatch):
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
    nabe = pd.DataFrame({
        "pdb_id": ["1URN", "1ASY"],
        "mutation": ["Y13A", "F127A"],
        "ddg": [2.5, 2.0],
        "method": ["Nabe", "Nabe"],
        "type": ["RNA-Protein", "RNA-Protein"],
        "wt_aa": ["Y", "F"],
        "position": [13, 127],
        "mut_aa": ["A", "A"],
    })

    monkeypatch.setattr(ftd, "fetch_pronab", lambda **kw: pronab)
    monkeypatch.setattr(ftd, "fetch_nabe", lambda **kw: nabe)

    merged = ftd.fetch_training_data(include_nabe=True, include_literature=False)
    assert len(merged) == 2
    assert (merged["source"] == "pronab").sum() == 1
    assert (merged["source"] == "nabe").sum() == 1
