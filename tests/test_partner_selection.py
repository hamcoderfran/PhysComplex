"""Tests for partner chain selection and pipeline hardening."""

from __future__ import annotations

from pathlib import Path

import pytest

from physrna_filter.structure.parse_complex import parse_complex
from physrna_filter.structure.partner_selection import (
    count_chain_contacts,
    select_partner_pair,
)
from physrna_filter.analysis.score import _combine_verdicts_af3


STRUCTURES = Path(__file__).parent.parent / "physrna_filter" / "data" / "structures"


def test_select_partner_pair_single_copy():
    parsed = parse_complex(str(STRUCTURES / "1urn.pdb"))
    prot, rna = select_partner_pair(parsed.protein_chains, parsed.rna_chains)
    assert len(prot) == 1
    assert len(rna) == 1
    assert count_chain_contacts(prot[0], rna[0]) > 0


def test_combine_verdicts_bio_warn_when_decisive():
    assert _combine_verdicts_af3("PASS", "WARN", "PASS", bio_decisive=True) == "WARN"
    assert _combine_verdicts_af3("PASS", "WARN", "PASS", bio_decisive=False) == "PASS"


def test_contact_thresholds_from_json(tmp_path):
    from physrna_filter.analysis.thresholds import save_thresholds, reset_threshold_cache
    from physrna_filter.analysis.contact_score import _contact_verdict

    reset_threshold_cache()
    save_thresholds(
        {"contact_pass_per_residue": -2.0, "contact_warn_per_residue": -0.5},
        tmp_path / "t.json",
    )
    reset_threshold_cache()
    from physrna_filter.analysis import contact_score as cs
    from physrna_filter.analysis import thresholds as th

    old = th._CACHE
    th._CACHE = th.load_thresholds(tmp_path / "t.json")
    try:
        assert _contact_verdict(-3.0, 5, 1) == "PASS"
        assert _contact_verdict(-0.8, 3, 1) == "WARN"
    finally:
        th._CACHE = old


def test_gt_inference_context_reuse():
    from physrna_filter.analysis.gt_inference import GtInferenceContext

    ctx = GtInferenceContext()
    m1, meta1 = ctx.ensure_loaded(None, use_esm=False, use_rnafm=False)
    m2, meta2 = ctx.ensure_loaded(None, use_esm=False, use_rnafm=False)
    assert m1 is m2
    assert meta1 is meta2
