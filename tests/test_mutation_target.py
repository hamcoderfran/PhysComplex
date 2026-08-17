"""Tests for RNA vs protein mutation target classification."""
from __future__ import annotations

import pandas as pd

import pytest

from physrna_filter.data.mutation_target import (
    MutationTarget,
    is_rna_mutation,
    resolve_mutation_target,
    validate_mutation_target_for_structure,
)


def test_is_rna_mutation():
    assert is_rna_mutation("G", "U")
    assert not is_rna_mutation("K", "A")


def test_resolve_rna_mutation_from_nabe_chain():
    row = pd.Series({"pdb_id": "4lck", "mutation": "G43U", "chain": "B", "source": "nabe"})
    target = resolve_mutation_target(row)
    assert target.kind == "rna"
    assert target.chain_id == "B"
    assert target.position == 43


def test_resolve_protein_mutation_with_chain():
    row = pd.Series({"pdb_id": "1asy", "mutation": "K293A", "chain": "A", "source": "nabe"})
    target = resolve_mutation_target(row)
    assert target.kind == "protein"
    assert target.chain_id == "A"


def test_validate_rejects_protein_codes_on_rna_chain():
    target = MutationTarget("protein", "X", 7, "R", "L")
    with pytest.raises(ValueError, match="RNA in structure"):
        validate_mutation_target_for_structure(
            target,
            protein_chain_ids={"A", "B"},
            rna_chain_ids={"X", "Y"},
        )
