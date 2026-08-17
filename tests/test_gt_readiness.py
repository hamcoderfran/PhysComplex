"""Tests for PhysGT readiness checks."""
from physrna_filter.validation.gt_readiness import (
    validate_checkpoint,
    run_gt_readiness,
)
from physrna_filter.analysis.gt_constants import CHECKPOINT_SCHEMA_VERSION


def test_validate_checkpoint_v1_warns():
    ok, warns = validate_checkpoint({
        "model_state": {"head.0.weight": "x"},
        "schema_version": 1,
        "interface_head_trained": False,
    })
    assert ok
    assert any("schema" in w.lower() for w in warns)


def test_run_gt_readiness_no_checkpoint():
    report = run_gt_readiness(use_esm=False, use_rnafm=False)
    assert "checks" in report
    assert report["schema_version"] == CHECKPOINT_SCHEMA_VERSION
