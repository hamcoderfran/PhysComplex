"""Tests for eval_gt checkpoint evaluation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import torch


def test_6d8p_sized_structures_bypass_pdbfixer_relaxation(monkeypatch, tmp_path):
    from physrna_filter.structure import mutate

    source = tmp_path / "6d8p.pdb"
    source.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
    wt_output = tmp_path / "wt.pdb"
    mut_output = tmp_path / "mut.pdb"

    # 6D8P has 14,004 coordinate records.  This is large enough for
    # PDBFixer's implicit addMissingAtoms() minimizer to stall evaluation.
    monkeypatch.setattr(
        mutate, "_count_pdb_atom_records",
        lambda _: 14_004,
    )
    monkeypatch.setattr(
        mutate, "_mutate_pdbfixer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("PDBFixer called")),
    )

    assert mutate.prepare_fixed_structure(
        str(source), "A", 1, "A", str(wt_output),
    ) == str(wt_output)
    assert wt_output.read_text() == source.read_text()

    simple = MagicMock(return_value=str(mut_output))
    monkeypatch.setattr(mutate, "_mutate_simple", simple)
    assert mutate.introduce_mutation(
        str(source), "A", 1, "L", str(mut_output),
    ) == str(mut_output)
    simple.assert_called_once_with(str(source), "A", 1, "L", str(mut_output))


def test_eval_gt_checkpoint_runs_predict(monkeypatch, tmp_path):
    from physrna_filter.validation import eval_gt

    ckpt = tmp_path / "gt_checkpoint.pt"
    torch.save({
        "model_state": {},
        "protein_node_dim": 344,
        "rna_node_dim": 648,
        "hidden_dim": 128,
        "n_layers": 3,
        "esm_dim": 320,
        "rnafm_dim": 640,
        "target_mean": 0.0,
        "target_std": 1.0,
        "interface_head_trained": True,
    }, ckpt)

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.return_value = torch.tensor(0.5)

    fake_graph = MagicMock()
    fake_graph.x_protein = torch.zeros(2, 344)
    fake_graph.x_rna = torch.zeros(1, 648)
    fake_graph.edge_index = torch.zeros(2, 0, dtype=torch.long)
    fake_graph.edge_attr = torch.zeros(0, 9)
    fake_graph.mutation_node_idx = 0

    dataset = [(fake_graph, fake_graph, 1.0, "1abc", "A12G", "pronab")]

    monkeypatch.setattr(
        eval_gt, "load_gt_model",
        lambda path: (mock_model, {
            "checkpoint": str(ckpt),
            "esm_dim": 320,
            "rnafm_dim": 640,
            "use_esm": True,
            "use_rnafm": True,
            "target_mean": 0.0,
            "target_std": 1.0,
        }),
    )
    monkeypatch.setattr(eval_gt, "_load_or_build_dataset", lambda **kwargs: (dataset, []))
    monkeypatch.setattr(eval_gt, "fetch_training_data", lambda **kw: __import__("pandas").DataFrame({
        "pdb_id": ["1abc"], "mutation": ["A12G"], "ddg": [1.0], "source": ["pronab"],
    }))

    out = tmp_path / "results.csv"
    df = eval_gt.eval_gt_checkpoint(
        checkpoint_path=str(ckpt),
        results_csv=str(out),
    )
    assert len(df) == 1
    assert out.exists()


def test_eval_gt_holdout_writes_summary_json(monkeypatch, tmp_path):
    from physrna_filter.validation import eval_gt

    ckpt = tmp_path / "gt_checkpoint.pt"
    torch.save({
        "model_state": {},
        "protein_node_dim": 344,
        "rna_node_dim": 648,
        "hidden_dim": 128,
        "n_layers": 3,
        "esm_dim": 320,
        "rnafm_dim": 640,
        "target_mean": 0.0,
        "target_std": 1.0,
        "interface_head_trained": True,
    }, ckpt)

    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.return_value = torch.tensor(0.5)

    fake_graph = MagicMock()
    fake_graph.x_protein = torch.zeros(2, 344)
    fake_graph.x_rna = torch.zeros(1, 648)
    fake_graph.edge_index = torch.zeros(2, 0, dtype=torch.long)
    fake_graph.edge_attr = torch.zeros(0, 9)
    fake_graph.mutation_node_idx = 0

    dataset = [
        (fake_graph, fake_graph, float(i), f"pdb{i}", f"A{i}G", "pronab" if i % 2 == 0 else "nabe")
        for i in range(5)
    ]

    monkeypatch.setattr(
        eval_gt, "load_gt_model",
        lambda path: (mock_model, {
            "checkpoint": str(ckpt),
            "esm_dim": 320,
            "rnafm_dim": 640,
            "use_esm": True,
            "use_rnafm": True,
            "target_mean": 0.0,
            "target_std": 1.0,
        }),
    )
    monkeypatch.setattr(eval_gt, "_load_or_build_dataset", lambda **kwargs: (dataset, []))
    monkeypatch.setattr(
        eval_gt,
        "_split_holdout_by_pdb",
        lambda dataset, **kwargs: ([], [], dataset),
    )
    monkeypatch.setattr(eval_gt, "fetch_training_data", lambda **kw: __import__("pandas").DataFrame({
        "pdb_id": [f"pdb{i}" for i in range(5)],
        "mutation": [f"A{i}G" for i in range(5)],
        "ddg": [float(i) for i in range(5)],
        "source": ["pronab" if i % 2 == 0 else "nabe" for i in range(5)],
    }))

    out = tmp_path / "holdout.csv"
    eval_gt.eval_gt_checkpoint(
        checkpoint_path=str(ckpt),
        results_csv=str(out),
        holdout=True,
    )
    summary = tmp_path / "holdout_summary.json"
    assert summary.is_file()
    report = json.loads(summary.read_text(encoding="utf-8"))
    assert report["n_scored"] == 5
    assert report["claim_eligible"] is False
