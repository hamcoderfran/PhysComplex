"""Summarize PhysGT holdout evaluation CSV into a claim-aware JSON report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float | int | None]:
    if len(labels) < 2:
        return {"n": int(len(labels)), "pearson_r": None, "spearman_r": None}
    r_p = pearsonr(labels, preds)
    r_s = spearmanr(labels, preds)
    return {
        "n": int(len(labels)),
        "pearson_r": float(r_p.statistic),
        "pearson_p": float(r_p.pvalue),
        "spearman_r": float(r_s.statistic),
        "spearman_p": float(r_s.pvalue),
    }


def summarize_holdout_csv(
    csv_path: str | Path,
    *,
    expected_test_n: int = 150,
) -> dict:
    frame = pd.read_csv(csv_path)
    labels = frame["experimental_ddg"].to_numpy(dtype=float)
    preds = frame["gt_pred_ddg"].to_numpy(dtype=float)
    report: dict = {
        "schema_version": 1,
        "protocol": "pdb_grouped_holdout_test",
        "claim_eligible": False,
        "n_scored": int(len(frame)),
        "n_expected_test": expected_test_n,
        "coverage": float(len(frame) / expected_test_n) if expected_test_n else 1.0,
        "overall": _metrics(labels, preds),
        "by_source": {},
        "notes": [],
    }
    if len(frame) < expected_test_n:
        report["notes"].append(
            f"Incomplete holdout coverage ({len(frame)}/{expected_test_n} test rows)."
        )
    for source, group in frame.groupby("source"):
        report["by_source"][str(source)] = _metrics(
            group["experimental_ddg"].to_numpy(dtype=float),
            group["gt_pred_ddg"].to_numpy(dtype=float),
        )
    if report["coverage"] >= 1.0 and report["overall"]["pearson_r"] is not None:
        report["notes"].append(
            "Regression claim still requires frozen manifest + calibration artifact; "
            "this report is descriptive only."
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize PhysGT holdout predictions")
    ap.add_argument("--csv", default="/tmp/physgt_holdout_full.csv")
    ap.add_argument("--output", default="/tmp/physgt_holdout_summary.json")
    ap.add_argument("--expected-test-n", type=int, default=150)
    args = ap.parse_args()

    report = summarize_holdout_csv(
        args.csv,
        expected_test_n=args.expected_test_n,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
