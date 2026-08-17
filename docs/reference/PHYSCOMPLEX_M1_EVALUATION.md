# PhysComplex M1 evaluation protocol

M1 measures current models and simple baselines on identical, versioned
examples. It is deliberately a reporting layer, not model training.

## Protein–RNA partner panel

```bash
physcomplex m1-protein-rna-panel \
  --zips af3_predictions \
  --fast \
  --output-dir physcomplex_reports
```

The command writes:

* `protein_rna_panel_rows.csv` — one row per expected panel item, including
  `ERROR` rows for missing predictions;
* `protein_rna_panel_pipeline_metrics.json` — current pipeline metrics;
* `protein_rna_m1_baselines.json` — same-row metrics for:
  * predictor confidence only (`af3_iptm`);
  * PhysGT only (`gt_score_norm`);
  * full fast composite (`gt_score_norm` plus clash/biology penalties).

The report contains coverage, missing/error count, score direction, AUROC, and
partner-group top-1 accuracy. It sets `claim_eligible=false` whenever panel
coverage is incomplete. It also records whether the checkpoint was panel
fine-tuned; a fine-tuned-panel result is not an out-of-distribution claim.

## Boltz bundle

```bash
# Normalize nested model outputs as they appear.
physcomplex sync-boltz --bundle boltz_test_100

# Score only completed structures; coverage remains part of the result.
physcomplex score-boltz --bundle boltz_test_100 --fast \
  --output-csv boltz_test_100/physcomplex_scores.csv \
  --output-json boltz_test_100/physcomplex_metrics.json
```

The output reports manifest count, scored count, missing count, errors,
coverage, composite AUROC, and evaluable partner-group top-1 accuracy. A
partial bundle must never be described as a full benchmark.

## Required next reports

1. Complete the 10-job AF3 panel without auto-finetuning, then rerun M1.
2. Complete Boltz-100 and rerun its coverage-aware score.
3. Finish full PDB-grouped PhysGT holdout evaluation.
4. Generate FoldBench predictions and compare PhysComplex/PhysRNA scores to
   predictor confidence on identical targets.
5. Freeze family/scaffold-disjoint manifests before selecting a new model,
   threshold, physics weight, or evolutionary feature.
