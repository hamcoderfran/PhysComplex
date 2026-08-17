# PhysComplex universal validator

PhysComplex now provides a universal **validation framework**, not a universal
trained model. Every structure can be classified, statically checked, assigned
to a modality/task, and passed through a strict uncertainty gate. Only
protein–RNA currently has a learned PhysRNA/PhysGT adapter.

## Operational capabilities

| Capability | All modalities | Protein–RNA |
|---|---|---|
| PDB/mmCIF/AF3-ZIP chain classification | yes | yes |
| Modality inference | protein–protein, protein–DNA, protein–RNA, protein–ligand, RNA–RNA, RNA–ligand | yes |
| Static inter-chain clash/distance checks | yes | yes |
| Physics-only baseline | yes, marked `UNCALIBRATED` | yes |
| Predictor score normalizer | validation-fit only | validation-fit only |
| Calibrated PASS/WARN/FAIL/ABSTAIN decision | only with frozen validation artifact | only with frozen validation artifact |
| Learned interface head | no | PhysGT only |

## Commands

```bash
# Inspect what a structure contains.
physcomplex classify physrna_filter/data/structures/2sic.pdb

# Create a PDB-grouped SKEMPI protein–protein ΔΔG split.
physcomplex freeze-skempi --output splits/protein_protein_skempi_v1.json

# Existing protein–RNA split.
physcomplex freeze-rna-ddg --output splits/protein_rna_ddg_v1.json
```

## Calibration rule

`UNCALIBRATED` is the correct outcome when no validation-fitted artifact exists.
The validator will also return `ABSTAIN` when required evidence is missing.
Only `CalibratedDecisionGate.fit_from_validation()` may create an actionable
threshold, and it rejects records outside a frozen manifest's validation
partition. Test records must never be read during fit.

## Before enabling a new modality

1. Freeze labels and family/scaffold-disjoint splits.
2. Run confidence-only and physics-only baselines on the same records.
3. Fit score normalizers and abstention thresholds on validation only.
4. Evaluate once on the frozen test set, reporting coverage and failures.
5. Add a modality-specific learned graph/head only if it improves a predeclared
   endpoint without coverage regression.

This prevents an RNA-trained checkpoint from being presented as a universal
validator while still making every structure type operationally inspectable
today.
