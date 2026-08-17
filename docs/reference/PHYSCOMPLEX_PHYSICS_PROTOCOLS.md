# PhysComplex physics adjudication protocols

Physics is used here to discriminate among plausible AI-generated candidates
and to identify uncertainty. It is not treated as a universal truth score or a
substitute for held-out validation.

## Universal stage: run on every candidate

`static_physical_checks()` records:

* number of parsed chains/heavy atoms;
* minimum inter-chain heavy-atom distance;
* count of inter-chain contacts below a 1.5 Å clash threshold;
* explicitly unavailable chemistry checks: bond geometry, chirality,
  protonation/tautomer, and solvation free energy.

`aggregate_uncalibrated_risk()` combines normalized model disagreement,
confidence gap, physical strain, replica variance, and missing evidence into a
triage score. Its `calibrated` flag is always false until a modality-specific
validation set maps it to observed error. Use `risk_coverage_curve()` only with
independent labels.

## Expensive stage: capability-gated plans

| Modality | Plan | Required before execution | Output |
|---|---|---|---|
| Protein–protein | replicated short explicit-solvent MD | prepared protonation/force field, solvent/ion model, multiple replicas | contact persistence, interface drift, buried area, replica variance |
| Protein–RNA | restrained minimization + MM/GBSA | protein/RNA force fields, protonation, ion conditions | minimized energy, MM/GBSA score, restraint displacement |
| Protein–RNA finalists | explicit-ion/polarizable MD or QM/MM | metal/ion identity, QM region, replicas | ion occupancy/H-bond stability/variance |
| Protein–ligand | local induced-fit preparation then RB-FEP/ABFE | ligand microstates, force-field parameters, congeneric-series or ABFE protocol | ΔG, uncertainty, convergence diagnostics |
| Protein–DNA | physics-descriptor affinity model | DNA sequence, quantitative assays, salt/context, family groups | calibrated affinity and physical descriptors |

Use:

```bash
physcomplex adjudication-plan protein_rna
```

The command produces an auditable plan; it does not claim that unavailable
engines have run. OpenMM is installed in the Cloud environment, but Amber
MMPBSA, a polarizable force field, QM/MM, and an FEP engine are not. Prepared
structures, protonation/tautomer choices, force-field parameters, solvent/ion
conditions, and independent labels are all required before an expensive
calculation can become a result.

## Evidence and limits

* Protein–RNA minimized MM/GBSA showed near-native top-10 enrichment in a
  148-system study: [Sun et al. (2018)](https://rnajournal.cshlp.org/content/24/9/1183).
* Protein–protein MD stability features can discriminate plausible decoys, but
  MD may improve local packing while worsening a globally wrong interface:
  [Martins et al. (2021)](https://doi.org/10.1021/acs.jctc.1c00336) and
  [Méndez et al. (2008)](https://doi.org/10.1002/prot.21698).
* Raw MD ΔΔG is unreliable for protein–DNA specificity; a hybrid,
  assay-trained approach is required: [Zandarashvili et al. (2016)](https://doi.org/10.1021/acs.jpcb.6b12450).
* RNA ion and H-bond treatment is force-field-sensitive:
  [Pokorná et al. (2018)](https://doi.org/10.1021/acs.jctc.8b00670).

## Acceptance criteria

Before enabling any plan as a default:

1. freeze structure/family/scaffold-disjoint splits;
2. compare AI confidence alone, static checks, and the expensive protocol on
   the same ensemble;
3. report top-k structural enrichment or affinity error, coverage, runtime,
   failures, and replicate uncertainty;
4. calibrate risk/coverage on validation data and evaluate it once on a frozen
   test set.
