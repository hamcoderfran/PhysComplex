# PhysComplex: a modality-aware path beyond PhysRNA

PhysComplex is the shared evaluation and provenance layer for future
protein–RNA, protein–protein, protein–DNA, protein–ligand, RNA–RNA, and
RNA–ligand validators. It does **not** claim that PhysRNA/PhysGT weights work
for those modalities.

## What is implemented now

* a versioned `ComplexScoreResult` contract carrying modality, score direction,
  verdict, checkpoint hash/path, and simulation method;
* a dataset catalog with task, provenance, access note, and leakage-safe split
  unit;
* deterministic grouped split manifests with a content hash;
* an identical-record baseline runner that reports coverage and errors rather
  than dropping failed predictions;
* a `ProteinRnaAdapter` around the existing PhysRNA pipeline;
* explicit unsupported adapters for every non-RNA modality.

This is deliberately infrastructure-first. The only currently implemented
scoring modality is `protein_rna`.

## What is not implemented

No massive external dataset is downloaded, no cross-modality model is trained,
and no result is presented as a PhysComplex benchmark. Data access, curation,
licenses, chemical standardization, homology clustering, compute, and
independent evaluation remain required before a modality is enabled.

## Why a single model is not enough

The current PhysGT graph uses RNA-FM embeddings, RNA backbone/pucker features,
RNA phosphate/contact heuristics, free-RNA simulation, and RBP biological
evidence. These are meaningful for protein–RNA but do not represent:

* protein–protein solvent/exposure and symmetry;
* DNA sequence/motif and B/A/Z-form context;
* ligand protonation, tautomer, conformer, and chemical scaffold;
* RNA–ligand or RNA–RNA structural-motif space.

PhysComplex therefore shares contracts, split logic, reporting, and eventually
the graph-transformer *backbone*, while requiring modality-specific graph
builders, heads, labels, calibrations, and held-out benchmarks.

## Data catalog and evidence types

The registry is a catalog, not a merged training corpus. It keeps these label
types separate:

| Modality | Potential sources | Do not collapse into one target |
|---|---|---|
| Protein–RNA | ProNAB, Nabe, FoldBench, PDB/NAKB, ENCODE eCLIP | structural pose, ΔΔG, cellular occupancy |
| Protein–protein | Docking Benchmark 5.5, SKEMPI 2.0, PDB, IntAct | docking success, ΔΔG, interaction evidence |
| Protein–DNA | Protein–DNA Docking Benchmark, ProNAB, PDB/NAKB, JASPAR | pose, ΔΔG, sequence preference |
| Protein–ligand | PDB, BioLiP2, BindingDB, ChEMBL | pose/contact, affinity/activity |
| RNA–ligand | PDB/NAKB, RNALigands, R-BIND | pose/contact, motif/association |
| RNA–RNA | PDB/NAKB, RNA 3D Motif Atlas, RISE, RNAcentral | tertiary geometry, motif, interaction edges |

Primary sources:

* ProNAB: [Mishra et al. (2022)](https://doi.org/10.1093/nar/gkab848)
* Nabe: [Wang et al. (2021)](https://doi.org/10.1093/database/baab050)
* SKEMPI 2.0: [Jankauskaitė et al. (2019)](https://doi.org/10.1093/bioinformatics/bty635)
* Docking Benchmark 5: [Vreven et al. (2015)](https://doi.org/10.1016/j.jmb.2015.07.016)
* Protein–DNA Docking Benchmark: [van Zundert et al.](https://github.com/haddocking/Prot-DNABenchmark)
* BioLiP2: [Yang et al. (2024)](https://doi.org/10.1093/nar/gkad630)
* RNA 3D Motif Atlas: [BGSU RNA 3D Hub](https://rna.bgsu.edu/rna3dhub/motifs)
* RISE: [Gong et al. (2018)](https://doi.org/10.1093/nar/gkx864)

Licenses and redistribution conditions differ. For example, source data can be
publicly queryable without granting permission to redistribute a derived bulk
dataset. Review each source's current terms before fetching or committing data.

## Commands

```bash
# List sources; this does not download anything.
python -m physrna_filter.physcomplex catalog --modality protein_rna --json

# Freeze a custom manifest. Each record needs id and group.
python -m physrna_filter.physcomplex freeze-split records.json \
  --output splits.json --manifest-id my-protein-protein-v1 \
  --modality protein_protein --task ddg_regression --source skempi2

# Freeze the existing merged RNA–protein ΔΔG records by PDB complex.
python -m physrna_filter.physcomplex freeze-rna-ddg \
  --output protein_rna_ddg_v1.json
```

The `physcomplex` console command provides the same commands after installation.

## Required rollout sequence

1. Freeze and publish source/version/split manifests before fitting a model.
2. Establish confidence-only and simple physics-only baselines on the same
   records; report coverage and errors.
3. Implement one modality adapter and graph builder.
4. Train a modality-specific head without reading frozen test labels.
5. Compare against market-predictor confidence and established baselines on
   untouched groups, with uncertainty intervals.
6. Only then enable that adapter as an actual PhysComplex modality.

The M1/M2 gates in [`DEVELOPMENT_OPERATING_SYSTEM.md`](DEVELOPMENT_OPERATING_SYSTEM.md)
remain binding. A broad architecture is not evidence of broad generalization.

For the implemented universal checks and the capability-gated MD/MM-GBSA/FEP/
QM-MM plans, see [`PHYSCOMPLEX_PHYSICS_PROTOCOLS.md`](PHYSCOMPLEX_PHYSICS_PROTOCOLS.md).
For diagnostic MSA conservation and paired protein–RNA coevolution evidence,
see [`PHYSCOMPLEX_EVOLUTIONARY_EVIDENCE.md`](PHYSCOMPLEX_EVOLUTIONARY_EVIDENCE.md).
For the executable same-row baseline and partial-coverage policy, see
[`PHYSCOMPLEX_M1_EVALUATION.md`](PHYSCOMPLEX_M1_EVALUATION.md).
For the modality-aware parser, physics-only baseline, and strict calibration
boundary, see [`PHYSCOMPLEX_UNIVERSAL_VALIDATOR.md`](PHYSCOMPLEX_UNIVERSAL_VALIDATOR.md).
