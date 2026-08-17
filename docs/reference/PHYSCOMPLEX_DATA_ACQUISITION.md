# PhysComplex data acquisition ledger

This document records what was actually acquired into the local ignored
`physcomplex_data/` directory. The data themselves are intentionally not
committed: benchmarks can be large, source licenses differ, and a Git clone
should not be mistaken for a curated training-ready corpus.

## Acquired on 2026-08-09

| Source | Modality | Local location | Acquisition | Status |
|---|---|---|---|---|
| [FoldBench](https://github.com/BEAM-Labs/FoldBench) | multi-modal | `physcomplex_data/foldbench` | shallow Git clone | acquired |
| FoldBench referenced assemblies | all FoldBench modalities | `physcomplex_data/foldbench_cifs` | resumable RCSB assembly-CIF download | 1,820/1,823 target rows available; 1,519 unique assembly files |
| [PRDB v3.0](https://github.com/shrikantcombio/PRDBv3_dataset) | protein–RNA | `physcomplex_data/prdbv3` | shallow Git clone | acquired |
| [Protein–DNA Docking Benchmark](https://github.com/haddocking/Prot-DNABenchmark) | protein–DNA | `physcomplex_data/prot_dna_benchmark` | shallow Git clone | acquired |
| [Docking Benchmark 5.5](https://zlab.wenglab.org/benchmark/) | protein–protein | `physcomplex_data/docking_benchmark_5` | upstream archive | acquired |
| [BioLiP2 nonredundant annotations](https://seq2fun.dcmb.med.umich.edu/BioLiP2/download.html) | protein–ligand | `physcomplex_data/biolip2` | upstream annotation archive | acquired |
| [ChEMBL 37](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/) | protein–ligand | `physcomplex_data/chembl37` | official SQLite archive | acquired |
| [BindingDB 2026-08 TSV](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp) | protein–ligand | `physcomplex_data/bindingdb` | official bulk TSV archive | acquired |
| [RNALigands query package](https://github.com/SaisaiSun/RNALigands) | RNA–ligand | `physcomplex_data/rnaligands` | shallow Git clone | acquired; database reuse still requires review |
| ProNAB | protein–RNA / protein–DNA | `physrna_filter/data/pronab_raw.csv` | repository-bundled subset | already available |
| Nabe | protein–RNA / protein–DNA | `physrna_filter/data/nabe_raw.csv` | repository-bundled subset | already available |
| SKEMPI 2.0 | protein–protein | `physrna_filter/data/skempi_v2.csv` | repository-bundled subset | already available |

The Docking Benchmark 5.5 archive SHA-256 is recorded in
`physcomplex_data/acquisition_ledger.json`; the ledger also records clone
revisions for Git sources when they are first acquired.

## Not automatically acquired

| Source class | Reason |
|---|---|
| PDBbind | Its terms restrict redistribution of raw and derivative data without permission. |
| R-BIND | Public resources, but bulk redistribution terms must be reviewed. |
| RISE | Aggregated RNA–RNA evidence needs source/assay provenance and is not a structural-pose label. |
| Full PDB/NAKB | Large, continually changing structural corpus; selection must be versioned and filtered by biological assembly, resolution, modality, and release date. |

This is a license and scientific-validity boundary, not a missing implementation.

## Next curation steps

1. Record each upstream revision/archive hash, license, release date, and
   original citation in a versioned manifest.
2. Normalize component identities and biological assemblies.
3. Keep pose labels, affinity/ΔΔG labels, and cellular interaction evidence as
   separate tasks.
4. Cluster before splitting: partner family + interface family for
   protein–protein, protein family + RNA/DNA family for nucleic acids, and
   protein family + ligand scaffold for protein–ligand.
5. Build frozen train/validation/test manifests before any training or
   threshold selection.

## Reproduction command

```bash
python -m physrna_filter.physcomplex acquire \
  foldbench prdbv3 prot_dna_benchmark docking_benchmark_5 biolip2 \
  --destination physcomplex_data
```

The command blocks license-review sources rather than downloading them
silently.

To build the 1,823-row cross-modality target inventory and acquire referenced
assemblies:

```bash
python -m physrna_filter.physcomplex foldbench-inventory
python -m physrna_filter.physcomplex acquire-foldbench-cifs --workers 4
```

The second command is deliberately resumable and writes a per-target ledger;
RCSB availability/errors are retained as coverage information rather than
silently dropped.
