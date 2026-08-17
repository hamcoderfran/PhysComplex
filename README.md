# PhysComplex Filter — starter kit

 **PhysRNA** (physics-informed AF3 screening) plus **PhysGT** (graph-transformer ΔΔG / interface scoring). Use this tree to build **PhysComplex Filter**, potentially expanding beyond RNA-RBP into other structure types.

## Quick start

```bash
cd physcomplex_filter
pip install -e ".[dev]"
python -m physrna_filter.cli init
python -m physrna_filter.cli doctor

If you want real oxDNA MD (Highly Recommended)
Install oxDNA separately, then point PhysRNA at it:
https://github.com/lorenzo-rovigatti/oxDNA

export OXDNA_BIN=/path/to/oxDNA/build/bin/oxDNA
# Windows + WSL example:
# export OXDNA_BIN=wsl:/home/you/oxDNA/build/bin/oxDNA
Then confirm:

physrna doctor
python -m physrna_filter.data.verify_oxrna

```

Hello-world score on bundled crystal:

```bash
python -m physrna_filter.pipeline physrna_filter/data/structures/1urn.pdb
```

AF3 ranking (drop Server zips in a folder):

```bash
physrna rank ./af3_predictions --rbp LIN28A
```

## What is included

| Layer | Purpose |
|-------|---------|
| `physrna_filter.pipeline` | 4-branch physics screen (clash, geometry, RMSD/MD, PhysGT) |
| `physrna_filter.analysis.gt_*` | PhysGT model, inference, graph features |
| `physrna_filter.validation.deploy_gt` | Train + deploy checkpoint |
| `physrna_filter.validation.train_gt` | LOCO / holdout ΔΔG training |
| `physrna_filter.validation.eval_gt` | Evaluate fixed checkpoint |
| `physrna_filter.validation.rank_af3_candidates` | Rank AF3 decoys |
| Shipped data | 1029-entry training CSVs, 357 cached PDBs, `gt_checkpoint.pt` |

## What is **not** included (build these as PhysComplex Filter)

- `physrna_filter/physcomplex/` — modality adapters, frozen splits, eval coverage
- Boltz / FoldBench / RBPBench benchmark modules
- `physcomplex score-*`, `eval-coverage`, acquisition ledger

See **`HANDOFF.md`** for architecture, file map, and recommended next steps.  
Reference designs live in **`docs/reference/`** (copied from upstream PhysComplex docs).

## Tests

```bash
python -m pytest tests/ -q   # 154 passed (starter-kit subset)
```

## Docs

- `docs/WALKTHROUGH.md` — pipeline concepts
- `docs/USER_GUIDE.md` — AF3 workflow
- `docs/AF3_WORKFLOW.md` — rank / screen / fine-tune
- `docs/DATA_SOURCES.md` — PhysGT training provenance
- `docs/reference/PHYSCOMPLEX*.md` — target architecture (read-only reference)

Original upstream README: `README_ORIGINAL.md`.
