# PhysRNA

AIYGO Computational Biology Group

Physics-informed validation of AI-predicted protein–RNA complex structures.

AlphaFold 3 and RoseTTAFold All-Atom predict binding poses with high confidence but
no representation of the thermodynamic cost of adopting that pose from free solution.
PhysRNA catches three classes of hallucination that structure-confidence scores miss:

1. **Entropic** — the bound RNA conformation is not sampled in the free ensemble (RMSD branch)
2. **Local geometry** — interface nucleotides adopt torsion angles / sugar puckers not seen in free RNA (geometry branch)
3. **Contact physics** — the interface lacks electrostatic and H-bond contacts consistent with claimed binding (contact + PhysGT branches)

<p align="center">
<img width="700" height="510" alt="PhysRNA pipeline overview" src="https://github.com/user-attachments/assets/b4377d9b-c6d1-4ee3-a9a0-0e5b0d9c3593">
</p>

**Further reading:** [Walkthrough](docs/WALKTHROUGH.md) · [Results](docs/RESULTS.md) · [Test AF3 complexes](docs/TEST_COMPLEXES.md) · [Training data sources](docs/DATA_SOURCES.md)

---

## Install

```bash
git clone https://github.com/hamcoderfran/PhysRNA.git
cd PhysRNA
pip install -e .
```

**Core dependencies** (installed via `requirements.txt`): numpy, scipy, biopython, pandas, requests, pdbfixer, openmm, freesasa, scikit-learn

**For full PhysGT** (recommended):

```bash
pip install torch torch-geometric fair-esm rna-fm
python -m physrna_filter.data.download_rnafm_weights   # ~1.2 GB, one-time
python -m physrna_filter.data.verify_rnafm_weights
```

**Optional:**

| Component | Purpose | Notes |
|-----------|---------|-------|
| `pyrosetta` | Best mutant-structure quality for ΔΔG training | Academic license; falls back to PDBFixer |
| [oxDNA](https://github.com/lorenzo-rovigatti/oxDNA) | Real free-RNA MD for RMSD branch | Without it, step 4 uses a CG Langevin fallback |
| `barnaba` | eRMSD metric (alternative RMSD branch) | Optional upgrade |
| GPU + CUDA | PhysGT training and ESM-2 inference | CPU works; training is slow |

**Windows oxDNA:** no native binary — install in WSL and set `OXDNA_BIN=wsl:/home/<you>/oxDNA/build/bin/oxDNA`. Verify with `python -m physrna_filter.data.verify_oxrna`.

---

## Typical research pipeline

End-to-end workflow from install to screening AF3 predictions. Run from the repo root, determined by your specific setup.

```bash
# 1. Install
pip install -e .
pip install torch torch-geometric fair-esm rna-fm

# 2. One-time setup
python -m physrna_filter.data.download_rnafm_weights
python -m physrna_filter.data.verify_rnafm_weights
python -c "from physrna_filter.data.fetch_training_data import fetch_training_data; fetch_training_data()"

# 3. (Optional) oxDNA for meaningful RMSD scores
python -m physrna_filter.data.verify_oxrna

# 4. Deploy PhysGT checkpoint (~1 LOCO fold — fast path)
python -m physrna_filter.validation.deploy_gt --n-folds 1 --gt-epochs 80 --interface-epochs 15

# 5. Sanity-check checkpoint (holdout = unbiased; all entries = in-sample)
python -m physrna_filter.validation.eval_gt --holdout --graph-cache physrna_filter/validation/gt_graphs.pt
python -m physrna_filter.validation.eval_gt --graph-cache physrna_filter/validation/gt_graphs.pt

# 6. Score one AF3 structure (PDB, mmCIF, or AF3 Server .zip)
python -m physrna_filter.pipeline my_af3_complex.pdb --require-gt-checkpoint

# 7. Batch-screen a folder of AF3 outputs
python -m physrna_filter.validation.screen_af3 ./af3_predictions/ --require-gt-checkpoint --output af3_screen.csv
```

**Verdicts:** `PASS` = plausible · `WARN` = review · `FAIL` = likely hallucination. For AF3 screening, `combined_verdict` uses PhysGT + biology + true steric clashes; RMSD/geometry are diagnostic (bound pose vs free ensemble).
Combined verdict = worst branch (RMSD, geometry, contacts, clashes, PhysGT, biological).

**Holdout AF3 test panel** (10 complexes not in ProNAB training): see [docs/TEST_COMPLEXES.md](docs/TEST_COMPLEXES.md).

**Post-AF3 ranking:** see [docs/USER_GUIDE.md](docs/USER_GUIDE.md) and [docs/AF3_WORKFLOW.md](docs/AF3_WORKFLOW.md).

```bash
physrna init
physrna rank af3_predictions --rbp LIN28A
```

For AF3 screening, `combined_verdict` uses **PhysGT + biological partner checks + steric clashes**. RMSD, geometry, and contact energy are reported as diagnostics in the CSV.

---

## Quick start

Score a structure without training (physics branches only):

```bash
python -m physrna_filter.pipeline my_af3_complex.pdb
```

```python
from physrna_filter.pipeline import run_pipeline
result = run_pipeline("my_af3_complex.pdb")
print(result["combined_verdict"], result["confidence"])
```

Exit code: `0` = PASS, `1` = WARN/FAIL.

With PhysGT (requires `gt_checkpoint.pt` from deploy step above):

```bash
python -m physrna_filter.pipeline fold_6sqn_u1a_hairpin.zip --require-gt-checkpoint
```

---

## Command reference

All commands are one line each. Run from the repo root after `pip install -e .`.

### Score structures

Supports **PDB**, **mmCIF** (`.cif`), and **AF3 Server zip** archives (auto-extracts `model_0.cif`).

```bash
python -m physrna_filter.pipeline my_af3_complex.pdb
python -m physrna_filter.pipeline fold_6sqn_u1a_hairpin.zip --require-gt-checkpoint
python -m physrna_filter.pipeline fold_6sqn_u1a_hairpin.zip --model-rank 1
python -m physrna_filter.pipeline my_af3_complex.pdb --cutoff 5.0 --quiet
python -m physrna_filter.pipeline my_af3_complex.pdb --rbp-name ELAVL1 --rna-sequence AUGCAUGC
```

### Batch screening

```bash
python -m physrna_filter.validation.screen_af3 ./af3_predictions/ --require-gt-checkpoint --output af3_screen.csv
python -m physrna_filter.validation.screen_af3 fold_6sqn_u1a_hairpin.zip reference.pdb --require-gt-checkpoint
```

### RNA-FM weights

```bash
python -m physrna_filter.data.download_rnafm_weights
python -m physrna_filter.data.verify_rnafm_weights
```

Windows (optional env vars):

```cmd
set RNAFM_CHECKPOINT=C:\path\to\RNA-FM_pretrained.pth
set RNAFM_SKIP_DOWNLOAD=1
```

Default path: `physrna_filter/data/rnafm_weights/RNA-FM_pretrained.pth`

### oxDNA (optional)

```bash
python -m physrna_filter.data.verify_oxrna
```

WSL install (inside Ubuntu):

```bash
sudo apt update && sudo apt install -y build-essential cmake git
git clone https://github.com/lorenzo-rovigatti/oxDNA.git
cd oxDNA && mkdir build && cd build && cmake .. -DPython=OFF && make -j4
export PATH=$PWD/bin:$PATH
```

Windows PowerShell → WSL binary:

```cmd
set OXDNA_BIN=wsl:/home/<you>/oxDNA/build/bin/oxDNA
python -m physrna_filter.data.verify_oxrna
```

### Training data

```bash
# ProNAB only (710 entries)
python -c "from physrna_filter.data.fetch_pronab import fetch_pronab; print(len(fetch_pronab()))"

# Merged: ProNAB + Nabe (RNA-only) + literature (~1,029 entries)
python -c "from physrna_filter.data.fetch_training_data import fetch_training_data; df=fetch_training_data(); print(df['source'].value_counts())"

# Holdout split report (fast — no graph build)
python -m physrna_filter.validation.holdout_eval_merged --report-only
```

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) and [docs/PRONAB_BULK_REQUEST.md](docs/PRONAB_BULK_REQUEST.md).

### Deploy PhysGT (production checkpoint)

**Recommended fast deploy** (interface head + 1 LOCO fold):

```bash
python -m physrna_filter.validation.deploy_gt --n-folds 1 --gt-epochs 80 --interface-epochs 15
```

ProNAB-only baseline:

```bash
python -m physrna_filter.validation.deploy_gt --pronab-only --n-folds 1 --gt-epochs 80
```

CPU smoke test (no language models):

```bash
python -m physrna_filter.validation.deploy_gt --no-esm --no-rnafm --n-folds 1 --gt-epochs 30
```

Full deploy (LOCO over all complexes — slow):

```bash
python -m physrna_filter.validation.deploy_gt
```

**Outputs:** `physrna_filter/validation/gt_checkpoint.pt`, `gt_interface_pretrain.pt`, `gt_results.csv`

### Evaluate checkpoint

```bash
# In-sample (fast)
python -m physrna_filter.validation.eval_gt --graph-cache physrna_filter/validation/gt_graphs.pt

# Unbiased holdout (~15% complexes by PDB)
python -m physrna_filter.validation.eval_gt --holdout --graph-cache physrna_filter/validation/gt_graphs.pt

# Custom paths
python -m physrna_filter.validation.eval_gt --checkpoint physrna_filter/validation/gt_checkpoint.pt --results-csv my_eval.csv
```

Training flags: `--pronab-only`, `--no-nabe`, `--no-literature`

### Train PhysGT directly (advanced)

Build graph cache once (reused by train/eval):

```bash
python -m physrna_filter.validation.train_gt --graph-cache physrna_filter/validation/gt_graphs_merged.pt --rebuild-cache --n-folds 1 --epochs 80
```

Holdout evaluation (faster than full LOCO):

```bash
python -m physrna_filter.validation.train_gt --cv-mode holdout --graph-cache physrna_filter/validation/gt_graphs_merged.pt --epochs 150
```

Compare ProNAB-only vs merged holdout:

```bash
python -m physrna_filter.validation.holdout_eval_merged --report-only
python -m physrna_filter.validation.train_gt --cv-mode holdout --pronab-only --graph-cache physrna_filter/validation/gt_graphs_pronab.pt --rebuild-cache
python -m physrna_filter.validation.train_gt --cv-mode holdout --graph-cache physrna_filter/validation/gt_graphs_merged.pt --rebuild-cache
```

Full LOCO benchmark (publication-grade, very slow):

```bash
python -m physrna_filter.validation.train_gt --graph-cache physrna_filter/validation/gt_graphs_merged.pt --epochs 150
```

GPU orchestrator:

```bash
python -m physrna_filter.validation.run_full_training --gpu
python -m physrna_filter.validation.run_full_training --smoke-test
```

### Other utilities

```bash
# ProNAB contact-scorer baseline
python -c "from physrna_filter.validation.benchmark_pronab import run_pronab_benchmark; run_pronab_benchmark(max_entries=200, output_csv='pilot_200.csv')"

# Calibrate RMSD / geometry thresholds
python -m physrna_filter.validation.calibrate_thresholds

# AF3 mutation re-prediction loop
python -m physrna_filter.validation.af3_mutation_loop --pdb-id 1urn --mutation F56A --experimental-ddg 2.5

# Unit tests
python -m pytest tests/ -v
```

---

## Physics pipeline (no PhysGT)

```
PDB / CIF / AF3 zip
    → [1] Parse complex          (parse_complex.py)
    → [2] Find interface         (extract_interface.py, 5 Å cutoff)
    → [3] AF3 local geometry     (local_geometry.py — torsions + puckering)
    → [4] Simulate free RNA      (run_simulation.py — oxDNA or fallback)
    → [5] Cluster ensemble       (cluster.py — k-medoids + circular k-means)
    → [6] Score + verdict        (score.py — RMSD + geometry branches)
```

---

## PhysGT — Physics-Informed Graph Transformer

Siamese graph transformer for protein-side mutation ΔΔG at RNA interfaces. Learns from
merged training data (ProNAB + Nabe RNA-only + literature curation) with ESM-2 and RNA-FM
embeddings. Architecture details: `physrna_filter/analysis/gt_model.py`.

| Goal | Command |
|------|---------|
| Fast deploy | `deploy_gt --n-folds 1 --gt-epochs 80 --interface-epochs 15` |
| Eval (holdout) | `eval_gt --holdout --graph-cache validation/gt_graphs.pt` |
| Full LOCO | `train_gt --graph-cache validation/gt_graphs_merged.pt` |

Key flags (`train_gt` / `deploy_gt` / `eval_gt`):

| Flag | Description |
|------|-------------|
| `--pronab-only` | Exclude Nabe and literature supplements |
| `--no-nabe` / `--no-literature` | Drop individual supplemental sources |
| `--graph-cache PATH` | Cache built interface graphs |
| `--rebuild-cache` | Force rebuild of graph cache |
| `--cv-mode holdout` | Single PDB-grouped train/val/test split |
| `--no-esm` / `--no-rnafm` | Disable language-model embeddings |
| `--n-folds N` | Cap LOCO folds (smoke tests) |

Results CSVs include a `source` column (`pronab` / `nabe` / `literature`) and per-source Pearson r.

---

## Benchmarks (summary)

| Benchmark | n | Pearson r | Spearman r |
|-----------|---|-----------|------------|
| PhysGT holdout (ProNAB, user GPU run) | 101 | **+0.854** | +0.843 |
| PhysGT in-sample (ESM-2 + RNA-FM, full ProNAB) | 706 | +0.836 | +0.858 |
| PhysGT holdout (ESM-2 only, CPU pilot) | 98 | +0.334 | +0.248 |
| ProNAB linear contact scorer (full) | 617 | 0.180 | 0.275 |
| U1A mini (1URN, ITC) | 6 | 0.706 | 0.486 |

Full tables and per-mutation breakdowns: [docs/RESULTS.md](docs/RESULTS.md).

---

## Package layout

```
physrna_filter/
  pipeline.py              Main entry — orchestrates all validation branches
  structure/               PDB/CIF/zip parsing, interface, geometry, mutations
  simulation/              oxDNA MD + free-RNA folding
  analysis/                Scoring, graphs, PhysGT model, ESM-2 / RNA-FM embeddings
  data/                    ProNAB, Nabe, literature fetchers; embedding caches
  validation/              deploy_gt, train_gt, eval_gt, screen_af3, benchmarks
benchmarks/                U1A / multi-system contact-scorer benchmarks
docs/                      WALKTHROUGH, RESULTS, TEST_COMPLEXES, DATA_SOURCES
tests/                     pytest suite
```

---

## Tests

```bash
python -m pytest tests/ -v
```

Covers RMSD (Kabsch + reflection guard), torsion geometry, contact physics, AF3 zip I/O,
PhysGT eval, and merged training-data fetchers.

---

## Scientific background

See [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) for why AF3 hallucinates RNA conformations,
algorithm details (Kabsch, torsion angles, contact energy), ProNAB benchmarking protocol,
PhysGT training, and future directions.
