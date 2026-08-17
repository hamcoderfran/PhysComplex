# PhysComplex Filter — handoff document

**Date:** 2026-08-17  
**Source repo:** [PhysRNA](https://github.com/hamcoderfran/PhysRNA) @ `13384e3` (`main`)  
**This folder:** `physcomplex_filter/` — minimal extract for greenfield **PhysComplex Filter** development

---

## 1. Purpose

You asked for a **standalone copy of the original PhysRNA tool + PhysGT**, without the evaluation/benchmark layer that grew into PhysComplex on `main`. This directory is that starter kit.

**Use it to:**

- Score and rank AF3 (or mmCIF) protein–RNA predictions with physics + PhysGT
- Train, evaluate, and fine-tune PhysGT on merged ProNAB / Nabe / literature ΔΔG data
- Implement **PhysComplex Filter** (modality-aware adapter, calibration, coverage reporting) on top of a known-good base

**Do not expect here:**

- `physcomplex` CLI (`eval-coverage`, `score-boltz`, FoldBench crystal screen, frozen splits)
- Boltz-100/1000 bundles, benchmark CSV pipelines, scorecards
- Full CI matrix from the monorepo (subset of tests only)

---

## 2. Architecture (what you are extending)

```
                    ┌─────────────────────────────────────┐
                    │   PhysComplex Filter (YOU BUILD)    │
                    │   adapters · calibration · coverage │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │  physrna_filter.pipeline            │
                    │  clash → geom → RMSD/MD → PhysGT    │
                    └─────────────────┬───────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  structure/*                  simulation/*                 analysis/*
  parse · partner · clash      oxDNA / C4' MD              score · gt_inference
```

### Core entry points

| Command | Module | Role |
|---------|--------|------|
| `physrna init` | `config.init_user_environment` | Copy checkpoint → `~/.physrna/` |
| `physrna doctor` | `config.doctor_report` | Health check |
| `physrna rank DIR` | `validation.rank_af3_candidates` | Rank AF3 zips vs manifest |
| `physrna screen DIR` | `validation.screen_af3` | Partner-group screening |
| `physrna predict …` | `validation.predict_validate` | AF3 Server API workflow |
| `python -m physrna_filter.pipeline FILE` | `pipeline.run_pipeline` | Single-structure score |
| `python -m physrna_filter.validation.deploy_gt` | Full PhysGT deploy | Interface head + LOCO train |
| `python -m physrna_filter.validation.eval_gt --holdout` | Holdout Pearson/Spearman | Unbiased ΔΔG eval |

### PhysGT checkpoint resolution

1. `PHYSRNA_CHECKPOINT` env  
2. `~/.physrna/gt_checkpoint.pt` (writable; fine-tunes land here)  
3. Shipped `physrna_filter/validation/gt_checkpoint.pt`

The shipped checkpoint is **AF3 panel fine-tuned** (`af3_panel_finetuned=true`). For independent generalization benchmarks, retrain without panel fine-tune (see §5).

---

## 3. Directory map

```
physcomplex_filter/
├── HANDOFF.md                 ← this file
├── README.md
├── pyproject.toml             ← package name: physcomplex-filter
├── physrna_filter/
│   ├── pipeline.py            ← main orchestrator
│   ├── cli.py                 ← physrna CLI (benchmark subcmds lazy-import; see §6)
│   ├── config.py
│   ├── analysis/              ← 15 modules (no desolvation_score)
│   ├── structure/             ← 8 modules
│   ├── simulation/            ← 6 modules
│   ├── data/
│   │   ├── pronab_raw.csv     ← 706 ProNAB mutations
│   │   ├── nabe_raw.csv       ← 318 Nabe-only
│   │   ├── literature_mined.csv
│   │   ├── calibrated_thresholds.json
│   │   ├── af3_eval_panel.json (+ extended, holdout metadata)
│   │   ├── mutation_target.py ← RNA/protein site resolution for train_gt
│   │   ├── boltz_training_pairs.py  ← optional contrastive data for interface head
│   │   └── structures/        ← 357 cached PDBs (~124 MB)
│   └── validation/
│       ├── gt_checkpoint.pt   ← ~5 MB shipped weights
│       ├── deploy_gt.py · train_gt.py · eval_gt.py
│       ├── train_interface_head.py · merge_gt_checkpoint.py
│       ├── rank_af3_candidates.py · screen_af3.py · report_af3.py
│       ├── finetune_af3_panel.py · predict_validate.py
│       └── holdout_report.py
├── docs/                      ← core user docs (transferred)
├── docs/reference/            ← PhysComplex design docs (transferred, read-only)
├── tests/                     ← 154 core tests (subset of monorepo)
└── scripts/install_oxdna.sh   ← optional real MD
```

**Module count:** ~58 Python files (vs ~120+ in full monorepo with benchmarks).

---

## 4. Bundled data

| Asset | Path | Notes |
|-------|------|-------|
| PhysGT weights | `validation/gt_checkpoint.pt` | Panel-finetuned; see §5 for base retrain |
| Thresholds | `data/calibrated_thresholds.json` | PASS/WARN/FAIL cutoffs |
| Training table | `data/pronab_raw.csv`, `nabe_raw.csv`, `literature_mined.csv` | 1029 merged entries |
| AF3 panel | `data/af3_eval_panel.json` | 10-job P/N panel for rank/finetune |
| Structures | `data/structures/*.pdb` | Offline cache; `fetch_pdb` also downloads missing |
| Sample hello-world | `data/structures/1urn.pdb`, `1a9n.pdb` | Documented in upstream AGENTS.md |

**Not bundled (~1.2 GB):** RNA-FM weights. Download when training:

```bash
python -m physrna_filter.data.download_rnafm_weights
python -m physrna_filter.data.verify_rnafm_weights
```

**Optional:** oxDNA for real free-RNA MD (`scripts/install_oxdna.sh`). Without it, pipeline uses internal C4′ Langevin (expected, not an error).

---

## 5. Verified workflows

Run from `physcomplex_filter/` after `pip install -e ".[dev]"`.

### Health

```bash
python -m physrna_filter.cli init
python -m physrna_filter.cli doctor
python -m pytest tests/ -q
```

### Score one structure

```bash
python -m physrna_filter.pipeline physrna_filter/data/structures/1urn.pdb
# Exit 0 = PASS/WARN acceptable; 1 = FAIL
```

### Rank AF3 folder

```bash
mkdir -p af3_predictions   # drop fold_*.zip files here
physrna rank af3_predictions --rbp LIN28A --fast
```

### PhysGT holdout eval (fixed checkpoint)

```bash
python -m physrna_filter.validation.eval_gt --holdout --fast-mutations
```

### Full PhysGT retrain (long; GPU recommended)

```bash
python -m physrna_filter.validation.deploy_gt --n-folds 3   # smoke
# Production:
python -m physrna_filter.validation.deploy_gt
```

To produce a **non-panel-finetuned** checkpoint for unbiased M1 eval, run `train_gt` / `deploy_gt` and **do not** run `finetune_af3_panel` afterward.

---

## 6. CLI caveats (intentional slimming)

`cli.py` is copied verbatim from upstream. These subcommands **lazy-import** modules that were **not** copied:

| Subcommand | Missing module | Action |
|------------|----------------|--------|
| `physrna boltz prepare` | `data.boltz_benchmark` | Copy from monorepo or implement in PhysComplex Filter |
| `physrna benchmark foldbench` | `validation.benchmark_foldbench` | Same |
| `physrna benchmark pronab/rnapedia/pdb` | `validation.benchmark_*` | Same |

Core commands (`init`, `doctor`, `predict`, `rank`, `screen`, `panel`) work.

---

## 7. Suggested PhysComplex Filter build order

Reference specs are in `docs/reference/` (copied from upstream).

1. **Adapter layer** — wrap `run_pipeline()` behind modality-specific adapters (protein–RNA first). See `docs/reference/PHYSCOMPLEX_UNIVERSAL_VALIDATOR.md`.
2. **Static physics gate** — inter-chain clash / geometry before ML branches. See `docs/reference/PHYSCOMPLEX_PHYSICS_PROTOCOLS.md`.
3. **Uncalibrated risk aggregation** — composite score contract from `validation/screen_af3._composite_rank_score` and `analysis/score.py`.
4. **Calibration gate** — fit on held-out validation; do not tune on AF3 panel then claim unseen-panel accuracy (`docs/reference/PHYSCOMPLEX_M1_EVALUATION.md`).
5. **Coverage reporting** — never report partial bundles as full benchmarks (pattern from upstream scorecard).

Suggested new package layout (your choice):

```
physcomplex_filter/
├── physrna_filter/          ← keep as dependency / vendored core
└── physcomplex_filter/      ← NEW: your filter API
    ├── adapters/
    ├── calibration.py
    └── __main__.py          ← physcomplex-filter CLI
```

---

## 8. Transferred markdown files

### Core docs (`docs/`)

| File | Content |
|------|---------|
| `WALKTHROUGH.md` | Pipeline branch explanations |
| `USER_GUIDE.md` | Accuracy-first AF3 workflow |
| `AF3_WORKFLOW.md` | Rank / screen / fine-tune |
| `PREDICT_VALIDATE.md` | `physrna predict` end-to-end |
| `DATA_SOURCES.md` | ProNAB / Nabe / literature provenance |
| `context.md` | Problem statement & architecture |
| `TEST_COMPLEXES.md` | 10 holdout AF3 submission sequences |

### Reference docs (`docs/reference/`) — design targets, not implemented here

| File | Content |
|------|---------|
| `PHYSCOMPLEX.md` | Foundation overview |
| `PHYSCOMPLEX_UNIVERSAL_VALIDATOR.md` | Validator core contract |
| `PHYSCOMPLEX_PHYSICS_PROTOCOLS.md` | Modality physics plans |
| `PHYSCOMPLEX_M1_EVALUATION.md` | M1 panel methodology |
| `PHYSCOMPLEX_DATA_ACQUISITION.md` | Public dataset ledger pattern |
| `PHYSCOMPLEX_EVOLUTIONARY_EVIDENCE.md` | Coevolution diagnostics |

Upstream full README preserved as `README_ORIGINAL.md`.

---

## 9. Dependencies

See `pyproject.toml` and `requirements.txt`.

**Required:** PyTorch, torch-geometric, fair-esm, BioPython, OpenMM/PDBFixer, MDAnalysis, pandas, scipy, scikit-learn.

**Recommended for PhysGT training:** `pip install -e ".[full]"` (RNA-FM + Hugging Face hub).

---

## 10. Syncing with upstream

To pull fixes from PhysRNA monorepo without re-copying benchmarks:

```bash
# From monorepo root — example: sync pipeline only
cp ../PhysRNA/physrna_filter/pipeline.py physrna_filter/
cp ../PhysRNA/physrna_filter/analysis/gt_inference.py physrna_filter/analysis/
python -m pytest tests/ -q
```

Avoid blindly copying `physrna_filter/physcomplex/` until you are ready to merge that layer into your filter design.

---

## 11. Contact / provenance

- **Upstream:** PhysRNA / AIYGO Computational Biology Group  
- **License:** MIT (see `LICENSE`)  
- **Extract created by:** Cursor Cloud Agent handoff, 2026-08-17

When PhysComplex Filter is ready, consider publishing as a separate package that depends on `physrna-filter` rather than vendoring forever.
