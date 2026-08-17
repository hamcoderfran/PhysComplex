# PhysRNA user guide

One-page reference for the **accuracy-first** workflow. All proven techniques stay enabled by default: full oxRNA MD, PhysGT, contrastive fine-tune when applicable, partner biology checks.

---

## Install

```bash
git clone https://github.com/hamcoderfran/PhysRNA.git
cd PhysRNA
pip install -e .

physrna init          # checkpoint → ~/.physrna/
physrna doctor        # verify checkpoint + oxDNA
```

Windows + oxDNA: `.\scripts\bootstrap_wsl.ps1`

---

## Commands (use these)

| Command | When |
|---------|------|
| `physrna init` | First-time setup |
| `physrna doctor` | Troubleshoot missing checkpoint / oxDNA |
| `physrna configure af3 --api-key KEY` | Save AlphaFold API key (optional) |
| `physrna predict --protein SEQ --rna SEQ --rbp NAME` | **Predict + validate** one complex |
| `physrna boltz prepare` | Generate 100 Boltz test complexes (YAML + manifest) |
| `physrna rank FOLDER --rbp NAME` | **Main workflow** — rank your AF3 candidates |
| `physrna panel FOLDER` | Benchmark 20-job P/N panel |
| `physrna panel FOLDER --prepare-missing jobs.json` | Export AF3 JSON for missing panel zips |
| `physrna benchmark foldbench --prepare` | Export 70 FoldBench jobs for AF3 Server |
| `physrna benchmark foldbench --predictions DIR` | Evaluate PhysRNA vs crystal labels |
| `physrna report results.csv` | Regenerate HTML report |

Legacy aliases still work: `physrna-rank`, `physrna-screen`, `physrna-filter`.

---

## Typical lab workflow

### Option A — sequences in, verdict out (single complex)

```bash
physrna init
physrna configure af3 --api-key YOUR_ALPHAFOLD_SERVER_KEY   # optional
physrna predict --protein MKTIIAL... --rna AUGCAUGC... --rbp LIN28A
```

Without an API key, `predict` writes AlphaFold Server JSON and prints upload instructions.
Re-run with `--af3-zip fold_myjob.zip` after downloading from alphafoldserver.com.

### Option B — rank many AF3 candidates

```bash
# 1. Download AF3 Server zips into af3_predictions/

# 2. Rank (full accuracy — no --fast)
physrna rank af3_predictions --rbp LIN28A

# 3. Open report
# ranked_candidates_report.html
```

### Option C — Boltz batch (100 test complexes, no AF3 Server quota)

```bash
# Bundled set (or regenerate):
physrna boltz prepare -o boltz_test_100 --count 100

# Predict with Boltz (GPU + pip install boltz):
boltz predict boltz_test_100/inputs --use_msa_server

# Copy/symlink top-model .cif files into boltz_test_100/predictions/
physrna rank boltz_test_100/predictions --manifest ../manifest.csv --rbp U1A
```

The 100-complex set spans 10 RBP families (U1A, MS2, LIN28, PUM1, …) with native
positives, wrong-partner swaps, and shuffled decoys — ideal for screening benchmarks.

**Verdicts (AF3 mode):**

| Verdict | Meaning |
|---------|---------|
| PASS | PhysGT + biology + clashes support the pose |
| WARN | Review manually (physics-only GT, borderline scores, or missing branch) |
| FAIL | Do not pursue (wrong partner, clashes, or bad PhysGT) |

Combined verdict uses **PhysGT + biology + steric clashes**. RMSD/geometry/contact are diagnostic columns in the CSV.

---

## Data & checkpoints

| Path | Purpose |
|------|---------|
| `~/.physrna/gt_checkpoint.pt` | **Your** weights (fine-tunes never overwritten by git pull) |
| `physrna_filter/validation/gt_checkpoint.pt` | Shipped base model (~4.5 MB) |
| `physrna_filter/validation/gt_interface_pretrain.pt` | Interface-head training output (safe default) |

Override: `PHYSRNA_CHECKPOINT=/path/to.pt` or `PHYSRNA_HOME=/custom/dir`

Fine-tune backups: `gt_checkpoint.pt.bak` written before panel fine-tune.

---

## Accuracy options

| Flag | Effect |
|------|--------|
| *(default)* | Full oxRNA MD + auto fine-tune when 2+ RNAs / same protein |
| `--fast` | Quick triage only (skip oxRNA) — not for final decisions |
| `--no-finetune` | Skip contrastive fine-tune |
| `--require-oxrna` | Fail if oxDNA missing (no CG fallback) |
| `--allow-physics-only` | Allow run without trained PhysGT (not recommended) |

---

## Training (advanced)

```bash
# Interface head only → safe path (does not overwrite production ckpt)
python -m physrna_filter.validation.train_interface_head
python -m physrna_filter.validation.merge_gt_checkpoint

# Full deploy
python -m physrna_filter.validation.deploy_gt --n-folds 1 --gt-epochs 80

# Panel fine-tune (writes to ~/.physrna/)
physrna panel af3_predictions
```

---

## Further reading

- [AF3_WORKFLOW.md](AF3_WORKFLOW.md) — detailed AF3 playbook
- [AF3_EVAL_PANEL.md](AF3_EVAL_PANEL.md) — P/N panel job list
- [TEST_COMPLEXES.md](TEST_COMPLEXES.md) — holdout crystal sequences
- [README.md](../README.md) — full command reference & training corpus
