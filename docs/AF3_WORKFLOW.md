# AlphaFold 3 workflow with PhysRNA

**Goal:** After you download AF3 Server results, run PhysRNA before ordering RNA oligos.

Three commands cover the full workflow. Full oxRNA MD runs by default — researchers get the most accurate ranking without flags.

---

## Step 0 — One-time setup

### Any platform

```bash
pip install physrna-filter
# or from a clone:
pip install -e .

physrna init
```

`physrna init` copies the pre-trained checkpoint to `~/.physrna/gt_checkpoint.pt`. Fine-tunes are saved there and are **never overwritten** by `git pull`.

### Windows (recommended: WSL for oxRNA)

```powershell
cd C:\path\to\PhysRNA
.\scripts\bootstrap_wsl.ps1
```

This runs `physrna init` and auto-detects oxDNA in WSL. For this PowerShell session it sets:

```powershell
$env:OXDNA_BIN = "wsl:/home/you/oxDNA/build/bin/oxDNA"
$env:OAT_BIN = "wsl:/home/you/miniconda3/bin/oat"
$env:PHYSRNA_OXRNA_RELAX_STEPS = "100000"
```

Re-run the bootstrap script (or add those lines to your profile) before ranking if you want real oxRNA MD on Windows.

### Docker

```bash
docker build -t physrna .
docker run --rm -v "$PWD/af3_predictions:/data" physrna rank /data --rbp LIN28A
```

---

## Step 1 — Download AF3 Server zips

1. Submit jobs at [alphafoldserver.com](https://alphafoldserver.com)
2. Download each `fold_*.zip`
3. Put them in one folder, e.g. `af3_predictions/`

**No panel JSON required.** PhysRNA reads sequences from each zip automatically.

---

## Step 2 — Rank candidates (default: full accuracy)

```bash
physrna rank af3_predictions --rbp LIN28A
```

This runs:

- Full pipeline on every candidate (oxRNA MD when available, otherwise internal fallback with a warning)
- Auto contrastive fine-tune when you have **2+ RNAs for the same protein** and no local fine-tuned checkpoint yet
- Your existing fine-tuned weights in `~/.physrna/` if present

Outputs:

| File | Purpose |
|------|---------|
| `ranked_candidates.csv` | Sortable table |
| `ranked_candidates_report.html` | Open in browser — go/no-go per candidate |
| `ranked_candidates_metrics.json` | Summary stats |

```powershell
# Windows
start ranked_candidates_report.html
```

**Read the table:**

| Column | Meaning |
|--------|---------|
| `rank` | 1 = best candidate |
| `af3_iptm` | AF3 interface confidence (higher = better) |
| `composite_score` | PhysRNA score (lower = better) |
| `bio_verdict` | Motif / partner check |
| `combined_verdict` | PASS / WARN / FAIL |

**Rule of thumb:** Trust `bio_verdict=FAIL` as do-not-pursue. Within passing biology, rank by `composite_score`, not `af3_iptm` alone.

### Filter vs rank-all

```bash
# Only LIN28A jobs (wrong-partner controls excluded):
physrna rank af3_predictions --rbp LIN28A

# Every zip in folder (mixed panel / benchmark):
physrna rank af3_predictions --all
```

---

## Step 3 — Benchmark panel (P1–P10 / N1–N10)

```bash
physrna panel af3_predictions
```

Uses the extended 20-job panel, auto fine-tunes unless you already have a tuned checkpoint, and writes `eval_panel_report.html`.

---

## Optional flags

| Flag | When to use |
|------|-------------|
| `--fast` | Quick triage only (~1 min/job, skips oxRNA). Not for final decisions. |
| `--deep-top 3` | With `--fast`: oxRNA on top 3 finalists only |
| `--no-finetune` | Skip auto fine-tune |
| `--finetune` | Force re-fine-tune even if checkpoint is already tuned |
| `--require-oxrna` | Fail instead of CG fallback when oxDNA missing |

---

## Keeping your fine-tuned model

| Location | Role |
|----------|------|
| `~/.physrna/gt_checkpoint.pt` | **Your** weights (fine-tunes, backups) |
| `physrna_filter/validation/gt_checkpoint.pt` | Shipped base model (read-only after pip install) |

```bash
# Verify which checkpoint is active:
python -m physrna_filter.validation.download_gt_checkpoint
```

Override with `PHYSRNA_CHECKPOINT=/path/to/my.pt` or `PHYSRNA_HOME=/custom/dir`.

---

## Optional manifest (eCLIP coordinates)

```bash
physrna rank --write-manifest-template candidates.csv
# edit CSV, then:
physrna rank af3_predictions --manifest candidates.csv
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Checkpoint not found | `physrna init` |
| oxRNA fails on Windows | `.\scripts\bootstrap_wsl.ps1` and set `OXDNA_BIN=wsl:...` |
| Empty ranked table | Ensure zips are real AF3 downloads (>1 KB) |
| Results look worse than before | You may have run with `--fast` or `--rbp-name` on a mixed folder — use defaults above |

---

## FoldBench benchmark (70 protein–RNA targets)

Publication-grade evaluation against the [FoldBench](https://github.com/BEAM-Labs/FoldBench) protein–RNA set (post–AF3-training-cutoff crystal structures).

```bash
# Step 1: export AlphaFold Server JSON (upload at alphafoldserver.com)
physrna benchmark foldbench --prepare -o foldbench_af3_jobs.json

# Pilot: first 10 jobs only
physrna benchmark foldbench --prepare --max-targets 10 -o foldbench_pilot.json

# Step 2: submit jobs at alphafoldserver.com, save zips to foldbench_af3/

# Step 3: run benchmark — structural labels + PhysRNA + AUROC
physrna benchmark foldbench --predictions foldbench_af3/ -o foldbench_results.csv

# Check which predictions you still need:
physrna benchmark foldbench --list --predictions foldbench_af3/
```

**Metrics in `foldbench_metrics.json`:**

| Metric | Meaning |
|--------|---------|
| `auroc_composite_neg` | Can PhysRNA `composite_score` separate good vs bad AF3 poses? (higher = better) |
| `auroc_af3_iptm` | Same for AF3 ipTM alone |
| `physrna_fail_on_structural_fail_rate` | Fraction of structurally failed AF3 jobs that PhysRNA marks FAIL |

Structural success uses CAPRI acceptable cutoffs (iRMSD ≤ 4 Å, L-RMSD ≤ 10 Å vs crystal). Override with `--labels-csv` if you have FoldBench DockQ outputs.

---

## Related docs

- [AF3_EVAL_PANEL.md](AF3_EVAL_PANEL.md) — panel job definitions
- [TEST_COMPLEXES.md](TEST_COMPLEXES.md) — Holdout crystal sequences
