# Predict & Validate — sequences to PhysRNA report

One-stop workflow for **Use Case 3**: input protein + RNA sequences, obtain an
AlphaFold 3 model, run PhysRNA (oxRNA + PhysGT + physics branches).

## Quick start (unified CLI)

```bash
physrna init
physrna configure af3 --api-key YOUR_KEY    # optional — enables automatic AF3 submit
physrna predict --protein MKTIIAL... --rna AUGCAUGC... --rbp LIN28A
```

With a saved API key, `predict` submits to AlphaFold, waits for the zip, and runs
PhysRNA immediately. Without a key, it writes Server JSON for manual upload.

After manual AF3 Server upload:

```bash
physrna predict --protein MKTIIAL... --rna AUGCAUGC... \
  --af3-zip fold_myjob.zip --rbp LIN28A
```

Resume a long-running API job:

```bash
physrna predict --af3-job-id JOB_ID --protein ... --rna ...
```

## Important: how AF3 access works

| Method | Fully automatic? | Requirements |
|--------|------------------|--------------|
| **server-json** | Two-step | Free [AlphaFold Server](https://alphafoldserver.com) account; upload JSON, download zip |
| **local** | Yes (one command) | Linux/WSL, GPU, AF3 weights from Google, ~TB databases |
| **api** | Yes (if key works) | `physrna configure af3 --api-key KEY` or `AF3_API_KEY` |
| **zip** | Validate only | You already ran AF3 |

There is **no official public REST API** for AlphaFold Server job submission
for most users. PhysRNA generates the correct JSON and automates everything
**after** you have a structure file.

## Quick start (Windows — recommended)

### Step 1 — Generate AF3 job JSON

```powershell
cd C:\Users\frann\Downloads\PhysRNA-main

python -m physrna_filter.validation.predict_validate `
  --protein MKTIIALSYIFCLVFADYKDYNLKWNIKALNISLPSYYEIKLQAKKDITKGLHIFQK `
  --rna AUGCAUGCAUGCAUGCAUGC `
  --mode server-json `
  --job-name my_rbp_hairpin `
  --server-json-out af3_jobs\my_rbp_hairpin.json
```

### Step 2 — Run on AlphaFold Server

1. Open https://alphafoldserver.com
2. **Upload JSON** → select `my_rbp_hairpin.json`
3. Submit and wait for the job
4. Download `fold_my_rbp_hairpin.zip`

### Step 3 — Validate with PhysRNA

```powershell
$env:OXDNA_BIN = "wsl:/home/frann/oxDNA/build/bin/oxDNA"

python -m physrna_filter.validation.predict_validate `
  --protein MKTIIALSYIFCLVFADYKDYNLKWNIKALNISLPSYYEIKLQAKKDITKGLHIFQK `
  --rna AUGCAUGCAUGCAUGCAUGC `
  --af3-zip C:\path\to\fold_my_rbp_hairpin.zip `
  --require-gt-checkpoint `
  --rbp-name "MyRBP" `
  --output-json results\my_rbp_report.json
```

Exit codes: `0` = PASS, `1` = WARN/FAIL, `2` = JSON written only (AF3 pending).

## Local AF3 (WSL/Linux + GPU)

```bash
export AF3_MODEL_DIR=$HOME/af3_models
export AF3_DB_DIR=$HOME/public_databases
export AF3_DOCKER_IMAGE=alphafold3

python -m physrna_filter.validation.predict_validate \
  --protein ... --rna ... \
  --mode local \
  --require-gt-checkpoint
```

See [AlphaFold 3 installation](https://github.com/google-deepmind/alphafold3).

## Environment variables

| Variable | Purpose |
|----------|---------|
| `AF3_MODEL_DIR` | Local AF3 model parameters |
| `AF3_DB_DIR` | Genetic databases for local AF3 |
| `AF3_DOCKER_IMAGE` | Docker image (default `alphafold3`) |
| `AF3_API_KEY` / `ALPHAFOLD_API_KEY` | API mode (or use `physrna configure af3`) |
| `AF3_API_URL` | API base URL (default `https://alphafoldserver.com/api`) |
| `AF3_API_POLL_S` | Poll interval in seconds (default 30) |
| `OXDNA_BIN` | WSL path to oxDNA for RMSD branch |
| `PHYRNA_AF3_WORK` | Cache dir for JSON/jobs (default `~/.cache/physrna_filter/af3_jobs`) |

## Product positioning

**PhysRNA Predict & Validate** = AF3 structure generation hook + hallucination
filter trained on experimental protein–RNA ΔΔG data. Sell it as QC for
AI-predicted RBP–RNA complexes, not as an AF3 replacement.
