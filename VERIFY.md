# Verification checklist (run before publishing to GitHub)

Last run: 2026-08-17 — all checks passed.

## 1. Install (clean copy)

```bash
cd physcomplex_filter
python3 -m pip install -e ".[dev]"
```

## 2. Tests

```bash
python3 -m pytest tests/ -q
# Expected: 154 passed
```

## 3. Module import sweep

Every `.py` file under `physrna_filter/` must import without error:

```bash
python3 -c "
import importlib, pkgutil
from pathlib import Path
failures = []
for m in pkgutil.walk_packages(['physrna_filter'], 'physrna_filter.'):
    if m.ispkg: continue
    try: importlib.import_module(m.name)
    except Exception as e: failures.append((m.name, e))
assert not failures, failures
print('All modules import OK')
"
```

## 4. CLI smoke

```bash
python3 -m physrna_filter.cli doctor
python3 -m physrna_filter.pipeline physrna_filter/data/structures/1urn.pdb
python3 -m physrna_filter.pipeline physrna_filter/data/structures/1a9n.pdb
python3 -m physrna_filter.validation.gt_readiness
python3 -m physrna_filter.validation.deploy_gt --help
python3 -m physrna_filter.validation.eval_gt --help
python3 -m physrna_filter.validation.rank_af3_candidates --help
python3 -m physrna_filter.validation.screen_af3 --help
```

Console scripts (after `pip install -e .`):

```bash
physrna doctor
physrna-filter physrna_filter/data/structures/1urn.pdb
```

## 5. Data & checkpoint

```bash
python3 -c "
from physrna_filter.data.fetch_training_data import fetch_training_data
from physrna_filter.data.af3_eval_panel import load_af3_eval_panel
from physrna_filter.analysis.gt_inference import load_gt_model
assert len(fetch_training_data()) == 1029
assert len(load_af3_eval_panel()) == 10
_, meta = load_gt_model('physrna_filter/validation/gt_checkpoint.pt')
assert meta['interface_head_trained']
print('Data + checkpoint OK')
"
```

## 6. Init workflow (isolated home)

```bash
PHYSRNA_HOME=/tmp/physrna_handoff_test python3 -m physrna_filter.cli init
test -f /tmp/physrna_handoff_test/gt_checkpoint.pt && echo "init OK"
```

## 7. Known exclusions (expected failures)

These subcommands are **not** in the starter kit and should fail with `ModuleNotFoundError`:

```bash
python3 -m physrna_filter.cli boltz prepare -o /tmp/x   # exit 1
python3 -m physrna_filter.cli benchmark foldbench --help  # exit 1
```

## 8. Optional (not required for AF3 screening)

- RNA-FM weights: `python3 -m physrna_filter.data.download_rnafm_weights` (~1.2 GB)
- oxDNA: `scripts/install_oxdna.sh`

Without RNA-FM, PhysGT training uses fallback encoding; AF3 screening still works with the shipped checkpoint.
