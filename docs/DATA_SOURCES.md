# PhysGT training data sources

PhysGT learns from **experimentally measured protein-side mutation ΔΔG** at
protein–RNA interfaces (PDB structure + mutation + ΔΔG in kcal/mol).

Current pipeline entry point: `fetch_pronab()` → 710 entries (PRA-MutPred scrape).  
Expanded merge: `fetch_training_data(include_nabe=True)` → **~1,029 entries** (incl. literature).

---

## Tier 1 — Use now (trusted, structured ΔΔG)

### 1. ProNAB / PRA-MutPred (current primary)

| Field | Value |
|-------|-------|
| URL | [ProNAB](https://web.iitm.ac.in/bioinfo2/pronab/) · [PRA-MutPred dataset](https://web.iitm.ac.in/bioinfo2/pramutpred/datasetdetails.html) |
| Paper | Harini et al., *Nucleic Acids Research* (ProNAB, 2023+) |
| Entries | **710** mutations, **134–142** PDB complexes (3D structures) |
| In repo | `physrna_filter/data/pronab_raw.csv` via `fetch_pronab()` |
| Quality | Curated IIT Madras lab; same set used by PRA-MutPred (reported r=0.75 LOCO) |

**Upside:** Full parent database has **~5,326 protein-RNA affinity entries** and
**20,219 total** mutation/affinity records, but bulk CSV requires contacting
ProNAB authors (search UI only for automated download).

**Action:** Email ProNAB for full structured export — largest single source.

---

### 2. Nabe (implemented: `fetch_nabe()`)

| Field | Value |
|-------|-------|
| URL | [nabe.denglab.org](http://nabe.denglab.org/) · [PMC8363842](https://pmc.ncbi.nlm.nih.gov/articles/PMC8363842/) |
| Paper | Luo et al., *Nucleic Acids Research*, 2021 |
| Full DB | 2,506 mutations (778 RNA, 1,728 DNA), 473 complexes |
| RNA benchmark | **PRNA90** — 400 mutations, 89 PDBs (CD-HIT non-redundant) |
| Download | `http://nabe.denglab.org/download/Nabe_database.tar.gz` (verified) |
| Overlap vs ProNAB | ~357 shared; **~362 parseable RNA-only additions** |

**Action (ready):**

```bash
python -c "from physrna_filter.data.fetch_training_data import fetch_training_data; fetch_training_data()"
```

Use `include_nabe=True` in training after validating graph-build success rate on new PDBs.

---

## Tier 2 — Supplemental / specialized

### 3. dbAMEPNI (alanine scanning)

| Field | Value |
|-------|-------|
| URL | [zhulab.org.cn/dbAMEPNI](http://zhulab.org.cn/dbAMEPNI/) |
| Entries | 577 quantitative alanine ΔΔG (859 total incl. qualitative) |
| Note | Mix of DNA/RNA; download link was 404 when checked (2026-06). Manual curation from browse UI may be needed. |
| Value | Hotspot-focused; good for interface residue weighting, not full ΔΔG regression |

### 4. PDBbind v2020 (binding affinity, not mutation ΔΔG)

| Field | Value |
|-------|-------|
| URL | [pdbbind-cn.org](http://www.pdbbind-cn.org/) |
| Content | ~19.5k complexes including protein–nucleic acid with **Kd/ΔG** |
| Limitation | Wild-type affinity only — **no mutation ΔΔG** for PhysGT Siamese training |
| Use | Pretraining decoy discrimination, not ΔΔG head |

### 5. Literature mini-panels (manual curation)

| System | Source | Entries | In repo |
|--------|--------|---------|---------|
| U1A / 1URN | Nolan 1999 PNAS; Showalter 2002 | 6 ITC ΔΔG | `benchmarks/run_u1a_benchmark.py` |
| Multi-RBP panel | `run_multi_benchmark.py` | 5 systems | benchmarks/ |
| RBM39, SARS-CoV-2 NTD | Nat Commun 2023–2024 ITC/MST | 5 | `literature_mined.csv` via `fetch_literature()` |

Good for **sanity checks** and **supplemental training**; curated from 2023–2026 papers.

---

## Tier 3 — Not recommended for PhysGT

| Source | Why skip |
|--------|----------|
| **SKEMPI 2.0** | Protein–protein only in practice (see `docs/RESULTS.md` §4) |
| **PROXiMATE** | Protein–protein mutation thermodynamics |
| **AF3 predictions** | No experimental ΔΔG — use for screening only |
| **RNAsite decoys** | Native vs decoy structures, no mutation labels |

---

## Recommended expansion roadmap

1. **Immediate (+50% data):** Train on `fetch_training_data(include_nabe=True)` (~1,070 entries). Re-run LOCO/holdout to confirm no regression.
2. **Medium term:** Obtain full ProNAB CSV from authors (~5k RNA entries; filter to single-point mutations with PDB + ΔΔG).
3. **Curation:** Mine recent literature (2023–2026) for ITC/SPR RNA-binding mutations not yet in Nabe/ProNAB — expect 50–150 new high-quality points.
4. **Do not mix:** DNA-protein mutations (Nabe DNA subset, PDNA160) unless DNA nodes/force fields are added to PhysGT.

---

## Structural positives (crystal screening)

These are **not** mutation ΔΔG training sets. Use them to validate that PhysRNA
scores experimental crystals as plausible (positives), distinct from AF3 decoys.

### RNApedia (~56k interfaces)

| Field | Value |
|-------|-------|
| URL | [bioinfo.dcc.ufmg.br/rnapedia3](https://bioinfo.dcc.ufmg.br/rnapedia3/) |
| Entries | **56,133** protein–RNA pairs, **5,015** unique PDB codes |
| In repo | `fetch_rnapedia_manifest()`; bundled sample `rnapedia_database_sample.tsv` |
| Optional | `fetch_rnapedia_affinity()` — Zenodo affinity subset (~251 MB) |

```bash
physrna fetch rnapedia --update
physrna benchmark rnapedia --max-targets 50 --fast
```

### RCSB PDB protein–RNA catalog (~7.9k entries)

| Field | Value |
|-------|-------|
| Source | RCSB Search API (RNA + protein polymers) |
| Entries | **7,876** crystal structures (X-ray / EM) |
| In repo | `fetch_pdb_catalog()`; bundled sample `pdb_protein_rna_sample.json` |

```bash
physrna fetch pdb --update
physrna benchmark pdb --max-targets 50 --fast
```

### Smoke-test all three

```bash
physrna fetch all --sample          # offline manifests
physrna test datasets --sample --max-targets 3
physrna benchmark pronab --max-targets 20
```

---

## Quality filters before training

Apply the same gates as current `train_gt.py`:

- Single-point protein mutations parseable as `X123Y`
- PDB downloadable from RCSB; graph builds for WT + mutant
- Exclude entries with \|ΔΔG\| > 8 kcal/mol (measurement/outlier artifacts)
- LOCO split by **PDB complex**, never by mutation within same PDB
- Track `source` column (`pronab` vs `nabe`) in eval CSV for per-source Pearson r

---

## Commands

```bash
# Current ProNAB only
python -c "from physrna_filter.data.fetch_pronab import fetch_pronab; print(len(fetch_pronab()))"

# Download + merge ProNAB + Nabe + literature
python -c "from physrna_filter.data.fetch_training_data import fetch_training_data; df=fetch_training_data(); print(df['source'].value_counts())"

# Holdout split report (fast — no graph build)
python -m physrna_filter.validation.holdout_eval_merged --report-only

# Full deploy on merged data (after graph cache built)
python -m physrna_filter.validation.deploy_gt --graph-cache physrna_filter/validation/gt_graphs_merged.pt

# ProNAB bulk CSV (after author response)
python -m physrna_filter.data.import_pronab_bulk path/to/pronab_bulk.csv

# Refresh from upstream
python -c "from physrna_filter.data.fetch_training_data import fetch_training_data; fetch_training_data(force_download=True)"
```
