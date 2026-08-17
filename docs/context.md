# PhysRNA-Filter — Project Context

## What This Is

PhysRNA-Filter is a physics-informed validation pipeline for AI-generated protein-RNA structural complexes.
It targets a specific failure mode in tools like AlphaFold 3 and RoseTTAFold All-Atom: **hallucinatory binding**,
where the model forces an RNA into a protein binding pocket regardless of whether that pose is physically achievable.

The filter operates downstream of any structure predictor. You feed it a `.pdb` or `.cif` file, and it
returns a verdict on whether the predicted RNA-protein interface is:

1. Geometrically plausible (no clashes, correct bond geometry)
2. Conformationally accessible (the RNA can actually adopt the bound pose from free solution)
3. Geometrically consistent (per-nucleotide torsion angles match conformations seen in simulation)

---

## The Core Problem

AlphaFold 3 was trained on crystal structures — snapshots of already-frozen, already-bound complexes.
It has no representation of:

- The free RNA conformational ensemble in solution
- The probability of the RNA sampling the bound conformation spontaneously
- The conformational entropy cost (TΔS) of locking the RNA into the bound pose
- Whether each nucleotide's local geometry (sugar pucker, backbone torsions) is physically reasonable

This produces two classes of hallucination:

| Class | What's wrong | How detected |
|---|---|---|
| Geometric hallucination | Atom clashes, wrong bond geometry | Static analysis |
| Entropic hallucination | Pose is geometrically fine but conformationally inaccessible | Ensemble comparison |
| Local geometry hallucination | Individual nucleotide torsions are outside normal ranges | Per-residue torsion scoring |

Most existing validators only catch the first class. PhysRNA-Filter catches all three.

---

## Pipeline Architecture

```
[AF3 / RoseTTAFold .pdb input]
         │
         ▼
[1. Parse Complex]
   — separate protein and RNA chains
   — identify chain types from residue names
         │
         ▼
[2. Find Interface Residues]
   — NeighborSearch: RNA atoms within 5Å of protein
   — outputs list of (chain_id, residue_number) pairs
         │
         ▼
[3. Extract Interface Coordinates + Local Geometry]
   — C4' backbone coords for RMSD branch
   — 7 backbone torsions (α β γ δ ε ζ χ) per nucleotide
   — sugar pseudorotation phase P and amplitude ν_max
   — base stacking geometry (distance + angle to neighbors)
         │
         ▼
[4. Simulate Free RNA]
   — fold sequence with RNAfold → 2D structure
   — build 3D with FARFAR2 or SimRNA
   — run oxRNA coarse-grained MD (10–50 ns equivalent)
   — collect N snapshots (typically 5,000–20,000)
         │
         ▼
[5. Extract Geometry Features from Each Snapshot]
   — same torsion + sugar pucker features as step 3
   — focused on interface nucleotides only
         │
         ▼
[6. Dual-Branch Clustering]
   ┌─────────────────────┐  ┌─────────────────────────┐
   │  RMSD branch        │  │  Geometry branch         │
   │  cluster on C4'     │  │  cluster on torsion      │
   │  coordinates        │  │  angle vectors           │
   │  → coordinate       │  │  → torsion-space         │
   │    medoids          │  │    centroids             │
   └──────────┬──────────┘  └────────────┬────────────┘
              │                          │
              ▼                          ▼
[7. Score AF3 Bound Pose Against Both Cluster Sets]
   — RMSD score: min Kabsch-aligned RMSD to nearest coord medoid
   — Geometry score: min angular distance to nearest torsion centroid
   — Per-nucleotide geometry deviation (flags specific residues)
         │
         ▼
[8. Combined Verdict]
   — PASS / WARN / FAIL
   — per-residue breakdown
   — highlighted suspicious nucleotides
```

---

## Key Algorithms

### Kabsch RMSD

Computes the minimum RMSD between two coordinate sets after optimal rigid-body alignment.

1. Center both structures at origin
2. Compute covariance matrix H = M^T · R
3. SVD decompose: H = U · S · V^T
4. Handle reflection: d = det(V^T · U^T)
5. Rotation matrix: R = V · diag(1,1,d) · U^T
6. RMSD = sqrt(mean(||R_aligned - M_centered||²))

### RNA Backbone Torsion Angles

Seven angles define the complete conformation of each nucleotide:

| Angle | Atoms | Range |
|---|---|---|
| α | O3'(i-1)–P–O5'–C5' | −180° to 180° |
| β | P–O5'–C5'–C4' | −180° to 180° |
| γ | O5'–C5'–C4'–C3' | −180° to 180° |
| δ | C5'–C4'–C3'–O3' | −180° to 180° |
| ε | C4'–C3'–O3'–P(i+1) | −180° to 180° |
| ζ | C3'–O3'–P(i+1)–O5'(i+1) | −180° to 180° |
| χ | O4'–C1'–N9–C4 (purines) / O4'–C1'–N1–C2 (pyrimidines) | −180° to 180° |

### Sugar Pseudorotation

The ribose ring conformation is described by pseudorotation phase P and amplitude ν_max.
P ≈ 0° → C3'-endo (A-form RNA, common in structured regions)
P ≈ 180° → C2'-endo (B-form DNA-like, unusual for RNA — flags suspicious nucleotides)

### Angular Distance for Torsion Clustering

Torsion angles are circular. Distance between two angle vectors uses the circular mean:

```
d(θ₁, θ₂) = sqrt( Σᵢ min(|θ₁ᵢ - θ₂ᵢ|, 360° - |θ₁ᵢ - θ₂ᵢ|)² )
```

### Entropic Plausibility Score

```
score = min over all cluster medoids { Kabsch_RMSD(AF3_bound, medoid) }
```

Threshold calibration (from RNA-Puzzles data):
- < 2.0 Å → PASS (bound conformation well-sampled in free ensemble)
- 2.0–4.0 Å → WARN (rarely sampled, elevated entropic cost)
- > 4.0 Å → FAIL (conformationally inaccessible)

---

## Datasets

### Training and Benchmarking

| Dataset | URL | Purpose |
|---|---|---|
| ProNAB | http://pronab.ibbr.umd.edu | Mutation ΔΔG ground truth — primary benchmark |
| SKEMPI 2.0 | https://life.bsc.es/pid/skempi2 | Secondary ΔΔG benchmark |
| PDB RNA-protein complexes | https://www.rcsb.org | Training structures |
| PRIDB | http://pridb.gdcb.iastate.edu | Curated interface annotations |
| RNAsite benchmark | Published with RNAsite paper | Native + decoy structures |
| Dockground RNA | https://dockground.compbio.ku.edu | Docking decoys (negative examples) |

### Biological Validation

| Dataset | URL | Purpose |
|---|---|---|
| ENCODE eCLIP | https://www.encodeproject.org | 150+ RBPs, in vivo binding sites |
| POSTAR3 | http://postar.ncrnalab.org | Aggregated CLIP-seq |
| RBPDB | http://rbpdb.ccbr.utoronto.ca | Verified binding sequences + Kd |
| ATtRACT | https://attract.cnic.es | RBP sequence motifs |

### RNA Conformational Reference

| Dataset | URL | Purpose |
|---|---|---|
| RNA-Puzzles | http://www.rnapuzzles.org | RMSD threshold calibration |
| Rfam | https://rfam.org | RNA family conformational variation |
| BGSU RNA 3D Hub | http://rna.bgsu.edu | RNA 3D motif atlas |

---

## Scoring Output

For each input complex, PhysRNA-Filter reports:

```
PhysRNA-Filter Report
─────────────────────────────────────────────
Input:              complex.pdb
Interface residues: A:14 A:15 A:16 A:17 A:18

RMSD Score
  Nearest cluster:    3  (of 8 clusters)
  RMSD to medoid:     1.84 Å
  Verdict:            PASS

Geometry Score
  Nearest cluster:    1  (of 6 torsion clusters)
  Angular distance:   28.3°
  Verdict:            WARN

Per-Nucleotide Breakdown
  A:14  PASS   δ=124°  P=18° (C3'-endo, normal)
  A:15  PASS   δ=138°  P=22° (C3'-endo, normal)
  A:16  WARN   δ=157°  P=148° (near C2'-endo, unusual for A-form)
  A:17  FAIL   δ=172°  P=179° (C2'-endo, rare in structured RNA)
  A:18  PASS   δ=119°  P=15° (C3'-endo, normal)

Combined Verdict:  WARN
  — A:17 adopts C2'-endo sugar pucker not observed in free ensemble
  — Geometry cluster mismatch at interface position 4
─────────────────────────────────────────────
```

---

## Mutation Sensitivity Test

Beyond scoring a single structure, the pipeline supports the mutation sensitivity test:

1. Load wild-type complex
2. Introduce user-specified mutation computationally (PyRosetta or FoldX)
3. Re-extract interface geometry
4. Compare filter score change to experimental ΔΔG from ProNAB

A correctly working filter should show score increase (worse) when experimental ΔΔG is large and positive.

---

## Realistic Timeline

| Phase | Weeks | Deliverable |
|---|---|---|
| Geometric validator | 1–3 | Clash + torsion checker |
| Free RNA simulation | 4–6 | oxRNA integration + snapshot extraction |
| RMSD cluster scoring | 7–9 | Kabsch + k-medoids pipeline |
| Local geometry scoring | 10–12 | Per-nucleotide torsion comparison |
| ProNAB benchmark | 13–15 | Pearson r vs experimental ΔΔG |
| Combined scoring | 16–17 | Fused RMSD + geometry verdict |
| Preprint | 18–22 | bioRxiv submission |

---

## Dependencies

```
biopython       — PDB parsing, structure manipulation
MDAnalysis      — trajectory analysis, atom selection
numpy           — array math, SVD for Kabsch
scipy           — spatial math, circular statistics
scikit-learn    — KMeans, silhouette scoring
pandas          — dataset loading and manipulation
requests        — PDB / ProNAB download
pyrosetta       — mutation introduction (optional, requires license)
```

Optional (for simulation):
```
oxDNA / oxRNA   — coarse-grained RNA MD
OpenMM          — all-atom MD (higher accuracy, more compute)
```

---

## Citing This Tool

If benchmarked against ProNAB and ENCODE eCLIP and published:

- Primary comparison: RNAsite (Chen et al.), DFIRE-RNA, 3dRNAscore
- Novel contribution: per-nucleotide local geometry scoring against free-ensemble clusters
- Novel contribution: explicit entropic plausibility via coarse-grained simulation + RMSD to cluster medoids

The combination of static geometry validation + conformational ensemble comparison + per-residue torsion scoring
against simulation-derived clusters is not present in any existing published tool as of mid-2025.
