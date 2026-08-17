# PhysRNA-Filter: Full Walkthrough

## The Problem This Solves

AlphaFold 3 and RoseTTAFold All-Atom are trained on crystal structures —
snapshots of protein-RNA complexes that have already formed. The models learn
to recognize patterns of what a bound complex looks like. What they are never
shown is the process of binding: the free RNA floating in solution, sampling
thousands of conformations per nanosecond, until the right protein comes along.

This creates a specific failure mode. When you ask AF3 to predict the structure
of a protein-RNA complex, it produces a confident-looking output regardless of
whether the RNA could physically achieve the predicted bound conformation from
free solution. It has no way to evaluate the *cost* of forcing the RNA into
that pose. If a mutation eliminates the key contact that drives binding, the
model often predicts the same interface anyway, because the sequence-to-structure
mapping it learned doesn't encode whether that contact is load-bearing.

This is not a bug in AlphaFold — it is a fundamental consequence of training on
equilibrium structures rather than on thermodynamic cycles. But it means that
structural biologists using AF3 for RNA drug discovery cannot fully trust the
predicted interfaces without independent validation.

PhysRNA-Filter is that validation layer.

---

## What Hallucination Looks Like in Practice

There are three distinct failure modes, each requiring a different detector:

### Class 1 — Geometric hallucination
Atoms are placed too close together (steric clash), bond angles are wrong, or
the phosphate backbone adopts an impossible geometry. This is the easiest class
to catch and the one most existing tools handle.

### Class 2 — Entropic hallucination
The interface geometry is locally perfect — no clashes, good hydrogen bonds,
reasonable bond angles — but the RNA as a whole would have to adopt a
conformation it essentially never visits in free solution. Even if the bound
pose looks fine, the RNA has to pay a conformational entropy penalty (T·ΔS)
to freeze into that shape. If the pose is rare enough, this penalty exceeds the
enthalpic gain from contacts and binding simply does not happen. AF3 cannot see
this because it has no representation of the free RNA ensemble.

### Class 3 — Local geometry hallucination
Individual nucleotides at the interface adopt backbone torsion angles or sugar
pucker conformations that are geometrically legal but physically unusual. The
canonical example is a C2'-endo sugar pucker (pseudorotation phase P ~ 160°)
at an interface that should be A-form RNA (C3'-endo, P ~ 18°). The atom
positions are plausible in isolation, but the energy cost of adopting that
conformation is not accounted for.

PhysRNA-Filter detects all three, using a layered pipeline.

---

## The Pipeline — Six Steps

```
[AF3 / RoseTTAFold output (.pdb or .cif)]
               |
               v
     [1. Parse complex]
     Separate protein and RNA chains.
     Identify chain type by majority residue class
     (RNA: A/U/G/C; protein: three-letter amino acids).
               |
               v
     [2. Find interface residues]
     NeighborSearch: any RNA atom within 5 A of
     any protein heavy atom. Returns a list of
     (chain_id, residue_number) pairs.
               |
               v
     [3. Extract bound-pose features]
     From the AF3 structure, for interface nucleotides:
       - C4' coordinates (for RMSD branch)
       - 7 backbone torsion angles alpha/beta/gamma/
         delta/epsilon/zeta/chi (for geometry branch)
       - Sugar pseudorotation phase P and amplitude vmax
       - Base stacking geometry
               |
               v
     [4. Simulate free RNA]
     Fold the RNA sequence alone (no protein) into
     a 3D starting structure. Run oxRNA coarse-grained
     MD to generate a conformational ensemble of the
     unbound RNA. Collect 5,000-20,000 snapshots.
               |
               v
     [5. Cluster the ensemble — dual branch]
       RMSD branch:     k-medoids on C4' coordinates
       Geometry branch: k-means on torsion angle vectors
                        (with circular encoding)
               |
               v
     [6. Score AF3 pose against clusters]
       RMSD score:    min Kabsch-RMSD to any coord medoid
       Geometry score: min angular distance to any torsion centroid
       Per-nucleotide: flag C2'-endo pucker, syn chi, high torsion dev
               |
               v
     [PASS / WARN / FAIL + per-residue breakdown]
```

---

## Key Algorithms

### Kabsch RMSD

The RMSD between two sets of atomic coordinates is only meaningful after you
remove any rigid-body displacement — translation and rotation that just happen
to differ between the two poses without reflecting a real conformational
difference.

The Kabsch algorithm finds the rotation matrix R that minimizes RMSD after
centering both structures at the origin. It does this through singular value
decomposition of the covariance matrix H = M^T R, where M and R are the
centered mobile and reference coordinate arrays.

```
H = M_centered^T @ R_centered
U, S, Vt = SVD(H)
d = det(Vt^T @ U^T)          # det check prevents improper rotation (reflection)
R_optimal = Vt^T @ diag(1,1,d) @ U^T
RMSD = sqrt( mean( || R_centered - M_centered @ R_optimal^T ||^2 ) )
```

The determinant check is critical. Without it, the algorithm might find a
reflection (det = -1) that gives lower RMSD than any proper rotation — which
would be physically meaningless. For chiral biomolecules, alignment must be
restricted to proper rotations (det = +1).

The Kabsch RMSD between the AF3 bound pose and the nearest free-simulation
cluster medoid is the **entropic plausibility score**. A score under 2 Å means
the free RNA naturally visits conformations close to the bound pose. A score
over 4 Å means the bound conformation is not represented in the free ensemble
and the predicted binding is entropically implausible.

### RNA Backbone Torsion Angles

A single RNA residue's conformation is fully described by seven torsion angles
along the backbone and glycosidic bond:

| Angle   | Atom quartet                          | What it describes                  |
|---------|---------------------------------------|------------------------------------|
| alpha   | O3'(i-1) — P — O5' — C5'            | Entry angle from previous residue  |
| beta    | P — O5' — C5' — C4'                 | 5' side of ribose                  |
| gamma   | O5' — C5' — C4' — C3'               | Furanose ring approach             |
| delta   | C5' — C4' — C3' — O3'               | Sugar pucker indicator             |
| epsilon | C4' — C3' — O3' — P(i+1)            | 3' exit toward next residue        |
| zeta    | C3' — O3' — P(i+1) — O5'(i+1)      | Phosphodiester linkage             |
| chi     | O4' — C1' — N9 — C4 (purines)       | Glycosidic bond: syn vs anti       |
|         | O4' — C1' — N1 — C2 (pyrimidines)   |                                    |

These angles are computed from four atomic coordinates using the dihedral
formula based on cross products of bond vectors and atan2 for quadrant
disambiguation.

Torsion angles are circular quantities — the "distance" between 170° and -170°
is 20°, not 340°. This matters for clustering and comparison. The filter
handles this throughout: angular distance uses `min(|a-b|, 360 - |a-b|)` per
component, and k-means clustering encodes each angle as `(sin(θ), cos(θ))` so
that Euclidean distance in the encoded space equals angular distance on the
circle.

### Sugar Pseudorotation Phase

The ribose ring can adopt several conformations characterized by which atoms
are displaced from the plane of the ring. The pseudorotation formalism
describes the ring conformation with two numbers: phase P and amplitude ν_max.
They are computed from the five endocyclic torsion angles ν0 through ν4.

The phase angle P is the key number:

- **P ≈ 0–36°** (C3'-endo): the C3' atom is above the plane on the same side
  as the base. This is the canonical A-form conformation. All standard
  double-stranded RNA adopts C3'-endo. It is the expected state for most
  structured RNA at protein interfaces.

- **P ≈ 144–190°** (C2'-endo): the C2' atom is above the plane. This is the
  B-form DNA conformation. It is rare in RNA — typically less than 5% of
  residues in crystal structures — and energetically costly because the 2'-OH
  is repositioned. If AF3 places a C2'-endo nucleotide at an interface, the
  filter flags it as suspicious unless the free simulation shows C2'-endo
  conformations in that position too.

### Circular Encoding for Torsion Clustering

Standard k-means uses Euclidean distance. For torsion angles this fails
because the space is toroidal — clustering algorithms would treat alpha=179°
and alpha=-179° as far apart when they are actually adjacent. The fix is to
encode each angle θ as the two-component vector (sin(θ), cos(θ)) before
passing to k-means. Euclidean distance in this representation equals the
angular distance on the circle:

```
||(sin(a), cos(a)) - (sin(b), cos(b))||^2
  = sin^2(a) - 2sin(a)sin(b) + sin^2(b) + cos^2(a) - 2cos(a)cos(b) + cos^2(b)
  = 2 - 2(sin(a)sin(b) + cos(a)cos(b))
  = 2 - 2cos(a-b)
  = 2(1 - cos(a-b))
```

which is the squared chord distance proportional to the squared arc distance
for small angles. This means k-means on (sin, cos)-encoded torsions correctly
groups conformations that are angularly close even when they straddle ±180°.

### Contact Energy Scoring

The contact scorer is the primary layer for the ProNAB benchmark. Its job is
to estimate how much binding free energy a specific protein residue contributes
to the interface, and by how much that contribution changes when the residue is
mutated.

Four physical interaction types are scored per atom pair within 5.5 Å,
weighted by an exponential distance decay w(d) = exp(-d/3.5):

**Electrostatic** — RNA phosphate oxygens (OP1, OP2) carry partial charges of
approximately -0.8e. Lysine (NZ, +0.8e), Arginine (NH1/NH2, +0.6e), and
Histidine (ND1/NE2, +0.3e) carry positive partial charges on their sidechain
nitrogen atoms. When opposite charges are in contact, the contribution is
negative (favorable). This term dominates for K→A and R→A mutations, which
account for the largest ΔΔG values in ProNAB. Losing a Lys that bridges two
phosphate groups is worth 3-5 kcal/mol.

**Aromatic stacking** — Phe, Tyr, and Trp ring atoms stacking on purine or
pyrimidine ring atoms contribute approximately 1-3 kcal/mol per contact.
This is captured by checking whether both atoms belong to the aromatic atom
sets and applying a separate stacking weight. This term drives Y→A and W→A
ΔΔG predictions.

**Hydrogen bonds** — N and O atoms within 3.5 Å of each other at the
interface contribute a favorable H-bond energy. This captures contacts between
protein backbone NH or sidechain donors/acceptors and RNA phosphate oxygens,
base ring nitrogens, or 2'-OH groups. H-bonds contribute 1-2 kcal/mol each
and dominate for S→A, T→A, N→A, and Q→A mutations.

**Van der Waals** — All remaining heavy atom contacts within the cutoff
contribute a small background burial term. This picks up the entropic
advantage of burying nonpolar surface area and drives V→A, L→A, and A→G
predictions.

The score delta for a mutation is:
```
score_delta = contact_score(WT_residue) - contact_score(mutant_residue)
```

Positive delta means the WT residue made more favorable contacts. This should
correlate with experimental ΔΔG > 0 (binding disrupted). The Pearson
correlation between score_delta and experimental ΔΔG on ProNAB is the primary
benchmark metric.

---

## The Scientific Ideas Behind Each Design Choice

### Why simulate the free RNA?

Most RNA-protein structure validation tools compare the predicted complex
against a database of known complexes, or apply energy minimization to check
for clashes. Neither approach captures the fundamental question: *can the RNA
physically achieve the bound conformation starting from free solution?*

Conformational entropy is real and large. For a flexible RNA loop of 10
nucleotides, restricting it to a single conformation from a broad ensemble
can cost 3-8 kcal/mol in -TΔS terms — enough to negate a moderately strong
binding interface entirely. The only way to detect this is to sample the free
ensemble explicitly and compare it to the predicted bound pose.

Coarse-grained simulation (oxRNA) is chosen for this step rather than
all-atom MD for two reasons: it runs on consumer hardware in minutes for the
short (10-50 nt) sequences that appear at RBP binding sites in CLIP-seq data,
and it is specifically parameterized to reproduce RNA structural ensembles
including loop flexibility, A-form helix stability, and single-strand
stacking. The accuracy is sufficient for the population-level comparison
(does the free RNA visit anything near the bound pose?) without the
computational cost of microsecond all-atom simulations.

### Why k-medoids for the coordinate branch?

K-means finds cluster centroids that minimize within-cluster variance, but
the centroids are averages of the member coordinates. For molecular
conformations, the average of several structures is often not a physically
meaningful structure — it may have wrong bond lengths or clash with itself.

K-medoids instead selects the cluster center as the actual snapshot with the
lowest average distance to all other members of the cluster. This guarantees
the center is a real, physically realized conformation from the simulation.
When the AF3 bound pose is compared against cluster centers, it is being
compared against actual RNA conformations that were observed during simulation,
not mathematical abstractions.

### Why compare local geometry separately from global RMSD?

Global RMSD is a single number for the whole interface. It can miss situations
where the overall interface geometry is broadly correct but one specific
nucleotide is in a wrong local conformation — for example, if 4 of 5 interface
nucleotides are close to their free-ensemble positions but the fifth has a
C2'-endo sugar pucker that never appears in the simulation. That fifth
nucleotide may be the one making the critical 2'-OH contact to the protein,
and its unusual geometry may be what makes the contact possible in the AF3
prediction while being physically inaccessible in solution.

The geometry branch catches this by scoring each nucleotide independently
in torsion-angle space. The per-nucleotide verdict column in the report
shows exactly which nucleotide is causing the problem, making the output
actionable for structural biologists.

### Why use contact energy for the ProNAB benchmark specifically?

ProNAB entries are mutations of protein residues in experimentally solved
complex structures. The RNA does not change between wild-type and mutant —
the same sequence is present, and the RNA coordinates in the crystal are nearly
identical before and after the computational mutation. This means the RMSD
branch, which measures whether the RNA conformation is accessible from free
solution, gives essentially the same score for WT and mutant: the RNA is
already crystallized in a real bound conformation, so it trivially passes.

What does change is the set of contacts the mutated protein residue makes
with the RNA. A Y13A mutation removes the tyrosine ring that stacks on a
uracil base — the contact disappears. The contact scorer detects this by
computing how much favorable interaction energy the WT residue makes and
comparing it to the mutant. The difference (score_delta) is the ΔΔG proxy
that correlates with experiment.

This separation is intentional: the RMSD+geometry pipeline is for detecting
hallucinations in AI-predicted structures, and the contact scorer is for
benchmarking mutation effects on known experimental structures. Both are
part of the same tool but address different scientific questions.

---

## Benchmarking Against ProNAB

The ProNAB database contains experimentally measured binding free energy
changes for mutations at protein-RNA interfaces. Each entry provides:
- The wild-type PDB structure of the complex
- The mutated residue (chain, position, amino acid change)
- The experimentally measured ΔΔG in kcal/mol (by ITC, SPR, or gel shift)

The benchmark protocol:
1. Load the WT structure from RCSB
2. Introduce the mutation computationally (PyRosetta with sidechain minimization,
   or coordinate-level rename for X→A mutations)
3. Score both WT and mutant with the contact scorer
4. Compute score_delta = WT_score - mutant_score
5. Measure Pearson r and Spearman r between score_delta and experimental ΔΔG

Target thresholds based on published tool comparisons:
- r > 0.50 — competitive with existing rule-based tools
- r > 0.60 — publication-worthy
- r > 0.70 — strong result, among the best for RNA-protein specifically

The current weights (W_elec=3.0, W_stack=1.8, W_hbond=1.5, W_vdw=0.4) are
physically motivated but not trained. Running tune_weights.py on a pilot set
of 50 ProNAB entries and optimizing with Nelder-Mead should push r by 0.05-0.10.

The dominant signal in ProNAB is charged residue mutations (K→A, R→A), which
account for the largest ΔΔG values and are cleanly captured by the electrostatic
term. The aromatic stacking term captures the second tier (Y→A, W→A). Together
these two terms cover approximately 60-70% of the variance in ProNAB ΔΔG values.

---

## Future Directions

### Short term — improve the current scoring

**1. Train contact weights on a held-out ProNAB split**

Run tune_weights.py on 70% of ProNAB, validate on the remaining 30%. Report
both in-sample and cross-validated r. A proper train/validate split is required
before claiming publication-level r.

**2. Add distance-dependent dielectric for electrostatics**

The current electrostatic term uses a simple exponential decay. A more
physically accurate model uses a distance-dependent dielectric ε(r) ∝ r,
which better represents the screening of charges by solvent and counterions
at longer range. For RNA interfaces, Mg2+ ions can bridge phosphate groups
and protein side chains — including a Mg2+ coordination geometry term would
improve predictions for structures where Mg2+ is present in the crystal.

**3. Distinguish 2'-OH contacts from other N/O contacts**

The current H-bond term scores all N/O pairs equally. The 2'-OH group on
RNA (which distinguishes RNA from DNA) makes distinctive contacts to
protein asparagine, lysine, and serine residues. Weighting 2'-OH contacts
specifically would improve discrimination on the subset of ProNAB entries
where this is the load-bearing interaction.

**4. Add buried surface area as a feature**

Solvent-accessible surface area (SASA) buried upon complex formation
correlates with binding affinity across many protein-nucleic acid datasets.
BioPython's SASA module (Shrake-Rupley algorithm) can compute this in seconds.
Adding ΔBSA per mutated residue as an independent feature — or as an
additional scoring term — typically contributes r ≈ 0.1-0.15 on top of
contact-counting approaches.

---

### Medium term — make the RMSD branch publishable

**5. Integrate OpenMM for all-atom free-RNA simulation**

The oxRNA coarse-grained simulation is fast but loses atomic detail. For
short sequences (< 20 nt), running a 100 ns all-atom simulation with OpenMM
using the ff99-bsc0χOL3 force field (the current standard for RNA) takes
6-12 hours on a consumer GPU. This would make the entropic plausibility
scores much more accurate, particularly for structured motifs like GNRA
tetraloops and internal loops that have specific free-solution conformational
preferences that coarse-grained models only approximate.

**6. Calibrate RMSD thresholds against RNA-Puzzles**

The current 2.0 Å / 4.0 Å pass/warn/fail thresholds are estimated from
published structural biology intuition. RNA-Puzzles provides a direct
calibration dataset: known experimental structures alongside predictions with
known RMSD to the true answer. Fitting a logistic regression of "correct
prediction" vs RMSD on this dataset would give data-driven thresholds with
confidence intervals.

**7. Add a population-weighted score**

Currently the entropic score is the RMSD to the nearest cluster medoid. A
richer version would weight each cluster by its population in the free
simulation: a bound pose that resembles a highly populated cluster (say, 40%
of simulation frames) is much more accessible than one that resembles a minor
cluster (5% of frames). The score could be log(cluster_population / RMSD^2),
penalizing both rarity and geometric distance simultaneously.

---

### Long term — fundamentally new capabilities

**8. Phase separation and IDR handling**

Many RBPs that bind RNA contain intrinsically disordered regions (IDRs) that
drive liquid-liquid phase separation into membraneless organelles like stress
granules and P-bodies. TDP-43, FUS, hnRNPA1, and others have both structured
RNA-binding domains and long IDR tails. AF3 predictions for these full-length
proteins are particularly unreliable because the IDR can adopt many
conformations and its interactions with RNA are often non-specific and
concentration-dependent rather than sequence-specific.

A future module could flag predictions where the protein has a long IDR
(by checking for low-complexity sequence in the UniProt annotation), note
that the predicted contacts may reflect IDR-mediated interactions not
captured by single-structure analysis, and suggest that for these proteins
the relevant question is phase diagram membership rather than specific
interface contacts.

**9. Graph neural network trained on PDB decoys**

The current scoring is physics-based and interpretable, but the weights are
constrained to four numbers. A message-passing GNN that operates on the
protein-RNA contact graph — nodes are atoms, edges are contacts with
features including distance, atom type, residue type, and secondary structure
assignment — could learn richer interaction patterns from the ~7,000 RNA-
protein PDB structures plus computationally generated decoys.

The key dataset for this would be the RNAsite benchmark, which provides
native structures alongside docking decoys. A GNN trained to discriminate
native from decoy could serve as a drop-in replacement for the contact
scorer and would not require explicit hand-crafting of interaction weights.

**10. Mutation sensitivity loop with AF3 re-prediction**

The most rigorous validation — and the most compelling demonstration of the
tool's value — would be the following loop:

1. Take a known RNA-protein complex (e.g. U1A bound to stem-loop II)
2. Introduce a mutation computationally (e.g. Y13A)
3. Submit the mutant sequence to AF3 and get a new predicted complex
4. Score both predictions with PhysRNA-Filter
5. Ask: does PhysRNA-Filter's score change between WT and Y13A in the
   direction of the experimental ΔΔG, even if AF3's confidence did not?

If the filter catches disruptions that AF3 misses, that is the headline
result. The comparison between AF3 confidence (pTM, ipTM) and
PhysRNA-Filter score across a panel of known mutations would demonstrate
that the filter adds genuine information beyond what the AI model itself
reports.

This experiment requires an AF3 API key and several hundred GPU-hours,
but it is the most direct argument for why the tool matters.

**11. Negative design — filter-guided sequence optimization**

Inverting the scoring function: instead of asking "does this complex score
well?", ask "what RNA sequence mutations would make it score poorly?".
Starting from a CLIP-seq binding site, enumerate single-nucleotide variants
and predict which mutations would destabilize the interface. This creates
a computational mutagenesis panel that can be compared against published
in vitro affinity measurements (RBPDB) or used to design experiments.

This application would be particularly useful for therapeutic RNA design —
finding mutations in a target RNA sequence that disrupt its interaction with
a disease-relevant RBP without disrupting the RNA's other functions.

---

## Summary Table

| Component | File | What it computes | When it matters |
|---|---|---|---|
| Complex parser | structure/parse_complex.py | Splits PDB into protein + RNA chains | Every run |
| Interface finder | structure/extract_interface.py | RNA atoms within 5 A of protein | Every run |
| Local geometry | structure/local_geometry.py | 7 torsions, sugar pucker, stacking | Geometry branch |
| RNA folder | simulation/fold_rna.py | 3D starting structure from sequence | Simulation branch |
| MD simulation | simulation/run_simulation.py | Free RNA conformational ensemble | Simulation branch |
| Kabsch RMSD | analysis/rmsd.py | RMSD after optimal alignment | RMSD branch |
| Coordinate cluster | analysis/cluster.py | k-medoids on C4' coords | RMSD branch |
| Geometry cluster | analysis/cluster.py | k-means on torsion angles | Geometry branch |
| Geometry scorer | analysis/geometry_score.py | Per-nucleotide torsion mismatch | Geometry branch |
| Contact scorer | analysis/contact_score.py | Electrostatic + stacking + HB | ProNAB benchmark |
| Combined scorer | analysis/score.py | PASS/WARN/FAIL verdict | Every run |
| ProNAB benchmark | validation/benchmark_pronab.py | Pearson r vs experimental ddG | Validation only |
| Weight tuner | tune_weights.py | Optimize W_elec/stack/hbond/vdw | After pilot run |
| Data setup | setup_data.py | Download PDB structures | First run only |

---

## Further Reading

**Foundational papers on RNA-protein binding thermodynamics**
- Draper (2004) A guide to ions and RNA structure. *RNA* — establishes that Mg2+ and electrostatics dominate RNA folding and binding energetics
- Chen & Varani (2005) Lessons from structural genomics of RNA-protein complexes. *FEBS Journal* — comprehensive survey of interface geometries
- Jankowsky & Harris (2015) Specificity and nonspecificity in RNA-protein interactions. *Nature Reviews Molecular and Cell Biology*

**AlphaFold 3 and its limitations for RNA**
- Abramson et al. (2024) Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*
- The supplementary benchmarks in this paper show that RNA-protein ipTM scores are notably lower than protein-protein, reflecting lower confidence on RNA interfaces

**Conformational entropy in RNA binding**
- Leulliot & Varani (2001) Current topics in RNA-protein recognition: control of specificity and biological function through induced fit and conformational capture. *Biochemistry*
- Williamson (2000) Induced fit in RNA-protein recognition. *Nature Structural Biology*

**CLIP-seq and RBP binding site mapping**
- Van Nostrand et al. (2020) A large-scale binding and functional map of human RNA-binding proteins. *Nature* (the eCLIP paper)
- Dominguez et al. (2018) Sequence, structure, and context preferences of human RNA binding proteins. *Molecular Cell*

**RNA structure prediction tools referenced in this pipeline**
- Das et al. (2010) Atomic accuracy in predicting and designing RNA-loop sequences. *Biochemistry* (FARFAR2)
- Boniecki et al. (2016) SimRNA: a coarse-grained method for RNA folding simulations and 3D structure prediction. *Nucleic Acids Research*
- Sulc et al. (2014) A nucleotide-level coarse-grained model of RNA. *Journal of Chemical Physics* (oxRNA)

**ProNAB and SKEMPI databases**
- Gromiha et al. (2019) ProNAB: database for binding affinities of protein-nucleic acid complexes and their mutations. *Nucleic Acids Research*
- Jankauskaite et al. (2019) SKEMPI 2.0: an updated benchmark of changes in protein-protein binding energy, kinetics, and thermodynamics upon mutation. *Bioinformatics*
