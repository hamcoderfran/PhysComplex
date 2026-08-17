# Evolutionary evidence for PhysComplex

AlphaFold-style evolutionary signal is powerful because an MSA summarizes many
independent evolutionary constraints. It is not a hidden oracle for arbitrary
complexes. In particular, **inter-chain coevolution requires correctly paired
orthologs**; combining unrelated protein and RNA alignments by row order creates
false correlations.

## Implemented diagnostic signals

`physcomplex evolutionary-check` accepts user-supplied alignments and returns:

1. protein interface conservation and MSA coverage;
2. RNA interface conservation and MSA coverage;
3. paired protein–RNA mutual information (MI) and average-product-corrected
   MI (APC), only when an explicit pairing-provenance statement is supplied.

It accepts:

```bash
physcomplex evolutionary-check \
  --protein-msa protein.a3m \
  --rna-msa rna.a3m \
  --paired-alignment ortholog_pairs.tsv \
  --protein-positions 10,14,31 \
  --rna-positions 3,7,8 \
  --pairing-provenance "same-genome ortholog pairs; accession mapping in manifest"
```

The paired file has two tab-separated, rectangular aligned columns per row:

```text
PROTEIN_ALIGNMENT<TAB>RNA_ALIGNMENT
```

No score produced by this command changes `run_pipeline()` or PhysGT verdicts.
All evidence is marked diagnostic until calibrated against frozen,
structure/family-disjoint data.

## Mandatory interpretation guards

* Conservation supports functional constraint, not a particular binding partner.
* Absence of coevolution is inconclusive when paired depth, taxonomic diversity,
  or pairing confidence is low.
* Raw MI is confounded by ancestry, gaps, composition, indirect correlations,
  and bad pairing. The current MI/APC output is a diagnostic baseline; a
  trainable feature requires sequence reweighting, phylogeny-aware shuffled
  controls, and preferably direct-coupling analysis.
* Rfam/Infernal covariance supports RNA family/secondary-structure compatibility,
  not a protein docking pose. R-scape non-significance must be interpreted with
  its power.
* Genome-neighborhood and phylogenetic-profile evidence is valuable for
  conserved prokaryotic modules but does not transfer directly to eukaryotic or
  transient RBP interactions.

## Why this could add value

* **Conserved interface residues/nucleotides:** a prediction that buries highly
  conserved positions in incompatible geometry is suspicious; a candidate using
  co-conserved surfaces is biologically more plausible.
* **Direct paired coevolution:** high inter-chain MI/APC at predicted contacting
  positions can support an interface when paired homolog data genuinely exists.
* **RNA covariance:** Rfam/Infernal covariance models can test whether a
  predicted RNA preserves family sequence-and-secondary-structure constraints,
  rather than relying only on a single-sequence RNA language embedding.

Protein–RNA paired data are most realistic for conserved microbial assemblies
(for example, ribosomal proteins plus rRNA). They are often unavailable for
eukaryotic RBP targets, long noncoding RNAs, or condition-specific interactions;
missing paired evidence must increase uncertainty, not become a negative label.

## Literature grounding

* Protein–RNA residue-triplet MI found direct coevolutionary evidence in the
  L22–23S ribosomal interface: [Wang et al. (2012)](https://doi.org/10.1371/journal.pone.0030022).
* Paired evolutionary information can identify protein–protein interface
  contacts when alignments are sufficiently deep:
  [Ovchinnikov et al. (2014)](https://elifesciences.org/articles/02030).
* Rfam covariance models combine RNA sequence and conserved secondary
  structure; Infernal can search/align against these models:
  [Nawrocki and Eddy (2013)](https://doi.org/10.1093/bioinformatics/btt509).

## Next research advancement

The high-risk/high-reward advance is not simply adding MI to a neural network.
It is a **phylogenetically validated interface-evidence model**:

1. obtain trusted paired protein/RNA ortholog tables, including genomic context
   or experimentally justified association;
2. infer protein and RNA MSAs independently, then pair only documented
   orthologs;
3. compute MI/APC and, where depth permits, direct-coupling features;
4. evaluate whether they improve top-k native ranking over AF3/Boltz confidence,
   static physics, and PhysGT on a frozen family-held-out benchmark;
5. add a new trained head only if the improvement survives ablation and
   risk/coverage calibration.

Do not tune pair selection, evolutionary thresholds, or interface positions on
the same evaluation panel used for the final result.
