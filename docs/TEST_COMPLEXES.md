# AF3 test complexes (holdout panel)

Ten protein–RNA systems **not in the ProNAB training set** (`in_pronab: false`).
Use them to validate AF3 predictions and PhysRNA screening on unseen complexes.

Machine-readable list: `physrna_filter/data/af3_holdout_complexes.json`

Submit each pair to the [AlphaFold 3 Server](https://alphafoldserver.com) with the protein
and RNA sequences below. Save the output zip (e.g. `fold_6sqn_u1a_hairpin.zip`) and screen
with PhysRNA.

---

## 1. 6SQN — U1A + hairpin

| | Sequence |
|---|----------|
| **Protein** | `AVPETRPNHTIYINNLNEKIKKDELKKSLHAIFSRFGQILDILVSRSLKMRGQAWVIFKEVSSATNALRSMQGFPFYDKPMRIQYAKTDSDIIAKMK` |
| **RNA** | `AAUCCAUUGCACUCCGGAUUU` |

Training has **1URN** (mutant structures) but not **6SQN** (native hairpin).

## 2. 1U1Y — MS2 + F5 aptamer

| | Sequence |
|---|----------|
| **Protein** | `ASNFTQFVLVDNGGTGDVTVAPSNFANGVAEWISSNSRSQAYKVTCSVRQSSAQNRKYTIKVEVPKVATQTVGGVELPVAAWRSYLNMELTIPIFATNSDCELIVKAMQGLLKDGNPIPSAIAANSGIY` |
| **RNA** | `CCGGGGGAUCACCACGG` |

## 3. 3TRZ — Lin28 + pre-let-7

| | Sequence |
|---|----------|
| **Protein** | `AADEPQLLHGAGICKWFNVRMGFGFLSMTARAGVALDPPVDVFVHQSKLHMEGFRSLKEGEAVEFTFKKSAKGLESIRVTGPGGVFCIGSERRPKGGDRCYNCGGLDHHAKECKLPPQPKKCHFCQSINHMVASCPLKAQQGPSSQGK` |
| **RNA** | `GGCAGGGAUUUUGCCCGGAG` |

## 4. 1EC6 — Nova-2 + hairpin

| | Sequence |
|---|----------|
| **Protein** | `MKELVEIAVPENLVGAILGKGGKTLVEYQELTGARIQISKKGEFLPGTRNRRVTITGSPAATQAAQYLISQRVTYEQGVRASNPQKV` |
| **RNA** | `GAGGACCUAGAUCACCCCUC` |

## 5. 4PMI — HIV Rev + RRE

| | Sequence |
|---|----------|
| **Protein** | `AGRSGDSDEDSLKAVRLIKFLYQSNPPPNPEGTRQARRNRRRRWRARQRQIHSISERIRSTYLGRSAEP` |
| **RNA** | `GGGAGUAUAUGGGCGCACUUCGGUGACGGUACAGGCUCCU` |

## 6. 2N82 — RBFOX1 + element

| | Sequence |
|---|----------|
| **Protein** | `MNTENKSQPKRLHVSNIPFRFRDPDLRQMFGQFGKILDVEIIFNERGSKGFGFVTFENSADADRAREKLHGTVVEGRKIEVNNATARVMTNKKTVNPYTNG` |
| **RNA** | `GGUAGUUUUGGCAUGACUCUACC` |

## 7. 1M8X — Pumilio + UGUA

| | Sequence |
|---|----------|
| **Protein** | `GRSRLLEDFRNNRYPNLQLREIAGHIMEFSQDQHGSRFIQLKLERATPAERQLVFNEILQAAYQLMVDVFGNYVIQKFFEFGSLEQKLALAERIRGHVLSLALQMYGCRVIQKALEFIPSDQQNEMVRELDGHVLKCVKDQNGNHVVQKCIECVQPQSLQFIIDAFKGQVFALSTHPYGCRVIQRILEHCLPDQTLPILEELHQHTEQLVQDQYGNYVIQHVLEHGRPEDKSKIVAEIRGNVLVLSQHKFASNVVEKCVTHASRTERAVLIDEVCTMNDGPHSALYTMMKDQYANYVVQKMIDVAEPGQRKIVMHKIRPHIATLRKYTYGKHILAKLEKYYMKNGVDLG` |
| **RNA** | `UUGUAUAU` |

## 8. 4YOE — hnRNP A1 UP1 + AGU

| | Sequence |
|---|----------|
| **Protein** | `MGMSKSESPKEPEQLRKLFIGGLSFETTDESLRSHFEQWGTLTDCVVMRDPNTKRSRGFGFVTYATVEEVDAAMNARPHKVDGRVVEPKRAVSREDSQRPGAHLTVKKIFVGGIKEDTEEHHLRDYFEQYGKIEVIEIMTDRGSGKKRGFAFVTFDDHDSVDKIVIQKYHTVNGHNCEVRKALSKQEMASASSSQRGR` |
| **RNA** | `AGU` |

## 9. 1CVJ — PABP + poly(A)

| | Sequence |
|---|----------|
| **Protein** | `MNPSAPSYPMASLYVGDLHPDVTEAMLYEKFSPAGPILSIRVCRDMITRRSLGYAYVNFQQPADAERALDTMNFDVIKGKPVRIMWSQRDPSLRKSGVGNIFIKNLDKSIDNKALYDTFSAFGNILSCKVVCDENGSKGYGFVHFETQEAAERAIEKMNGMLLNDRKVFVGRFKSRKEREAELGARAKEF` |
| **RNA** | `AAAAAAAAAAA` |

## 10. 1FXL — HuD + ARE

| | Sequence |
|---|----------|
| **Protein** | `SKTNLIVNYLPQNMTQEEFRSLFGSIGEIESCKLVRDKITGQSLGYGFVNYIDPKDAEKAINTLNGLRLQTKTIKVSYARPSSASIRDANLYVSGLPKTMTQKELEQLFSQYGRIITSRILVDQVTGVSRGVGFIRFDKRIEAEEAIKGLNGQKPSGATEPITVKFA` |
| **RNA** | `UUUUAUUUU` |

---

## Screening commands

After downloading AF3 Server zips:

```bash
# Single structure (PDB, mmCIF, or AF3 zip)
python -m physrna_filter.pipeline fold_6sqn_u1a_hairpin.zip --require-gt-checkpoint

# Batch folder
python -m physrna_filter.validation.screen_af3 ./af3_predictions/ --require-gt-checkpoint --output af3_screen.csv
```

### Fair comparison vs crystal (U1A example)

AF3 outputs often contain one protein + one RNA chain. Crystal PDBs may have multiple
chains — extract the matching pair before comparing:

```bash
python -m physrna_filter.data.extract_chain_pair physrna_filter/data/structures/1urn.pdb A P -o 1urn_single.pdb
python -m physrna_filter.validation.screen_af3 fold_6sqn_u1a_hairpin.zip 1urn_single.pdb --require-gt-checkpoint
```

**Interpretation:** Trust **PhysGT + contact** branches for interface plausibility.
RMSD/geometry may **FAIL** without oxDNA installed (internal CG fallback is not discriminative).
See `docs/WALKTHROUGH.md` for branch details.

For the **P1–P5 / N1–N5 evaluation panel** (partner discrimination), see
`docs/AF3_EVAL_PANEL.md` and `physrna_filter/data/af3_eval_panel.json`.
