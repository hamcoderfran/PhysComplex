"""
Builds protein-RNA interface graphs for the Physics-Informed Graph Transformer.

Graph structure
---------------
Nodes: every protein residue within INTERFACE_CUTOFF of any RNA atom,
       every RNA nucleotide within INTERFACE_CUTOFF of any protein atom,
       PLUS the mutation-site residue (always included even if distal).

Node features:
  protein: [one-hot AA (20d), volume (1d), charge (1d),
            hydrophobicity (1d), is_mutation_site (1d), ESM-2 embedding]
  RNA:     [one-hot nucleotide (8d), RNA-FM embedding or fallback]

Edge features (all pairs within EDGE_CUTOFF, bidirectional):
  [dist_norm, dist_weight, elec, stack, hbond, vdw, edge_types (3),
   pi_cation, dir_hbond, salt_bridge, RBF×3]  — EDGE_DIM=15

The mutation is encoded at the graph level — the mutation site node has
is_mutation_site=1, and in the Siamese GT the difference between WT and mutant
embeddings at that node carries the ΔΔG signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .gt_constants import EDGE_DIM
from .physics_edge import (
    EDGE_CUTOFF as _PHYS_EDGE_CUTOFF,
    aggregate_prot_rna_pair,
    backbone_edge_vector,
    build_edge_feature_vector,
)

INTERFACE_CUTOFF = 12.0   # Å: nodes this close to partner chain are included
EDGE_CUTOFF      = _PHYS_EDGE_CUTOFF
BACKBONE_PROT    = 6.0    # Å: protein Cα–Cα for sequential backbone edge
BACKBONE_RNA     = 8.0    # Å: RNA C4'–C4' for sequential backbone edge

AA_VOCAB  = list("ACDEFGHIKLMNPQRSTVWY")
_AA_IDX   = {aa: i for i, aa in enumerate(AA_VOCAB)}

NUC_VOCAB = ["A", "G", "C", "U", "DA", "DG", "DC", "DT"]
_NUC_IDX  = {n: i for i, n in enumerate(NUC_VOCAB)}
_RNA_RESNAME_TO_ONE = {
    "A": "A", "ADE": "A", "U": "U", "URA": "U", "G": "G", "GUA": "G", "C": "C", "CYT": "C",
}


def _rna_one_letter(resname: str) -> str:
    return _RNA_RESNAME_TO_ONE.get(resname.strip().upper(), resname.strip().upper()[:1])

_AA_VOLUME: dict[str, float] = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "E": 138.4, "Q": 143.8, "G": 60.0,  "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0,  "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}
_AA_CHARGE: dict[str, float] = {
    "K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0,
}
_AA_KD: dict[str, float] = {
    "I": 4.5,  "V": 4.2,  "L": 3.8,  "F": 2.8,  "C": 2.5,
    "M": 1.9,  "A": 1.8,  "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HSD": "H", "HSE": "H", "HSP": "H", "HIP": "H", "HIE": "H",
    "MSE": "M", "SEC": "C",
    "MLZ": "K", "M3L": "K", "LLY": "K",
}


def _is_protein_residue(residue) -> bool:
    """Include standard and MODRES protein residues (e.g. MLZ → lysine)."""
    if residue.id[0] == " ":
        return True
    return residue.resname.strip().upper() in _THREE_TO_ONE


@dataclass
class InterfaceGraph:
    """All tensors needed for one forward pass through the graph transformer."""
    x_protein:          torch.Tensor   # [n_prot, protein_node_dim]
    x_rna:              torch.Tensor   # [n_rna,  rna_node_dim]
    edge_index:         torch.Tensor   # [2, n_edges]
    edge_attr:          torch.Tensor   # [n_edges, EDGE_DIM]
    node_types:         torch.Tensor   # [n_nodes] 0=protein 1=RNA
    mutation_node_idx:  int            # global node index (protein idx or n_prot + rna idx)
    protein_residues:   list = field(default_factory=list)   # (chain, resnum, aa1)
    rna_residues:       list = field(default_factory=list)   # (chain, resnum, resname)
    prot_coords:        torch.Tensor | None = None  # [n_prot, 3] Cα
    rna_coords:         torch.Tensor | None = None  # [n_rna, 3] C4'


def build_interface_graph(
    pdb_path:          str,
    chain_id:          str,
    position:          int,
    pdb_id:            str = "unknown",
    esm_embeddings:    dict | None = None,    # (chain_id, resnum) → Tensor
    rnafm_embeddings:  dict | None = None,    # (chain_id, resnum) → Tensor
    esm_dim:           int = 320,
    rnafm_dim:         int = 640,
    edge_dim:          int = EDGE_DIM,
    *,
    mutation_on:       str = "protein",
    partner_chain_id:  str | None = None,
) -> InterfaceGraph:
    """
    Build an InterfaceGraph from a PDB file.

    Args:
        pdb_path:         path to structure (WT or mutant)
        chain_id:         chain carrying the mutation (protein or RNA)
        position:         residue number of mutation site
        pdb_id:           for embedding cache lookup
        esm_embeddings:   pre-computed ESM-2 embeddings (or None → zeros)
        rnafm_embeddings: pre-computed RNA-FM embeddings (or None → zeros)
        esm_dim:          ESM-2 embedding dimension (must match what's cached)
        rnafm_dim:        RNA-FM embedding dimension (or fallback dim)
        mutation_on:      ``protein`` (default) or ``rna``
        partner_chain_id: optional partner chain when ``mutation_on='rna'``
    """
    from ..structure.parse_complex import RNA_RESIDUE_NAMES, parse_complex
    from ..structure.partner_selection import select_partner_pair, select_partner_rna_chains

    parsed     = parse_complex(pdb_path)
    if mutation_on == "rna":
        protein_chains, rna_chains = select_partner_pair(
            parsed.protein_chains,
            parsed.rna_chains,
            protein_chain_id=partner_chain_id,
            rna_chain_id=chain_id,
        )
        protein_chain_id = protein_chains[0].id if protein_chains else partner_chain_id
    else:
        protein_chains = [ch for ch in parsed.protein_chains if ch.id == chain_id]
        if not protein_chains:
            raise ValueError(
                f"Protein mutation chain {chain_id} not found in {pdb_path}"
            )
        protein_chain_id = chain_id
        rna_chains = select_partner_rna_chains(
            parsed.protein_chains, parsed.rna_chains, chain_id
        )

    rna_atoms_all = [
        a for ch in rna_chains
        for res in ch
        if res.id[0] == " " and res.resname.strip().upper() in RNA_RESIDUE_NAMES
        for a in res
        if a.element != "H"
    ]
    prot_atoms_all = [
        a for ch in protein_chains
        for res in ch
        if _is_protein_residue(res)
        for a in res
        if a.element != "H"
    ]

    # ── collect interface protein residues ──────────────────────────────────
    prot_residue_info = []
    protein_mutation_node_idx: int | None = None

    for chain in protein_chains:
        for residue in chain:
            if not _is_protein_residue(residue):
                continue
            ca = next((a for a in residue if a.name == "CA"), None)
            if ca is None:
                continue
            is_site = (
                mutation_on == "protein"
                and chain.id == chain_id
                and residue.id[1] == position
            )
            if not is_site and rna_atoms_all:
                min_d = min(
                    np.linalg.norm(ca.coord - ra.coord) for ra in rna_atoms_all
                )
                if min_d > INTERFACE_CUTOFF:
                    continue
            aa1 = _THREE_TO_ONE.get(residue.resname, "X")
            prot_residue_info.append({
                "chain":   chain.id,
                "resnum":  residue.id[1],
                "icode":   residue.id[2],
                "resname": residue.resname,
                "aa1":     aa1,
                "residue": residue,
                "ca":      ca.coord.copy(),
                "is_site": is_site,
            })

    if mutation_on == "protein":
        for idx, r in enumerate(prot_residue_info):
            if r["is_site"]:
                protein_mutation_node_idx = idx
                break
        if protein_mutation_node_idx is None:
            raise ValueError(
                f"Mutation site {chain_id}:{position} not found in {pdb_path}"
            )

    # ── collect interface RNA residues ──────────────────────────────────────
    rna_residue_info = []
    rna_mutation_node_idx: int | None = None
    for chain in rna_chains:
        for residue in chain:
            if (
                residue.id[0] != " "
                or residue.resname.strip().upper() not in RNA_RESIDUE_NAMES
            ):
                continue
            rep = next(
                (a for a in residue if a.name in ("C4'", "C1'", "P")), None
            )
            if rep is None:
                continue
            is_site = (
                mutation_on == "rna"
                and chain.id == chain_id
                and residue.id[1] == position
            )
            if not is_site and prot_atoms_all:
                min_d = min(
                    np.linalg.norm(rep.coord - pa.coord) for pa in prot_atoms_all
                )
                if min_d > INTERFACE_CUTOFF:
                    continue
            rna_residue_info.append({
                "chain":   chain.id,
                "resnum":  residue.id[1],
                "icode":   residue.id[2],
                "resname": residue.resname.strip(),
                "residue": residue,
                "c4":      rep.coord.copy(),
                "is_site": is_site,
            })

    if mutation_on == "rna":
        for idx, r in enumerate(rna_residue_info):
            if r["is_site"]:
                rna_mutation_node_idx = idx
                break
        if rna_mutation_node_idx is None:
            raise ValueError(
                f"RNA mutation site {chain_id}:{position} not found in {pdb_path}"
            )

    n_prot = max(len(prot_residue_info), 1)
    n_rna  = max(len(rna_residue_info),  1)
    if mutation_on == "rna":
        mutation_node_idx = n_prot + rna_mutation_node_idx
    else:
        mutation_node_idx = protein_mutation_node_idx

    # ── build protein node feature matrix ──────────────────────────────────
    prot_node_dim = 20 + 3 + 1 + esm_dim   # one_hot + physics + is_site + esm2
    x_protein_rows = []
    for r in (prot_residue_info or [{"chain": chain_id, "resnum": position,
                                     "aa1": "X", "is_site": True}]):
        aa1 = r["aa1"]
        one_hot = torch.zeros(20)
        idx = _AA_IDX.get(aa1, -1)
        if idx >= 0:
            one_hot[idx] = 1.0

        physics = torch.tensor([
            _AA_VOLUME.get(aa1, 120.0) / 227.8,    # normalise to max (TRP)
            _AA_CHARGE.get(aa1, 0.0),
            _AA_KD.get(aa1, 0.0) / 4.5,
        ])
        is_site = torch.tensor([1.0 if r.get("is_site") else 0.0])

        if esm_embeddings:
            esm_feat = esm_embeddings.get(
                (r["chain"], r["resnum"]), torch.zeros(esm_dim)
            )
        else:
            esm_feat = torch.zeros(esm_dim)

        x_protein_rows.append(torch.cat([one_hot, physics, is_site, esm_feat]))

    x_protein = torch.stack(x_protein_rows)

    # ── build RNA node feature matrix ──────────────────────────────────────
    rna_node_dim = len(NUC_VOCAB) + rnafm_dim
    x_rna_rows = []
    for r in (rna_residue_info or [{"chain": "X", "resnum": 1, "resname": "A"}]):
        one_hot = torch.zeros(len(NUC_VOCAB))
        idx = _NUC_IDX.get(_rna_one_letter(r["resname"]), -1)
        if idx >= 0:
            one_hot[idx] = 1.0

        if rnafm_embeddings:
            rf_feat = rnafm_embeddings.get(
                (r["chain"], r["resnum"]), torch.zeros(rnafm_dim)
            )
        else:
            rf_feat = torch.zeros(rnafm_dim)

        if rf_feat.numel() != rnafm_dim:
            padded = torch.zeros(rnafm_dim)
            n = min(rnafm_dim, rf_feat.numel())
            padded[:n] = rf_feat.flatten()[:n]
            rf_feat = padded

        x_rna_rows.append(torch.cat([one_hot, rf_feat]))

    x_rna = torch.stack(x_rna_rows)

    # ── build edges ─────────────────────────────────────────────────────────
    edge_indices: list[list[int]] = []
    edge_attrs:   list[torch.Tensor] = []

    def _add_edge(i: int, j: int, feat: torch.Tensor):
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(feat)
        edge_attrs.append(feat)

    # protein↔RNA contact edges
    for pi, pres in enumerate(prot_residue_info):
        p_atoms = [a for a in pres["residue"] if a.element != "H"]
        for ri, rres in enumerate(rna_residue_info):
            if np.linalg.norm(pres["ca"] - rres["c4"]) > EDGE_CUTOFF + 5.0:
                continue

            r_atoms = [a for a in rres["residue"] if a.element != "H"]
            min_d, elec, stack, hbond, vdw, pi_c, dir_h, salt = aggregate_prot_rna_pair(
                p_atoms, r_atoms, pres, rres
            )
            if min_d > EDGE_CUTOFF:
                continue

            feat = build_edge_feature_vector(
                min_d, elec, stack, hbond, vdw, pi_c, dir_h, salt,
                prot_rna=True,
                edge_dim=edge_dim,
            )
            _add_edge(pi, n_prot + ri, feat)

    # protein backbone (sequential Cα–Cα)
    for pi in range(len(prot_residue_info) - 1):
        curr = prot_residue_info[pi]
        nxt = prot_residue_info[pi + 1]
        if curr["chain"] != nxt["chain"] or nxt["resnum"] != curr["resnum"] + 1:
            continue
        d = float(np.linalg.norm(
            curr["ca"] - nxt["ca"]
        ))
        if d < BACKBONE_PROT:
            feat = backbone_edge_vector(d, prot_prot=True, edge_dim=edge_dim)
            _add_edge(pi, pi + 1, feat)

    # RNA backbone (sequential C4'–C4')
    for ri in range(len(rna_residue_info) - 1):
        curr = rna_residue_info[ri]
        nxt = rna_residue_info[ri + 1]
        if curr["chain"] != nxt["chain"] or nxt["resnum"] != curr["resnum"] + 1:
            continue
        d = float(np.linalg.norm(
            curr["c4"] - nxt["c4"]
        ))
        if d < BACKBONE_RNA:
            feat = backbone_edge_vector(d, rna_rna=True, edge_dim=edge_dim)
            _add_edge(n_prot + ri, n_prot + ri + 1, feat)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr  = torch.stack(edge_attrs)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, edge_dim), dtype=torch.float32)

    node_types = torch.cat([
        torch.zeros(n_prot, dtype=torch.long),
        torch.ones(n_rna, dtype=torch.long),
    ])

    prot_coords = torch.tensor(
        [r["ca"] for r in prot_residue_info], dtype=torch.float32
    ) if prot_residue_info else None
    rna_coords = torch.tensor(
        [r["c4"] for r in rna_residue_info], dtype=torch.float32
    ) if rna_residue_info else None

    return InterfaceGraph(
        x_protein=x_protein,
        x_rna=x_rna,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_types=node_types,
        mutation_node_idx=mutation_node_idx,
        protein_residues=[
            (r["chain"], r["resnum"], r["aa1"]) for r in prot_residue_info
        ],
        rna_residues=[
            (r["chain"], r["resnum"], r["resname"]) for r in rna_residue_info
        ],
        prot_coords=prot_coords,
        rna_coords=rna_coords,
    )


def build_af3_interface_graph(
    pdb_path:          str,
    pdb_id:            str = "unknown",
    interface_cutoff:  float = INTERFACE_CUTOFF,
    esm_embeddings:    dict | None = None,
    rnafm_embeddings:  dict | None = None,
    esm_dim:           int = 320,
    rnafm_dim:         int = 640,
    edge_dim:          int = EDGE_DIM,
    *,
    model_rank:        int = 0,
    parsed:            object | None = None,
    protein_chains:    list | None = None,
    rna_chains:        list | None = None,
) -> InterfaceGraph:
    """
    Build an InterfaceGraph from an AF3/RoseTTAFold complex for plausibility
    scoring (no mutation site required).

    Uses the same node/edge featurisation as the ProNAB training graphs but
    includes all interface residues within `interface_cutoff` of the partner
    chain.  mutation_node_idx is set to 0 (unused by interface scoring).
    """
    from ..structure.parse_complex import RNA_RESIDUE_NAMES, parse_complex
    from ..structure.partner_selection import select_partner_pair

    if parsed is None:
        parsed = parse_complex(pdb_path, model_rank=model_rank)
    if protein_chains is None or rna_chains is None:
        protein_chains, rna_chains = select_partner_pair(
            parsed.protein_chains, parsed.rna_chains
        )

    rna_atoms_all = [
        a for ch in rna_chains
        for res in ch
        if res.id[0] == " " and res.resname.strip().upper() in RNA_RESIDUE_NAMES
        for a in res
        if a.element != "H"
    ]
    prot_atoms_all = [
        a for ch in protein_chains
        for res in ch
        if _is_protein_residue(res)
        for a in res
        if a.element != "H"
    ]

    prot_residue_info = []
    for chain in protein_chains:
        for residue in chain:
            if not _is_protein_residue(residue):
                continue
            ca = next((a for a in residue if a.name == "CA"), None)
            if ca is None:
                continue
            if rna_atoms_all:
                min_d = min(
                    np.linalg.norm(ca.coord - ra.coord) for ra in rna_atoms_all
                )
                if min_d > interface_cutoff:
                    continue
            aa1 = _THREE_TO_ONE.get(residue.resname, "X")
            prot_residue_info.append({
                "chain": chain.id, "resnum": residue.id[1],
                "resname": residue.resname, "aa1": aa1,
                "residue": residue, "ca": ca.coord.copy(), "is_site": False,
            })

    rna_residue_info = []
    for chain in rna_chains:
        for residue in chain:
            if (
                residue.id[0] != " "
                or residue.resname.strip().upper() not in RNA_RESIDUE_NAMES
            ):
                continue
            rep = next(
                (a for a in residue if a.name in ("C4'", "C1'", "P")), None
            )
            if rep is None:
                continue
            if prot_atoms_all:
                min_d = min(
                    np.linalg.norm(rep.coord - pa.coord) for pa in prot_atoms_all
                )
                if min_d > interface_cutoff:
                    continue
            rna_residue_info.append({
                "chain": chain.id, "resnum": residue.id[1],
                "resname": residue.resname.strip(),
                "residue": residue, "c4": rep.coord.copy(),
            })

    if not prot_residue_info and not rna_residue_info:
        raise ValueError(f"No interface residues found in {pdb_path}")

    n_prot = max(len(prot_residue_info), 1)
    n_rna  = max(len(rna_residue_info), 1)

    prot_node_dim = 20 + 3 + 1 + esm_dim
    x_protein_rows = []
    for r in (prot_residue_info or [{"chain": "X", "resnum": 1, "aa1": "X", "is_site": False}]):
        aa1 = r["aa1"]
        one_hot = torch.zeros(20)
        idx = _AA_IDX.get(aa1, -1)
        if idx >= 0:
            one_hot[idx] = 1.0
        physics = torch.tensor([
            _AA_VOLUME.get(aa1, 120.0) / 227.8,
            _AA_CHARGE.get(aa1, 0.0),
            _AA_KD.get(aa1, 0.0) / 4.5,
        ])
        is_site = torch.tensor([0.0])
        if esm_embeddings:
            esm_feat = esm_embeddings.get(
                (r["chain"], r["resnum"]), torch.zeros(esm_dim)
            )
        else:
            esm_feat = torch.zeros(esm_dim)
        x_protein_rows.append(torch.cat([one_hot, physics, is_site, esm_feat]))
    x_protein = torch.stack(x_protein_rows)

    x_rna_rows = []
    for r in (rna_residue_info or [{"chain": "X", "resnum": 1, "resname": "A"}]):
        one_hot = torch.zeros(len(NUC_VOCAB))
        idx = _NUC_IDX.get(_rna_one_letter(r["resname"]), -1)
        if idx >= 0:
            one_hot[idx] = 1.0
        if rnafm_embeddings:
            rf_feat = rnafm_embeddings.get(
                (r["chain"], r["resnum"]), torch.zeros(rnafm_dim)
            )
        else:
            rf_feat = torch.zeros(rnafm_dim)
        if rf_feat.numel() != rnafm_dim:
            padded = torch.zeros(rnafm_dim)
            n = min(rnafm_dim, rf_feat.numel())
            padded[:n] = rf_feat.flatten()[:n]
            rf_feat = padded
        x_rna_rows.append(torch.cat([one_hot, rf_feat]))
    x_rna = torch.stack(x_rna_rows)

    edge_indices: list[list[int]] = []
    edge_attrs:   list[torch.Tensor] = []

    def _add_edge(i: int, j: int, feat: torch.Tensor):
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(feat)
        edge_attrs.append(feat)

    for pi, pres in enumerate(prot_residue_info):
        p_atoms = [a for a in pres["residue"] if a.element != "H"]
        for ri, rres in enumerate(rna_residue_info):
            if np.linalg.norm(pres["ca"] - rres["c4"]) > EDGE_CUTOFF + 5.0:
                continue
            r_atoms = [a for a in rres["residue"] if a.element != "H"]
            min_d, elec, stack, hbond, vdw, pi_c, dir_h, salt = aggregate_prot_rna_pair(
                p_atoms, r_atoms, pres, rres
            )
            if min_d > EDGE_CUTOFF:
                continue
            feat = build_edge_feature_vector(
                min_d, elec, stack, hbond, vdw, pi_c, dir_h, salt,
                prot_rna=True,
                edge_dim=edge_dim,
            )
            _add_edge(pi, n_prot + ri, feat)

    for pi in range(len(prot_residue_info) - 1):
        curr, nxt = prot_residue_info[pi], prot_residue_info[pi + 1]
        if curr["chain"] != nxt["chain"] or nxt["resnum"] != curr["resnum"] + 1:
            continue
        d = float(np.linalg.norm(curr["ca"] - nxt["ca"]))
        if d < BACKBONE_PROT:
            feat = backbone_edge_vector(d, prot_prot=True, edge_dim=edge_dim)
            _add_edge(pi, pi + 1, feat)

    for ri in range(len(rna_residue_info) - 1):
        curr, nxt = rna_residue_info[ri], rna_residue_info[ri + 1]
        if curr["chain"] != nxt["chain"] or nxt["resnum"] != curr["resnum"] + 1:
            continue
        d = float(np.linalg.norm(curr["c4"] - nxt["c4"]))
        if d < BACKBONE_RNA:
            feat = backbone_edge_vector(d, rna_rna=True, edge_dim=edge_dim)
            _add_edge(n_prot + ri, n_prot + ri + 1, feat)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr  = torch.stack(edge_attrs)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, edge_dim), dtype=torch.float32)

    node_types = torch.cat([
        torch.zeros(n_prot, dtype=torch.long),
        torch.ones(n_rna, dtype=torch.long),
    ])

    prot_coords = (
        torch.from_numpy(np.stack([r["ca"] for r in prot_residue_info], axis=0).astype(np.float32))
        if prot_residue_info else None
    )
    rna_coords = (
        torch.from_numpy(np.stack([r["c4"] for r in rna_residue_info], axis=0).astype(np.float32))
        if rna_residue_info else None
    )

    return InterfaceGraph(
        x_protein=x_protein,
        x_rna=x_rna,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_types=node_types,
        mutation_node_idx=0,
        protein_residues=[
            (r["chain"], r["resnum"], r["aa1"]) for r in prot_residue_info
        ],
        rna_residues=[
            (r["chain"], r["resnum"], r["resname"]) for r in rna_residue_info
        ],
        prot_coords=prot_coords,
        rna_coords=rna_coords,
    )
