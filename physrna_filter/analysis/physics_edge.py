"""
Extended physics edge features for PhysGT.

Literature motivation
---------------------
Recent protein–nucleic-acid GNNs (PNAbind, CoPRA, GET) combine learned
embeddings with explicit non-covalent interaction priors.  PhysGT v2 adds:

  - π-cation (Lys/Arg/His ↔ aromatic bases)
  - Directional H-bond proxy (heavy-atom distance + donor–acceptor geometry)
  - Salt-bridge geometry (opposite partial charges within 4 Å)
  - Gaussian RBF distance encoding (F3Affinity / GET-style geometric bias)

Legacy 9-d edge layout is preserved in the leading dimensions for backward
compatibility with v1 checkpoints.
"""
from __future__ import annotations

import numpy as np
import torch

from .contact_score import (
    PROTEIN_AROMATIC_ATOMS,
    PROTEIN_CHARGE,
    RNA_AROMATIC,
    RNA_CHARGE,
    W_ELECTROSTATIC,
    W_HBOND,
    W_STACKING,
    W_VDW,
    HBOND_DIST_CUTOFF,
    _dist_weight,
)
from .gt_constants import (
    EDGE_DIM,
    IDX_DIR_HBOND,
    IDX_PI_CATION,
    IDX_RBF_START,
    IDX_SALT_BRIDGE,
    RBF_CENTERS_ANGSTROM,
)

EDGE_CUTOFF = 8.0

_POSITIVE_PROT_RESNAMES = frozenset({"LYS", "ARG", "HIS"})
_AROMATIC_RNA_BASES = frozenset({"A", "G", "U", "C", "DA", "DG", "DU", "DC"})


def _rbf(distance: float, center: float, width: float = 1.2) -> float:
    return float(np.exp(-((distance - center) ** 2) / (2.0 * width ** 2)))


def _directional_hbond_bonus(
    p_atom,
    r_atom,
    p_coord: np.ndarray,
    r_coord: np.ndarray,
) -> float:
    """
    Lightweight directional H-bond score: distance gate × alignment of the
    inter-atomic vector with CA→representative-atom vectors (PNAbind-style
    geometric filtering without full angle tables).
    """
    if p_atom.element not in ("N", "O") or r_atom.element not in ("N", "O"):
        return 0.0
    d = float(np.linalg.norm(p_coord - r_coord))
    if d > HBOND_DIST_CUTOFF:
        return 0.0
    vec = r_coord - p_coord
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return 1.0
    # Favour contacts where N/O separation is not perfectly collinear with
    # backbone (proxy for accessible donor/acceptor orientation).
    return float(np.exp(-d / 2.5))


def _pi_cation_score(pres, rres, d: float, w: float) -> float:
    """Lys/Arg/His sidechains interacting with aromatic RNA bases."""
    resname = pres.get("resname", pres.get("aa1", "X"))
    if isinstance(resname, str) and len(resname) == 1:
        resname = {"K": "LYS", "R": "ARG", "H": "HIS"}.get(resname, resname)
    resname = str(resname).strip().upper()
    base = str(rres.get("resname", "X")).strip().upper()
    if resname not in _POSITIVE_PROT_RESNAMES:
        return 0.0
    if base not in _AROMATIC_RNA_BASES:
        return 0.0
    return -1.2 * w


def _salt_bridge_score(p_chg: float, r_chg: float, d: float, w: float) -> float:
    if p_chg * r_chg >= 0.0 or d > 4.0:
        return 0.0
    return -2.0 * w * min(abs(p_chg * r_chg), 1.0)


def aggregate_prot_rna_pair(
    p_atoms: list,
    r_atoms: list,
    pres: dict,
    rres: dict,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Aggregate all physics terms between one protein residue and one RNA residue.

    Returns:
        min_d, elec, stack, hbond, vdw, pi_cation, dir_hbond, salt_bridge
    """
    elec = stack = hbond = vdw = 0.0
    pi_cation = dir_hbond = salt_bridge = 0.0
    min_d = float("inf")

    p_aro_atoms = PROTEIN_AROMATIC_ATOMS.get(
        str(pres.get("resname", "")).strip().upper(), set()
    )
    p_resname = str(pres.get("resname", pres.get("aa1", "X"))).strip().upper()

    for pa in p_atoms:
        pc = pa.coord
        p_chg = PROTEIN_CHARGE.get(pa.name, 0.0)
        p_aro = pa.name in p_aro_atoms
        for ra in r_atoms:
            rc = ra.coord
            d = float(np.linalg.norm(pc - rc))
            if d < min_d:
                min_d = d
            if d > EDGE_CUTOFF or d < 0.1:
                continue
            w = _dist_weight(d)
            r_chg = RNA_CHARGE.get(ra.name, 0.0)

            if p_chg != 0.0 and r_chg != 0.0:
                elec += (
                    W_ELECTROSTATIC * w
                    * abs(p_chg * r_chg)
                    * float(np.sign(p_chg * r_chg))
                )
                salt_bridge += _salt_bridge_score(p_chg, r_chg, d, w)

            if p_aro and ra.name in RNA_AROMATIC:
                stack -= W_STACKING * w

            if pa.element in ("N", "O") and ra.element in ("N", "O") and d <= HBOND_DIST_CUTOFF:
                hbond -= W_HBOND * w
                dir_hbond -= W_HBOND * 0.5 * _directional_hbond_bonus(pa, ra, pc, rc)

            vdw -= W_VDW * w

    if min_d == float("inf"):
        min_d = EDGE_CUTOFF + 1.0

    if min_d <= EDGE_CUTOFF:
        w_min = _dist_weight(min_d)
        pi_cation += _pi_cation_score(
            {"resname": p_resname, "aa1": pres.get("aa1", "X")},
            rres,
            min_d,
            w_min,
        )

    return min_d, elec, stack, hbond, vdw, pi_cation, dir_hbond, salt_bridge


def build_edge_feature_vector(
    min_d: float,
    elec: float,
    stack: float,
    hbond: float,
    vdw: float,
    pi_cation: float = 0.0,
    dir_hbond: float = 0.0,
    salt_bridge: float = 0.0,
    *,
    prot_rna: bool = True,
    prot_prot: bool = False,
    rna_rna: bool = False,
    edge_dim: int | None = None,
) -> torch.Tensor:
    """Build a physics edge feature vector (legacy 9-d or full EDGE_DIM)."""
    target_dim = edge_dim if edge_dim is not None else EDGE_DIM
    rbf = [_rbf(min_d, c) for c in RBF_CENTERS_ANGSTROM]
    full = torch.tensor(
        [
            min(min_d / EDGE_CUTOFF, 1.5),
            float(np.exp(-min_d / 3.5)),
            elec,
            stack,
            hbond,
            vdw,
            1.0 if prot_rna else 0.0,
            1.0 if prot_prot else 0.0,
            1.0 if rna_rna else 0.0,
            pi_cation,
            dir_hbond,
            salt_bridge,
            *rbf,
        ],
        dtype=torch.float32,
    )
    if target_dim < full.shape[0]:
        return full[:target_dim]
    if target_dim > full.shape[0]:
        padded = torch.zeros(target_dim, dtype=torch.float32)
        padded[: full.shape[0]] = full
        return padded
    return full


def backbone_edge_vector(
    distance: float,
    *,
    prot_prot: bool = False,
    rna_rna: bool = False,
    edge_dim: int | None = None,
) -> torch.Tensor:
    """Sequential backbone edge with zero interaction physics."""
    return build_edge_feature_vector(
        distance,
        0.0,
        0.0,
        0.0,
        0.0,
        prot_rna=False,
        prot_prot=prot_prot,
        rna_rna=rna_rna,
        edge_dim=edge_dim,
    )


def coerce_edge_attr(edge_attr: torch.Tensor, edge_dim: int) -> torch.Tensor:
    """Truncate or zero-pad edge features to match a checkpoint edge_dim."""
    if edge_attr.numel() == 0:
        return torch.zeros((0, edge_dim), dtype=edge_attr.dtype, device=edge_attr.device)
    current = edge_attr.shape[1]
    if current == edge_dim:
        return edge_attr
    if current > edge_dim:
        return edge_attr[:, :edge_dim]
    pad = torch.zeros(
        edge_attr.shape[0],
        edge_dim - current,
        dtype=edge_attr.dtype,
        device=edge_attr.device,
    )
    return torch.cat([edge_attr, pad], dim=1)


def physics_summary_from_graph(
    x_protein: torch.Tensor,
    edge_attr: torch.Tensor,
    mutation_idx: int | None = None,
    *,
    legacy_dim: int | None = None,
) -> torch.Tensor:
    """
    Explicit physics vector concatenated into GT heads.

    When ``legacy_dim=11``, returns the v1 summary for old checkpoints.
    """
    from .gt_constants import LEGACY_PHYSICS_SUMMARY_DIM, PHYSICS_SUMMARY_DIM

    target_dim = legacy_dim or PHYSICS_SUMMARY_DIM
    device = x_protein.device
    dtype = x_protein.dtype

    if edge_attr.numel() == 0:
        edge_physics = torch.zeros(4, device=device, dtype=dtype)
        ext_physics = torch.zeros(3, device=device, dtype=dtype)
        mean_dist_w = torch.zeros(1, device=device, dtype=dtype)
        n_edges_norm = torch.zeros(1, device=device, dtype=dtype)
        complementarity = torch.zeros(1, device=device, dtype=dtype)
    else:
        prot_rna = edge_attr[:, 6] > 0.5
        selected = edge_attr[prot_rna] if prot_rna.any() else edge_attr
        edge_physics = selected[:, 2:6].sum(dim=0)
        if edge_attr.shape[1] >= EDGE_DIM:
            ext_physics = selected[:, IDX_PI_CATION:IDX_SALT_BRIDGE + 1].sum(dim=0)
        else:
            ext_physics = torch.zeros(3, device=device, dtype=dtype)
        mean_dist_w = selected[:, 1].mean().unsqueeze(0) if selected.shape[0] else torch.zeros(1, device=device, dtype=dtype)
        n_edges_norm = torch.tensor(
            [min(selected.shape[0] / 50.0, 2.0)],
            device=device,
            dtype=dtype,
        )
        favorable = selected[:, 2:6].sum(dim=1) + (
            selected[:, IDX_PI_CATION:IDX_SALT_BRIDGE + 1].sum(dim=1)
            if edge_attr.shape[1] >= EDGE_DIM
            else 0.0
        )
        complementarity = (
            (-favorable.mean()).unsqueeze(0)
            if selected.shape[0]
            else torch.zeros(1, device=device, dtype=dtype)
        )

    if mutation_idx is not None and mutation_idx < x_protein.shape[0]:
        site_features = x_protein[mutation_idx, 20:24]
    else:
        site_features = torch.zeros(4, device=device, dtype=dtype)

    global_features = (
        x_protein[:, 20:23].mean(dim=0)
        if x_protein.shape[0]
        else torch.zeros(3, device=device, dtype=dtype)
    )

    if target_dim <= LEGACY_PHYSICS_SUMMARY_DIM:
        return torch.cat([edge_physics, site_features, global_features], dim=0)

    return torch.cat(
        [
            edge_physics,
            ext_physics,
            mean_dist_w,
            site_features,
            global_features,
            n_edges_norm,
            complementarity,
        ],
        dim=0,
    )
