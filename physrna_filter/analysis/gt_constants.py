"""
Shared PhysGT schema constants.

Version 2 extends edge features (pi-cation, directional H-bonds, salt bridges,
RBF distance encoding) and the explicit physics-summary vector used by the
regression / interface heads.  Checkpoints record ``schema_version`` so
inference can rebuild models with matching dimensions.
"""
from __future__ import annotations

CHECKPOINT_SCHEMA_VERSION = 2

LEGACY_EDGE_DIM = 9
EDGE_DIM = 15

LEGACY_PHYSICS_SUMMARY_DIM = 11
PHYSICS_SUMMARY_DIM = 17

# RBF centres (Å) for pairwise distance encoding on prot–RNA edges
RBF_CENTERS_ANGSTROM = (2.5, 4.0, 6.5)

# Extended physics term indices inside EDGE_DIM vectors (after legacy 9-d block)
IDX_PI_CATION = 9
IDX_DIR_HBOND = 10
IDX_SALT_BRIDGE = 11
IDX_RBF_START = 12
