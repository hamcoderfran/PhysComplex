"""
Extract one protein chain + one RNA chain into a minimal complex PDB.

Useful for fair AF3 vs crystal comparisons (e.g. 1URN chain A + P).

Example
-------
    python -m physrna_filter.data.extract_chain_pair physrna_filter/data/structures/1urn.pdb A P -o 1urn_single.pdb
"""
from __future__ import annotations

import argparse
import sys

from Bio.PDB import PDBIO, PDBParser, Select


class _ChainSelect(Select):
    def __init__(self, chain_ids: set[str]):
        self.chain_ids = chain_ids

    def accept_chain(self, chain):
        return chain.id in self.chain_ids


def extract_chain_pair(
    input_pdb: str,
    protein_chain: str,
    rna_chain: str,
    output_pdb: str,
) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    chains = {protein_chain, rna_chain}
    io.save(output_pdb, select=_ChainSelect(chains))
    return output_pdb


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extract protein+RNA chains to a single PDB")
    ap.add_argument("input_pdb")
    ap.add_argument("protein_chain", help="Protein chain ID, e.g. A")
    ap.add_argument("rna_chain", help="RNA chain ID, e.g. P")
    ap.add_argument("-o", "--output", required=True, help="Output PDB path")
    args = ap.parse_args(argv)

    out = extract_chain_pair(
        args.input_pdb,
        args.protein_chain,
        args.rna_chain,
        args.output,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
