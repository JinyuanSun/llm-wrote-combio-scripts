#!/usr/bin/env python3
"""
Calculate the correlation of SASA (Solvent Accessible Surface Area) values for each atom between two PDB files.

Usage:
    python sasa_correlation.py 1pga_freesasa.pdb 1pga_sasa.pdb

Dependencies:
    - Biopython
    - NumPy
    - SciPy

Author: Jinyuan Sun
Date: 2024-09-13
"""

import sys
from Bio.PDB import PDBParser
import numpy as np
from scipy.stats import pearsonr, spearmanr

def get_atom_sasa(pdb_file):
    """
    Extract the SASA value for each atom from a PDB file.

    Parameters:
    - pdb_file (str): Path to the PDB file.

    Returns:
    - dict: A dictionary with atom IDs as keys and SASA values as values.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    atom_sasa = {}
    for atom in structure.get_atoms():
        element = atom.element.strip()
        if element == 'H':
            continue  # Skip hydrogen atoms
        # Create a unique identifier for the atom
        atom_id = (
            atom.get_parent().get_parent().get_id(),               # Chain ID
            atom.get_parent().get_id()[1],                         # Residue number
            atom.get_parent().get_resname(),                       # Residue name
            atom.get_name()                                        # Atom name
        )
        sasa_value = atom.get_bfactor()  # SASA value is stored in the B-factor field
        atom_sasa[atom_id] = sasa_value
    return atom_sasa

def main():
    if len(sys.argv) != 3:
        print("Usage: python sasa_correlation.py 1pga_freesasa.pdb 1pga_sasa.pdb")
        sys.exit(1)

    pdb_file1 = sys.argv[1]
    pdb_file2 = sys.argv[2]

    # Extract SASA values from the two PDB files
    sasa_dict1 = get_atom_sasa(pdb_file1)
    sasa_dict2 = get_atom_sasa(pdb_file2)

    # Find common atoms in both files
    common_atoms = set(sasa_dict1.keys()).intersection(set(sasa_dict2.keys()))
    if not common_atoms:
        print("No common atoms found between the two PDB files.")
        sys.exit(1)

    # Extract SASA values for common atoms
    sasa_values1 = []
    sasa_values2 = []
    for atom_id in sorted(common_atoms):
        sasa_values1.append(sasa_dict1[atom_id])
        sasa_values2.append(sasa_dict2[atom_id])

    # Convert to NumPy arrays
    sasa_values1 = np.array(sasa_values1)
    sasa_values2 = np.array(sasa_values2)

    # Calculate correlation coefficients
    pearson_corr, pearson_p = pearsonr(sasa_values1, sasa_values2)
    spearman_corr, spearman_p = spearmanr(sasa_values1, sasa_values2)

    print(f"Number of common atoms: {len(common_atoms)}")
    print(f"Pearson correlation coefficient: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
    print(f"Spearman correlation coefficient: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")

if __name__ == "__main__":
    main()
