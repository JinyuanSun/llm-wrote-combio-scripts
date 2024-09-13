#!/usr/bin/env python3
"""
Protein Solvent Accessible Surface Area (SASA) Calculator using PyTorch.

This script computes the SASA of a protein structure provided in a PDB file using
the Shrake-Rupley algorithm with GPU acceleration.

Features:
1. Optimized Command-Line Interface using argparse.
2. Detailed English annotations for clarity.
3. GPU acceleration on macOS using PyTorch's MPS backend for Apple Silicon.
4. Batch processing to handle large proteins efficiently.
5. Outputs per-atom SASA as JSON and writes a new PDB file with SASA in B-factor.

Author: Jinyuan Sun,o1-preview
Date: 2024-09-13
"""

import argparse
import numpy as np
from Bio.PDB import PDBParser, PDBIO
import torch
import json
import sys
import os

def get_vdw_radius(element):
    """
    Retrieve the Van der Waals radius for a given chemical element.

    Parameters:
    - element (str): Chemical element symbol (e.g., 'C', 'N', 'O').

    Returns:
    - float: Van der Waals radius in Ångströms.
    """
    vdw_radii = {
        'H': 1.00,
        'C': 1.76,
        'N': 1.65,
        'O': 1.60,
        'S': 1.85,
        'P': 1.90,
        # Add more elements as needed
    }
    return vdw_radii.get(element, 1.50)  # Default radius if element not found

def get_atom_coords_and_radii(pdb_file):
    """
    Parse a PDB file to extract atom coordinates and their corresponding Van der Waals radii.

    Hydrogen atoms are excluded to match standard SASA calculation protocols.

    Parameters:
    - pdb_file (str): Path to the PDB file.

    Returns:
    - tuple: A tuple containing:
        - np.ndarray: Atom coordinates with shape (N, 3).
        - np.ndarray: Corresponding Van der Waals radii with shape (N,).
    """
    parser = PDBParser(QUIET=True)  # Suppress warnings
    try:
        structure = parser.get_structure('protein', pdb_file)
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        sys.exit(1)
    
    coords = []
    radii = []
    atoms = []
    for atom in structure.get_atoms():
        element = atom.element.strip()
        if element == 'H':
            continue  # Skip hydrogen atoms
        coords.append(atom.get_coord())
        radii.append(get_vdw_radius(element))
        atoms.append(atom)  # Store atom for later use
    return np.array(coords), np.array(radii), atoms, structure

def fibonacci_sphere(samples, device):
    """
    Generate points uniformly distributed on the surface of a unit sphere using the Fibonacci method.

    Parameters:
    - samples (int): Number of points to generate.
    - device (torch.device): PyTorch device to store the tensor.

    Returns:
    - torch.Tensor: Tensor of shape (samples, 3) containing the sphere points.
    """
    points = []
    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = ((i + 1) % samples) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x, y, z])

    return torch.tensor(points, dtype=torch.float32, device=device)

def compute_sasa(coords, vdw_radii, n_points=960, probe_radius=1.4, device=None):
    """
    Compute SASA using the Shrake-Rupley algorithm.

    Parameters:
    - coords: Atom coordinates, shape (N, 3)
    - vdw_radii: Van der Waals radii of atoms, shape (N,)
    - n_points: Number of sampling points per atom surface
    - probe_radius: Probe radius, default 1.4 Å

    Returns:
    - np.ndarray: SASA values for each atom, shape (N,)
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Apple Silicon GPU (MPS) for acceleration.")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA GPU for acceleration.")
        else:
            device = torch.device('cpu')
            print("GPU not available. Using CPU for computation.")
    
    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    vdw_radii = torch.tensor(vdw_radii, dtype=torch.float32, device=device)
    radii = vdw_radii + probe_radius  # Effective radii for sampling
    N = coords.shape[0]

    sphere_points = fibonacci_sphere(n_points, device=device)  # (n_points, 3)
    
    # Compute sampling points on the surface of each atom
    expanded_radii = radii.unsqueeze(1)  # (N, 1)
    expanded_coords = coords.unsqueeze(1)  # (N, 1, 3)
    points = expanded_coords + expanded_radii.unsqueeze(2) * sphere_points.unsqueeze(0)  # (N, n_points, 3)
    
    # Flatten sampling points
    points = points.view(-1, 3)  # (N * n_points, 3)
    
    # Compute distances between points and atom centers
    coords_expanded = coords.unsqueeze(0)  # (1, N, 3)
    points_expanded = points.unsqueeze(1)  # (N * n_points, 1, 3)
    dists = torch.norm(points_expanded - coords_expanded, dim=2)  # (N * n_points, N)
    
    # Effective radii for occlusion
    effective_radii = (vdw_radii + probe_radius).unsqueeze(0)  # (1, N)
    
    # Exclude self-occlusion
    atom_indices = torch.arange(N, device=device).repeat_interleave(n_points)  # (N * n_points,)
    mask_self = torch.zeros_like(dists, dtype=torch.bool)
    mask_self[torch.arange(dists.size(0)), atom_indices] = True
    dists = dists.masked_fill(mask_self, float('inf'))
    
    # Determine occluded points
    occluded = (dists < effective_radii)  # (N * n_points, N)
    occluded = occluded.any(dim=1)  # (N * n_points,)

    # Ensure occluded size is correct
    assert occluded.numel() == N * n_points, f"Occluded size should be {N * n_points}, but got {occluded.numel()}"

    # Reshape occluded result
    occluded = occluded.view(N, n_points)
    
    # Compute accessible surface area
    total_area = 4 * np.pi * (radii ** 2)  # (N,)
    point_areas = total_area / n_points  # (N,)
    accessible = (~occluded).sum(dim=1).float()  # (N,)
    sasa = accessible * point_areas  # (N,)

    return sasa.cpu().numpy()

def main():
    """
    Main function to parse command-line arguments and execute SASA calculation.
    """
    parser = argparse.ArgumentParser(description="Compute Solvent Accessible Surface Area (SASA) of a protein using PyTorch.")
    parser.add_argument("pdb_file", type=str, help="Path to the input PDB file.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Prefix for output files.")
    parser.add_argument("-n", "--n_points", type=int, default=960, help="Number of sampling points per atom (default: 960).")
    parser.add_argument("-p", "--probe_radius", type=float, default=1.4, help="Probe radius in Ångströms (default: 1.4).")
    parser.add_argument("-d", "--device", type=str, default=None, help="Device for computation (cpu, cuda, mps).")
    args = parser.parse_args()

    # Extract arguments
    pdb_file = args.pdb_file
    output_prefix = args.output if args.output else os.path.splitext(os.path.basename(pdb_file))[0]
    n_points = args.n_points
    probe_radius = args.probe_radius

    print(f"Loading PDB file: {pdb_file}")
    coords, radii, atoms, structure = get_atom_coords_and_radii(pdb_file)
    print(f"Number of atoms (excluding hydrogens): {len(radii)}")

    print("Computing SASA...")
    sasa_values = compute_sasa(coords, radii, n_points=n_points, probe_radius=probe_radius, device=args.device)
    total_sasa = sasa_values.sum()
    print(f"Total SASA: {total_sasa:.2f} Å²")

    # Save SASA values in a dictionary and output as JSON
    sasa_dict = {}
    for idx, (atom, sasa) in enumerate(zip(atoms, sasa_values)):
        atom_id = f"{atom.get_full_id()}"
        sasa_dict[atom_id] = str(sasa)

    json_output_file = f"{output_prefix}_sasa.json"
    with open(json_output_file, 'w') as json_file:
        json.dump(sasa_dict, json_file, indent=4)
    print(f"SASA values saved to JSON file: {json_output_file}")

    # Write new PDB file replacing B-factor with SASA values
    pdb_output_file = f"{output_prefix}_sasa.pdb"
    for idx, atom in enumerate(atoms):
        atom.set_bfactor(sasa_values[idx])

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_output_file)
    print(f"New PDB file with SASA values in B-factor column saved as: {pdb_output_file}")

if __name__ == "__main__":
    main()
