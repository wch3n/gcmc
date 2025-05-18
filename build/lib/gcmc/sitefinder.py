import numpy as np
from scipy.spatial import Delaunay

def find_fcc_hollow_sites(
    slab,
    substrate_elements=("Ti", "C"),
    z_threshold=2.0,
    fcc_offset=1.8
):
    """
    Find FCC hollow sites on a surface, *excluding functional groups*.
    
    Parameters
    ----------
    slab : ASE Atoms
        The full system including functional groups.
    substrate_elements : tuple
        Atom symbols to use for site-finding (e.g., ("Ti", "C")).
    z_threshold : float
        Thickness window to include "top" surface atoms for Delaunay.
    fcc_offset : float
        z-offset to place the Cu atom above the average surface plane.
    """
    # Get indices and z positions of non-functional surface atoms
    substrate_atoms = [atom for atom in slab if atom.symbol in substrate_elements]
    # Now find the *top* z among substrate atoms
    z_surface = max(atom.position[2] for atom in substrate_atoms)
    surface_atoms = [
        atom for atom in substrate_atoms if z_surface - atom.position[2] < z_threshold
    ]
    if not surface_atoms:
        raise RuntimeError("No surface atoms found! Check substrate_elements or z_threshold.")

    positions = np.array([atom.position[:2] for atom in surface_atoms])
    tri = Delaunay(positions)
    fcc_sites = []
    for simplex in tri.simplices:
        pts = positions[simplex]
        center = pts.mean(axis=0)
        # Average z of the 3 surface substrate atoms
        z = np.mean([surface_atoms[i].position[2] for i in simplex])
        fcc_sites.append([center[0], center[1], z + fcc_offset])
    return fcc_sites
