import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry import get_distances

def find_cu_clusters(atoms, cutoff=3.0, element="Cu"):
    cu_indices = [i for i, atom in enumerate(atoms) if atom.symbol == element]
    if not cu_indices:
        return []

    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    G = nx.Graph()
    G.add_nodes_from(cu_indices)

    for ii, idx1 in enumerate(cu_indices):
        for idx2 in cu_indices[ii + 1 :]:
            _, dist = get_distances(
                positions[idx1],
                positions[idx2],
                cell=cell,
                pbc=pbc,
            )
            if dist < cutoff:
                G.add_edge(idx1, idx2)

    clusters = list(nx.connected_components(G))
    sizes = [len(c) for c in clusters]
    return sizes

def analyze_and_plot(traj_file, cutoff=3.0, element='Cu'):
    traj = list(read(traj_file, index=':'))
    largest_cluster = []
    n_clusters = []
    total_cu = []

    for atoms in traj:
        sizes = find_cu_clusters(atoms, cutoff=cutoff, element=element)
        largest_cluster.append(max(sizes) if sizes else 0)
        n_clusters.append(len(sizes))
        total_cu.append(len([a for a in atoms if a.symbol == element]))

    print(f"Final frame: Largest Cu cluster size = {largest_cluster[-1]}, Number of clusters = {n_clusters[-1]}, Total Cu = {total_cu[-1]}")

    plt.figure(figsize=(8,5))
    plt.plot(largest_cluster, label='Largest cluster size')
    plt.plot(total_cu, label='Total Cu')
    plt.xlabel('GCMC Step')
    plt.ylabel('Cu count')
    plt.title('Cu Clustering during GCMC')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(n_clusters, label='Number of clusters')
    plt.xlabel('GCMC Step')
    plt.ylabel('Number of clusters')
    plt.title('Cu Cluster Number during GCMC')
    plt.tight_layout()
    plt.show()
