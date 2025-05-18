import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from ase.io import read

def find_cu_clusters(atoms, cutoff=3.0, element='Cu'):
    cu_indices = [i for i, atom in enumerate(atoms) if atom.symbol == element]
    positions = np.array([atoms[i].position for i in cu_indices])
    G = nx.Graph()
    for i, idx1 in enumerate(cu_indices):
        for j, idx2 in enumerate(cu_indices):
            if j <= i:
                continue
            if np.linalg.norm(positions[i] - positions[j]) < cutoff:
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
