import os
import pickle
import numpy as np
import random
from ase import Atom
from ase.optimize import LBFGS
from ase.io import write, read, Trajectory
from scipy.spatial import Delaunay, cKDTree

class MultiLayerSiteGCMC:
    """
    Grand Canonical Monte Carlo (GCMC) class with adaptive site generation,
    efficient occupation checking, robust "surface site" logic,
    maximum layer restriction, and LBFGS optimization. Adsorbate-neutral.
    """

    def __init__(
        self,
        atoms,
        calculator,
        mu,
        T,
        substrate_elements=("Ti", "C"),
        z_threshold=2.0,
        fcc_offset=1.8,
        element='Cu',
        nsteps=1000,
        relax=False,
        relax_steps=200,
        traj_file='gcmc.traj',
        checkpoint_traj='gcmc_checkpoint.traj',
        checkpoint_data='gcmc_checkpoint.pkl',
        checkpoint_interval=1000,
        seed=81,
        tol=0.5,
        xy_tol=0.4,
        z_tol=0.1,
        max_layers=None,
        layer_spacing=2.0,
    ):
        self.atoms = atoms
        self.calculator = calculator
        self.mu = mu
        self.T = T
        self.beta = 1 / (8.617333e-5 * T)
        self.substrate_elements = substrate_elements
        self.z_threshold = z_threshold
        self.fcc_offset = fcc_offset
        self.element = element
        self.nsteps = nsteps
        self.relax = relax
        self.relax_steps = relax_steps
        self.traj_file = traj_file
        self.checkpoint_traj = checkpoint_traj
        self.checkpoint_data = checkpoint_data
        self.checkpoint_interval = checkpoint_interval
        self.tol = tol
        self.xy_tol = xy_tol
        self.z_tol = z_tol
        self.max_layers = max_layers
        self.layer_spacing = layer_spacing

        self.rng = random.Random(seed)
        self.seed = seed

        self.attempted_insertions = 0
        self.accepted_insertions = 0
        self.attempted_deletions = 0
        self.accepted_deletions = 0
        self.attempted_displacements = 0
        self.accepted_displacements = 0

        self.step = 0

        # Reference substrate z for layer capping
        self.substrate_z = max(atom.position[2] for atom in self.atoms if atom.symbol in self.substrate_elements)

        if os.path.exists(self.checkpoint_traj) and os.path.exists(self.checkpoint_data):
            self._load_checkpoint()
            print(f"Resuming from checkpoint at step {self.step}")
        else:
            self.atoms.calc = self.calculator
            self.e_old = self.atoms.get_potential_energy()

        self._update_sites()

    def _update_sites(self):
        """Find adaptive FCC hollow sites on the current (relaxed) surface, including adatom islands."""
        all_sites = self.find_adaptive_fcc_sites(
            self.atoms,
            substrate_elements=self.substrate_elements,
            adatom_element=self.element,
            z_threshold=self.z_threshold,
            fcc_offset=self.fcc_offset
        )
        # Restrict to max_layers if requested
        if self.max_layers is not None:
            max_z = self.substrate_z + self.max_layers * self.layer_spacing
            all_sites = np.array([s for s in all_sites if s[2] <= max_z])
        self.all_sites = all_sites

    @staticmethod
    def find_adaptive_fcc_sites(atoms, substrate_elements=("Ti", "C"), adatom_element="Cu", z_threshold=2.0, fcc_offset=1.8):
        candidates = [atom for atom in atoms if atom.symbol in substrate_elements or atom.symbol == adatom_element]
        positions = np.array([atom.position for atom in candidates])
        xy = positions[:, :2]
        rounded_xy = np.round(xy / 0.2) * 0.2
        surface_atoms = []
        seen = set()
        for i, rxy in enumerate(rounded_xy):
            key = tuple(rxy)
            if key not in seen:
                same = np.all(np.isclose(rounded_xy, rxy, atol=1e-3), axis=1)
                idx = np.argmax(positions[same, 2])
                surface_atoms.append(candidates[np.where(same)[0][idx]])
                seen.add(key)
        surface_xy = np.array([atom.position[:2] for atom in surface_atoms])
        tri = Delaunay(surface_xy)
        fcc_sites = []
        for simplex in tri.simplices:
            pts = surface_xy[simplex]
            center = pts.mean(axis=0)
            z = np.mean([surface_atoms[i].position[2] for i in simplex])
            fcc_sites.append([center[0], center[1], z + fcc_offset])
        return np.array(fcc_sites)

    def _get_adsorbate_atoms(self):
        return [atom for atom in self.atoms if atom.symbol == self.element]

    def _get_occupied_mask(self, all_sites):
        adsorbate_atoms = self._get_adsorbate_atoms()
        if not adsorbate_atoms or len(all_sites) == 0:
            return np.zeros(len(all_sites), dtype=bool)
        adsorbate_positions = np.array([atom.position for atom in adsorbate_atoms])
        site_pos = np.array(all_sites)
        kdtree = cKDTree(adsorbate_positions)
        dists, _ = kdtree.query(site_pos, distance_upper_bound=self.tol)
        return dists < self.tol

    def get_exposed_surface_sites(self, sites=None):
        if sites is None:
            sites = self.all_sites
        adsorbate_atoms = self._get_adsorbate_atoms()
        adsorbate_positions = np.array([atom.position for atom in adsorbate_atoms]) if adsorbate_atoms else np.empty((0,3))
        occupied_mask = self._get_occupied_mask(sites)
        exposed_vacant = []
        exposed_occupied = []
        for idx, site in enumerate(sites):
            site_xy = site[:2]
            site_z = site[2]
            if adsorbate_positions.shape[0] > 0:
                xy_dists = np.linalg.norm(adsorbate_positions[:, :2] - site_xy, axis=1)
                above = (adsorbate_positions[:,2] > site_z + self.z_tol) & (xy_dists < self.xy_tol)
                if np.any(above):
                    continue
            if occupied_mask[idx]:
                exposed_occupied.append(idx)
            else:
                exposed_vacant.append(idx)
        return exposed_vacant, exposed_occupied

    def attempt_insertion(self):
        exposed_vacant, _ = self.get_exposed_surface_sites()
        if not exposed_vacant:
            return None
        # Layer cap
        if self.max_layers is not None:
            max_z = self.substrate_z + self.max_layers * self.layer_spacing
            exposed_vacant = [idx for idx in exposed_vacant if self.all_sites[idx][2] <= max_z]
            if not exposed_vacant:
                return None
        idx = self.rng.choice(exposed_vacant)
        site = self.all_sites[idx]
        atoms_new = self.atoms.copy()
        atoms_new.append(Atom(self.element, site))
        return atoms_new

    def attempt_deletion(self):
        _, exposed_occupied = self.get_exposed_surface_sites()
        if not exposed_occupied:
            return None
        # Only delete adsorbate within allowed layers
        if self.max_layers is not None:
            max_z = self.substrate_z + self.max_layers * self.layer_spacing
            exposed_occupied = [idx for idx in exposed_occupied if self.all_sites[idx][2] <= max_z]
            if not exposed_occupied:
                return None
        site_idx = self.rng.choice(exposed_occupied)
        site = self.all_sites[site_idx]
        atom_idx = None
        for i, atom in enumerate(self.atoms):
            if atom.symbol == self.element and np.linalg.norm(atom.position - site) < self.tol:
                atom_idx = i
                break
        if atom_idx is None:
            return None
        atoms_new = self.atoms.copy()
        del atoms_new[atom_idx]
        return atoms_new

    def attempt_displacement(self):
        exposed_vacant, exposed_occupied = self.get_exposed_surface_sites()
        # Layer cap: only allow move from and to allowed layers
        if self.max_layers is not None:
            max_z = self.substrate_z + self.max_layers * self.layer_spacing
            exposed_vacant = [i for i in exposed_vacant if self.all_sites[i][2] <= max_z]
            exposed_occupied = [i for i in exposed_occupied if self.all_sites[i][2] <= max_z]
        if not exposed_occupied or not exposed_vacant:
            return None
        src_idx = self.rng.choice(exposed_occupied)
        dest_indices = [i for i in exposed_vacant if not np.allclose(self.all_sites[i], self.all_sites[src_idx], atol=1e-3)]
        if not dest_indices:
            return None
        dst_idx = self.rng.choice(dest_indices)
        atom_idx = None
        for i, atom in enumerate(self.atoms):
            if atom.symbol == self.element and np.linalg.norm(atom.position - self.all_sites[src_idx]) < self.tol:
                atom_idx = i
                break
        if atom_idx is None:
            return None
        atoms_new = self.atoms.copy()
        atoms_new[atom_idx].position = self.all_sites[dst_idx]
        return atoms_new

    def relax_structure(self, atoms):
        """Relax structure using LBFGS optimizer after move."""
        if self.relax:
            dyn = LBFGS(atoms, logfile=None)
            dyn.run(fmax=0.01, steps=self.relax_steps)
        return atoms

    def metropolis_accept(self, e_new, move_type):
        delta_e = e_new - self.e_old
        if move_type == 'insert':
            tau = -self.beta * (delta_e - self.mu)
        elif move_type == 'delete':
            tau = -self.beta * (delta_e + self.mu)
        elif move_type == 'displace':
            tau = -self.beta * delta_e
        else:
            raise ValueError("Unknown move_type")
        return np.exp(min(tau, 0)), tau

    def _save_checkpoint(self):
        atoms_to_save = self.atoms.copy()
        atoms_to_save.calc = None
        write(self.checkpoint_traj, atoms_to_save)
        state = {
            'step': self.step,
            'e_old': self.e_old,
            'attempted_insertions': self.attempted_insertions,
            'accepted_insertions': self.accepted_insertions,
            'attempted_deletions': self.attempted_deletions,
            'accepted_deletions': self.accepted_deletions,
            'attempted_displacements': self.attempted_displacements,
            'accepted_displacements': self.accepted_displacements,
            'rng_state': self.rng.getstate(),
        }
        with open(self.checkpoint_data, 'wb') as f:
            pickle.dump(state, f)
        print(f"Checkpoint saved at step {self.step}.")

    def _load_checkpoint(self):
        self.atoms = read(self.checkpoint_traj)
        self.atoms.calc = self.calculator
        with open(self.checkpoint_data, 'rb') as f:
            state = pickle.load(f)
        self.step = state['step']
        self.e_old = state['e_old']
        self.attempted_insertions = state.get('attempted_insertions', 0)
        self.accepted_insertions = state.get('accepted_insertions', 0)
        self.attempted_deletions = state.get('attempted_deletions', 0)
        self.accepted_deletions = state.get('accepted_deletions', 0)
        self.attempted_displacements = state.get('attempted_displacements', 0)
        self.accepted_displacements = state.get('accepted_displacements', 0)
        if 'rng_state' in state:
            self.rng.setstate(state['rng_state'])
        self._update_sites()

    def run(self, w_insert=1, w_delete=1, w_displace=1):
        traj_mode = 'a' if os.path.exists(self.traj_file) else 'w'
        with Trajectory(self.traj_file, traj_mode) as traj:
            for s in range(self.step, self.nsteps):
                move_type = self.rng.choices(
                    ['insert', 'delete', 'displace'],
                    weights=[w_insert, w_delete, w_displace]
                )[0]
                if move_type == 'insert':
                    self.attempted_insertions += 1
                    atoms_new = self.attempt_insertion()
                    if atoms_new is None:
                        continue
                elif move_type == 'delete':
                    self.attempted_deletions += 1
                    atoms_new = self.attempt_deletion()
                    if atoms_new is None:
                        continue
                elif move_type == 'displace':
                    self.attempted_displacements += 1
                    atoms_new = self.attempt_displacement()
                    if atoms_new is None:
                        continue
                else:
                    continue

                atoms_new.calc = self.calculator
                if self.relax:
                    atoms_new = self.relax_structure(atoms_new)
                e_new = atoms_new.get_potential_energy()
                prob, tau = self.metropolis_accept(e_new, move_type)
                if self.rng.random() < prob:
                    # Accept move
                    self.atoms = atoms_new
                    self.e_old = e_new
                    if move_type == 'insert':
                        self.accepted_insertions += 1
                    elif move_type == 'delete':
                        self.accepted_deletions += 1
                    elif move_type == 'displace':
                        self.accepted_displacements += 1
                    print(f"Step {s:6d}: Accepted {move_type:<8} (tau = {tau:8.2f}, E = {e_new:8.2f} eV)")
                    self._update_sites()
                else:
                    print(f"Step {s:6d}: Rejected {move_type:<8} (tau = {tau:8.2f})")

                self.atoms.calc = None
                if len(self.atoms) > 0:
                    traj.write(self.atoms, step=s, move=move_type, energy=self.e_old)

                self.step = s + 1
                if (self.checkpoint_interval and (self.step % self.checkpoint_interval == 0)):
                    self._save_checkpoint()
            # Final checkpoint
            self._save_checkpoint()
            print("GCMC completed.")
            print(f"Insertions: attempted {self.attempted_insertions}, accepted {self.accepted_insertions}")
            print(f"Deletions: attempted {self.attempted_deletions}, accepted {self.accepted_deletions}")
            print(f"Displacements: attempted {self.attempted_displacements}, accepted {self.accepted_displacements}")

