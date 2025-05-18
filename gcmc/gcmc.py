import os
import pickle
import numpy as np
import random
from ase import Atom
from ase.optimize import LBFGS, FIRE
from ase.io import write, read
from ase.io.trajectory import Trajectory

class MultiLayerSiteGCMC:
    """
    Grand Canonical Monte Carlo (GCMC) class for multilayer adsorption of Cu
    on surface sites, with checkpoint/restart support.
    """

    def __init__(
        self,
        atoms,
        calculator,
        mu,
        T,
        all_sites,
        element='Cu',
        nsteps=1000,
        relax=False,
        relax_steps=20,
        traj_file='gcmc.traj',
        checkpoint_traj='gcmc_checkpoint.traj',
        checkpoint_data='gcmc_checkpoint.pkl',
        checkpoint_interval=1000,
        seed=81
    ):
        """
        Initialize the GCMC simulation.

        Parameters
        ----------
        atoms : ase.Atoms
            The starting structure.
        calculator : ASE calculator
            Calculator for energy/force evaluation.
        mu : float
            Chemical potential (eV).
        T : float
            Temperature (K).
        all_sites : array-like, shape (n_sites, 3)
            List of all possible adsorption site positions.
        element : str
            Adsorbate symbol (e.g. 'Cu').
        nsteps : int
            Number of GCMC steps to perform (including any previously run).
        relax : bool
            If True, relaxes after each MC move.
        relax_steps : int
            Number of relaxation steps.
        traj_file : str
            Output trajectory file.
        checkpoint_traj : str
            Path to checkpoint structure.
        checkpoint_data : str
            Path to checkpoint MC state.
        checkpoint_interval : int
            Steps between automatic checkpoint saves.
        """
        self.atoms = atoms
        self.calculator = calculator
        self.mu = mu
        self.T = T
        self.beta = 1 / (8.617333e-5 * T)
        self.all_sites = np.array(all_sites)
        self.element = element
        self.nsteps = nsteps
        self.relax = relax
        self.relax_steps = relax_steps
        self.traj_file = traj_file
        self.checkpoint_traj = checkpoint_traj
        self.checkpoint_data = checkpoint_data
        self.checkpoint_interval = checkpoint_interval
        self.rng = random.Random(seed)
        self.seed = seed

        self.pillar_xy = np.unique(np.round(self.all_sites[:, :2], 3), axis=0)

        # MC counters
        self.attempted_insertions = 0
        self.accepted_insertions = 0
        self.attempted_deletions = 0
        self.accepted_deletions = 0

        self.step = 0  # will be incremented as we go

        # Are we resuming?
        if os.path.exists(self.checkpoint_traj) and os.path.exists(self.checkpoint_data):
            self._load_checkpoint()
            print(f"Resuming GCMC from checkpoint at step {self.step}")
        else:
            self.atoms.calc = self.calculator
            self.e_old = self.atoms.get_potential_energy()

    def site_occupied(self, site):
        for atom in self.atoms:
            if atom.symbol == self.element and np.linalg.norm(atom.position - site) < 0.5:
                return True
        return False

    def get_pillar_sites(self):
        pillars = []
        for xy in self.pillar_xy:
            mask = np.all(np.isclose(self.all_sites[:, :2], xy, atol=1e-3), axis=1)
            pillar = self.all_sites[mask]
            pillar = pillar[pillar[:, 2].argsort()]
            pillars.append(pillar)
        return pillars

    def highest_occupied_layer(self, pillar):
        for l in reversed(range(len(pillar))):
            if self.site_occupied(pillar[l]):
                return l
        return -1

    def attempt_insertion(self):
        pillars = self.get_pillar_sites()
        possible = []
        for pillar in pillars:
            l = self.highest_occupied_layer(pillar)
            if l + 1 < len(pillar):
                site = pillar[l + 1]
                if not self.site_occupied(site):
                    possible.append(site)
        if not possible:
            return None
        site = self.rng.choice(possible)
        atoms_new = self.atoms.copy()
        atoms_new.append(Atom(self.element, site))
        return atoms_new

    def attempt_deletion(self):
        pillars = self.get_pillar_sites()
        possible = []
        for pillar in pillars:
            l = self.highest_occupied_layer(pillar)
            if l >= 0:
                site = pillar[l]
                possible.append(site)
        if not possible:
            return None
        site = self.rng.choice(possible)
        atoms_new = self.atoms.copy()
        for i, atom in enumerate(atoms_new):
            if atom.symbol == self.element and np.linalg.norm(atom.position - site) < 0.5:
                del atoms_new[i]
                break
        return atoms_new

    def relax_structure(self, atoms):
        if self.relax:
            dyn = LBFGS(atoms, logfile=None)
            dyn.run(fmax=0.01, steps=self.relax_steps)
        return atoms

    def metropolis_accept(self, e_new, move_type):
        n_old = len([a for a in self.atoms if a.symbol == self.element])
        n_new = n_old + (1 if move_type == 'insert' else -1)
        delta_e = e_new - self.e_old
        if move_type == 'insert':
            tau = -self.beta * (delta_e - self.mu)
        elif move_type == 'delete':
            tau = -self.beta * (delta_e + self.mu)
        else:
            raise ValueError("move_type must be 'insert' or 'delete'")
        return np.exp(min(tau, 0)), tau

    def _save_checkpoint(self):
        """Save the current structure and MC counters/step."""
        write(self.checkpoint_traj, self.atoms)
        state = {
            'step': self.step,
            'e_old': self.e_old,
            'attempted_insertions': self.attempted_insertions,
            'accepted_insertions': self.accepted_insertions,
            'attempted_deletions': self.attempted_deletions,
            'accepted_deletions': self.accepted_deletions,
        }
        with open(self.checkpoint_data, 'wb') as f:
            pickle.dump(state, f)
        print(f"Checkpoint saved at step {self.step}.")

    def _load_checkpoint(self):
        """Load the structure and MC counters/step."""
        self.atoms = read(self.checkpoint_traj)
        self.atoms.calc = self.calculator
        with open(self.checkpoint_data, 'rb') as f:
            state = pickle.load(f)
        self.step = state['step']
        self.e_old = state['e_old']
        self.attempted_insertions = state['attempted_insertions']
        self.accepted_insertions = state['accepted_insertions']
        self.attempted_deletions = state['attempted_deletions']
        self.accepted_deletions = state['accepted_deletions']

    def run(self):
        """Run the GCMC simulation with checkpointing and robust trajectory output."""
        # Determine starting mode for trajectory file
        traj_mode = 'a' if os.path.exists(self.traj_file) else 'w'
        with Trajectory(self.traj_file, traj_mode) as traj:
            for s in range(self.step, self.nsteps):
                move_type = self.rng.choice(['insert', 'delete'])
                if move_type == 'insert':
                    self.attempted_insertions += 1
                    atoms_new = self.attempt_insertion()
                    if atoms_new is None:
                        continue
                else:
                    self.attempted_deletions += 1
                    atoms_new = self.attempt_deletion()
                    if atoms_new is None:
                        continue
                atoms_new.calc = self.calculator
                if self.relax:
                    atoms_new = self.relax_structure(atoms_new)
                e_new = atoms_new.get_potential_energy()
                prob, tau = self.metropolis_accept(e_new, move_type)
                if self.rng.random() < prob:
                    # Accept the move
                    self.atoms = atoms_new
                    self.e_old = e_new
                    if move_type == 'insert':
                        self.accepted_insertions += 1
                    else:
                        self.accepted_deletions += 1
                    print(f"Step {s:5d}: Accepted {move_type} (tau = {tau:8.2f}, E = {e_new:10.2f} eV)")
                else:
                    print(f"Step {s:5d}: Rejected {move_type} (tau = {tau:8.2f})")
                # Save snapshot (strip calculator, and only if not empty)
                traj.write(self.atoms, step=s, move=move_type)
                # Checkpoint every checkpoint_interval steps
                self.step = s + 1
                if (self.step % self.checkpoint_interval == 0) and (self.step > 0):
                    self._save_checkpoint()
            # Final checkpoint on normal exit
            self._save_checkpoint()
            print("GCMC completed.")
            print(f"Insertions: attempted {self.attempted_insertions}, accepted {self.accepted_insertions}")
            print(f"Deletions: attempted {self.attempted_deletions}, accepted {self.accepted_deletions}")

