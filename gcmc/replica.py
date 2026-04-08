import numpy as np
import logging
import os
import pickle
import time
from ase import Atoms
from .constants import ADSORBATE_TAG_OFFSET, KB_EV_PER_K
from .execution_backends import build_replica_backend
from .process_replica import _restore_atoms_from_snapshot
from .utils import generate_nonuniform_temperature_grid

logger = logging.getLogger("mc")


def _count_tagged_adsorbate_groups(atoms):
    tags = np.asarray(atoms.get_tags(), dtype=int)
    if len(tags) != len(atoms):
        return 0
    return int(np.sum(np.unique(tags) >= ADSORBATE_TAG_OFFSET))


def _count_move_units(atoms, mc_kwargs):
    n_adsorbates = _count_tagged_adsorbate_groups(atoms)
    if n_adsorbates > 0:
        return n_adsorbates
    swap_elements = mc_kwargs.get("swap_elements")
    if not swap_elements:
        return len(atoms)
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    return int(np.isin(symbols, list(swap_elements)).sum())


class ReplicaExchange:
    """
    Parallel tempering driver for MC engines.

    Notes:
    - Per-replica RNG streams are anchored to the temperature slot, not the
      physical configuration. After an accepted swap, the configuration moves to
      a new slot while each slot keeps its own RNG state.
    - Heat-capacity estimates use the population variance convention
      ``E[E^2] - E[E]^2`` (division by ``n`` rather than ``n-1``).
    """

    def __init__(
        self,
        n_gpus,
        workers_per_gpu,
        replica_states,
        swap_interval=10,
        report_interval=10,
        sampling_interval=1,
        checkpoint_interval=10,
        swap_stride=1,
        local_eq_fraction: float = 0.2,
        stats_file="replica_stats.csv",
        results_file="results.csv",
        checkpoint_file="pt_state.pkl",
        resume=False,
        worker_init_info=None,
        track_composition=None,
        seed: int = 67,
        seed_nonce: int = 0,
        execution_backend: str = "multiprocessing",
        backend_kwargs: dict = None,
    ):
        self.n_gpus = n_gpus
        self.workers_per_gpu = workers_per_gpu
        self.replica_states = replica_states
        self.swap_interval = swap_interval
        self.report_interval = report_interval
        self.sampling_interval = sampling_interval
        self.checkpoint_interval = checkpoint_interval
        self.swap_stride = swap_stride
        self.local_eq_fraction = float(local_eq_fraction)
        if not (0.0 <= self.local_eq_fraction <= 1.0):
            raise ValueError("local_eq_fraction must be in [0, 1].")
        self.worker_init_info = worker_init_info
        self.execution_backend = execution_backend
        self.backend_kwargs = backend_kwargs or {}
        self.seed = seed
        self.seed_nonce = seed_nonce
        self.rng = np.random.default_rng(seed)

        # Store elements to track (for example, ['Ti', 'Mo']).
        self.track_composition = track_composition if track_composition else []

        self.stats_file = stats_file
        self.results_file = results_file
        self.checkpoint_file = checkpoint_file
        self.cycle_start = 0
        self.has_adsorbate_observables = any(
            (
                _count_tagged_adsorbate_groups(state["atoms"]) > 0
                or "adsorbate" in state.get("mc_kwargs", {})
            )
            for state in self.replica_states
        )

        self.backend = build_replica_backend(
            execution_backend=self.execution_backend,
            n_gpus=self.n_gpus,
            workers_per_gpu=self.workers_per_gpu,
            worker_init_info=self.worker_init_info,
            backend_kwargs=self.backend_kwargs,
        )

        # Keep RNG streams deterministic and isolated per temperature slot,
        # independent of worker scheduling and collection order.
        for state in self.replica_states:
            if "rng_state" not in state or state["rng_state"] is None:
                rid = state.get("id", 0)
                state["rng_state"] = np.random.default_rng(
                    self.seed + self.seed_nonce + rid
                ).bit_generator.state

        if not os.path.exists(self.stats_file) and not resume:
            with open(self.stats_file, "w") as f:
                f.write("Cycle,T_i,T_j,E_i,E_j,Accepted\n")

        if not os.path.exists(self.results_file) and not resume:
            with open(self.results_file, "w") as f:
                header = (
                    "cycle,T_K,n_atoms,E_eV,E_eV_per_atom,"
                    "Cv_cycle_eV_per_K,Cv_cum_eV_per_K,acc_pct"
                )
                if self.has_adsorbate_observables:
                    header += ",n_adsorbates"
                for el in self.track_composition:
                    header += f",N_{el}_avg,chi_{el}"
                f.write(header + "\n")

        if resume and os.path.exists(self.checkpoint_file):
            self._load_master_checkpoint()

    def _start_workers(self):
        self.backend.start()

    @classmethod
    def from_auto_config(
        cls,
        atoms_template,
        T_start,
        T_end,
        T_step=None,
        *,
        calculator_class=None,
        mc_class=None,
        calc_kwargs=None,
        mc_kwargs=None,
        n_gpus=4,
        workers_per_gpu=2,
        swap_stride=1,
        resume=False,
        results_file="results.csv",
        track_composition=None,
        seed_nonce: int = 0,
        n_replicas: int = None,
        fine_grid_temps=None,
        fine_grid_weights=None,
        fine_grid_strength: float = 4.0,
        fine_grid_width: float = None,
        grid_space: str = "temperature",
        execution_backend: str = "multiprocessing",
        backend_kwargs: dict = None,
        **pt_kwargs,
    ):

        if calculator_class is None:
            raise ValueError("calculator_class must be provided.")
        if mc_class is None:
            raise ValueError("mc_class must be provided.")
        if calc_kwargs is None:
            calc_kwargs = {}
        if mc_kwargs is None:
            mc_kwargs = {}
        pt_mc_kwargs = dict(mc_kwargs)
        if "checkpoint_interval" not in pt_mc_kwargs:
            pt_mc_kwargs["checkpoint_interval"] = 0
            logger.info(
                "Replica workers: checkpoint_interval not set in mc_kwargs; defaulting worker checkpointing to 0 (master PT checkpoint remains active)."
            )

        if n_replicas is not None:
            temps = generate_nonuniform_temperature_grid(
                T_start=T_start,
                T_end=T_end,
                n_replicas=n_replicas,
                focus_temps=fine_grid_temps,
                focus_weights=fine_grid_weights,
                focus_strength=fine_grid_strength,
                focus_width=fine_grid_width,
                grid_space=grid_space,
            )
        else:
            if T_step is None or np.isclose(T_step, 0.0):
                raise ValueError("T_step must be non-zero when n_replicas is not set.")
            if T_start > T_end:
                temps = np.arange(
                    T_start, T_end - abs(T_step) / 2, -abs(T_step)
                ).tolist()
            else:
                temps = np.arange(
                    T_start, T_end + abs(T_step) / 2, abs(T_step)
                ).tolist()

        logger.info(
            f"Configuration: {len(temps)} Replicas | {n_gpus} GPUs | {workers_per_gpu} Workers/GPU"
        )

        if isinstance(atoms_template, Atoms):
            initial_structures = [atoms_template.copy() for _ in temps]
        else:
            try:
                initial_structures = [atoms.copy() for atoms in atoms_template]
            except TypeError as exc:
                raise TypeError(
                    "atoms_template must be an ASE Atoms object or a sequence of ASE Atoms."
                ) from exc
            if not initial_structures:
                raise ValueError("atoms_template sequence must not be empty.")
            if len(initial_structures) == 1 and len(temps) > 1:
                initial_structures = [initial_structures[0].copy() for _ in temps]
            elif len(initial_structures) != len(temps):
                raise ValueError(
                    "When atoms_template is a sequence, its length must match the number "
                    "of replicas (or be length 1)."
                )
            if not all(isinstance(atoms, Atoms) for atoms in initial_structures):
                raise TypeError(
                    "All entries in atoms_template must be ASE Atoms objects."
                )

        n_atoms_reference = len(initial_structures[0])
        if any(len(atoms) != n_atoms_reference for atoms in initial_structures):
            raise ValueError(
                "All initial replica structures must contain the same number of atoms."
            )

        replica_states = []
        for i, T in enumerate(temps):
            t_str = f"{T:.0f}"
            atoms_i = initial_structures[i].copy()
            atoms_i.calc = None
            state = {
                "id": i,
                "T": T,
                "atoms": atoms_i,
                "e_old": 0.0,
                "sweep": 0,
                # Cumulative counters kept in memory for logging.
                "cum_sum_E": 0.0,
                "cum_sum_E_sq": 0.0,
                "cum_n_samples": 0,
                "traj_file": f"replica_{t_str}K.traj",
                "thermo_file": f"replica_{t_str}K.dat",
                "checkpoint_file": f"checkpoint_{t_str}K.pkl",
                "mc_kwargs": pt_mc_kwargs,
            }
            replica_states.append(state)

        worker_init_info = {
            "calculator_module": calculator_class.__module__,
            "calculator_class_name": calculator_class.__name__,
            "mc_module": mc_class.__module__,
            "mc_class": mc_class.__name__,
            "calc_kwargs": calc_kwargs,
            "mc_kwargs": pt_mc_kwargs,
            "atoms_template": initial_structures[0].copy(),
        }

        return cls(
            n_gpus,
            workers_per_gpu,
            replica_states,
            worker_init_info=worker_init_info,
            swap_stride=swap_stride,
            resume=resume,
            results_file=results_file,
            track_composition=track_composition,
            seed_nonce=seed_nonce,
            execution_backend=execution_backend,
            backend_kwargs=backend_kwargs,
            **pt_kwargs,
        )

    def run(self, n_cycles, equilibration_cycles=0):
        self._start_workers()
        logger.info(f"Starting PT Loop: Cycles {self.cycle_start} -> {n_cycles}")
        kB = KB_EV_PER_K

        try:
            for cycle in range(self.cycle_start, n_cycles):
                logger.info(f"--- PT Cycle {cycle+1}/{n_cycles} ---")

                is_equilibrating = cycle < equilibration_cycles
                nsweeps = self.swap_interval
                cycle_start_sweep = cycle * self.swap_interval

                # Local equilibration after swaps.
                local_eq_steps = int(nsweeps * self.local_eq_fraction)
                eq_steps = nsweeps if is_equilibrating else local_eq_steps

                t_start = time.time()
                total_atom_trial_moves_in_cycle = 0
                total_move_units_in_cycle = 0

                # A. Submit tasks.
                for state in self.replica_states:
                    atoms = state["atoms"]
                    n_atoms = len(atoms)
                    n_move_units = _count_move_units(
                        atoms, state.get("mc_kwargs", {})
                    )
                    total_move_units_in_cycle += n_move_units * nsweeps
                    total_atom_trial_moves_in_cycle += n_atoms * n_move_units * nsweeps

                    task_data = {
                        "T": state["T"],
                        "positions": atoms.get_positions(),
                        "numbers": atoms.get_atomic_numbers(),
                        "tags": atoms.get_tags(),
                        "cell": atoms.get_cell(),
                        "pbc": atoms.get_pbc(),
                        "e_old": state["e_old"],
                        "rng_state": state["rng_state"],
                        "sweep": cycle_start_sweep,
                        "nsweeps": nsweeps,
                        "traj_file": state["traj_file"],
                        "thermo_file": state["thermo_file"],
                        "checkpoint_file": state["checkpoint_file"],
                        "report_interval": self.report_interval,
                        "sample_interval": self.sampling_interval,
                        "eq_steps": eq_steps,
                    }
                    target_gpu = state["id"] % self.n_gpus
                    self.backend.submit(
                        replica_id=state["id"],
                        target_gpu=target_gpu,
                        task_data=task_data,
                    )

                # B. Collect results.
                completed = 0
                cycle_log_rows = []
                while completed < len(self.replica_states):
                    res = self.backend.get_result()
                    if isinstance(res, tuple) and res[0] == "ERROR":
                        raise RuntimeError(f"Worker Error: {res[1]}")

                    rid = res["replica_id"]
                    state = self.replica_states[rid]

                    # 1. Update state.
                    state["atoms"].set_positions(res["positions"])
                    state["atoms"].set_atomic_numbers(res["numbers"])
                    state["atoms"].set_tags(res["tags"])
                    state["atoms"].set_cell(res["cell"])
                    state["atoms"].pbc = res["pbc"]
                    state["e_old"] = res["e_old"]
                    state["rng_state"] = res["rng_state"]
                    state["sweep"] = res["sweep"]

                    # 2. Cycle stats from worker.
                    cycle_stats = res["local_stats"]
                    cycle_E = cycle_stats["energy"]
                    cycle_Cv = cycle_stats.get("cv", 0.0)
                    acc = cycle_stats["acceptance"]
                    n_adsorbates = cycle_stats.get("n_adsorbates")

                    # 3. Update cumulative stats in memory.
                    if not is_equilibrating:
                        state["cum_sum_E"] += res["cycle_sum_E"]
                        state["cum_sum_E_sq"] += res["cycle_sum_E_sq"]
                        state["cum_n_samples"] += res["cycle_n_samples"]

                    N = state["cum_n_samples"]
                    if N > 1:
                        cum_avg_E = state["cum_sum_E"] / N
                        # Use the population-variance convention for consistency
                        # with the per-cycle Cv estimator returned by the workers.
                        cum_var = (state["cum_sum_E_sq"] / N) - (cum_avg_E**2)
                        cum_Cv = cum_var / (kB * state["T"] ** 2)
                    else:
                        cum_Cv = 0.0

                    # 4. Write CSV with both local-cycle and cumulative Cv estimators.
                    n_atoms = len(state["atoms"])
                    e_per_atom = cycle_E / n_atoms if n_atoms > 0 else 0.0
                    line = (
                        f"{cycle+1},"
                        f"{state['T']:.6f},"
                        f"{n_atoms},"
                        f"{cycle_E:.6f},"
                        f"{e_per_atom:.8f},"
                        f"{cycle_Cv:.6f},"
                        f"{cum_Cv:.6f},"
                        f"{acc:.2f}"
                    )
                    if self.has_adsorbate_observables:
                        if n_adsorbates is None:
                            line += ","
                        else:
                            line += f",{int(n_adsorbates)}"

                    if self.track_composition and "composition" in cycle_stats:
                        for el in self.track_composition:
                            comp = cycle_stats["composition"].get(el, 0.0)
                            susc = cycle_stats["susceptibility"].get(el, 0.0)
                            line += f",{comp:.4f},{susc:.4f}"

                    with open(self.results_file, "a") as f:
                        f.write(line + "\n")

                    cycle_log_rows.append(
                        {
                            "rid": rid,
                            "T": state["T"],
                            "E": cycle_E,
                            "e_per_atom": e_per_atom,
                            "cv_cycle": cycle_Cv,
                            "cv_cum": cum_Cv,
                            "acc": acc,
                            "n_adsorbates": n_adsorbates,
                        }
                    )

                    completed += 1

                # Keep driver-side PT reporting available across all execution backends.
                if self.report_interval > 0 and (
                    (cycle + 1) % self.report_interval == 0
                    or cycle == self.cycle_start
                    or cycle == (n_cycles - 1)
                ):
                    rows_sorted = sorted(cycle_log_rows, key=lambda row: row["T"])
                    for row in rows_sorted:
                        logger.info(
                            "[Replica %02d] T=%7.3f K | E=%12.6f eV | E/N=%10.6f eV | "
                            "Cv(cycle)=%10.6f | Cv(cum)=%10.6f | Acc=%6.2f%%%s",
                            row["rid"],
                            row["T"],
                            row["E"],
                            row["e_per_atom"],
                            row["cv_cycle"],
                            row["cv_cum"],
                            row["acc"],
                            (
                                f" | Nads={int(row['n_adsorbates']):4d}"
                                if row["n_adsorbates"] is not None
                                else ""
                            ),
                        )

                t_end = time.time()
                duration = t_end - t_start
                move_unit_rate = (
                    total_move_units_in_cycle / duration if duration > 0 else 0.0
                )
                speed = (
                    total_atom_trial_moves_in_cycle / duration / 1.0e6
                    if duration > 0
                    else 0
                )
                logger.info(
                    f"[Timing] {duration:.2f}s | {speed:.2e} Matom-trial-moves/s | "
                    f"{move_unit_rate:.2e} move-proposals/s"
                )

                self._attempt_swaps(cycle)

                if (
                    self.checkpoint_interval > 0
                    and (cycle + 1) % self.checkpoint_interval == 0
                ):
                    self._save_master_checkpoint(cycle + 1)

            if (
                self.checkpoint_interval > 0
                and (n_cycles == 0 or n_cycles % self.checkpoint_interval != 0)
            ):
                self._save_master_checkpoint(n_cycles)

        finally:
            self.stop()
            logger.info("PT Completed.")

    def _attempt_swaps(self, cycle):
        kB = KB_EV_PER_K
        stride = self.swap_stride
        n = len(self.replica_states)
        phase = self.rng.integers(0, stride)
        is_odd_cycle = cycle % 2 == 1
        start_idx = phase + (stride if is_odd_cycle else 0)

        for i in range(start_idx, n - stride, 2 * stride):
            j = i + stride
            s_i = self.replica_states[i]
            s_j = self.replica_states[j]
            e_i_before = s_i["e_old"]
            e_j_before = s_j["e_old"]
            delta = (1.0 / (kB * s_i["T"]) - 1.0 / (kB * s_j["T"])) * (
                e_i_before - e_j_before
            )
            accepted = False
            if delta > 0 or self.rng.random() < np.exp(delta):
                accepted = True
                # Swap the sampled configuration and cached observables only;
                # the RNG state intentionally stays with the temperature slot.
                s_i["atoms"], s_j["atoms"] = s_j["atoms"], s_i["atoms"]
                s_i["e_old"], s_j["e_old"] = s_j["e_old"], s_i["e_old"]
                logger.info(f"  [Swap] {s_i['T']:.0f}K <-> {s_j['T']:.0f}K | ACCEPTED")
            with open(self.stats_file, "a") as f:
                f.write(
                    f"{cycle},{s_i['T']},{s_j['T']},{e_i_before:.4f},{e_j_before:.4f},{accepted}\n"
                )

    def _save_master_checkpoint(self, cycle):
        replica_snapshots = []
        for state in self.replica_states:
            snapshot = {
                "id": state["id"],
                "e_old": state["e_old"],
                "cum_sum_E": state["cum_sum_E"],
                "cum_sum_E_sq": state["cum_sum_E_sq"],
                "cum_n_samples": state["cum_n_samples"],
                "rng_state": state["rng_state"],
                "positions": state["atoms"].get_positions(),
                "numbers": state["atoms"].get_atomic_numbers(),
                "tags": state["atoms"].get_tags(),
                "cell": state["atoms"].get_cell(),
                "pbc": state["atoms"].get_pbc(),
            }
            replica_snapshots.append(snapshot)
        data = {
            "cycle": cycle,
            "rng_state": self.rng.bit_generator.state,
            "seed_nonce": self.seed_nonce,
            "replica_snapshots": replica_snapshots,
        }
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Checkpoint cycle {cycle}")

    def _load_master_checkpoint(self):
        with open(self.checkpoint_file, "rb") as f:
            data = pickle.load(f)
            self.cycle_start = data.get("cycle", 0)
            rng_state = data.get("rng_state")
            if rng_state is not None:
                self.rng.bit_generator.state = rng_state
            self.seed_nonce = data.get("seed_nonce", self.seed_nonce)
            saved_snapshots = data.get("replica_snapshots", [])
            logger.info(f"Resuming from Cycle {self.cycle_start}")
            for s_data in saved_snapshots:
                rid = s_data["id"]
                if rid < len(self.replica_states):
                    state = self.replica_states[rid]
                    state["e_old"] = s_data["e_old"]
                    state["cum_sum_E"] = s_data["cum_sum_E"]
                    state["cum_sum_E_sq"] = s_data["cum_sum_E_sq"]
                    state["cum_n_samples"] = s_data["cum_n_samples"]
                    state["rng_state"] = s_data.get("rng_state")
                    if state["rng_state"] is None:
                        state["rng_state"] = np.random.default_rng(
                            self.seed + self.seed_nonce + rid
                        ).bit_generator.state
                    state["atoms"].set_positions(s_data["positions"])
                    state["atoms"].set_atomic_numbers(s_data["numbers"])
                    if "tags" in s_data:
                        state["atoms"].set_tags(s_data["tags"])
                    state["atoms"].set_cell(s_data["cell"])
                    state["atoms"].pbc = s_data["pbc"]

    def stop(self):
        self.backend.stop()


class MuReplicaExchange:
    """
    Fixed-temperature replica exchange in chemical-potential space for GCMC.

    Notes:
    - Each replica slot is defined by a fixed chemical potential ``mu`` and a
      common temperature ``T``.
    - On accepted swaps, the sampled configurations move between ``mu`` slots;
      RNG state and output files remain attached to the slot.
    """

    def __init__(
        self,
        n_gpus,
        workers_per_gpu,
        replica_states,
        *,
        swap_interval=10,
        swap_stride=1,
        report_interval=1,
        checkpoint_interval=10,
        stats_file="mu_exchange_stats.csv",
        results_file="mu_exchange_results.csv",
        checkpoint_file="mu_exchange_state.pkl",
        resume=False,
        worker_init_info=None,
        seed: int = 67,
        seed_nonce: int = 0,
        execution_backend: str = "multiprocessing",
        backend_kwargs: dict = None,
    ):
        self.n_gpus = int(n_gpus)
        self.workers_per_gpu = int(workers_per_gpu)
        self.replica_states = replica_states
        self.swap_interval = int(swap_interval)
        self.swap_stride = int(swap_stride)
        self.report_interval = int(report_interval)
        self.checkpoint_interval = int(checkpoint_interval)
        self.worker_init_info = worker_init_info
        self.execution_backend = execution_backend
        self.backend_kwargs = backend_kwargs or {}
        self.stats_file = stats_file
        self.results_file = results_file
        self.checkpoint_file = checkpoint_file
        self.seed = seed
        self.seed_nonce = seed_nonce
        self.rng = np.random.default_rng(seed)
        self.cycle_start = 0

        if self.swap_interval < 1:
            raise ValueError("swap_interval must be >= 1.")
        if self.swap_stride < 1:
            raise ValueError("swap_stride must be >= 1.")

        self.backend = build_replica_backend(
            execution_backend=self.execution_backend,
            n_gpus=self.n_gpus,
            workers_per_gpu=self.workers_per_gpu,
            worker_init_info=self.worker_init_info,
            backend_kwargs=self.backend_kwargs,
        )

        for state in self.replica_states:
            state.setdefault("cum_sum_E", 0.0)
            state.setdefault("cum_sum_E_sq", 0.0)
            state.setdefault("cum_sum_N", 0.0)
            state.setdefault("cum_sum_N_sq", 0.0)
            state.setdefault("cum_n_samples", 0)
            state.setdefault("cum_n_hist", {})
            state.setdefault("cum_insert_attempted", 0)
            state.setdefault("cum_insert_accepted", 0)
            state.setdefault("cum_delete_attempted", 0)
            state.setdefault("cum_delete_accepted", 0)
            state.setdefault("cum_canonical_attempted", 0)
            state.setdefault("cum_canonical_accepted", 0)
            state.setdefault("cum_md_attempted", 0)
            state.setdefault("cum_md_accepted", 0)
            state.setdefault("cum_total_attempted", 0)
            state.setdefault("cum_total_accepted", 0)
            state.setdefault("sweep", 0)
            if "rng_state" not in state or state["rng_state"] is None:
                rid = state.get("id", 0)
                state["rng_state"] = np.random.default_rng(
                    self.seed + self.seed_nonce + rid
                ).bit_generator.state

        if not os.path.exists(self.stats_file) and not resume:
            with open(self.stats_file, "w") as handle:
                handle.write("cycle,mu_i,mu_j,N_i,N_j,E_i,E_j,accepted\n")

        if not os.path.exists(self.results_file) and not resume:
            with open(self.results_file, "w") as handle:
                handle.write(
                    "cycle,mu_eV,n_atoms,E_eV,Cv_cycle_eV_per_K,"
                    "Cv_cum_eV_per_K,acc_pct,n_adsorbates,n_adsorbates_avg\n"
                )

        if resume and os.path.exists(self.checkpoint_file):
            self._load_master_checkpoint()

    def _start_workers(self):
        self.backend.start()

    def stop(self):
        self.backend.stop()

    def run(self, total_sweeps: int, equilibration_sweeps: int = 0) -> list[dict]:
        total_sweeps = int(total_sweeps)
        equilibration_sweeps = int(equilibration_sweeps)
        if total_sweeps < 0:
            raise ValueError("total_sweeps must be >= 0.")
        if equilibration_sweeps < 0:
            raise ValueError("equilibration_sweeps must be >= 0.")

        n_cycles = (
            0 if total_sweeps == 0 else int(np.ceil(total_sweeps / self.swap_interval))
        )
        self._start_workers()
        logger.info(
            "Starting mu-exchange GCMC: cycles %d -> %d | swap_interval=%d sweeps",
            self.cycle_start,
            n_cycles,
            self.swap_interval,
        )

        try:
            for cycle in range(self.cycle_start, n_cycles):
                cycle_start_sweep = cycle * self.swap_interval
                nsweeps = min(self.swap_interval, total_sweeps - cycle_start_sweep)
                if nsweeps <= 0:
                    break

                logger.info("--- Mu-Exchange Cycle %d/%d ---", cycle + 1, n_cycles)
                t_start = time.time()
                completed = 0
                cycle_log_rows = []

                for state in self.replica_states:
                    eq_steps = max(
                        0,
                        min(nsweeps, equilibration_sweeps - cycle_start_sweep),
                    )
                    atoms = state["atoms"]
                    task_data = {
                        "T": state["T"],
                        "mu": state["mu"],
                        "positions": atoms.get_positions(),
                        "numbers": atoms.get_atomic_numbers(),
                        "tags": atoms.get_tags(),
                        "cell": atoms.get_cell(),
                        "pbc": atoms.get_pbc(),
                        "e_old": state["e_old"],
                        "rng_state": state["rng_state"],
                        "sweep": state["sweep"],
                        "nsweeps": nsweeps,
                        "traj_file": state["traj_file"],
                        "thermo_file": state["thermo_file"],
                        "checkpoint_file": state["checkpoint_file"],
                        "report_interval": state.get("write_interval", 10),
                        "sample_interval": state.get("sample_interval", 1),
                        "eq_steps": eq_steps,
                    }
                    target_gpu = state["id"] % self.n_gpus
                    self.backend.submit(
                        replica_id=state["id"],
                        target_gpu=target_gpu,
                        task_data=task_data,
                    )

                while completed < len(self.replica_states):
                    res = self.backend.get_result()
                    if isinstance(res, tuple) and res[0] == "ERROR":
                        raise RuntimeError(f"Worker Error: {res[1]}")

                    rid = res["replica_id"]
                    state = self.replica_states[rid]
                    if len(state["atoms"]) != len(res["positions"]):
                        state["atoms"] = _restore_atoms_from_snapshot(
                            self.worker_init_info["atoms_template"],
                            res["positions"],
                            res["numbers"],
                            res["cell"],
                            res["pbc"],
                            tags=res.get("tags"),
                        )
                    else:
                        state["atoms"].set_positions(res["positions"])
                        state["atoms"].set_atomic_numbers(res["numbers"])
                        state["atoms"].set_tags(res["tags"])
                        state["atoms"].set_cell(res["cell"])
                        state["atoms"].pbc = res["pbc"]
                    state["e_old"] = res["e_old"]
                    state["rng_state"] = res["rng_state"]
                    state["sweep"] = res["sweep"]

                    state["cum_sum_E"] += res.get("cycle_sum_E", 0.0)
                    state["cum_sum_E_sq"] += res.get("cycle_sum_E_sq", 0.0)
                    state["cum_sum_N"] += res.get("cycle_sum_N", 0.0)
                    state["cum_sum_N_sq"] += res.get("cycle_sum_N_sq", 0.0)
                    state["cum_n_samples"] += res.get("cycle_n_samples", 0)

                    local_stats = res["local_stats"]
                    for key, target in (
                        ("insert_attempted", "cum_insert_attempted"),
                        ("insert_accepted", "cum_insert_accepted"),
                        ("delete_attempted", "cum_delete_attempted"),
                        ("delete_accepted", "cum_delete_accepted"),
                        ("canonical_attempted", "cum_canonical_attempted"),
                        ("canonical_accepted", "cum_canonical_accepted"),
                        ("md_attempted", "cum_md_attempted"),
                        ("md_accepted", "cum_md_accepted"),
                    ):
                        state[target] += int(local_stats.get(key, 0))

                    attempted_this_cycle = (
                        int(local_stats.get("insert_attempted", 0))
                        + int(local_stats.get("delete_attempted", 0))
                        + int(local_stats.get("canonical_attempted", 0))
                        + int(local_stats.get("md_attempted", 0))
                    )
                    accepted_this_cycle = (
                        int(local_stats.get("insert_accepted", 0))
                        + int(local_stats.get("delete_accepted", 0))
                        + int(local_stats.get("canonical_accepted", 0))
                        + int(local_stats.get("md_accepted", 0))
                    )
                    state["cum_total_attempted"] += attempted_this_cycle
                    state["cum_total_accepted"] += accepted_this_cycle

                    for n_key, count in dict(local_stats.get("n_hist", {})).items():
                        bucket = int(n_key)
                        state["cum_n_hist"][bucket] = (
                            state["cum_n_hist"].get(bucket, 0) + int(count)
                        )

                    N = state["cum_n_samples"]
                    if N > 1:
                        avg_E = state["cum_sum_E"] / N
                        var_E = (state["cum_sum_E_sq"] / N) - (avg_E**2)
                        cum_cv = var_E / (KB_EV_PER_K * state["T"] ** 2)
                        avg_N = state["cum_sum_N"] / N
                    else:
                        cum_cv = 0.0
                        avg_N = float(local_stats.get("n_adsorbates_avg", 0.0))

                    with open(self.results_file, "a") as handle:
                        handle.write(
                            f"{cycle + 1},"
                            f"{state['mu']:.8f},"
                            f"{len(state['atoms'])},"
                            f"{float(local_stats['energy']):.10f},"
                            f"{float(local_stats.get('cv', 0.0)):.10f},"
                            f"{cum_cv:.10f},"
                            f"{float(local_stats.get('acceptance', 0.0)):.6f},"
                            f"{int(local_stats.get('n_adsorbates', 0))},"
                            f"{avg_N:.10f}\n"
                        )

                    cycle_log_rows.append(
                        {
                            "mu": state["mu"],
                            "E": float(local_stats["energy"]),
                            "cv_cycle": float(local_stats.get("cv", 0.0)),
                            "cv_cum": float(cum_cv),
                            "acc": float(local_stats.get("acceptance", 0.0)),
                            "N": int(local_stats.get("n_adsorbates", 0)),
                            "Navg": float(avg_N),
                        }
                    )
                    completed += 1

                if self.report_interval > 0 and (
                    (cycle + 1) % self.report_interval == 0
                    or cycle == self.cycle_start
                    or cycle == (n_cycles - 1)
                ):
                    for row in sorted(cycle_log_rows, key=lambda item: item["mu"]):
                        logger.info(
                            "[mu=%+.4f eV] E=%12.6f eV | Cv(cycle)=%10.6f | "
                            "Cv(cum)=%10.6f | Acc=%6.2f%% | N=%3d | Navg=%6.3f",
                            row["mu"],
                            row["E"],
                            row["cv_cycle"],
                            row["cv_cum"],
                            row["acc"],
                            row["N"],
                            row["Navg"],
                        )

                duration = time.time() - t_start
                logger.info("[Timing] %.2fs per mu-exchange cycle", duration)
                self._attempt_swaps(cycle)

                if (
                    self.checkpoint_interval > 0
                    and (cycle + 1) % self.checkpoint_interval == 0
                ):
                    self._save_master_checkpoint(cycle + 1)

            if (
                self.checkpoint_interval > 0
                and (n_cycles == 0 or n_cycles % self.checkpoint_interval != 0)
            ):
                self._save_master_checkpoint(n_cycles)
        finally:
            self.stop()
            logger.info("Mu-exchange GCMC completed.")

        return self._final_stats()

    def _attempt_swaps(self, cycle):
        beta = 1.0 / (KB_EV_PER_K * float(self.replica_states[0]["T"]))
        stride = self.swap_stride
        n = len(self.replica_states)
        phase = self.rng.integers(0, stride)
        is_odd_cycle = cycle % 2 == 1
        start_idx = phase + (stride if is_odd_cycle else 0)

        for i in range(start_idx, n - stride, 2 * stride):
            j = i + stride
            s_i = self.replica_states[i]
            s_j = self.replica_states[j]
            n_i = _count_tagged_adsorbate_groups(s_i["atoms"])
            n_j = _count_tagged_adsorbate_groups(s_j["atoms"])
            e_i = s_i["e_old"]
            e_j = s_j["e_old"]
            delta = beta * (float(s_i["mu"]) - float(s_j["mu"])) * (n_i - n_j)
            accepted = False
            if delta <= 0.0 or self.rng.random() < np.exp(-delta):
                accepted = True
                s_i["atoms"], s_j["atoms"] = s_j["atoms"], s_i["atoms"]
                s_i["e_old"], s_j["e_old"] = s_j["e_old"], s_i["e_old"]
                logger.info(
                    "  [Swap] mu=%+.4f <-> mu=%+.4f | N=(%d,%d) | ACCEPTED",
                    s_i["mu"],
                    s_j["mu"],
                    n_i,
                    n_j,
                )
            with open(self.stats_file, "a") as handle:
                handle.write(
                    f"{cycle},{float(s_i['mu']):.8f},{float(s_j['mu']):.8f},"
                    f"{n_i},{n_j},{e_i:.10f},{e_j:.10f},{accepted}\n"
                )

    def _save_master_checkpoint(self, cycle):
        replica_snapshots = []
        for state in self.replica_states:
            replica_snapshots.append(
                {
                    "id": state["id"],
                    "mu": state["mu"],
                    "T": state["T"],
                    "e_old": state["e_old"],
                    "cum_sum_E": state["cum_sum_E"],
                    "cum_sum_E_sq": state["cum_sum_E_sq"],
                    "cum_sum_N": state["cum_sum_N"],
                    "cum_sum_N_sq": state["cum_sum_N_sq"],
                    "cum_n_samples": state["cum_n_samples"],
                    "cum_n_hist": dict(state["cum_n_hist"]),
                    "cum_insert_attempted": state["cum_insert_attempted"],
                    "cum_insert_accepted": state["cum_insert_accepted"],
                    "cum_delete_attempted": state["cum_delete_attempted"],
                    "cum_delete_accepted": state["cum_delete_accepted"],
                    "cum_canonical_attempted": state["cum_canonical_attempted"],
                    "cum_canonical_accepted": state["cum_canonical_accepted"],
                    "cum_md_attempted": state["cum_md_attempted"],
                    "cum_md_accepted": state["cum_md_accepted"],
                    "cum_total_attempted": state["cum_total_attempted"],
                    "cum_total_accepted": state["cum_total_accepted"],
                    "rng_state": state["rng_state"],
                    "sweep": state["sweep"],
                    "positions": state["atoms"].get_positions(),
                    "numbers": state["atoms"].get_atomic_numbers(),
                    "tags": state["atoms"].get_tags(),
                    "cell": state["atoms"].get_cell(),
                    "pbc": state["atoms"].get_pbc(),
                }
            )

        payload = {
            "cycle_start": cycle,
            "rng_state": self.rng.bit_generator.state,
            "replica_states": replica_snapshots,
        }
        with open(self.checkpoint_file, "wb") as handle:
            pickle.dump(payload, handle)

    def _load_master_checkpoint(self):
        with open(self.checkpoint_file, "rb") as handle:
            payload = pickle.load(handle)
        self.cycle_start = int(payload.get("cycle_start", 0))
        if payload.get("rng_state") is not None:
            self.rng.bit_generator.state = payload["rng_state"]

        snapshots = payload.get("replica_states", [])
        state_by_id = {int(state["id"]): state for state in self.replica_states}
        for snap in snapshots:
            state = state_by_id[int(snap["id"])]
            state["mu"] = float(snap["mu"])
            state["T"] = float(snap["T"])
            state["e_old"] = float(snap["e_old"])
            state["cum_sum_E"] = float(snap.get("cum_sum_E", 0.0))
            state["cum_sum_E_sq"] = float(snap.get("cum_sum_E_sq", 0.0))
            state["cum_sum_N"] = float(snap.get("cum_sum_N", 0.0))
            state["cum_sum_N_sq"] = float(snap.get("cum_sum_N_sq", 0.0))
            state["cum_n_samples"] = int(snap.get("cum_n_samples", 0))
            state["cum_n_hist"] = {
                int(key): int(value)
                for key, value in dict(snap.get("cum_n_hist", {})).items()
            }
            for key in (
                "cum_insert_attempted",
                "cum_insert_accepted",
                "cum_delete_attempted",
                "cum_delete_accepted",
                "cum_canonical_attempted",
                "cum_canonical_accepted",
                "cum_md_attempted",
                "cum_md_accepted",
                "cum_total_attempted",
                "cum_total_accepted",
            ):
                state[key] = int(snap.get(key, 0))
            state["rng_state"] = snap.get("rng_state")
            state["sweep"] = int(snap.get("sweep", 0))
            if len(state["atoms"]) != len(snap["positions"]):
                state["atoms"] = _restore_atoms_from_snapshot(
                    self.worker_init_info["atoms_template"],
                    snap["positions"],
                    snap["numbers"],
                    snap["cell"],
                    snap["pbc"],
                    tags=snap.get("tags"),
                )
            else:
                atoms = state["atoms"]
                atoms.set_positions(snap["positions"])
                atoms.set_atomic_numbers(snap["numbers"])
                atoms.set_tags(snap["tags"])
                atoms.set_cell(snap["cell"])
                atoms.pbc = snap["pbc"]

    def _final_stats(self) -> list[dict]:
        rows = []
        for state in self.replica_states:
            n_samples = int(state["cum_n_samples"])
            current_e = 0.0 if state["e_old"] is None else float(state["e_old"])
            avg_E = (
                float(state["cum_sum_E"]) / n_samples if n_samples else current_e
            )
            avg_N = (
                float(state["cum_sum_N"]) / n_samples
                if n_samples
                else float(_count_tagged_adsorbate_groups(state["atoms"]))
            )
            cv = 0.0
            if n_samples > 1:
                var_E = (state["cum_sum_E_sq"] / n_samples) - (avg_E**2)
                cv = var_E / (KB_EV_PER_K * float(state["T"]) ** 2)
            acceptance = (
                100.0 * state["cum_total_accepted"] / state["cum_total_attempted"]
                if state["cum_total_attempted"]
                else 0.0
            )
            rows.append(
                {
                    "mu": float(state["mu"]),
                    "T": float(state["T"]),
                    "energy": avg_E,
                    "cv": cv,
                    "acceptance": acceptance,
                    "n_adsorbates": _count_tagged_adsorbate_groups(state["atoms"]),
                    "n_adsorbates_avg": avg_N,
                    "n_hist": dict(sorted(state["cum_n_hist"].items())),
                    "insert_attempted": int(state["cum_insert_attempted"]),
                    "insert_accepted": int(state["cum_insert_accepted"]),
                    "delete_attempted": int(state["cum_delete_attempted"]),
                    "delete_accepted": int(state["cum_delete_accepted"]),
                    "canonical_attempted": int(state["cum_canonical_attempted"]),
                    "canonical_accepted": int(state["cum_canonical_accepted"]),
                    "md_attempted": int(state["cum_md_attempted"]),
                    "md_accepted": int(state["cum_md_accepted"]),
                }
            )
        rows.sort(key=lambda row: row["mu"])
        return rows
