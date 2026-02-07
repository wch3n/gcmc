import numpy as np
import logging
import os
import pickle
import multiprocessing
import time
from .process_replica import ReplicaWorker, ctx
from .utils import generate_nonuniform_temperature_grid

logger = logging.getLogger("mc")


class ReplicaExchange:
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
        stats_file="replica_stats.csv",
        results_file="results.csv",
        checkpoint_file="pt_state.pkl",
        resume=False,
        worker_init_info=None,
        track_composition=None,
        seed: int = 67,
        seed_nonce: int = 0,
    ):
        self.n_gpus = n_gpus
        self.workers_per_gpu = workers_per_gpu
        self.replica_states = replica_states
        self.swap_interval = swap_interval
        self.report_interval = report_interval
        self.sampling_interval = sampling_interval
        self.checkpoint_interval = checkpoint_interval
        self.swap_stride = swap_stride
        self.worker_init_info = worker_init_info
        self.seed = seed
        self.seed_nonce = seed_nonce
        self.rng = np.random.default_rng(seed)

        # Store elements to track (for example, ['Ti', 'Mo']).
        self.track_composition = track_composition if track_composition else []

        self.stats_file = stats_file
        self.results_file = results_file
        self.checkpoint_file = checkpoint_file
        self.cycle_start = 0

        # Load balancing: one queue per GPU.
        self.gpu_queues = [ctx.Queue() for _ in range(n_gpus)]
        self.result_queue = ctx.Queue()
        self.workers = []

        # Keep RNG streams deterministic and isolated per replica, independent of worker scheduling and ordering.
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
                header = "cycle,T_K,n_atoms,E_eV,E_eV_per_atom,Cv_eV_per_K,acc_pct"
                for el in self.track_composition:
                    header += f",N_{el}_avg,chi_{el}"
                f.write(header + "\n")

        if resume and os.path.exists(self.checkpoint_file):
            self._load_master_checkpoint()

    def _start_workers(self):
        total_workers = self.n_gpus * self.workers_per_gpu
        logger.info(
            f"Spawning {total_workers} persistent workers ({self.workers_per_gpu} per GPU)..."
        )

        for i in range(total_workers):
            assigned_gpu = i % self.n_gpus
            specific_queue = self.gpu_queues[assigned_gpu]
            w = ReplicaWorker(
                i,
                assigned_gpu,
                specific_queue,
                self.result_queue,
                self.worker_init_info,
            )
            w.start()
            self.workers.append(w)

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
        **pt_kwargs,
    ):

        atoms_clean = atoms_template.copy()
        atoms_clean.calc = None
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

        replica_states = []
        for i, T in enumerate(temps):
            t_str = f"{T:.0f}"
            state = {
                "id": i,
                "T": T,
                "atoms": atoms_clean.copy(),
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
            "atoms_template": atoms_clean,
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
            **pt_kwargs,
        )

    def run(self, n_cycles, equilibration_cycles=0):
        self._start_workers()
        logger.info(f"Starting PT Loop: Cycles {self.cycle_start} -> {n_cycles}")
        kB = 8.617333e-5

        try:
            for cycle in range(self.cycle_start, n_cycles):
                logger.info(f"--- PT Cycle {cycle+1}/{n_cycles} ---")

                is_equilibrating = cycle < equilibration_cycles
                nsweeps = self.swap_interval
                cycle_start_sweep = cycle * self.swap_interval

                # Local equilibration.
                # Discard the first 20% of stats to remove swap shock.
                local_eq_steps = int(nsweeps * 0.2)
                eq_steps = nsweeps if is_equilibrating else local_eq_steps

                t_start = time.time()
                total_atoms_in_cycle = 0

                # A. Submit tasks.
                for state in self.replica_states:
                    atoms = state["atoms"]
                    total_atoms_in_cycle += len(atoms)

                    task_data = {
                        "T": state["T"],
                        "positions": atoms.get_positions(),
                        "numbers": atoms.get_atomic_numbers(),
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
                    self.gpu_queues[target_gpu].put((state["id"], task_data))

                # B. Collect results.
                completed = 0
                while completed < len(self.replica_states):
                    res = self.result_queue.get()
                    if isinstance(res, tuple) and res[0] == "ERROR":
                        raise RuntimeError(f"Worker Error: {res[1]}")

                    rid = res["replica_id"]
                    state = self.replica_states[rid]

                    # 1. Update state.
                    state["atoms"].set_positions(res["positions"])
                    state["atoms"].set_atomic_numbers(res["numbers"])
                    state["atoms"].set_cell(res["cell"])
                    state["atoms"].pbc = res["pbc"]
                    state["e_old"] = res["e_old"]
                    state["rng_state"] = res["rng_state"]
                    state["sweep"] = res["sweep"]

                    # 2. Cycle stats from worker.
                    cycle_stats = res["local_stats"]
                    cycle_E = cycle_stats["energy"]
                    acc = cycle_stats["acceptance"]

                    # 3. Update cumulative stats in memory.
                    if not is_equilibrating:
                        state["cum_sum_E"] += res["cycle_sum_E"]
                        state["cum_sum_E_sq"] += res["cycle_sum_E_sq"]
                        state["cum_n_samples"] += res["cycle_n_samples"]

                    N = state["cum_n_samples"]
                    if N > 1:
                        cum_avg_E = state["cum_sum_E"] / N
                        cum_var = (state["cum_sum_E_sq"] / N) - (cum_avg_E**2)
                        cum_Cv = cum_var / (kB * state["T"] ** 2)
                    else:
                        cum_Cv = 0.0

                    # 4. Write CSV using cumulative Cv estimator.
                    n_atoms = len(state["atoms"])
                    e_per_atom = cycle_E / n_atoms if n_atoms > 0 else 0.0
                    line = (
                        f"{cycle+1},"
                        f"{state['T']:.6f},"
                        f"{n_atoms},"
                        f"{cycle_E:.6f},"
                        f"{e_per_atom:.8f},"
                        f"{cum_Cv:.6f},"
                        f"{acc:.2f}"
                    )

                    if self.track_composition and "composition" in cycle_stats:
                        for el in self.track_composition:
                            comp = cycle_stats["composition"].get(el, 0.0)
                            susc = cycle_stats["susceptibility"].get(el, 0.0)
                            line += f",{comp:.4f},{susc:.4f}"

                    with open(self.results_file, "a") as f:
                        f.write(line + "\n")

                    completed += 1

                t_end = time.time()
                duration = t_end - t_start
                total_ops = total_atoms_in_cycle * nsweeps
                speed = total_ops / duration if duration > 0 else 0
                logger.info(f"[Timing] {duration:.2f}s | {speed:.2e} atom_sweeps/s")

                self._attempt_swaps(cycle)

                if (cycle + 1) % self.checkpoint_interval == 0:
                    self._save_master_checkpoint(cycle + 1)

            self._save_master_checkpoint(n_cycles)

        finally:
            self.stop()
            logger.info("PT Completed.")

    def _attempt_swaps(self, cycle):
        kB = 8.617333e-5
        stride = self.swap_stride
        n = len(self.replica_states)
        phase = self.rng.integers(0, stride)
        is_odd_cycle = cycle % 2 == 1
        start_idx = phase + (stride if is_odd_cycle else 0)

        for i in range(start_idx, n - stride, 2 * stride):
            j = i + stride
            s_i = self.replica_states[i]
            s_j = self.replica_states[j]
            delta = (1.0 / (kB * s_i["T"]) - 1.0 / (kB * s_j["T"])) * (
                s_j["e_old"] - s_i["e_old"]
            )
            accepted = False
            if delta > 0 or self.rng.random() < np.exp(delta):
                accepted = True
                s_i["atoms"], s_j["atoms"] = s_j["atoms"], s_i["atoms"]
                s_i["e_old"], s_j["e_old"] = s_j["e_old"], s_i["e_old"]
                logger.info(f"  [Swap] {s_i['T']:.0f}K <-> {s_j['T']:.0f}K | ACCEPTED")
            with open(self.stats_file, "a") as f:
                f.write(
                    f"{cycle},{s_i['T']},{s_j['T']},{s_i['e_old']:.4f},{s_j['e_old']:.4f},{accepted}\n"
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
                        # Backfill a deterministic per-replica stream for older checkpoints.
                        state["rng_state"] = np.random.default_rng(
                            self.seed + self.seed_nonce + rid
                        ).bit_generator.state
                    state["atoms"].set_positions(s_data["positions"])
                    state["atoms"].set_atomic_numbers(s_data["numbers"])
                    state["atoms"].set_cell(s_data["cell"])
                    state["atoms"].pbc = s_data["pbc"]

    def stop(self):
        for q in self.gpu_queues:
            for _ in range(self.workers_per_gpu):
                q.put("STOP")
        for w in self.workers:
            w.join()
