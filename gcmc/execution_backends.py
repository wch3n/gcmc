import importlib
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from scipy.spatial import cKDTree

from .process_replica import ReplicaWorker, ctx

logger = logging.getLogger("mc")


class ReplicaExecutionBackend(ABC):
    """Execution backend interface for replica worker orchestration."""

    @abstractmethod
    def start(self) -> None:
        """Start workers/resources."""

    @abstractmethod
    def submit(self, replica_id: int, target_gpu: int, task_data: Dict[str, Any]) -> None:
        """Submit one replica task."""

    @abstractmethod
    def get_result(self) -> Any:
        """Return one completed result."""

    @abstractmethod
    def stop(self) -> None:
        """Stop workers/resources."""


class MultiprocessingReplicaBackend(ReplicaExecutionBackend):
    """Local multiprocessing backend preserving current behavior."""

    def __init__(
        self,
        n_gpus: int,
        workers_per_gpu: int,
        worker_init_info: Optional[Dict[str, Any]],
    ):
        self.n_gpus = int(n_gpus)
        self.workers_per_gpu = int(workers_per_gpu)
        if self.n_gpus < 1:
            raise ValueError("n_gpus must be >= 1.")
        if self.workers_per_gpu < 1:
            raise ValueError("workers_per_gpu must be >= 1.")
        self.worker_init_info = worker_init_info

        self._gpu_queues = [ctx.Queue() for _ in range(self.n_gpus)]
        self._result_queue = ctx.Queue()
        self._workers = []

    def start(self) -> None:
        total_workers = self.n_gpus * self.workers_per_gpu
        logger.info(
            f"Spawning {total_workers} persistent workers ({self.workers_per_gpu} per GPU)..."
        )

        for rank in range(total_workers):
            assigned_gpu = rank % self.n_gpus
            worker = ReplicaWorker(
                rank,
                assigned_gpu,
                self._gpu_queues[assigned_gpu],
                self._result_queue,
                self.worker_init_info,
            )
            worker.start()
            self._workers.append(worker)

    def submit(self, replica_id: int, target_gpu: int, task_data: Dict[str, Any]) -> None:
        self._gpu_queues[target_gpu].put((replica_id, task_data))

    def get_result(self) -> Any:
        return self._result_queue.get()

    def stop(self) -> None:
        for queue in self._gpu_queues:
            for _ in range(self.workers_per_gpu):
                queue.put("STOP")
        for worker in self._workers:
            worker.join()


class _RayReplicaActor:
    """
    Persistent Ray worker actor mirroring process_replica.ReplicaWorker logic.

    Ray sets CUDA visibility per actor according to allocated resources.
    """

    def __init__(self, rank: int, worker_init_info: Dict[str, Any]):
        self.rank = rank
        self.worker_init_info = worker_init_info

        mc_module_name = self.worker_init_info.get("mc_module", "gcmc.alloy_cmc")
        mc_class_name = self.worker_init_info.get("mc_class", "AlloyCMC")
        mc_module = importlib.import_module(mc_module_name)
        mc_class = getattr(mc_module, mc_class_name)

        calc_module_name = self.worker_init_info["calculator_module"]
        calc_class_name = self.worker_init_info["calculator_class_name"]
        calc_module = importlib.import_module(calc_module_name)
        calc_class = getattr(calc_module, calc_class_name)

        calc_args = self.worker_init_info.get("calc_kwargs", {}).copy()
        if "device" in calc_args and "cuda" in str(calc_args["device"]):
            calc_args["device"] = "cuda:0"

        calculator = calc_class(**calc_args)
        self.atoms_template = self.worker_init_info["atoms_template"].copy()

        self.sim = mc_class(
            atoms=self.atoms_template,
            calculator=calculator,
            T=300,
            traj_file="placeholder.traj",
            thermo_file="placeholder.dat",
            checkpoint_file="placeholder.pkl",
            **self.worker_init_info["mc_kwargs"],
        )

    def run_task(self, replica_id: int, data: Dict[str, Any]) -> Any:
        try:
            sim = self.sim
            sim.T = data["T"]

            if len(sim.atoms) != len(data["positions"]):
                sim.atoms = self.atoms_template.copy()

            sim.atoms.set_positions(data["positions"])
            sim.atoms.set_atomic_numbers(data["numbers"])
            sim.atoms.set_cell(data["cell"])
            sim.atoms.pbc = data["pbc"]

            sim.e_old = data["e_old"]
            if "rng_state" in data and data["rng_state"] is not None:
                sim.rng.bit_generator.state = data["rng_state"]
            sim.sweep = data["sweep"]

            sim.traj_file = data["traj_file"]
            sim.thermo_file = data["thermo_file"]
            sim.checkpoint_file = data["checkpoint_file"]

            if hasattr(sim, "tree"):
                sim.tree = cKDTree(data["positions"])

            stats = sim.run(
                nsweeps=data["nsweeps"],
                traj_file=data["traj_file"],
                interval=data["report_interval"],
                sample_interval=data["sample_interval"],
                equilibration=data["eq_steps"],
            )

            return {
                "replica_id": replica_id,
                "positions": sim.atoms.get_positions(),
                "numbers": sim.atoms.get_atomic_numbers(),
                "cell": sim.atoms.get_cell(),
                "pbc": sim.atoms.get_pbc(),
                "e_old": sim.e_old,
                "rng_state": sim.rng.bit_generator.state,
                "sweep": sim.sweep,
                "cycle_sum_E": sim.sum_E,
                "cycle_sum_E_sq": sim.sum_E_sq,
                "cycle_n_samples": sim.n_samples,
                "local_stats": stats,
            }
        except Exception as exc:
            traceback.print_exc()
            return ("ERROR", str(exc))


class RayReplicaBackend(ReplicaExecutionBackend):
    """
    Ray backend for multi-node/multi-GPU replica execution.

    backend_kwargs supports:
      - init_kwargs: dict passed to ray.init() when needed.
      - actor_options: dict merged into actor options. If num_gpus is not provided,
        default is set to 1/workers_per_gpu so worker over-subscription maps cleanly.
      - use_placement_group: bool. If True, place actors via a Ray placement group.
      - placement_group_strategy: strategy string for placement group (default "SPREAD").
      - placement_group_bundles: optional explicit bundles list. If omitted, one bundle
        per actor is generated automatically from CPU/GPU per actor requirements.
      - placement_group_name: optional placement group name.
      - placement_group_lifetime: optional placement group lifetime.
      - placement_group_capture_child_tasks: bool passed to scheduling strategy.
      - remove_placement_group_on_stop: bool (default True).
      - shutdown_on_stop: bool, if True call ray.shutdown() when this backend owns runtime.
    """

    def __init__(
        self,
        n_gpus: int,
        workers_per_gpu: int,
        worker_init_info: Optional[Dict[str, Any]],
        backend_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.n_gpus = int(n_gpus)
        self.workers_per_gpu = int(workers_per_gpu)
        if self.n_gpus < 1:
            raise ValueError("n_gpus must be >= 1.")
        if self.workers_per_gpu < 1:
            raise ValueError("workers_per_gpu must be >= 1.")
        self.worker_init_info = worker_init_info
        self.backend_kwargs = backend_kwargs or {}

        self._ray = None
        self._owns_ray_runtime = False
        self._actors_by_gpu = {}
        self._next_actor_by_gpu = {}
        self._pending_refs = []
        self._placement_group = None
        self._remove_placement_group = None
        self._owns_placement_group = False

    def start(self) -> None:
        try:
            import ray
        except ImportError as exc:
            raise ImportError(
                "execution_backend='ray' requested but package 'ray' is not installed."
            ) from exc

        self._ray = ray
        if not ray.is_initialized():
            ray.init(**self.backend_kwargs.get("init_kwargs", {}))
            self._owns_ray_runtime = True

        actor_options = dict(self.backend_kwargs.get("actor_options", {}))
        if "num_gpus" in actor_options:
            num_gpus_per_actor = float(actor_options["num_gpus"])
            if num_gpus_per_actor < 0:
                raise ValueError("actor_options['num_gpus'] must be >= 0 for Ray backend.")
        else:
            num_gpus_per_actor = 1.0 / float(self.workers_per_gpu)
            actor_options["num_gpus"] = num_gpus_per_actor

        gpus_per_slot = num_gpus_per_actor * self.workers_per_gpu
        if gpus_per_slot > 1.0 + 1e-8:
            logger.warning(
                "Ray backend requests %.3f GPUs per slot "
                "(workers_per_gpu=%d * num_gpus_per_actor=%.3f). "
                "This exceeds 1 GPU per logical slot and may cause unschedulable actors.",
                gpus_per_slot,
                self.workers_per_gpu,
                num_gpus_per_actor,
            )
        actor_cls = ray.remote(_RayReplicaActor)

        total_workers = self.n_gpus * self.workers_per_gpu
        cpus_per_actor = float(actor_options.get("num_cpus", 1))
        logger.info(
            "Spawning %d Ray workers (%d per GPU slot, %.3f GPU per actor, %.3f CPU per actor)...",
            total_workers,
            self.workers_per_gpu,
            num_gpus_per_actor,
            cpus_per_actor,
        )

        use_placement_group = bool(self.backend_kwargs.get("use_placement_group", False))
        actor_options_per_worker = [dict(actor_options) for _ in range(total_workers)]
        if use_placement_group:
            pg_mod = importlib.import_module("ray.util.placement_group")
            sched_mod = importlib.import_module("ray.util.scheduling_strategies")
            placement_group = getattr(pg_mod, "placement_group")
            self._remove_placement_group = getattr(pg_mod, "remove_placement_group")
            placement_group_strategy_cls = getattr(
                sched_mod, "PlacementGroupSchedulingStrategy"
            )

            bundles = self.backend_kwargs.get("placement_group_bundles")
            if bundles is None:
                bundle = {"CPU": cpus_per_actor}
                if num_gpus_per_actor > 0:
                    bundle["GPU"] = num_gpus_per_actor
                bundles = [dict(bundle) for _ in range(total_workers)]
            elif len(bundles) < total_workers:
                raise ValueError(
                    "placement_group_bundles must have at least n_gpus * workers_per_gpu bundles."
                )

            pg = placement_group(
                bundles=bundles,
                strategy=self.backend_kwargs.get("placement_group_strategy", "SPREAD"),
                name=self.backend_kwargs.get("placement_group_name"),
                lifetime=self.backend_kwargs.get("placement_group_lifetime"),
            )
            ray.get(pg.ready())
            self._placement_group = pg
            self._owns_placement_group = True

            capture_children = bool(
                self.backend_kwargs.get("placement_group_capture_child_tasks", True)
            )
            for worker_idx in range(total_workers):
                actor_options_per_worker[worker_idx]["scheduling_strategy"] = (
                    placement_group_strategy_cls(
                        placement_group=pg,
                        placement_group_bundle_index=worker_idx,
                        placement_group_capture_child_tasks=capture_children,
                    )
                )
            logger.info(
                "Ray placement group enabled: strategy=%s bundles=%d",
                self.backend_kwargs.get("placement_group_strategy", "SPREAD"),
                len(bundles),
            )

        for gpu in range(self.n_gpus):
            self._actors_by_gpu[gpu] = []
            self._next_actor_by_gpu[gpu] = 0

        for rank in range(total_workers):
            target_slot = rank % self.n_gpus
            actor = actor_cls.options(**actor_options_per_worker[rank]).remote(
                rank=rank, worker_init_info=self.worker_init_info
            )
            self._actors_by_gpu[target_slot].append(actor)

    def submit(self, replica_id: int, target_gpu: int, task_data: Dict[str, Any]) -> None:
        actor_pool = self._actors_by_gpu[target_gpu]
        if not actor_pool:
            raise RuntimeError(f"No Ray workers available for GPU slot {target_gpu}.")
        idx = self._next_actor_by_gpu[target_gpu]
        actor = actor_pool[idx]
        self._next_actor_by_gpu[target_gpu] = (idx + 1) % len(actor_pool)
        self._pending_refs.append(actor.run_task.remote(replica_id, task_data))

    def get_result(self) -> Any:
        if not self._pending_refs:
            raise RuntimeError("No pending Ray tasks to collect.")
        ready, pending = self._ray.wait(self._pending_refs, num_returns=1)
        self._pending_refs = pending
        return self._ray.get(ready[0])

    def stop(self) -> None:
        if self._ray is not None:
            for actor_pool in self._actors_by_gpu.values():
                for actor in actor_pool:
                    try:
                        self._ray.kill(actor, no_restart=True)
                    except Exception:
                        pass
        self._actors_by_gpu = {}
        self._next_actor_by_gpu = {}
        self._pending_refs = []
        if (
            self._owns_placement_group
            and self._placement_group is not None
            and self._remove_placement_group is not None
            and self.backend_kwargs.get("remove_placement_group_on_stop", True)
        ):
            try:
                self._remove_placement_group(self._placement_group)
            except Exception:
                pass
        self._placement_group = None
        self._remove_placement_group = None
        self._owns_placement_group = False
        if self._owns_ray_runtime and self.backend_kwargs.get("shutdown_on_stop", False):
            self._ray.shutdown()


def build_replica_backend(
    execution_backend: str,
    n_gpus: int,
    workers_per_gpu: int,
    worker_init_info: Optional[Dict[str, Any]],
    backend_kwargs: Optional[Dict[str, Any]] = None,
) -> ReplicaExecutionBackend:
    """Factory for replica execution backends."""
    name = str(execution_backend).strip().lower()
    if name in {"multiprocessing", "mp", "local"}:
        return MultiprocessingReplicaBackend(n_gpus, workers_per_gpu, worker_init_info)
    if name == "ray":
        return RayReplicaBackend(
            n_gpus=n_gpus,
            workers_per_gpu=workers_per_gpu,
            worker_init_info=worker_init_info,
            backend_kwargs=backend_kwargs,
        )
    raise ValueError(
        f"Unsupported execution_backend '{execution_backend}'. "
        "Use 'multiprocessing' or 'ray'."
    )
