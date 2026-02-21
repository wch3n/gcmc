import sys
import types
import unittest
from unittest.mock import patch

from gcmc.execution_backends import RayReplicaBackend


def _make_fake_ray_module():
    mod = types.ModuleType("ray")
    mod._initialized = False
    mod.init_calls = []
    mod.actor_option_calls = []

    def is_initialized():
        return mod._initialized

    def init(**kwargs):
        mod.init_calls.append(kwargs)
        mod._initialized = True

    def remote(_cls):
        class _Template:
            def options(self, **opts):
                mod.actor_option_calls.append(dict(opts))

                class _Invoker:
                    def remote(self, **_kwargs):
                        return object()

                return _Invoker()

        return _Template()

    def kill(_actor, no_restart=True):
        return None

    def shutdown():
        mod._initialized = False

    mod.is_initialized = is_initialized
    mod.init = init
    mod.remote = remote
    mod.kill = kill
    mod.shutdown = shutdown
    mod.get = lambda _obj: None
    return mod


class RayBackendConfigTests(unittest.TestCase):
    def test_default_fractional_gpu_per_actor(self):
        fake_ray = _make_fake_ray_module()
        with patch.dict(sys.modules, {"ray": fake_ray}):
            backend = RayReplicaBackend(
                n_gpus=2,
                workers_per_gpu=2,
                worker_init_info={},
                backend_kwargs={},
            )
            backend.start()

        # 2 GPU slots * 2 workers/slot = 4 actors.
        self.assertEqual(len(fake_ray.actor_option_calls), 4)
        self.assertTrue(
            all(abs(opts["num_gpus"] - 0.5) < 1e-12 for opts in fake_ray.actor_option_calls)
        )

    def test_respects_explicit_num_gpus_override(self):
        fake_ray = _make_fake_ray_module()
        with patch.dict(sys.modules, {"ray": fake_ray}):
            backend = RayReplicaBackend(
                n_gpus=1,
                workers_per_gpu=2,
                worker_init_info={},
                backend_kwargs={"actor_options": {"num_gpus": 0.25}},
            )
            backend.start()

        self.assertEqual(len(fake_ray.actor_option_calls), 2)
        self.assertTrue(
            all(abs(opts["num_gpus"] - 0.25) < 1e-12 for opts in fake_ray.actor_option_calls)
        )

    def test_rejects_negative_num_gpus_override(self):
        fake_ray = _make_fake_ray_module()
        with patch.dict(sys.modules, {"ray": fake_ray}):
            backend = RayReplicaBackend(
                n_gpus=1,
                workers_per_gpu=2,
                worker_init_info={},
                backend_kwargs={"actor_options": {"num_gpus": -0.1}},
            )
            with self.assertRaises(ValueError):
                backend.start()

    def test_placement_group_scheduling_strategy_is_attached(self):
        fake_ray = _make_fake_ray_module()
        pg_calls = []
        sched_calls = []

        pg_mod = types.ModuleType("ray.util.placement_group")

        class _FakePG:
            def ready(self):
                return object()

        def placement_group(**kwargs):
            pg_calls.append(dict(kwargs))
            return _FakePG()

        def remove_placement_group(_pg):
            return None

        pg_mod.placement_group = placement_group
        pg_mod.remove_placement_group = remove_placement_group

        sched_mod = types.ModuleType("ray.util.scheduling_strategies")

        class PlacementGroupSchedulingStrategy:
            def __init__(self, **kwargs):
                sched_calls.append(dict(kwargs))

        sched_mod.PlacementGroupSchedulingStrategy = PlacementGroupSchedulingStrategy

        with patch.dict(
            sys.modules,
            {
                "ray": fake_ray,
                "ray.util.placement_group": pg_mod,
                "ray.util.scheduling_strategies": sched_mod,
            },
        ):
            backend = RayReplicaBackend(
                n_gpus=1,
                workers_per_gpu=2,
                worker_init_info={},
                backend_kwargs={"use_placement_group": True},
            )
            backend.start()

        self.assertEqual(len(pg_calls), 1)
        self.assertEqual(pg_calls[0]["strategy"], "SPREAD")
        self.assertEqual(len(pg_calls[0]["bundles"]), 2)
        self.assertTrue(all("CPU" in b for b in pg_calls[0]["bundles"]))
        self.assertEqual(len(sched_calls), 2)
        self.assertTrue(
            all(call["placement_group_bundle_index"] in {0, 1} for call in sched_calls)
        )


if __name__ == "__main__":
    unittest.main()
