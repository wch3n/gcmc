import numpy as np
import logging
import os
import pickle
import multiprocessing
from typing import List, Any, Dict, Type
from .process_replica import ReplicaWorker, ctx

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
        worker_init_info=None
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
        
        self.stats_file = stats_file
        self.results_file = results_file
        self.checkpoint_file = checkpoint_file
        
        self.cycle_start = 0

        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.workers = []

        # Init Files
        if not os.path.exists(self.stats_file) and not resume:
            with open(self.stats_file, "w") as f:
                f.write("Cycle,T_i,T_j,E_i,E_j,Accepted\n")
        
        if not os.path.exists(self.results_file) and not resume:
            with open(self.results_file, "w") as f:
                f.write("Cycle,Temperature,Avg_Potential_Energy,Cv,Acceptance_Rate\n")
        
        if resume and os.path.exists(checkpoint_file):
            self._load_master_checkpoint()

    def _start_workers(self):
        total_workers = self.n_gpus * self.workers_per_gpu
        logger.info(f"Spawning {total_workers} persistent workers ({self.workers_per_gpu} per GPU)...")
        
        for i in range(total_workers):
            # i = 0, 1, 2, 3...
            # if n_gpus=4: 0->GPU0, 1->GPU1, 2->GPU2, 3->GPU3, 4->GPU0 ...
            assigned_gpu = i % self.n_gpus
            
            w = ReplicaWorker(
                rank=i, 
                device_id=assigned_gpu, 
                task_queue=self.task_queue, 
                result_queue=self.result_queue, 
                init_kwargs=self.worker_init_info
            )
            w.start()
            self.workers.append(w)

    @classmethod
    def from_auto_config(cls, atoms_template, T_start, T_end, T_step, 
                         calculator_class, mc_class, calc_kwargs, mc_kwargs, 
                         n_gpus=4, workers_per_gpu=2, # Default 2 workers/GPU
                         swap_stride=1, resume=False, 
                         results_file="results.csv", **pt_kwargs):
        
        # Clean Template (remove calc for safe init transfer)
        atoms_clean = atoms_template.copy()
        atoms_clean.calc = None

        # Generate Temps
        if T_start > T_end:
            temps = np.arange(T_start, T_end - abs(T_step)/2, -abs(T_step)).tolist()
        else:
            temps = np.arange(T_start, T_end + abs(T_step)/2, abs(T_step)).tolist()
            
        logger.info(f"Configuration: {len(temps)} Replicas | {n_gpus} GPUs | {workers_per_gpu} Workers/GPU")

        # Init Logical States
        replica_states = []
        for i, T in enumerate(temps):
            t_str = f"{T:.0f}"
            state = {
                'id': i,
                'T': T,
                'atoms': atoms_clean.copy(),
                'e_old': 0.0,
                'sweep': 0,
                'traj_file': f"replica_{t_str}K.traj",
                'thermo_file': f"replica_{t_str}K.dat",
                'checkpoint_file': f"checkpoint_{t_str}K.pkl",
                'mc_kwargs': mc_kwargs
            }
            if resume and os.path.exists(state['checkpoint_file']):
                 with open(state['checkpoint_file'], "rb") as f:
                    chk = pickle.load(f)
                    atoms_loaded = chk['atoms']
                    atoms_loaded.calc = None
                    state['atoms'] = atoms_loaded
                    state['e_old'] = chk.get('e_old', 0.0)
                    state['sweep'] = chk.get('sweep', 0)
            
            replica_states.append(state)

        worker_init_info = {
            'calculator_module': calculator_class.__module__,
            'calculator_class_name': calculator_class.__name__,
            'calc_kwargs': calc_kwargs,
            'mc_kwargs': mc_kwargs, 
            'atoms_template': atoms_clean # Passed so worker can init object once
        }

        return cls(n_gpus, workers_per_gpu, replica_states, 
                   worker_init_info=worker_init_info, 
                   swap_stride=swap_stride, resume=resume, 
                   results_file=results_file, **pt_kwargs)

    def run(self, n_cycles, equilibration_cycles=0):
        self._start_workers()
        logger.info(f"Starting PT Loop: Cycles {self.cycle_start} -> {n_cycles}")
        
        try:
            for cycle in range(self.cycle_start, n_cycles):
                logger.info(f"--- PT Cycle {cycle+1}/{n_cycles} ---")
                
                is_equilibrating = (cycle < equilibration_cycles)
                eq_steps = self.swap_interval if is_equilibrating else 0
                
                # A. SUBMIT TASKS (Arrays Only)
                for state in self.replica_states:
                    atoms = state['atoms']
                    
                    task_data = {
                        'T': state['T'],
                        'positions': atoms.get_positions(),
                        'numbers': atoms.get_atomic_numbers(),
                        'cell': atoms.get_cell(),
                        'pbc': atoms.get_pbc(),
                        'e_old': state['e_old'],
                        'sweep': state['sweep'],
                        'nsweeps': self.swap_interval,
                        'traj_file': state['traj_file'],
                        'thermo_file': state['thermo_file'],
                        'checkpoint_file': state['checkpoint_file'],
                        'report_interval': self.report_interval,
                        'sample_interval': self.sampling_interval,
                        'eq_steps': eq_steps
                    }
                    self.task_queue.put((state['id'], task_data))
                
                # B. COLLECT RESULTS
                completed = 0
                while completed < len(self.replica_states):
                    res = self.result_queue.get()
                    if isinstance(res, tuple) and res[0] == 'ERROR':
                        raise RuntimeError(f"Worker Error: {res[1]}")
                    
                    rid = res['replica_id']
                    
                    # Update Memory using Arrays (Fast)
                    self.replica_states[rid]['atoms'].set_positions(res['positions'])
                    self.replica_states[rid]['atoms'].set_atomic_numbers(res['numbers'])
                    
                    self.replica_states[rid]['e_old'] = res['e_old']
                    self.replica_states[rid]['sweep'] = res['sweep']
                    
                    # Write Results
                    stats = res['stats']
                    with open(self.results_file, "a") as f:
                         f.write(f"{cycle+1},{stats['T']},{stats['energy']:.6f},{stats['cv']:.6f},{stats['acceptance']:.2f}\n")
                    
                    completed += 1
                
                # C. EXCHANGE
                self._attempt_swaps(cycle)
                
                # D. CHECKPOINT
                if (cycle+1) % self.checkpoint_interval == 0:
                    self._save_master_checkpoint(cycle+1)

            self._save_master_checkpoint(n_cycles)
            
        finally:
            self.stop()
            logger.info("PT Completed.")

    def _attempt_swaps(self, cycle):
        kB = 8.617333e-5
        stride = self.swap_stride
        n = len(self.replica_states)
        
        phase = np.random.randint(0, stride)
        is_odd_cycle = (cycle % 2 == 1)
        start_idx = phase + (stride if is_odd_cycle else 0)
        
        for i in range(start_idx, n - stride, 2 * stride):
            j = i + stride
            s_i = self.replica_states[i]
            s_j = self.replica_states[j]
            
            delta = (1.0/(kB*s_i['T']) - 1.0/(kB*s_j['T'])) * (s_j['e_old'] - s_i['e_old'])
            
            accepted = False
            if delta > 0 or np.random.rand() < np.exp(delta):
                accepted = True
                s_i['atoms'], s_j['atoms'] = s_j['atoms'], s_i['atoms']
                s_i['e_old'], s_j['e_old'] = s_j['e_old'], s_i['e_old']
                logger.info(f"  [Swap] {s_i['T']:.0f}K <-> {s_j['T']:.0f}K | ACCEPTED")
            
            with open(self.stats_file, "a") as f:
                f.write(f"{cycle},{s_i['T']},{s_j['T']},{s_i['e_old']:.4f},{s_j['e_old']:.4f},{accepted}\n")

    def _save_master_checkpoint(self, cycle):
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump({"cycle": cycle}, f)
        logger.info(f"Checkpoint cycle {cycle}")

    def _load_master_checkpoint(self):
        with open(self.checkpoint_file, "rb") as f:
            self.cycle_start = pickle.load(f).get("cycle", 0)

    def stop(self):
        for _ in self.workers: self.task_queue.put('STOP')
        for w in self.workers: w.join()
