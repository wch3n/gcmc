import multiprocessing
import os
import logging
import traceback
import importlib
import numpy as np
from scipy.spatial import cKDTree

# Use 'spawn' to ensure clean CUDA context initialization in each worker
ctx = multiprocessing.get_context('spawn')

class ReplicaWorker(ctx.Process):
    """
    High-Performance Worker:
    1. Initializes Calculator & AlloyCMC only ONCE to save massive overhead.
    2. Accepts multiple workers per GPU to saturate compute.
    3. Communicates via raw Numpy arrays to minimize CPU pickling cost.
    """
    def __init__(self, rank, device_id, task_queue, result_queue, init_kwargs):
        super().__init__()
        self.rank = rank
        self.device_id = device_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.init_kwargs = init_kwargs
        self.daemon = True

    def run(self):
        # 1. Isolate GPU
        # Multiple workers with same device_id will share the GPU via time-slicing
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        
        # Unique logger per worker
        logging.basicConfig(format=f'[GPU {self.device_id}|W {self.rank}] %(message)s', level=logging.INFO)

        try:
            from gcmc.alloy_cmc import AlloyCMC
            
            # 2. Dynamic Import of Calculator
            module_name = self.init_kwargs['calculator_module']
            class_name = self.init_kwargs['calculator_class_name']
            module = importlib.import_module(module_name)
            CalculatorClass = getattr(module, class_name)
            
            calc_args = self.init_kwargs.get('calc_kwargs', {}).copy()
            # Special handling for MACE/Torch: mapping 'cuda' to 'cuda:0' inside the isolated process
            if 'device' in calc_args and 'cuda' in str(calc_args['device']):
                calc_args['device'] = 'cuda:0' 
            
            # 3. Persistent Initialization (Done ONCE)
            calc = CalculatorClass(**calc_args)
            
            # Use template to create persistent sim object
            atoms_template = self.init_kwargs['atoms_template'].copy()
            
            # Initialize the Engine
            # We use placeholders for T/files, they get updated every task
            sim = AlloyCMC(
                atoms=atoms_template,
                calculator=calc,
                T=300, 
                traj_file="placeholder.traj",
                thermo_file="placeholder.dat",
                checkpoint_file="placeholder.pkl",
                **self.init_kwargs['mc_kwargs']
            )

            # 4. High-Speed Task Loop
            while True:
                task = self.task_queue.get()
                if task == 'STOP': break
                
                replica_id, data = task
                
                # --- A. STATE INJECTION (Fast) ---
                # Update physical state without creating new objects
                sim.T = data['T']
                
                # Direct Array Update (Avoids ASE overhead)
                sim.atoms.set_positions(data['positions'])
                sim.atoms.set_atomic_numbers(data['numbers'])
                sim.atoms.set_cell(data['cell'])
                sim.atoms.pbc = data['pbc']
                
                # Restore Logic State
                sim.e_old = data['e_old']
                sim.sweep = data['sweep']
                
                # Update File Paths for this specific replica
                sim.traj_file = data['traj_file']
                sim.thermo_file = data['thermo_file']
                sim.checkpoint_file = data['checkpoint_file']
                
                # Force Rebuild Neighbor List (Crucial after position jump)
                if hasattr(sim, 'tree'):
                    sim.tree = cKDTree(data['positions'])
                
                # --- B. EXECUTION ---
                stats = sim.run(
                    nsweeps=data['nsweeps'],
                    traj_file=data['traj_file'],
                    interval=data['report_interval'],
                    sample_interval=data['sample_interval'],
                    equilibration=data['eq_steps']
                )
                
                # --- C. RETURN RESULTS (Fast) ---
                # Return arrays only. Never pickle the calculator or full atoms object.
                result_package = {
                    'replica_id': replica_id,
                    'positions': sim.atoms.get_positions(),
                    'numbers': sim.atoms.get_atomic_numbers(),
                    'e_old': sim.e_old,
                    'sweep': sim.sweep,
                    'stats': stats
                }
                
                self.result_queue.put(result_package)
        
        except Exception as e:
            traceback.print_exc()
            self.result_queue.put(('ERROR', str(e)))
