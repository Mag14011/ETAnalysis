import sys
import numpy as np
import argparse
from scipy.spatial import cKDTree
from datetime import datetime, timedelta
from . import derrida 
from tabulate import tabulate  
import time  
from typing import List, Tuple, Dict
from multiprocessing import Pool, Manager, Value, Lock
import ctypes
import os

def print_flush(*args, **kwargs):
    """Print with immediate flush to see output in real time."""
    kwargs['flush'] = True
    print(*args, **kwargs)

# Physical constants
T = 300.0
PI = 3.141592654
KB = 8.6173304E-5
HBAR = 6.582119514E-16

class PDBProcessor:
    def __init__(self):
        self.heme_atoms = [
            'FE', 'NA', 'C1A', 'C2A', 'C3A', 'C4A', 'CHB', 'C1B', 'NB',
            'C2B', 'C3B', 'C4B', 'CHC', 'C1C', 'NC', 'C2C', 'C3C', 'C4C',
            'CHD', 'C1D', 'ND', 'C2D', 'C3D', 'C4D', 'CHA'
        ]

    def create_pairs_from_sequence(self, seqids: str) -> List[Tuple[int, int]]:
        """Create pairs of consecutive residue IDs from a comma-separated string"""
        residues = [int(resid.strip()) for resid in seqids.split(',')]
        return list(zip(residues[:-1], residues[1:]))

    def read_pdb_atoms(self, pdb_file: str) -> Dict[int, Dict[str, np.ndarray]]:
        """Read atom coordinates from PDB file"""
        atoms_dict = {}
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM  ') or line.startswith('HETATM'):
                    try:
                        atom_name = line[12:16].strip()
                        resid = int(line[22:26])
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        
                        if atom_name in self.heme_atoms:
                            if resid not in atoms_dict:
                                atoms_dict[resid] = {}
                            atoms_dict[resid][atom_name] = np.array([x, y, z])
                    except (ValueError, IndexError):
                        continue
        
        return atoms_dict

    def calculate_min_distance(self, coords1: Dict[str, np.ndarray],
                             coords2: Dict[str, np.ndarray]) -> float:
        """Calculate minimum distance between two sets of coordinates"""
        min_dist = float('inf')
        
        for atom1 in self.heme_atoms:
            if atom1 not in coords1:
                continue
            for atom2 in self.heme_atoms:
                if atom2 not in coords2:
                    continue
                    
                dist = np.linalg.norm(coords1[atom1] - coords2[atom2])
                min_dist = min(min_dist, dist)
        
        return min_dist

    def measure_avg_dist(self, pdb_file: str, seqids: str) -> List[float]:
        """Calculate average distances between consecutive heme pairs"""
        residue_pairs = self.create_pairs_from_sequence(seqids)
        atoms_dict = self.read_pdb_atoms(pdb_file)
        
        distances = []
        for res1, res2 in residue_pairs:
            if res1 in atoms_dict and res2 in atoms_dict:
                min_dist = self.calculate_min_distance(atoms_dict[res1], atoms_dict[res2])
                distances.append(min_dist)
            else:
                print_flush(f"Warning: Missing residue {res1} or {res2} in PDB file")
        
        return distances

    def get_average_spacing(self, pdb_file: str, seqids: str) -> float:
        """Calculate the average spacing between hemes"""
        distances = self.measure_avg_dist(pdb_file, seqids)
        if not distances:
            raise ValueError("No valid distances calculated")
        return np.mean(distances)

class SharedCounter:
    """A counter that can be shared between processes"""
    def __init__(self):
        self.val = Value('i', 0)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

def parse_arguments(args):
    """Parse command line arguments of the form key=value."""
    if len(args) == 1 or args[1] in ['--help', '-h', 'help']:
        print_help()
        sys.exit(0)

    params = {}
    for arg in args[1:]:
        key, value = arg.split('=')
        # Handle boolean values
        if value.lower() in ['true', 'false']:
            params[key] = value
        # Handle scientific notation and float values
        elif 'E' in value.upper() or 'e' in value:
            params[key] = float(value)
        elif value.replace('.','').isdigit():
            params[key] = float(value)
        else:
            params[key] = value
    return params

def format_row_data(row):
    """Format each value in the row with consistent spacing matching header widths."""
    formatted = []
    for i, val in enumerate(row):
        if i == 0:  # Index
            formatted.append(f"{val:5d}")  # Matches 'Index' width
        elif isinstance(val, str):
            if i <= 6:  # V1-2(T) through V6-7(S)
                num = float(val)
                formatted.append(f"{num:7.3f}")  # Matches 'V1-2(T)' width
            elif i <= 12:  # L1-2 through L6-7
                num = float(val)
                formatted.append(f"{num:6.3f}")  # Matches 'L1-2' width
            elif i <= 18:  # dG1-2 through dG6-7
                num = float(val)
                formatted.append(f"{num:7.3f}")  # Matches 'dG1-2' width
            elif 'e' in val.lower():  # Rates and D
                num = float(val)
                formatted.append(f"{num:7.2e}")  # Matches 'Rf1-2' width
            else:
                num = float(val)
                formatted.append(f"{num:7.3f}")
        else:
            formatted.append(f"{val:7}")
    return formatted

def validate_inputs(num_tot, num_s, num_t, sequence):
    """
    Validate input parameters for heme chain analysis.
    
    Parameters:
    num_tot (int): Total number of hemes in the chain
    num_s (int): Number of slip-stacked (S) pairs
    num_t (int): Number of T-stacked (T) pairs
    sequence (str): String of S and T characters representing pair geometries
    
    The sequence length should be num_tot - 1 since for n hemes we have n-1 pairs.
    Example: For 7 hemes, we have 6 pairs, so sequence might be 'STSTST'
    """
    # Calculate expected number of pairs
    expected_pairs = num_tot - 1
    
    # Check if number of pairs matches heme count
    if num_s + num_t != expected_pairs:
        raise ValueError(
            f"num_s ({num_s}) + num_t ({num_t}) must equal number of pairs (num_tot - 1 = {expected_pairs})"
        )

    # Check if sequence length matches number of pairs
    if len(sequence) != expected_pairs:
        raise ValueError(
            f"Sequence length ({len(sequence)}) must equal number of pairs (num_tot - 1 = {expected_pairs})"
        )

    # Verify sequence composition matches specified S and T counts
    s_count = sequence.count('S')
    t_count = sequence.count('T')
    if s_count != num_s or t_count != num_t:
        raise ValueError(
            f"Sequence S/T counts ({s_count}/{t_count}) don't match num_s/num_t ({num_s}/{num_t})"
        )
    
    # Verify sequence only contains S and T characters
    invalid_chars = set(sequence) - {'S', 'T'}
    if invalid_chars:
        raise ValueError(
            f"Sequence contains invalid characters: {invalid_chars}. Only 'S' and 'T' are allowed."
        )

def generate_coupling(type_):
    """Generate a random coupling value based on heme type."""
    if type_ == 'S':
        return np.random.uniform(3, 13)  # S-type: 3-13 meV
    else:  # T-type
        return np.random.uniform(1, 4)   # T-type: 1-4 meV

def generate_parameters(sequence):
    """
    Generate all random parameters needed for the calculation.
    Each pair (not each heme) should have its own coupling, lambda, and deltaG values.
    
    Parameters:
    sequence (str): String of S and T characters representing pair geometries
    
    Returns:
    dict: Contains arrays of couplings, lambdas, and deltaG values (one per pair)
    """
    num_pairs = len(sequence)  # Number of pairs between hemes
    
    params = {
        # Couplings for each pair
        'couplings': [generate_coupling(type_) for type_ in sequence],
        # Lambda values for each pair
        'lambdas': [np.random.uniform(0.24, 1.1) for _ in range(num_pairs)],
        # Delta G values for each pair
        'deltaG': [np.random.uniform(-0.3, 0.3) for _ in range(num_pairs)]
    }
    return params

def compute_marcus_rate(coupling, lambda_val, deltaG=None):
    """Compute Marcus rate with or without free energy optimization."""
    # coupling is in meV, convert to eV by dividing by 1000
    prefactor = (2 * PI * (coupling/1000)**2) / (HBAR * np.sqrt(4 * PI * lambda_val * KB * T))

    if deltaG is None:  # free energy optimized case
        return prefactor
    else:  # full Marcus equation
        return prefactor * np.exp(-(lambda_val + deltaG)**2 / (4 * lambda_val * KB * T))

def create_headers(sequence, free_eng_opt):
    """
    Create headers based on sequence and calculation type.
    Each pair has its own coupling, lambda, and deltaG values.
    """
    headers = ["Index"]
    
    # Number of pairs
    num_pairs = len(sequence)
    
    # Coupling headers - one for each pair
    headers.extend(f"V{i+1}-{i+2}({t})" for i, t in enumerate(sequence))
    
    # Lambda headers - one for each pair
    headers.extend(f"L{i+1}-{i+2}" for i in range(num_pairs))
    
    if not free_eng_opt:
        # Delta G headers - one for each pair
        headers.extend(f"dG{i+1}-{i+2}" for i in range(num_pairs))
        # Forward and backward rate headers - one for each pair
        headers.extend(f"Rf{i+1}-{i+2}" for i in range(num_pairs))
        headers.extend(f"Rb{i+1}-{i+2}" for i in range(num_pairs))
    else:
        # Single rate headers for free energy optimized case - one for each pair
        headers.extend(f"R{i+1}-{i+2}" for i in range(num_pairs))
    
    headers.append("D")
    return headers

def worker_process(sequence, free_eng_opt, spacing_factor, d_target, d_tolerance, worker_id, 
                  target_sets, max_attempts):
    """Worker function for parallel processing with unique random seeds"""
    # Set a unique seed for this worker that's within numpy's valid range
    max_seed = 2**32 - 1
    base_seed = int(time.time()) % 1000000  # Get last 6 digits of current time
    worker_seed = (base_seed + worker_id) % max_seed
    np.random.seed(worker_seed)
    
    print_flush(f"DEBUG: Worker {worker_id} starting with seed {worker_seed}")
    print_flush(f"DEBUG: Worker {worker_id} targeting {target_sets} results with max {max_attempts} attempts")

    local_results = []
    local_attempts = 0
    local_accepted = 0

    while local_attempts < max_attempts and local_accepted < target_sets:
        local_attempts += 1

        # Generate and calculate parameters
        params = generate_parameters(sequence)

        if free_eng_opt:
            rates = [compute_marcus_rate(c, l) for c, l in
                    zip(params['couplings'], params['lambdas'])]
            V, D = derrida.VD(rates, rates)
        else:
            forward_rates = [compute_marcus_rate(c, l, dg) for c, l, dg in
                           zip(params['couplings'], params['lambdas'], params['deltaG'])]
            backward_rates = [compute_marcus_rate(c, l, -dg) for c, l, dg in
                            zip(params['couplings'], params['lambdas'], params['deltaG'])]
            V, D = derrida.VD(forward_rates, backward_rates)

        if D is None:
            continue

        # Scale D
        D_scaled = D * spacing_factor

        # Check if this result should be accepted
        if free_eng_opt:
            accept = True
        else:
            if d_target > 0:
                relative_error = abs(D_scaled - d_target) / d_target
                accept = relative_error <= d_tolerance
            else:
                accept = True

        if accept:
            row_data = [0]  # Index will be updated later
            row_data.extend(f"{v:.3f}" for v in params['couplings'])
            row_data.extend(f"{l:.3f}" for l in params['lambdas'])

            if free_eng_opt:
                row_data.extend(f"{r:.2e}" for r in rates)
            else:
                row_data.extend(f"{dg:.3f}" for dg in params['deltaG'])
                row_data.extend(f"{r:.2e}" for r in forward_rates)
                row_data.extend(f"{r:.2e}" for r in backward_rates)

            row_data.append(f"{D_scaled:.2e}")
            local_results.append((D_scaled, row_data))
            local_accepted += 1

        if local_attempts % 10000 == 0:
            print_flush(f"Worker {worker_id}: Attempts={local_attempts}, Accepted={local_accepted}, "
                       f"Seed={worker_seed}")

    print_flush(f"Worker {worker_id} finished: Attempts={local_attempts}, Accepted={local_accepted}, "
                f"Seed={worker_seed}")
    return local_results, local_attempts, local_accepted

def run_parallel_sampling(num_processes, num_sets, sequence, free_eng_opt,
                        spacing_factor, d_target, d_tolerance, max_attempts):
    """Run sampling in parallel with improved result collection"""
    print_flush("Starting parallel sampling...")
    print_flush(f"Target total sets: {num_sets}")
    print_flush(f"Max attempts allowed: {max_attempts}")
    print_flush(f"Number of processes: {num_processes}")
    
    # Calculate sets per worker to exactly match num_sets
    base_sets_per_worker = num_sets // num_processes
    extra_sets = num_sets % num_processes
    
    # Calculate attempts per worker
    attempts_per_worker = max_attempts // num_processes
    
    all_results = []
    total_attempts = 0
    total_accepted = 0
    
    with Pool(num_processes) as pool:
        print_flush(f"Created pool with {num_processes} workers")
        
        # Create tasks for workers with balanced workload
        tasks = []
        for i in range(num_processes):
            # Distribute extra sets among first few workers
            worker_target = base_sets_per_worker + (1 if i < extra_sets else 0)
            tasks.append((
                sequence, free_eng_opt, spacing_factor, d_target, d_tolerance,
                i, worker_target, attempts_per_worker
            ))
        
        # Start workers and collect results asynchronously
        async_results = [pool.apply_async(worker_process, t) for t in tasks]
        
        # Monitor progress
        while any(not r.ready() for r in async_results):
            time.sleep(1)
            finished = sum(1 for r in async_results if r.ready())
            current_results = sum(len(r.get()[0]) for r in async_results if r.ready())
            print_flush(f"Progress: {finished}/{num_processes} workers finished, "
                       f"{current_results}/{num_sets} results found", end='\r')
        
        # Collect and combine results from all workers
        print_flush("\nCollecting results from all workers...")
        for r in async_results:
            worker_results, worker_attempts, worker_accepted = r.get()
            all_results.extend(worker_results)
            total_attempts += worker_attempts
            total_accepted += worker_accepted
            
    # Sort results by D value
    all_results.sort(key=lambda x: x[0], reverse=True)
    
    print_flush(f"Found {len(all_results)} results")
    return all_results, total_attempts

def are_parameters_similar(params1, params2):
    """
    Compare only couplings, deltaGs, and lambdas with parameter-specific tolerances.

    Parameters ranges:
    - Couplings: 0-15 meV
    - DeltaG: -0.3 to 0.3 eV
    - Lambda: 0.24 to 1.1 eV
    """
    # Number of pairs (length of sequence)
    num_pairs = len(params1[1:7])  # First 6 values after index are couplings

    # Set specific tolerances for each parameter type
    coupling_tol = 1.0    # 1.0 meV tolerance for couplings
    deltaG_tol = 0.025    # 0.025 eV tolerance for deltaG
    lambda_tol = 0.025    # 0.004 eV tolerance for lambda

    # Compare couplings (V values)
    for i in range(num_pairs):
        v1 = float(params1[i + 1])  # +1 to skip index
        v2 = float(params2[i + 1])
        if abs(v1 - v2) > coupling_tol:
            return False

    # Compare lambdas (L values)
    lambda_start = 1 + num_pairs  # Start after couplings
    for i in range(num_pairs):
        l1 = float(params1[lambda_start + i])
        l2 = float(params2[lambda_start + i])
        if abs(l1 - l2) > lambda_tol:
            return False

    # Compare deltaGs (dG values)
    dg_start = 1 + 2*num_pairs  # Start after couplings and lambdas
    for i in range(num_pairs):
        dg1 = float(params1[dg_start + i])
        dg2 = float(params2[dg_start + i])
        if abs(dg1 - dg2) > deltaG_tol:
            return False

    return True

'''
def are_parameters_similar(params1, params2, tolerance=1e-3):
    """
    Check if two parameter sets are similar within a tolerance.
    params1 and params2 are row_data lists from the results.
    Ignores the first element (index) and last element (D value).
    """
    # Skip first (index) and last (D value) elements
    values1 = [float(x) for x in params1[1:-1]]
    values2 = [float(x) for x in params2[1:-1]]
    
    for v1, v2 in zip(values1, values2):
        # For very small values, use absolute difference
        if abs(v1) < tolerance and abs(v2) < tolerance:
            if abs(v1 - v2) > tolerance:
                return False
        # For larger values, use relative difference
        else:
            relative_diff = abs(v1 - v2) / max(abs(v1), abs(v2))
            if relative_diff > tolerance:
                return False
    return True
'''

def remove_redundant_sets(results, tolerance=1e-3):
    """
    Remove redundant parameter sets from results with progress tracking.
    Returns a list of unique parameter sets.
    """
    unique_results = []
    total_sets = len(results)
    last_progress = 0
    redundant_count = 0
    
    print_flush(f"\nStarting deduplication of {total_sets} parameter sets...")
    start_time = time.time()
    
    for i, (d_val, row_data) in enumerate(results, 1):
        # Update progress every 1%
        progress = (i * 100) // total_sets
        if progress > last_progress:
            elapsed_time = time.time() - start_time
            sets_per_second = i / elapsed_time if elapsed_time > 0 else 0
            eta = (total_sets - i) / sets_per_second if sets_per_second > 0 else 0
            
            print_flush(f"Progress: {progress}% ({i}/{total_sets} sets checked, "
                       f"{redundant_count} redundant sets found, "
                       f"ETA: {eta:.1f}s)")
            last_progress = progress
        
        # Check if this set is similar to any existing unique set
        is_redundant = False
        for _, existing_row in unique_results:
            if are_parameters_similar(row_data, existing_row, tolerance):
                is_redundant = True
                redundant_count += 1
                break
        
        if not is_redundant:
            unique_results.append((d_val, row_data))
    
    total_time = time.time() - start_time
    print_flush(f"\nDeduplication complete:")
    print_flush(f"- Original sets: {total_sets}")
    print_flush(f"- Unique sets: {len(unique_results)}")
    print_flush(f"- Redundant sets removed: {redundant_count}")
    print_flush(f"- Processing time: {total_time:.1f}s")
    print_flush(f"- Average processing speed: {total_sets/total_time:.1f} sets/second")
    
    return unique_results

def remove_redundant_sets_optimized(results, tolerance=1e-3):
    """
    Remove redundant parameter sets using KD-tree for efficient nearest neighbor search.
    """
    print_flush("\nPreparing data for optimized deduplication...")
    start_time = time.time()
    
    # Extract parameter values and convert to numpy array
    # Skip first column (index) and last column (D value)
    data = []
    for _, row in results:
        values = [float(x) for x in row[1:-1]]
        data.append(values)
    
    data = np.array(data)
    total_sets = len(data)
    print_flush(f"Converting {total_sets} parameter sets to numpy array... Done")
    
    # Normalize the data for better comparison
    # Use log scale for rates which can span many orders of magnitude
    num_params = data.shape[1]
    data_normalized = np.zeros_like(data)
    
    print_flush("Normalizing parameter values...")
    for i in range(num_params):
        column = data[:, i]
        if np.any(column < 0):  # For values that can be negative (like deltaG)
            max_abs = np.max(np.abs(column))
            if max_abs > 0:
                data_normalized[:, i] = column / max_abs
        else:  # For strictly positive values (like rates)
            if np.any(column > 0):
                # Use log scale for values spanning many orders of magnitude
                min_positive = np.min(column[column > 0])
                column_log = np.log10(column + min_positive)
                data_normalized[:, i] = (column_log - np.min(column_log)) / (np.max(column_log) - np.min(column_log))
    
    print_flush("Building KD-tree for efficient neighbor search...")
    tree = cKDTree(data_normalized)
    
    # Find pairs of points that are within tolerance
    print_flush(f"Searching for similar parameter sets (tolerance={tolerance})...")
    pairs = tree.query_pairs(tolerance, output_type='ndarray')
    
    if len(pairs) > 0:
        # Create a set of indices to keep
        indices_to_remove = set()
        for i, j in pairs:
            # Keep the one with higher D value
            d_val_i = float(results[i][0])
            d_val_j = float(results[j][0])
            if d_val_i < d_val_j:
                indices_to_remove.add(i)
            else:
                indices_to_remove.add(j)
        
        indices_to_keep = sorted(set(range(len(results))) - indices_to_remove)
        unique_results = [results[i] for i in indices_to_keep]
    else:
        unique_results = results
    
    total_time = time.time() - start_time
    print_flush(f"\nDeduplication complete:")
    print_flush(f"- Original sets: {total_sets}")
    print_flush(f"- Unique sets: {len(unique_results)}")
    print_flush(f"- Redundant sets removed: {total_sets - len(unique_results)}")
    print_flush(f"- Processing time: {total_time:.1f}s")
    print_flush(f"- Processing speed: {total_sets/total_time:.1f} sets/second")
    
    return unique_results

def print_help():
    """Print help information about the script usage."""
    help_text = """
Parameter Exploration Script for Electron Transfer Calculations
------------------------------------------------------------

Usage:
    python script.py [parameters]

Required Parameters:
    num_tot=<int>        : Total number of hemes in sequence
    num_s=<int>          : Number of S-type pairs
    num_t=<int>          : Number of T-type pairs
    seq=<string>         : Sequence of S and T pairs (e.g., 'TSTST' for 6 pairs)
    num_sets=<float>     : Number of parameter sets to generate (can use scientific notation, e.g., 1E6)
    free_eng_opt=<bool>  : Whether to use free energy optimization ('true' or 'false')
    pdbname=<string>     : Name of PDB file to read
    seqids=<string>      : Comma-separated list of residue IDs for hemes (e.g., '1280,1274,1268,1262,1256')

Optional Parameters:
    d_target=<float>     : Target diffusion coefficient (can use scientific notation)
    d_tolerance=<float>  : Tolerance for matching target D (default: 0.1, meaning ±10%)
    max_attempts=<float> : Maximum number of attempts to find matching sets (default: 1e6)
    num_processes=<int>  : Number of CPU processes to use (default: all available)

Parameter Ranges:
    S-type coupling : 3-13 meV
    T-type coupling : 1-4 meV
    Lambda         : 0.24-1.1 eV
    Delta G        : -0.3 to 0.3 eV (only used when free_eng_opt=false)

Examples:
    1. Basic usage without target D (for 7 hemes with 6 pairs):
       python script.py num_tot=7 num_s=3 num_t=3 seq=STSTST num_sets=1E6 free_eng_opt=false \\
           pdbname=min.pdb seqids=1280,1274,1268,1262,1256,1250,1244

    2. With target D and parallel processing:
       python script.py num_tot=7 num_s=3 num_t=3 seq=STSTST num_sets=1E6 free_eng_opt=false \\
           pdbname=min.pdb seqids=1280,1274,1268,1262,1256,1250,1244 d_target=2.05E-4 \\
           d_tolerance=0.1 max_attempts=1E6 num_processes=4

Notes:
    - For n hemes, specify n-1 pairs in the sequence
    - The script automatically scales diffusion coefficients by the square of the average heme spacing
    - Results are sorted by diffusion coefficient (largest to smallest)
    - Progress updates show acceptance rate and estimated time remaining
    - When using d_target, the acceptance rate may be low if the target is rare in the parameter space
    """
    print_flush(help_text)

def main():
    """Main function for parameter exploration calculations."""
    parser = argparse.ArgumentParser(
        description="Explore electron transfer parameters for multi-heme systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter Ranges:
    S-type coupling : 3-13 meV
    T-type coupling : 1-4 meV
    Lambda         : 0.24-1.1 eV
    Delta G        : -0.3 to 0.3 eV (only used when free_eng_opt=false)
        """)

    # Required arguments
    parser.add_argument("--num-tot", type=int, required=True,
                       help="Total number of hemes in sequence")
    parser.add_argument("--num-s", type=int, required=True,
                       help="Number of S-type pairs")
    parser.add_argument("--num-t", type=int, required=True,
                       help="Number of T-type pairs")
    parser.add_argument("--seq", type=str, required=True,
                       help="Sequence of S and T pairs (e.g., 'TSTST')")
    parser.add_argument("--num-sets", type=float, required=True,
                       help="Number of parameter sets to generate (can use scientific notation)")
    parser.add_argument("--free-eng-opt", type=str, required=True, choices=['true', 'false'],
                       help="Whether to use free energy optimization")
    parser.add_argument("--pdbname", type=str, required=True,
                       help="Path to PDB file")
    parser.add_argument("--seqids", type=str, required=True,
                       help="Comma-separated list of residue IDs (e.g., '1280,1274,1268')")

    # Optional arguments
    parser.add_argument("--d-target", type=float, default=0,
                       help="Target diffusion coefficient (can use scientific notation)")
    parser.add_argument("--d-tolerance", type=float, default=0.1,
                       help="Tolerance for matching target D (default: 0.1, meaning ±10%%)")
    parser.add_argument("--max-attempts", type=float, default=1e6,
                       help="Maximum attempts to find matching sets (default: 1e6)")
    parser.add_argument("--num-processes", type=int, default=None,
                       help="Number of CPU processes to use (default: all available)")

    args = parser.parse_args()

    try:
        # Convert scientific notation for num_sets
        num_sets = int(float(args.num_sets))
        max_attempts = int(float(args.max_attempts))
        free_eng_opt = args.free_eng_opt.lower() == 'true'
        num_processes = args.num_processes or os.cpu_count()

        # Validate inputs
        validate_inputs(args.num_tot, args.num_s, args.num_t, args.seq)

        # Get average spacing
        print_flush("Calculating heme spacing from PDB...")
        pdb_processor = PDBProcessor()
        avg_spacing_angstrom = pdb_processor.get_average_spacing(args.pdbname, args.seqids)
        avg_spacing_cm = avg_spacing_angstrom * 1e-8
        spacing_factor = avg_spacing_cm * avg_spacing_cm
        
        print_flush(f"Average heme spacing: {avg_spacing_angstrom:.2f} Å")
        print_flush(f"Spacing factor (squared): {spacing_factor:.2e} cm²")

        # Create output filename
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filename = f"{args.seq}_{timestamp}.txt"

        # Create headers
        headers = create_headers(args.seq, free_eng_opt)

        # Create parameter line for output file
        param_line = (f"# num_tot={args.num_tot} num_s={args.num_s} num_t={args.num_t} "
                     f"seq={args.seq} free_eng_opt={free_eng_opt} "
                     f"pdbname={args.pdbname} avg_spacing={avg_spacing_angstrom:.2f}")
        
        if args.d_target > 0:
            param_line += f" d_target={args.d_target:.2e} d_tolerance={args.d_tolerance:.2e}"

        print_flush(param_line)
        with open(filename, 'w') as f:
            f.write(param_line + '\n')

        # Run sampling
        start_time = time.time()
        results, total_attempts = run_parallel_sampling(
            num_processes=num_processes,
            num_sets=num_sets,
            sequence=args.seq,
            free_eng_opt=free_eng_opt,
            spacing_factor=spacing_factor,
            d_target=args.d_target,
            d_tolerance=args.d_tolerance,
            max_attempts=max_attempts
        )

        # Calculate statistics
        elapsed = time.time() - start_time
        accept_rate = (len(results) / total_attempts * 100) if total_attempts > 0 else 0.0

        print_flush(f"\nFinal Statistics:")
        print_flush(f"Total attempts: {total_attempts:,d}")
        print_flush(f"Accepted sets: {len(results):,d}")
        print_flush(f"Acceptance rate: {accept_rate:.2f}%")
        print_flush(f"Total time: {elapsed:.1f}s")

        if len(results) > 0:
            print_flush(f"\nTotal sets before deduplication: {len(results)}")
            print_flush(f"\nStarting deduplication process...")
            initial_sets = len(results)
            results = remove_redundant_sets_optimized(results, tolerance=1e-3)
            final_sets = len(results)

            print_flush(f"Proceeding with unique parameter sets...")
            print_flush(f"Unique sets after deduplication: {final_sets}")
            print_flush(f"Removed {initial_sets - final_sets} redundant sets")

            print_flush(f"Average time per accepted unique set: {elapsed/final_sets:.3f}s")
            print_flush(f"Processing rate: {total_attempts/elapsed:.1f} attempts/second")

            # Create statistics header
            header_stats = (
                f"# Calculation Statistics:\n"
                f"# Total calculation attempts: {total_attempts:,d}\n"
                f"# Initial accepted sets: {initial_sets:,d}\n"
                f"# Acceptance rate: {accept_rate:.2f}%\n"
                f"# Sets after deduplication: {final_sets:,d}\n"
                f"# Total calculation time: {elapsed:.1f}s\n"
                f"# Processing rate: {total_attempts/elapsed:.1f} attempts/second\n"
            )

            # Read existing content
            with open(filename, 'r') as f:
                content = f.read()

            # Write back with new header
            with open(filename, 'w') as f:
                f.write(param_line + '\n')
                f.write(header_stats)
                f.write(content[len(param_line)+1:])

            # Print headers for results
            header_table = tabulate([], headers, tablefmt="simple",
                                numalign="right", stralign="right")
            print_flush("\n" + header_table)
            with open(filename, 'a') as f:
                f.write('\n' + header_table + '\n')

            # Sort results by diffusion coefficient
            sorted_data = [row for _, row in results]

            # Print and save sorted results in batches
            batch_size = 100
            num_batches = (len(sorted_data) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(sorted_data))
                batch = sorted_data[start_idx:end_idx]

                # Update indices and format data
                formatted_batch = []
                for i, row in enumerate(batch, start=start_idx+1):
                    row[0] = i
                    formatted_batch.append(format_row_data(row))

                # Create table for this batch WITHOUT headers
                batch_table = ""
                for row in formatted_batch:
                    batch_table += "  ".join(row) + "\n"

                # Print and save this batch
                print_flush(batch_table.rstrip())
                with open(filename, 'a') as f:
                    f.write(batch_table)

            print_flush(f"\nResults have been saved to: {filename}")
        else:
            print_flush("\nNo valid results found.")
            if d_target > 0:
                print_flush(f"Consider adjusting d_target ({d_target:.2e}) or tolerance ({d_tolerance:.2f})")
            else:
                print_flush("Check your input parameters and constraints.")

        return 0

    except ValueError as ve:
        print_flush(f"Error: {str(ve)}", file=sys.stderr)
        return 1
    except Exception as e:
        print_flush(f"Error during execution: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
