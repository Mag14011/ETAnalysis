import os
import re
import sys
import math
import numpy as np
from scipy import stats
import argparse
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from tabulate import tabulate
from . import hopping as hp

@dataclass
class ElectronTransferStep:
    """Class to hold electron transfer step information."""
    structure: str
    donor: str
    acceptor: str
    dg: float
    lambda_reorg: float
    hda: float
    geometry: str

@dataclass
class RedoxState:
    """Class to hold redox state calculation results"""
    potentials: np.ndarray      # Redox potentials (meV)
    populations: np.ndarray     # Site populations
    forward_rates: np.ndarray   # Forward rates (s⁻¹)
    backward_rates: np.ndarray  # Backward rates (s⁻¹)
    forward_flux: float        # Net forward flux (s⁻¹)
    backward_flux: float       # Net backward flux (s⁻¹)
    convergence_iterations: int # Number of iterations to converge
    dgs: np.ndarray            # DG values (deltaA) for each step
    e_act_forward: np.ndarray  # Forward activation energies
    e_act_backward: np.ndarray # Backward activation energies

# Color scheme for different structures
DEFAULT_COLORS = {
    '6EF8': '#90EE90',  # light green
    '6NEF': '#228B22',  # forest green
    '7LQ5': '#DDA0DD',  # plum
    '8D9M': '#800080',  # purple
    '7TFS': '#FF8C00',  # dark orange
    '8E5F': '#ADD8E6',  # light blue
    '8E5G': '#00008B'   # dark blue
}

def get_geometry(hda: float) -> str:
    """Determine geometry type based on Hda value."""
    if hda > 0.006:  # Closer to 0.008 (8 meV)
        return 'S'
    elif hda < 0.004:  # Closer to 0.002 (2 meV)
        return 'T'
    else:  # Around 0.005 (5 meV)
        return 'U'

def parse_rates_file(filename: str) -> List[Tuple[str, str, float, float, str]]:
    """Parse rates file to extract donor-acceptor pairs and their forward/backward rates.
    Expected format:
    HEM-X -> HEM-Y; kf = X.XXE+XX s^-1; kb = X.XXE+XX s^-1; geometry = S/T/U

    Returns:
        List of tuples (donor, acceptor, k_forward, k_backward, geometry)
    """
    rates_data = []
    with open(filename, 'r') as f:
        for line in f:
            pattern = r'(HEM-\d+) -> (HEM-\d+); kf = ([\d.E+-]+) s\^-1; kb = ([\d.E+-]+) s\^-1; geometry = ([STU])'
            match = re.match(pattern, line)
            if match:
                donor, acceptor, kf, kb, geometry = match.groups()
                rates_data.append((donor, acceptor, float(kf), float(kb), geometry))
    return rates_data


def parse_dg_file(filename: str) -> List[Tuple[str, str, float]]:
    """Parse DG.txt file to extract donor-acceptor pairs and DG values."""
    dg_data = []
    with open(filename, 'r') as f:
        for line in f:
            pattern = r'\((HEM-\d+).+?\) -> \((HEM-\d+).+?\); DG =\s*([-\d.]+) eV'
            match = re.match(pattern, line)
            if match:
                donor, acceptor, dg = match.groups()
                dg_data.append((donor, acceptor, float(dg)))
    return dg_data

def read_redox_potentials(dg_file: str) -> np.ndarray:
    """Extract unique redox potentials following chain structure from DG.txt"""
    heme_potentials = {}  # Dictionary to store unique heme:potential pairs

    with open(dg_file, 'r') as f:
        for line in f:
            # Extract both donor and acceptor info
            matches = re.findall(r'(HEM-\d+)\s*=\s*([-\d.]+)\s*eV', line)
            if matches:
                for heme, potential in matches:
                    heme_potentials[heme] = float(potential)

    # Convert to ordered list following chain structure
    unique_potentials = []
    ordered_hemes = []

    with open(dg_file, 'r') as f:
        # Get first donor from first line
        first_line = f.readline()
        first_match = re.search(r'\((HEM-\d+)', first_line)
        if first_match:
            first_heme = first_match.group(1)
            ordered_hemes.append(first_heme)
            unique_potentials.append(heme_potentials[first_heme])

    # Follow chain to maintain order
    for heme in heme_potentials:
        if heme not in ordered_hemes:
            ordered_hemes.append(heme)
            unique_potentials.append(heme_potentials[heme])

    return np.array(unique_potentials)

def parse_lambda_file(filename: str) -> List[Tuple[str, str, float]]:
    """Parse Lambda.txt file to extract donor-acceptor pairs and reorganization energies."""
    lambda_data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if '->' in line:
                parts = line.split('->')
                donor = 'HEM' + parts[0].strip()[3:]
                acceptor = parts[1].strip().split()[0]
                while i < len(lines):
                    if 'Reorg. Eng.' in lines[i]:
                        reorg = float(lines[i].split('=')[1].strip())
                        lambda_data.append((donor, acceptor, reorg))
                        break
                    i += 1
            i += 1
    return lambda_data

def parse_hda_file(filename: str) -> List[Tuple[str, str, float]]:
    """Parse Hda.txt file to extract donor-acceptor pairs and Hda values in eV."""
    hda_data = []
    with open(filename, 'r') as f:
        for line in f:
            if 'Hda' in line:
                pattern = r'Hda\((HEM-\d+) <-> (HEM-\d+)\).+?Hda = +?([\d.]+) meV'
                match = re.match(pattern, line)
                if match:
                    donor, acceptor, hda = match.groups()
                    hda_data.append((donor, acceptor, float(hda)/1000.0))  # Convert meV to eV
    return hda_data

def read_E_matrix(filename: str) -> np.ndarray:
    """
    Read E matrix from StateEnergies file and convert from meV to eV.
    """
    matrix = []
    matrix_started = False

    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            if line.strip().startswith('['):
                matrix_started = True

            if matrix_started:
                clean_line = line.strip('[] \n')
                if clean_line:
                    row = [float(x.strip())/1000.0 for x in clean_line.split(',')]  # Convert meV to eV
                    matrix.append(row)

    if not matrix:
        raise ValueError("No matrix data found in file")

    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("Input matrix must be square")

    return np.array(matrix)

def modify_E_matrix(E: np.ndarray, diagonal_shift: float = 0.0,
                   offdiagonal_scale: float = 1.0) -> np.ndarray:
    """
    Modify E matrix according to specified transformations.
    """
    E_modified = E.copy()
    n = len(E)

    # Shift diagonal elements
    for i in range(n):
        E_modified[i,i] += diagonal_shift

    # Scale off-diagonal elements
    for i in range(n):
        for j in range(n):
            if i != j:
                E_modified[i,j] *= offdiagonal_scale

    return E_modified

def replicate_array(arr: np.ndarray, n_replications: int) -> np.ndarray:
    """
    Replicate a 1D array n times by repeating all values.
    """
    result = arr.copy()
    for _ in range(n_replications):
        result = np.concatenate([result, arr])
    return result

def replicate_matrix(matrix: np.ndarray, n_replications: int) -> np.ndarray:
    """
    Replicate a square matrix n times by sliding the original matrix.
    """
    orig_size = matrix.shape[0]

    for n in range(n_replications):
        current_size = matrix.shape[0]
        new_size = current_size + (orig_size - 1)
        new_matrix = np.zeros((new_size, new_size))

        # Copy existing matrix to top-left corner
        new_matrix[:current_size, :current_size] = matrix

        # Fill in the slid matrix
        for i in range(orig_size):
            for j in range(orig_size):
                if i == 0 and j == 0:
                    continue

                new_i = i + (current_size - 1)
                new_j = j + (current_size - 1)

                if new_i < new_size and new_j < new_size:
                    new_matrix[new_i, new_j] = matrix[i, j]

        matrix = new_matrix

    return matrix

def compute_marcus_rates(hda, lambda_reorg, dg):
    """Calculate Marcus electron transfer rates."""
    T = 300.0
    PI = 3.141592654
    KB = 8.6173304E-5
    HBAR = 6.582119514E-16

    prefactor = (2 * PI * hda**2) / (HBAR * math.sqrt(4 * PI * lambda_reorg * KB * T))

    e_act_forward = ((dg + lambda_reorg)**2) / (4 * lambda_reorg)
    e_act_backward = ((-1 * dg + lambda_reorg)**2) / (4 * lambda_reorg)

    k_forward = prefactor * math.exp(-1 * e_act_forward / (KB * T))
    k_backward = prefactor * math.exp(-1 * e_act_backward / (KB * T))

    return k_forward, k_backward, e_act_forward, e_act_backward

def adaptive_mixing_parameter(iteration: int,
                            max_diff: float,
                            prev_max_diff: float,
                            current_param: float,
                            max_diff_history: Optional[List[float]] = None,
                            window_size: int = 3) -> Tuple[float, List[float]]:
    """
    Adaptively adjust mixing parameter based on convergence behavior.

    Args:
        iteration: Current iteration number
        max_diff: Current maximum difference
        prev_max_diff: Previous maximum difference
        current_param: Current mixing parameter
        max_diff_history: List storing recent max_diff values
        window_size: Number of previous iterations to consider

    Returns:
        Tuple of (new_mixing_parameter, updated_history)
    """
    if max_diff_history is None:
        max_diff_history = []

    # Update history
    max_diff_history.append(max_diff)
    if len(max_diff_history) > window_size:
        max_diff_history.pop(0)

    # Don't adjust parameter until we have enough history
    if iteration < 2:
        return current_param, max_diff_history

    # Detect oscillations
    is_oscillating = False
    if len(max_diff_history) >= 3:
        diffs = [max_diff_history[i] - max_diff_history[i-1]
                for i in range(1, len(max_diff_history))]
        sign_changes = sum(1 for i in range(1, len(diffs))
                         if diffs[i] * diffs[i-1] < 0)
        is_oscillating = sign_changes >= (len(diffs) - 1) / 2

    # Determine new mixing parameter
    if is_oscillating:
        new_param = max(0.1, current_param * 0.7)  # More aggressive reduction
    elif max_diff > prev_max_diff:
        new_param = max(0.1, current_param * 0.9)  # Normal reduction
    else:
        new_param = min(0.8, current_param * 1.05)  # Gentle increase

    return new_param, max_diff_history

def compute_redox_potentials(E: np.ndarray, case: str,
                           populations: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute redox potentials based on case and populations.
    Convert interaction energies from meV to eV.
    """
    n_hemes = len(E)
    R = np.zeros(n_hemes)

    for i in range(n_hemes):
        if case == 'reduced':
            R[i] = E[i,i]
        elif case == 'oxidized':
            R[i] = E[i,i] + sum(E[i,j] for j in range(n_hemes) if j != i)
        elif case == 'mixed':
            if populations is None:
                raise ValueError("Populations required for mixed case")
            R[i] = E[i,i] + sum(E[i,j] * (1-populations[j])
                               for j in range(n_hemes) if j != i)
    return R

def calculate_redox_state(E: np.ndarray,
                         H: np.ndarray,
                         lambda_reorg: np.ndarray,
                         case: str,
                         initial_mixing_param: float = 0.3,
                         max_iterations: int = 1000,
                         convergence_threshold: float = 0.001,
                         adaptive_mixing: bool = True) -> RedoxState:
    """Calculate redox state properties using self-consistent iteration."""
    n_hemes = len(E)
    populations = np.zeros(n_hemes)
    mixing_param = initial_mixing_param
    prev_max_diff = float('inf')
    max_diff_history = []

    # Initialize redox potentials
    R = compute_redox_potentials(E, case, populations)

    print(f"  Starting potential updates for {case} state...")

    # Iteration loop for convergence
    for iteration in range(max_iterations):
        prev_R = R.copy()

        # Calculate intermediate rates
        kfor = np.zeros(n_hemes-1)
        kback = np.zeros(n_hemes-1)

        for i in range(n_hemes-1):
            deltaA = (R[i] - R[i+1])
            lambda_i = lambda_reorg[i]
            h_i = H[i]

            print(f"deltaA = {deltaA:>10.3f}; lambda_i = {lambda_i:>10.3f}; h_i = {h_i:>10.3f}")

            kfor[i] = compute_marcus_rates(h_i, lambda_i, deltaA)[0]
            kback[i] = compute_marcus_rates(h_i, lambda_i, -deltaA)[0]

        # Calculate flux and populations
        sol = hp.solve_flux(kfor.tolist(), kback.tolist(), verbose=False)
        populations = np.array(sol[:-1])

        # Calculate new potentials
        R_new = compute_redox_potentials(E, case, populations)
        max_diff = np.max(np.abs(R_new - prev_R))
        print(f"  Update {iteration + 1}: maximum change in potentials = {max_diff:.6f} meV")

        if iteration > 0:
            if adaptive_mixing:
                mixing_param, max_diff_history = adaptive_mixing_parameter(
                    iteration, max_diff, prev_max_diff, mixing_param, max_diff_history)
            R = mixing_param * R_new + (1 - mixing_param) * prev_R
        else:
            R = R_new

        if max_diff < convergence_threshold:
            print(f"  Converged after {iteration + 1} potential updates")
            break

        prev_max_diff = max_diff

    # Calculate final values
    n_steps = n_hemes - 1
    final_kfor = np.zeros(n_steps)
    final_kback = np.zeros(n_steps)
    dgs = np.zeros(n_steps)
    e_act_forward = np.zeros(n_steps)
    e_act_backward = np.zeros(n_steps)

    for i in range(n_steps):
        dgs[i] = R[i] - R[i+1]  # Store DG (deltaA)
        lambda_i = lambda_reorg[i]
        h_i = H[i]

        kf, kb, eaf, eab = compute_marcus_rates(h_i, lambda_i, dgs[i])
        final_kfor[i] = kf
        final_kback[i] = kb
        e_act_forward[i] = eaf
        e_act_backward[i] = eab

    forward_sol = hp.solve_flux(final_kfor.tolist(), final_kback.tolist(), verbose=False)
    backward_sol = hp.solve_flux(final_kback[::-1].tolist(), final_kfor[::-1].tolist(), verbose=False)

    return RedoxState(
        potentials=R,
        populations=populations,
        forward_rates=final_kfor,
        backward_rates=final_kback,
        forward_flux=forward_sol[-1],
        backward_flux=backward_sol[-1],
        convergence_iterations=iteration + 1,
        dgs=dgs,
        e_act_forward=e_act_forward,
        e_act_backward=e_act_backward
    )

def calculate_redox_states(E: np.ndarray,
                          H: np.ndarray,
                          lambda_reorg: np.ndarray,
                          adaptive_mixing: bool = True) -> Dict[str, RedoxState]:
    """
    Calculate all redox states (reduced, oxidized, mixed).

    Args:
        E: Interaction matrix (meV)
        H: Electronic coupling values (meV)
        lambda_reorg: Reorganization energies (meV)
        adaptive_mixing: Whether to use adaptive mixing parameter

    Returns:
        Dictionary containing RedoxState objects for each case
    """
    states = {}
    for case in ['reduced', 'oxidized', 'mixed']:
        print(f"\n  Calculating {case} state...")
        states[case] = calculate_redox_state(
            E=E,
            H=H,
            lambda_reorg=lambda_reorg,
            case=case,
            adaptive_mixing=adaptive_mixing
        )
        print(f"    Converged in {states[case].convergence_iterations} iterations")
        print(f"    Forward flux: {states[case].forward_flux:.2E} s⁻¹")
        print(f"    Backward flux: {states[case].backward_flux:.2E} s⁻¹")

    return states

def compute_flux_analysis(steps: List[ElectronTransferStep], results: List[tuple]) -> Tuple[float, float]:
    """
    Compute forward and backward flux for the electron transfer chain.
    Returns:
        Tuple of (forward_flux, backward_flux)
    """
    # Extract forward and backward rates
    forward_rates = [kf for kf, _, _, _ in results]
    backward_rates = [kb for _, kb, _, _ in results]

    # Print rate information
    print('\nHopping Flux Analysis:')
    print('-' * 50)
    print('Rates for each step:')
    for i, (kf, kb) in enumerate(zip(forward_rates, backward_rates)):
        step = steps[i]
        print(f"{step.donor} → {step.acceptor}:  | Forward: {kf:4.2e} | Backward: {kb:4.2e} |")

    try:
        # Forward flux calculation
        forward_sol = hp.solve_flux(forward_rates, backward_rates)
        forward_flux = forward_sol[-1]

        # Backward flux calculation (reverse the rates)
        backward_rates_rev = backward_rates[::-1]
        forward_rates_rev = forward_rates[::-1]
        backward_sol = hp.solve_flux(backward_rates_rev, forward_rates_rev)
        backward_flux = backward_sol[-1]

        # Convert fluxes to current (pA)
        E_CHARGE = 1.602E-19  # Elementary charge in Coulombs

        print(f"\nNet forward flux: {forward_flux:4.2e} s⁻¹")
        print(f"Net backward flux: {backward_flux:4.2e} s⁻¹")
        print(f"Current: {(forward_flux - backward_flux) * E_CHARGE * 1E12:4.2e} pA")

        return forward_flux, backward_flux

    except Exception as e:
        print(f"Error in flux calculation: {str(e)}")
        return None, None

def create_gradient_span(ax, y_min, y_max, y_mean, x_min, x_max, n_steps=1000):
    """Create spans with appropriate gradients and borders."""
    if y_min == 1e2 and y_max == 1e4:  # Enzymatic turnover range
        # Create array of y positions
        y_positions = np.concatenate([
            np.logspace(np.log10(y_min), np.log10(y_mean), n_steps//2),
            np.logspace(np.log10(y_mean), np.log10(y_max), n_steps//2)
        ])
        
        # Calculate alphas - highest at mean, lowest at extremes
        alphas = np.concatenate([
            np.linspace(0.05, 0.5, n_steps//2),  # Lower half
            np.linspace(0.5, 0.05, n_steps//2)   # Upper half
        ])
        
        # Create gradient spans
        for i in range(len(y_positions)-1):
            ax.axhspan(y_positions[i], y_positions[i+1],
                      xmin=x_min, xmax=x_max,
                      color='#F5DEB3', alpha=alphas[i], zorder=0)
    else:  # Experimental ranges
        # Solid gray span
        ax.axhspan(y_min, y_max, xmin=x_min, xmax=x_max,
                  color='gray', alpha=0.2, zorder=0)
        # Add black edges at extremes
        ax.axhline(y=y_min, xmin=x_min, xmax=x_max, 
                  color='black', linewidth=1, alpha=0.5, zorder=1)
        ax.axhline(y=y_max, xmin=x_min, xmax=x_max, 
                  color='black', linewidth=1, alpha=0.5, zorder=1)

def create_rate_distribution_plot(steps: List[ElectronTransferStep], results: List[tuple], 
                                output_prefix: str, structure_order: List[str]):
    """Create and save box plot of rate distributions."""
    # Organize data by structure and geometry
    data_dict = {struct: {'S': [], 'T': []} for struct in structure_order}
    
    for step, (kf, kb, _, _) in zip(steps, results):
        # Collect both forward and backward rates
        data_dict[step.structure][step.geometry].extend([kf, kb])
    
    # Print distribution statistics
    print_distribution_stats(data_dict, structure_order)
    
    # Prepare data in specified order
    s_data = [data_dict[struct]['S'] for struct in structure_order]
    t_data = [data_dict[struct]['T'] for struct in structure_order]
    colors = [DEFAULT_COLORS[struct] for struct in structure_order] * 2
    
    # Define experimental rates
    exp_rates_S = np.array([219E6, 14E6, 105E6, 1560E6])
    exp_rates_T = np.array([125E6, 8.7E6, 114E6, 87E6])
    exp_mean_S = np.mean(exp_rates_S)
    exp_mean_T = np.mean(exp_rates_T)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_yscale('log')
    ax.set_ylim(1e1, 1.1e13)
    
    # Calculate layout parameters
    num_structs = len(structure_order)
    total_width = 10
    middle = total_width / 2
    box_spacing = (total_width/2) / (num_structs + 1)
    
    # Calculate positions
    left_positions = [middle - (num_structs - i) * box_spacing for i in range(num_structs)]
    right_positions = [middle + (i + 1) * box_spacing for i in range(num_structs)]
    all_positions = left_positions + right_positions
    
    # Add experimental ranges
    create_gradient_span(ax, np.min(exp_rates_S), np.max(exp_rates_S), exp_mean_S,
                        0, middle/total_width)
    create_gradient_span(ax, np.min(exp_rates_T), np.max(exp_rates_T), exp_mean_T,
                        middle/total_width, 1)
    
    # Add enzymatic turnover range
    create_gradient_span(ax, 1e2, 1e4, np.sqrt(1e2 * 1e4), 0, 1)
    
    # Add reference lines
    pe = [path_effects.withStroke(linewidth=3, foreground='black')]
    ax.axhline(y=4.82E9, color=DEFAULT_COLORS['6EF8'], 
               linestyle='--', zorder=1, linewidth=2.5, path_effects=pe)
    ax.axhline(y=5.20E12, color=DEFAULT_COLORS['7LQ5'], 
               linestyle='--', zorder=1, linewidth=2.5, path_effects=pe)
    
    # Create box plots
    bp = ax.boxplot(s_data + t_data, patch_artist=True, showfliers=True,
                    positions=all_positions, medianprops=dict(color="black"))
    
    # Apply colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add dividing line and labels
    ax.plot([middle, middle], [1e4, ax.get_ylim()[1]], 
            color='black', linestyle='-', linewidth=1.5, zorder=5)
    
    ax.text(middle/2, ax.get_ylim()[1]*1.3, 'Slip-stacked', 
            horizontalalignment='center', fontsize=14, weight='bold')
    ax.text(middle + middle/2, ax.get_ylim()[1]*1.3, 'T-stacked',
            horizontalalignment='center', fontsize=14, weight='bold')
    
    # Add OmcZ and OmcS annotations
    ax.text(total_width*0.55, 5.20E12*0.4, 'Exp. OmcZ', 
            color=DEFAULT_COLORS['7LQ5'],
            path_effects=[path_effects.withStroke(linewidth=1, foreground='black')],
            fontweight='bold', fontsize=12)
    
    ax.text(total_width*0.55, 4.82E9*1.6, 'Exp. OmcS', 
            color=DEFAULT_COLORS['6EF8'],
            path_effects=[path_effects.withStroke(linewidth=1, foreground='black')],
            fontweight='bold', fontsize=12)
    
    # Add MtrC and STC annotations inside gray regions
    ax.text(middle/2, 1e8, 'Exp. MtrC', 
            color='black', fontweight='bold', fontsize=12,
            horizontalalignment='center')
    
    ax.text(middle + middle/2, 1e7, 'Exp. STC', 
            color='black', fontweight='bold', fontsize=12,
            horizontalalignment='center')
    
    # Add enzymatic turnover text
    ax.text(middle, np.sqrt(1e2 * 1e4), 'Typical Enzymatic Turnover', 
            color='black',
            path_effects=[path_effects.withStroke(linewidth=1, foreground='black')],
            fontweight='bold', fontsize=12,
            horizontalalignment='center')
    
    # Configure axes
    ax.set_ylabel('Rate (s⁻¹)')
    ax.set_xlabel('PDB ID')
    ax.set_xticks(all_positions)
    ax.set_xticklabels(structure_order + structure_order, rotation=90)
    ax.tick_params(axis='both', direction='in', which='both')
    ax.set_xlim(0, total_width)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_rate_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_redox_plot(oxidized_potentials: np.ndarray,
                     reduced_potentials: Optional[np.ndarray] = None,
                     mixed_potentials: Optional[np.ndarray] = None,
                     mixed_populations: Optional[np.ndarray] = None,
                     output_prefix: str = "redox",
                     label_step: int = 1):
    """Create redox potential and population plot."""
    # Create figure with two subplots
    fig = plt.figure(figsize=(3.3, 3.3))
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0)
    ax1 = plt.subplot(gs[0])  # Potentials
    ax2 = plt.subplot(gs[1])  # Populations
    
    # Get x values
    x = np.arange(1, len(oxidized_potentials) + 1)
    
    # Plot potentials
    ax1.plot(x, oxidized_potentials, ':', marker='s', mfc='none', 
             color='blue', label='Oxidized')
    
    if reduced_potentials is not None:
        ax1.plot(x, reduced_potentials, ':', marker='s', mfc='none',
                color='red', label='Reduced')
    
    if mixed_potentials is not None:
        ax1.plot(x, mixed_potentials, ':', marker='s', mfc='none',
                color='green', label='Mixed')
    
    # Set potential plot limits
    pot_min = np.min([p for p in [oxidized_potentials, reduced_potentials, mixed_potentials] 
                     if p is not None])
    pot_max = np.max([p for p in [oxidized_potentials, reduced_potentials, mixed_potentials] 
                     if p is not None])
    pot_range = pot_max - pot_min
    ax1.set_ylim(pot_min - 0.1*pot_range, pot_max + 0.1*pot_range)
    
    # Plot populations
    if mixed_populations is not None:
        ax2.plot(x, mixed_populations, ':k')
        for i, pop in enumerate(mixed_populations):
            gray_value = 1.0 - pop
            ax2.plot(x[i], pop, 'o',
                    markerfacecolor=f'{gray_value:.3f}',
                    markeredgecolor='black',
                    markersize=6)
    else:
        zeros = np.zeros_like(x)
        ax2.plot(x, zeros, ':k')
        ax2.plot(x, zeros, 'o',
                markerfacecolor='1.0',
                markeredgecolor='black',
                markersize=6)
    
    # Configure axis limits and labels
    x_min = x[0] - 0.2
    x_max = x[-1] + 0.2
    for ax in [ax1, ax2]:
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.2)
        ax.tick_params(direction='in', which='both', top=True)
    
    # Configure specific subplot settings
    ax1.set_ylabel('Potential (eV)')
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    ax2.set_xlabel('Heme Index')
    ax2.set_ylabel('Population')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xticks(x[::label_step])
    
    if label_step > 1:
        current_ticks = list(ax2.get_xticks())
        if 1 not in current_ticks:
            current_ticks = [1] + current_ticks
            ax2.set_xticks(current_ticks)
    
    plt.savefig(f"{output_prefix}_redox_plot.png", dpi=150, 
                bbox_inches='tight', pad_inches=0.1)
    plt.close()

def print_distribution_stats(data_dict: Dict[str, Dict[str, List[float]]], 
                           structure_order: List[str]):
    """Print statistical information about rate distributions."""
    # Define experimental rates
    exp_rates_S = np.array([219E6, 14E6, 105E6, 1560E6])
    exp_rates_T = np.array([125E6, 8.7E6, 114E6, 87E6])
    
    print("\nExperimental Rate Statistics:")
    print("-" * 80)
    
    for geom, rates in [('S', exp_rates_S), ('T', exp_rates_T)]:
        mean = np.mean(rates)
        std = np.std(rates)
        rsd = (std/mean) * 100
        
        print(f"\n{geom}-stack Experimental Statistics:")
        print(f"  Number of rates: {len(rates)}")
        print(f"  Min:     {np.min(rates):12.2E}")
        print(f"  Max:     {np.max(rates):12.2E}")
        print(f"  Median:  {np.median(rates):12.2E}")
        print(f"  Mean:    {mean:12.2E}")
        print(f"  Std Dev: {std:12.2E}")
        print(f"  RSD:     {rsd:12.1f}%")
    
    print("\nCalculated Rate Distribution Statistics:")
    print("-" * 100)
    
    for struct in structure_order:
        for geom in ['S', 'T']:
            rates = data_dict[struct][geom]
            if rates:
                print(f"\n{struct} {geom}-stack Statistics:")
                print(f"  Number of rates: {len(rates)}")
                print(f"  Min:     {np.min(rates):12.2E}")
                print(f"  Max:     {np.max(rates):12.2E}")
                print(f"  Median:  {np.median(rates):12.2E}")
                print(f"  Mean:    {np.mean(rates):12.2E}")
                print(f"  RSD:     {(np.std(rates)/np.mean(rates))*100:12.1f}%")

def process_structure(structure: str, input_dir: str,
                     diagonal_shift: float = 0.0,
                     offdiagonal_scale: float = 1.0,
                     n_replications: int = 0) -> Tuple[List[ElectronTransferStep], List[tuple], Optional[Dict], str, Tuple[float, float], np.ndarray]:
    """
    Process a structure using either DG.txt (PATH 1) or StateEnergies.txt (PATH 2).
    """

    # First check for rates file
    rates_file = os.path.join(input_dir, f"{structure}_rates.txt")
    if os.path.exists(rates_file):
        print(f"\nFound rates file for {structure}")
        return process_structure_rates(structure, rates_file)

    # Check for input files
    dg_file = os.path.join(input_dir, f"{structure}_DG.txt")
    state_energies_file = os.path.join(input_dir, f"{structure}_StateEnergies.txt")
    lambda_file = os.path.join(input_dir, f"{structure}_Lambda.txt")
    hda_file = os.path.join(input_dir, f"{structure}_Hda.txt")

    # Verify required files exist
    if not all(os.path.exists(f) for f in [lambda_file, hda_file]):
        raise FileNotFoundError(f"Missing required files for structure {structure}")

    # Determine which path to take
    if os.path.exists(state_energies_file):
        if os.path.exists(dg_file):
            print(f"\nWarning: Both StateEnergies.txt and DG.txt found for {structure}")
            print("Using StateEnergies.txt and ignoring DG.txt")
        return process_structure_path2(
            structure, state_energies_file, lambda_file, hda_file,
            diagonal_shift, offdiagonal_scale, n_replications
        )
    elif os.path.exists(dg_file):
        if diagonal_shift != 0.0 or offdiagonal_scale != 1.0 or n_replications > 0:
            print(f"\nWarning: Matrix modifications ignored for {structure} (using DG.txt)")
        return process_structure_path1(structure, dg_file, lambda_file, hda_file)
    else:
        raise FileNotFoundError(
            f"Neither {state_energies_file} nor {dg_file} found for {structure}"
        )

def process_structure_rates(structure: str, rates_file: str) -> Tuple[List[ElectronTransferStep], List[tuple], None, str, Tuple[float, float], np.ndarray]:
    """Process structure using pre-computed rates from rates file."""
    print(f"\nProcessing {structure} using rates file")

    # Parse rates file
    rates_data = parse_rates_file(rates_file)

    steps = []
    results = []

    # Create electron transfer steps with placeholder energetics
    for donor, acceptor, kf, kb, geometry in rates_data:
        step = ElectronTransferStep(
            structure=structure,
            donor=donor,
            acceptor=acceptor,
            dg=0.0,  # placeholder
            lambda_reorg=0.0,  # placeholder
            hda=0.0,  # placeholder
            geometry=geometry
        )
        steps.append(step)
        results.append((kf, kb, 0.0, 0.0))  # forward rate, backward rate, placeholder activation energies

    # Extract rates for hopping model
    forward_rates = [kf for kf, _, _, _ in results]
    backward_rates = [kb for _, kb, _, _ in results]

    # Calculate fluxes using hopping model
    forward_sol = hp.solve_flux(forward_rates, backward_rates, verbose=False)
    populations = forward_sol[:-1]
    forward_flux = forward_sol[-1]

    # Calculate backward flux
    backward_rates_rev = backward_rates[::-1]
    forward_rates_rev = forward_rates[::-1]
    backward_sol = hp.solve_flux(backward_rates_rev, forward_rates_rev, verbose=False)
    backward_flux = backward_sol[-1]

    return steps, results, None, rates_file, (forward_flux, backward_flux), populations

def process_structure_path1(structure: str, dg_file: str,
                          lambda_file: str, hda_file: str) -> Tuple[List[ElectronTransferStep], List[tuple], None, str, Tuple[float, float]]:
    """Process structure using DG.txt approach (PATH 1)."""
    print(f"\nProcessing {structure} using PATH 1 (DG.txt)")

    # Parse input files
    dg_data = parse_dg_file(dg_file)
    lambda_data = parse_lambda_file(lambda_file)
    hda_data = parse_hda_file(hda_file)

    steps = []
    results = []

    # Create electron transfer steps
    for donor, acceptor, dg in dg_data:
        matching_lambda = next(
            (l for d, a, l in lambda_data if d == donor and a == acceptor), None
        )
        matching_hda = next(
            (h for d, a, h in hda_data if d == donor and a == acceptor), None
        )

        if matching_lambda is not None and matching_hda is not None:
            geometry = get_geometry(matching_hda)
            step = ElectronTransferStep(
                structure=structure,
                donor=donor,
                acceptor=acceptor,
                dg=dg,
                lambda_reorg=matching_lambda,
                hda=matching_hda,
                geometry=geometry
            )
            steps.append(step)
            results.append(compute_marcus_rates(step.hda, step.lambda_reorg, step.dg))

    # Calculate fluxes
    forward_rates = [kf for kf, _, _, _ in results]
    backward_rates = [kb for _, kb, _, _ in results]

    forward_sol = hp.solve_flux(forward_rates, backward_rates, verbose=False)
    populations = forward_sol[:-1]
    forward_flux = forward_sol[-1]

    backward_rates_rev = backward_rates[::-1]
    forward_rates_rev = forward_rates[::-1]
    backward_sol = hp.solve_flux(backward_rates_rev, forward_rates_rev, verbose=False)
    backward_flux = backward_sol[-1]

    return steps, results, None, dg_file, (forward_flux, backward_flux), populations

def process_structure_path2(structure: str, state_energies_file: str,
                          lambda_file: str, hda_file: str,
                          diagonal_shift: float = 0.0,
                          offdiagonal_scale: float = 1.0,
                          n_replications: int = 0) -> Tuple[List[ElectronTransferStep], List[tuple], Dict]:
    """Process structure using StateEnergies.txt approach (PATH 2)."""
    print(f"\nProcessing {structure} using PATH 2 (StateEnergies.txt)")

    # Read and modify energy matrix
    E = read_E_matrix(state_energies_file)

    if diagonal_shift != 0.0 or offdiagonal_scale != 1.0:
        E = modify_E_matrix(E, diagonal_shift, offdiagonal_scale)

    if n_replications > 0:
        E = replicate_matrix(E, n_replications)

    # Parse lambda and hda data
    lambda_data = parse_lambda_file(lambda_file)
    hda_data = parse_hda_file(hda_file)

    # Convert to arrays
    lambda_values = np.array([l for _, _, l in lambda_data])
    hda_values = np.array([h for _, _, h in hda_data])

    if n_replications > 0:
        lambda_values = replicate_array(lambda_values, n_replications)
        hda_values = replicate_array(hda_values, n_replications)

    # Calculate redox states
    redox_states = calculate_redox_states(E, hda_values, lambda_values)

    # Generate steps and results using mixed state
    steps = []
    results = []
    n_sites = len(E)

    for i in range(n_sites - 1):
        donor = f"HEM-{i+1}"
        acceptor = f"HEM-{i+2}"
        geometry = get_geometry(hda_values[i])
        dg = redox_states['mixed'].potentials[i] - redox_states['mixed'].potentials[i+1]

        step = ElectronTransferStep(
            structure=structure,
            donor=donor,
            acceptor=acceptor,
            dg=dg,
            lambda_reorg=lambda_values[i],
            hda=hda_values[i],
            geometry=geometry
        )

        steps.append(step)
        results.append(compute_marcus_rates(
            hda=step.hda,
            lambda_reorg=step.lambda_reorg,
            dg=step.dg
        ))

    return steps, results, redox_states, state_energies_file, (redox_states['mixed'].forward_flux,
           redox_states['mixed'].backward_flux), redox_states['mixed'].populations

def save_rates(steps: List[ElectronTransferStep], results: List[tuple],
               redox_states: Optional[Dict[str, RedoxState]], filename: str):
    """Save rate calculations for all cases and create DG_flux.txt for mixed state."""
    # Save main rate analysis
    with open(filename, "w") as f:
        f.write("Rate Calculations from Different Energy Schemes\n")
        f.write("=" * 80 + "\n\n")

        # Write state results if available
        if redox_states is not None:
            for state in ['reduced', 'oxidized', 'mixed']:
                redox_state = redox_states[state]
                f.write(f"{state.capitalize()} State:\n")
                f.write("-" * (len(state) + 7) + "\n")
                f.write("Step,Geometry,DG(eV),Lambda(eV),Hda(meV),E_act_forward(eV),"
                       "E_act_backward(eV),k_forward(s⁻¹),k_backward(s⁻¹)\n")
                
                n_steps = len(redox_state.dgs)
                for i in range(n_steps):
                    step = steps[i]  # Get geometry and reorganization energy from original steps
                    f.write(f"HEM-{i+1}->HEM-{i+2},{step.geometry},"
                           f"{redox_state.dgs[i]:.3f},{step.lambda_reorg:.3f},"
                           f"{step.hda:.3f},{redox_state.e_act_forward[i]:.3f},"
                           f"{redox_state.e_act_backward[i]:.3f},"
                           f"{redox_state.forward_rates[i]:.2E},"
                           f"{redox_state.backward_rates[i]:.2E}\n")
        else:
            # DG.txt pathway
            f.write("DG.txt Pathway Results:\n")
            f.write("-" * 40 + "\n")
            f.write("Step,Geometry,DG(eV),Lambda(eV),Hda(meV),E_act_forward(eV),"
                    "E_act_backward(eV),k_forward(s⁻¹),k_backward(s⁻¹)\n")

            for step, (kf, kb, eaf, eab) in zip(steps, results):
                f.write(f"{step.donor}->{step.acceptor},{step.geometry},{step.dg:.3f},"
                       f"{step.lambda_reorg:.3f},{step.hda:.3f},{eaf:.3f},{eab:.3f},"
                       f"{kf:.2E},{kb:.2E}\n")

    # Create DG_flux.txt using mixed state data
    if redox_states is not None:
        mixed_state = redox_states['mixed']
        print(f"\nCreating DG_flux.txt using mixed state data:")  # Debug print
        potentials = mixed_state.potentials
        dgs = mixed_state.dgs
        
        base_dir = os.path.dirname(filename)
        dg_flux_file = os.path.join(base_dir, "DG_flux.txt")
        
        with open(dg_flux_file, "w") as f:
            for i in range(len(dgs)):
                donor_pot = potentials[i]
                acceptor_pot = potentials[i+1]
                dg = dgs[i]
                print(f"  Step {i+1}: Pot{i+1}={donor_pot:.3f}, "  # Debug print
                      f"Pot{i+2}={acceptor_pot:.3f}, DG={dg:.3f}")
                f.write(f"(HEM-{i+1} = {donor_pot:.3f} eV) -> "
                       f"(HEM-{i+2} = {acceptor_pot:.3f} eV); "
                       f"DG = {dg:.3f} eV\n")

def save_results(steps: List[ElectronTransferStep], results: List[tuple],
                redox_states: Optional[Dict[str, RedoxState]], 
                forward_flux: float, backward_flux: float,
                output_prefix: str):
    """Save all analysis results to a single combined file."""
    with open(f"{output_prefix}_analysis.txt", "w") as f:
        f.write("Rate and Flux Analysis Results\n")
        f.write("=" * 80 + "\n\n")

        # Write rate calculations
        f.write("Rate Calculations:\n")
        f.write("=" * 80 + "\n\n")

        # Write state results if available
        if redox_states is not None:
            for state in ['reduced', 'oxidized', 'mixed']:
                redox_state = redox_states[state]
                f.write(f"\n{state.capitalize()} State:\n")
                f.write("-" * (len(state) + 7) + "\n")
                f.write("Step,Geometry,DG(eV),Lambda(eV),Hda(meV),E_act_forward(eV),"
                       "E_act_backward(eV),k_forward(s⁻¹),k_backward(s⁻¹)\n")
                
                n_steps = len(redox_state.dgs)
                for i in range(n_steps):
                    step = steps[i]  # Get geometry and reorganization energy from original steps
                    f.write(f"HEM-{i+1}->HEM-{i+2},{step.geometry},"
                           f"{redox_state.dgs[i]:.3f},{step.lambda_reorg:.3f},"
                           f"{step.hda:.3f},{redox_state.e_act_forward[i]:.3f},"
                           f"{redox_state.e_act_backward[i]:.3f},"
                           f"{redox_state.forward_rates[i]:.2E},"
                           f"{redox_state.backward_rates[i]:.2E}\n")
        else:
            # Direct rates (either from DG.txt or rates.txt)
            f.write("Direct Rate Results:\n")
            f.write("-" * 40 + "\n")
            if any(step.dg != 0.0 for step in steps):  # DG.txt pathway
                f.write("Step,Geometry,DG(eV),Lambda(eV),Hda(meV),E_act_forward(eV),"
                        "E_act_backward(eV),k_forward(s⁻¹),k_backward(s⁻¹)\n")

                for step, (kf, kb, eaf, eab) in zip(steps, results):
                    f.write(f"{step.donor}->{step.acceptor},{step.geometry},{step.dg:.3f},"
                           f"{step.lambda_reorg:.3f},{step.hda:.3f},{eaf:.3f},{eab:.3f},"
                           f"{kf:.2E},{kb:.2E}\n")
            else:  # rates.txt pathway
                f.write("Step,Geometry,k_forward(s⁻¹),k_backward(s⁻¹)\n")
                for step, (kf, kb, _, _) in zip(steps, results):
                    f.write(f"{step.donor}->{step.acceptor},{step.geometry},"
                           f"{kf:.2E},{kb:.2E}\n")

        # Write flux results
        f.write("\nFlux Analysis Results\n")
        f.write("=" * 80 + "\n")
        
        # Calculate currents
        E_CHARGE = 1.602E-19  # Elementary charge in Coulombs
        forward_current_pA = forward_flux * E_CHARGE * 1E12
        backward_current_pA = backward_flux * E_CHARGE * 1E12
        net_flux = forward_flux - backward_flux
        net_current_pA = net_flux * E_CHARGE * 1E12
        
        f.write(f"\nNet Flux Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Forward Flux: {forward_flux:.2E} s⁻¹\n")
        f.write(f"Forward Current: {forward_current_pA:.2E} pA\n")
        f.write(f"Backward Flux: {backward_flux:.2E} s⁻¹\n")
        f.write(f"Backward Current: {backward_current_pA:.2E} pA\n")
        f.write(f"Net Flux: {net_flux:.2E} s⁻¹\n")
        f.write(f"Net Current: {net_current_pA:.2E} pA\n")
        
        # Write redox state information if available
        if redox_states is not None:
            f.write("\nRedox State Analysis:\n")
            f.write("-" * 20 + "\n")
            for state, data in redox_states.items():
                f.write(f"\n{state.capitalize()} State:\n")
                net = data.forward_flux - data.backward_flux
                forward_current = data.forward_flux * E_CHARGE * 1E12
                backward_current = data.backward_flux * E_CHARGE * 1E12
                net_current = net * E_CHARGE * 1E12
                
                f.write(f"  Forward Flux: {data.forward_flux:.2E} s⁻¹\n")
                f.write(f"  Forward Current: {forward_current:.2E} pA\n")
                f.write(f"  Backward Flux: {data.backward_flux:.2E} s⁻¹\n")
                f.write(f"  Backward Current: {backward_current:.2E} pA\n")
                f.write(f"  Net Flux: {net:.2E} s⁻¹\n")
                f.write(f"  Net Current: {net_current:.2E} pA\n")

            # Create DG_flux.txt using mixed state data
            mixed_state = redox_states['mixed']
            potentials = mixed_state.potentials
            dgs = mixed_state.dgs
            
            base_dir = os.path.dirname(f"{output_prefix}_analysis.txt")
            dg_flux_file = os.path.join(base_dir, "DG_flux.txt")
            
            with open(dg_flux_file, "w") as f:
                for i in range(len(dgs)):
                    donor_pot = potentials[i]
                    acceptor_pot = potentials[i+1]
                    dg = dgs[i]
                    f.write(f"(HEM-{i+1} = {donor_pot:.3f} eV) -> "
                           f"(HEM-{i+2} = {acceptor_pot:.3f} eV); "
                           f"DG = {dg:.3f} eV\n")

"""
def main():
    parser = argparse.ArgumentParser(description='Calculate electron transfer rates and fluxes')
    parser.add_argument('--structures', nargs='+', required=True,
                       help='Structure names (e.g., 8E5F 8E5G)')
    parser.add_argument('--input-dir', default='.',
                       help='Directory containing input files (default: current directory)')
    parser.add_argument('--output-prefix', default='flux',
                       help='Prefix for output files (default: flux)')
    parser.add_argument('--shift-diagonal', type=float, default=0.0,
                       help='Shift to apply to diagonal elements of E matrix (meV)')
    parser.add_argument('--scale-offdiagonal', type=float, default=1.0,
                       help='Scale factor for off-diagonal elements of E matrix')
    parser.add_argument('--replicate', type=int, default=0,
                       help='Number of times to replicate the system')

    args = parser.parse_args()

    all_steps = []
    all_results = []

    for structure in args.structures:
        print(f"\nProcessing structure: {structure}")
        print("=" * 50)

        try:
            steps, results, redox_states, _, (forward_flux, backward_flux), populations = process_structure(
                structure, args.input_dir,
                args.shift_diagonal, args.scale_offdiagonal, args.replicate
            )

            # Store for combined analysis
            all_steps.extend(steps)
            all_results.extend(results)

            # Generate outputs
            output_prefix = f"{args.output_prefix}_{structure}"

            # Create plots
            create_rate_distribution_plot(steps, results, output_prefix, [structure])

            # Create redox plot if we have the data
            oxidized_potentials = read_redox_potentials(f"{args.input_dir}/{structure}_DG.txt") \
                                if redox_states is None else redox_states['oxidized'].potentials

            # Only create redox plot if using DG/StateEnergies files
            if redox_states is not None or source_file.endswith('_DG.txt'):
                oxidized_potentials = read_redox_potentials(f"{args.input_dir}/{structure}_DG.txt") \
                                    if redox_states is None else redox_states['oxidized'].potentials

                create_redox_plot(
                    oxidized_potentials=oxidized_potentials,
                    reduced_potentials=None if redox_states is None else redox_states['reduced'].potentials,
                    mixed_potentials=None if redox_states is None else redox_states['mixed'].potentials,
                    mixed_populations=populations,
                    output_prefix=output_prefix
                )

            # Save rate information and flux analysis
            save_results(steps, results, redox_states, forward_flux, backward_flux, output_prefix)

        except Exception as e:
            print(f"Error processing structure {structure}: {str(e)}")
            continue

    # Create combined rate distribution plot if multiple structures
    if len(args.structures) > 1:
        create_rate_distribution_plot(all_steps, all_results,
                                    args.output_prefix, args.structures)
"""
def main():
    parser = argparse.ArgumentParser(description='Calculate electron transfer rates and fluxes')
    parser.add_argument('--structures', nargs='+', required=True,
                       help='Structure names (e.g., 8E5F 8E5G)')
    parser.add_argument('--input-dir', default='.',
                       help='Directory containing input files (default: current directory)')
    parser.add_argument('--output-prefix', default='flux',
                       help='Prefix for output files (default: flux)')
    parser.add_argument('--shift-diagonal', type=float, default=0.0,
                       help='Shift to apply to diagonal elements of E matrix (meV)')
    parser.add_argument('--scale-offdiagonal', type=float, default=1.0,
                       help='Scale factor for off-diagonal elements of E matrix')
    parser.add_argument('--replicate', type=int, default=0,
                       help='Number of times to replicate the system')

    args = parser.parse_args()

    all_steps = []
    all_results = []

    for structure in args.structures:
        print(f"\nProcessing structure: {structure}")
        print("=" * 50)

        try:
            steps, results, redox_states, source_file, (forward_flux, backward_flux), populations = process_structure(
                structure, args.input_dir,
                args.shift_diagonal, args.scale_offdiagonal, args.replicate
            )

            # Store for combined analysis
            all_steps.extend(steps)
            all_results.extend(results)

            # Generate outputs
            output_prefix = f"{args.output_prefix}_{structure}"

            # Create plots
            create_rate_distribution_plot(steps, results, output_prefix, [structure])

            # Only create redox plot if using DG/StateEnergies files
            if redox_states is not None or source_file.endswith('_DG.txt'):
                oxidized_potentials = read_redox_potentials(f"{args.input_dir}/{structure}_DG.txt") \
                                    if redox_states is None else redox_states['oxidized'].potentials

                create_redox_plot(
                    oxidized_potentials=oxidized_potentials,
                    reduced_potentials=None if redox_states is None else redox_states['reduced'].potentials,
                    mixed_potentials=None if redox_states is None else redox_states['mixed'].potentials,
                    mixed_populations=populations,
                    output_prefix=output_prefix
                )

            # Save rate information and flux analysis
            save_results(steps, results, redox_states, forward_flux, backward_flux, output_prefix)

        except Exception as e:
            print(f"Error processing structure {structure}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Create combined rate distribution plot if multiple structures
    if len(args.structures) > 1:
        create_rate_distribution_plot(all_steps, all_results,
                                    args.output_prefix, args.structures)
if __name__ == "__main__":
    main()


