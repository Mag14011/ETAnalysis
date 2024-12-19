import os
import re
import sys
import math
import numpy as np
from scipy import stats
import argparse
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from tabulate import tabulate
from .derrida import VD
from .structure_analyzer import PDBProcessor

@dataclass
class ElectronTransferStep:
    structure: str
    donor: str
    acceptor: str
    dg: float
    lambda_reorg: float
    hda: float
    geometry: str

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

def get_geometry(hda: float) -> str:
    """Determine geometry type based on Hda value."""
    if hda > 0.006:  # Closer to 0.008 (8 meV)
        return 'S'
    elif hda < 0.004:  # Closer to 0.002 (2 meV)
        return 'T'
    else:  # Around 0.005 (5 meV)
        return 'U'

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

def measure_fe_fe_distance(pdb_file: str, first_res: int, last_res: int) -> float:
    """Measure Fe-Fe distance between two residues in Angstroms."""
    fe_coords = {}
    with open(pdb_file, 'r') as f:
        for line in f:
            if line[0:6] in ('ATOM  ', 'HETATM') and 'FE' in line[12:16]:
                res_num = int(line[22:26])
                if res_num in (first_res, last_res):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    fe_coords[res_num] = np.array([x, y, z])
    
    if len(fe_coords) != 2:
        raise ValueError(f"Could not find FE atoms for both residues {first_res} and {last_res}")
    
    distance = np.linalg.norm(fe_coords[first_res] - fe_coords[last_res])
    return distance

def get_terminal_residues(file_path: str) -> Tuple[int, int]:
    """
    Extract first and last residue numbers from either Hda or rates file,
    preserving the sequence order they appear in.
    """
    with open(file_path, 'r') as f:
        if file_path.endswith('_rates.txt'):
            # For rates file, first donor is start, last acceptor is end
            content = f.readlines()
            if not content:
                raise ValueError(f"Empty file: {file_path}")

            # Get first donor from first line
            first_match = re.search(r'HEM-(\d+) ->', content[0])
            if not first_match:
                raise ValueError(f"Could not find donor in first line: {content[0]}")
            first_residue = int(first_match.group(1))

            # Get last acceptor from last non-empty line
            for line in reversed(content):
                if line.strip():
                    last_match = re.search(r'-> HEM-(\d+);', line)
                    if last_match:
                        last_residue = int(last_match.group(1))
                        break
            else:
                raise ValueError("Could not find last acceptor")

        else:
            # For Hda file
            first_residue = None
            last_residue = None
            for line in f:
                if 'Hda(' in line:
                    matches = re.findall(r'HEM-(\d+)', line)
                    if matches:
                        if first_residue is None:  # First appearance becomes start
                            first_residue = int(matches[0])
                        last_residue = int(matches[-1])  # Keep updating end with last acceptor

    if first_residue is None or last_residue is None:
        raise ValueError(f"Could not determine terminal residues from {file_path}")

    print(f"DEBUG: Terminal residues - First: {first_residue}, Last: {last_residue}")
    return first_residue, last_residue

def detect_sequence_with_fallback(atoms_dict: Dict[int, Dict[str, np.ndarray]],
                                processor: PDBProcessor) -> Tuple[List[int], str]:
    """
    Detect sequence using either linear or branched topology.

    Args:
        atoms_dict: Dictionary of atom coordinates
        processor: PDBProcessor instance

    Returns:
        Tuple of (sequence, topology_used)

    Raises:
        ValueError: If sequence detection fails for both topologies
    """
    try:
        sequence = processor.detect_sequence(atoms_dict, topology='linear')
        return sequence, 'linear'
    except ValueError as e:
        try:
            branched_sequences = processor.detect_sequence(atoms_dict, topology='branched')
            longest_chain = max(branched_sequences, key=len)
            return longest_chain, 'branched'
        except Exception as e:
            raise ValueError(f"Both linear and branched topology detection failed: {str(e)}")

def compute_derrida_parameters(steps: List[ElectronTransferStep], results: List[tuple],
                             pdb_file: str, rates_file: Optional[str] = None) -> Dict[str, Tuple[float, float, float, int, str, List[str]]]:
    """
    Compute Derrida velocity and diffusion coefficients using actual heme spacing.

    Args:
        steps: List of electron transfer steps
        results: List of rate calculation results
        pdb_file: Path to PDB file for structural analysis

    Returns:
        Dictionary containing tuple of:
        - velocity (sites/s)
        - diffusion coefficient (sites²/s)
        - average Fe-Fe spacing (Å)
        - subunit length from Hda terminals (Å)
        - full chain length (Å)
        - number of steps
        - chain string representation
        - list of geometries    
    """
    forward_rates = []
    backward_rates = []
    geometries = []
    chain_str_parts = []
    
    # Start with the first donor
    if steps:
        chain_str_parts.append(steps[0].donor)
    
    # Add each step in the sequence
    for step, (kf, kb, _, _) in zip(steps, results):
        forward_rates.append(kf)
        backward_rates.append(kb)
        geometries.append(step.geometry)
        chain_str_parts.append(step.acceptor)
    
    # Only compute Derrida parameters if we have at least 2 steps
    if len(forward_rates) < 2:
        raise ValueError("Need at least 2 steps to compute Derrida parameters")
        
    V, D = VD(forward_rates, backward_rates)
    if V is None or D is None:
        raise ValueError("VD function failed to compute parameters")
    
    print(f"\nDebug: Computing Derrida parameters")
        
    # Initialize PDB processor
    processor = PDBProcessor()
    processor.distance_cutoff = 15.0
    
    # Get atoms dictionary and topology
    atoms_dict = processor.read_pdb_atoms(pdb_file)
    print(f"  Found {len(atoms_dict)} residues in PDB")
    
    # Detect sequence
    try:
        sequence = processor.detect_sequence(atoms_dict, topology='linear')
        topology_used = 'linear'
        print("  Detected linear topology")
    except ValueError:
        try:
            sequences = processor.detect_sequence(atoms_dict, topology='branched')
            longest_sequence = max(sequences, key=len)
            sequence = longest_sequence
            topology_used = 'branched'
            print(f"  Detected branched topology, using longest path with {len(sequence)} residues")
        except Exception as e:
            raise ValueError(f"Could not detect valid sequence: {str(e)}")
    
    # 1. Calculate average Fe-Fe spacing using consecutive pairs in the detected sequence
    distances = processor.measure_avg_dist(pdb_file, topology=topology_used)
    if not distances:
        raise ValueError("No Fe-Fe distances found")
    avg_spacing = np.mean([d['distance'] for d in distances])  # In Angstroms
    avg_spacing_cm = avg_spacing * 1E-8  # Convert to cm
    print(f"  Average Fe-Fe spacing: {avg_spacing:.2f} Å")
    
    # 2. Get subunit length from Hda file terminals
    if rates_file and os.path.exists(rates_file):
        # If using rates file, get terminals from there
        first_hda, last_hda = get_terminal_residues(rates_file)
    else:
        # Otherwise use Hda file as before
        first_hda, last_hda = get_terminal_residues(os.path.join(os.path.dirname(pdb_file), f"{steps[0].structure}_Hda.txt"))

    subunit_length = measure_fe_fe_distance(pdb_file, first_hda, last_hda)
    print(f"  Subunit length (between Hda terminals {first_hda}-{last_hda}): {subunit_length:.2f} Å")
    
    # 3. Get full chain length using first and last residues from detected sequence
    full_length = measure_fe_fe_distance(pdb_file, sequence[0], sequence[-1])
    print(f"  Full chain length (residues {sequence[0]}-{sequence[-1]}): {full_length:.2f} Å")
    
    chain_str = ' → '.join(chain_str_parts)
    return {'chain': (V, D, avg_spacing, subunit_length, full_length, len(forward_rates), chain_str, geometries)}

def get_subunit_length(pdb_file: str, first_res: int, last_res: int) -> float:
    """
    Measure Fe-Fe distance between first and last heme specified in Hda file.
    Returns distance in Angstroms.
    """
    fe_coords = {}
    with open(pdb_file, 'r') as f:
        for line in f:
            if line[0:6] in ('ATOM  ', 'HETATM') and 'FE' in line[12:16]:
                res_num = int(line[22:26])
                if res_num in (first_res, last_res):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    fe_coords[res_num] = np.array([x, y, z])

    if len(fe_coords) != 2:
        raise ValueError(f"Could not find FE atoms for both residues {first_res} and {last_res}")

    distance = np.linalg.norm(fe_coords[first_res] - fe_coords[last_res])
    return distance

def compute_conductivity(D_cm2_s: float, rho: float, T: float = 300.0) -> float:
    """
    Compute conductivity from diffusion coefficient using Einstein relation.

    Args:
        D_cm2_s: Diffusion coefficient in cm²/s
        rho: Charge density in charges/cm³
        T: Temperature in Kelvin (default: 300K)

    Returns:
        Conductivity in S/cm (Siemens per centimeter)
    """
    # Constants
    KB = 1.38E-23  # J/K
    E_CHARGE = 1.602E-19  # C

    # Einstein relation: σ = (e²/kT)ρD
    conductivity = (E_CHARGE**2 * rho * D_cm2_s) / (KB * T)

    return conductivity

def compute_current(structure: str, pdb_file: str, D_cm2_s: float, 
                   subunit_length: float, full_length: float) -> Dict[str, Any]:
    """
    Compute diffusive current using different length measurements.
    
    Args:
        structure: Structure identifier
        pdb_file: Path to PDB structure file
        D_cm2_s: Diffusion coefficient in cm²/s
        subunit_length: Length between Hda file terminals in Angstroms
        full_length: Length of entire chain in Angstroms
    """
    # Constants
    KB = 1.38E-23
    T = 300
    V = 0.1
    FIXED_L = 8.27E-5
    R = 7.5E-8
    E_CHARGE = 1.602E-19

    # Convert lengths to cm
    subunit_length_cm = subunit_length * 1E-8
    full_length_cm = full_length * 1E-8

    # Initialize PDB processor
    processor = PDBProcessor()
    processor.distance_cutoff = 15.0
    
    # Get atoms dictionary and topology
    atoms_dict = processor.read_pdb_atoms(pdb_file)
    
    # Detect sequence to get the longest chain
    try:
        sequence = processor.detect_sequence(atoms_dict, topology='linear')
        print("  Using linear topology for heme count")
    except ValueError:
        try:
            sequences = processor.detect_sequence(atoms_dict, topology='branched')
            sequence = max(sequences, key=len)
            print(f"  Using longest branch ({len(sequence)} residues) for heme count")
        except Exception as e:
            raise ValueError(f"Could not detect valid sequence: {str(e)}")
    
    # Count hemes only in the detected sequence
    n_hemes = len(sequence)
    print(f"  Number of hemes in chain: {n_hemes}")

    # Calculate basic geometry using full length for volume
    A = np.pi * R * R
    V_element = A * full_length_cm
    rho = (0.5 * n_hemes) / V_element

    # Calculate conductivity
    conductivity = compute_conductivity(D_cm2_s, rho, T)

    # Calculate currents using subunit length
    I_diff_subunit = ((A * E_CHARGE * E_CHARGE * rho * D_cm2_s) / (KB * T * subunit_length_cm)) * V
    I_diff_fixed = ((A * E_CHARGE * E_CHARGE * rho * D_cm2_s) / (KB * T * FIXED_L)) * V

    return {
        'structure': structure,
        'subunit_length_ang': subunit_length,
        'full_length_ang': full_length,
        'n_hemes': n_hemes,
        'charge_density': rho,
        'cross_sectional_area': A,
        'diffusive_current_subunit_pA': I_diff_subunit * 1E12,
        'diffusive_current_fixed_pA': I_diff_fixed * 1E12,
        'D_cm2_s': D_cm2_s,
        'conductivity_S_cm': conductivity
    }

def save_analysis_results(structure: str, steps: List[ElectronTransferStep], 
                         results: List[tuple], current_results: Dict[str, Any],
                         V: float, D_cm2_s: float, avg_fe_spacing: float,
                         subunit_length: float, full_length: float,
                         output_file: str):
    """Save combined Derrida and current analysis results."""
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"Analysis Results for {structure}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write electron transfer rates with geometries
        f.write("Electron Transfer Rates and Geometries:\n")
        f.write("-" * 40 + "\n")
        headers = ["Donor", "Acceptor", "Geometry", "k_forward (s⁻¹)", "k_backward (s⁻¹)"]
        table_data = []
        
        for step, (kf, kb, _, _) in zip(steps, results):
            table_data.append([
                step.donor,
                step.acceptor,
                step.geometry,
                f"{kf:.2E}",
                f"{kb:.2E}"
            ])
        
        f.write(tabulate(table_data, headers=headers, tablefmt="grid", disable_numparse=True))
        f.write("\n\n")
        
        # Write Derrida parameters and length measurements
        f.write("Derrida Analysis and Length Measurements:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Velocity (V): {V:.2E} s⁻¹\n")
        f.write(f"Diffusion Coefficient (D): {D_cm2_s:.2E} cm²/s\n")
        f.write(f"Average Fe-Fe spacing: {avg_fe_spacing:.2f} Å\n")
        f.write(f"Subunit length: {subunit_length:.2f} Å\n")
        f.write(f"Full chain length: {full_length:.2f} Å\n\n")
        
        # Write current analysis
        f.write("Current and Conductivity Analysis:\n")
        f.write("-" * 30 + "\n")     # Slightly longer line of dashes
        f.write(f"Structure Parameters:\n")
        f.write(f"  Number of Hemes: {current_results['n_hemes']}\n")
        f.write(f"  Cross-sectional Area: {current_results['cross_sectional_area']:.2E} cm²\n")
        f.write(f"  Charge Density: {current_results['charge_density']:.2E} charges/cm³\n")
        f.write(f"  Conductivity: {current_results['conductivity_S_cm']:.2E} S/cm\n\n")   # New line

        f.write(f"Calculated Currents:\n")
        f.write(f"  Diffusive Current (Subunit Length): {current_results['diffusive_current_subunit_pA']:.2E} pA\n")
        f.write(f"  Diffusive Current (Fixed Length): {current_results['diffusive_current_fixed_pA']:.2E} pA\n")

def compute_current_vs_length(D_cm2_s: float, n_hemes: int, wire_length_ang: float,
                            start_nm: float = 1.0,
                            end_nm: float = 1000.0,
                            steps: int = 1000) -> List[Tuple[float, float]]:
    """
    Compute diffusive current as a function of length.

    Args:
        D_cm2_s: Diffusion coefficient in cm²/s
        n_hemes: Number of hemes in the longest chain (not total structure)
        wire_length_ang: Length of the longest chain in Angstroms
        start_nm: Starting length in nanometers
        end_nm: Ending length in nanometers
        steps: Number of length points to calculate

    Returns:
        List of tuples (length_nm, current_pA)
    """
    if wire_length_ang <= 0:
        raise ValueError(f"Wire length must be positive, got {wire_length_ang} Å")

    # Constants
    KB = 1.38E-23  # J/K
    T = 300  # K
    V = 0.1  # V
    R = 7.5E-8  # cm (wire radius)
    E_CHARGE = 1.602E-19  # C

    # Calculate geometry
    A = np.pi * R * R  # cm²

    # Calculate charge density using actual wire length and chain heme count
    wire_length_cm = wire_length_ang * 1E-8
    V_element = A * wire_length_cm
    rho = (0.5 * n_hemes) / V_element  # charges/cm³

    # Generate length points in nm
    lengths_nm = np.linspace(start_nm, end_nm, steps)

    results = []
    for L_nm in lengths_nm:
        # Convert length to cm for calculations
        L_cm = L_nm * 1E-7

        # Calculate current using fixed charge density
        I_diff = ((A * E_CHARGE * E_CHARGE * rho * D_cm2_s) / (KB * T * L_cm)) * V

        # Convert to picoamps
        I_diff_pA = I_diff * 1E12

        results.append((L_nm, I_diff_pA))

    return results

def save_current_vs_length(structure: str, results: List[Tuple[float, float]],
                          D_cm2_s: float, n_hemes: int, wire_length_ang: float,
                          output_file: str):
    """
    Save length-dependent current analysis results with fixed-width columns.

    Args:
        structure: Structure identifier
        results: List of (length_nm, current_pA) tuples
        D_cm2_s: Diffusion coefficient used
        n_hemes: Number of hemes in the longest chain
        wire_length_ang: Length of the longest chain in Angstroms
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        # Write parameters as comments
        f.write(f"# Length-Dependent Current Analysis for {structure}\n")
        f.write(f"# Diffusion Coefficient: {D_cm2_s:.2E} cm²/s\n")
        f.write(f"# Number of Hemes in Chain: {n_hemes}\n")
        f.write(f"# Chain Length: {wire_length_ang:.1f} Å ({wire_length_ang/10:.1f} nm)\n")
        f.write(f"# Applied Voltage: 0.1 V\n")
        f.write(f"# Temperature: 300 K\n")
        f.write("#\n")

        # Write column headers with fixed width
        f.write(f"{'length_nm':>12s}   {'current_pA':>12s}\n")

        # Write data in fixed-width columns
        for length, current in results:
            f.write(f"{length:12.1f}   {current:12.6E}\n")

def main():
    parser = argparse.ArgumentParser(description='Calculate Derrida parameters and diffusive currents')
    parser.add_argument('--structures', nargs='+', required=True,
                       help='Structure names (e.g., 8E5F 8E5G)')
    parser.add_argument('--input-dir', default='.',
                       help='Directory containing input files (default: current directory)')
    parser.add_argument('--output-prefix', default='derrida',
                       help='Prefix for output files (default: derrida)')
    parser.add_argument('--length-start', type=float, default=1.0,
                       help='Starting length in nm (default: 1.0)')
    parser.add_argument('--length-end', type=float, default=1000.0,
                       help='Ending length in nm (default: 1000.0)')
    parser.add_argument('--length-steps', type=int, default=1000,
                       help='Number of length points (default: 1000)')

    args = parser.parse_args()

    for structure in args.structures:
        print(f"\nProcessing structure: {structure}")
        print("=" * 50)

        try:
            # Check for rates file first
            rates_file = os.path.join(args.input_dir, f"{structure}_rates.txt")
            pdb_file = os.path.join(args.input_dir, f"{structure}.pdb")

            if not os.path.exists(pdb_file):
                print(f"Error: Missing PDB file for {structure}")
                continue

            steps = []
            results = []

            if os.path.exists(rates_file):
                print(f"Found rates file: {rates_file}")
                print("Using pre-computed rates...")

                rates_data = parse_rates_file(rates_file)

                for donor, acceptor, kf, kb, geometry in rates_data:
                    step = ElectronTransferStep(
                        structure=structure,
                        donor=donor,
                        acceptor=acceptor,
                        dg=0.0,  # placeholder since we're using direct rates
                        lambda_reorg=0.0,  # placeholder
                        hda=0.0,  # placeholder
                        geometry=geometry
                    )
                    steps.append(step)
                    results.append((kf, kb, 0.0, 0.0))  # activation energies not needed

            else:
                print("No rates file found. Computing rates from energetic quantities...")
                # Original rate computation logic
                dg_file = os.path.join(args.input_dir, f"{structure}_DG.txt")
                lambda_file = os.path.join(args.input_dir, f"{structure}_Lambda.txt")
                hda_file = os.path.join(args.input_dir, f"{structure}_Hda.txt")

                if not all(os.path.exists(f) for f in [dg_file, lambda_file, hda_file]):
                    print(f"Error: Missing required files for {structure}")
                    continue

                dg_data = parse_dg_file(dg_file)
                lambda_data = parse_lambda_file(lambda_file)
                hda_data = parse_hda_file(hda_file)

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
                        rates = compute_marcus_rates(step.hda, step.lambda_reorg, step.dg)
                        results.append(rates)

            # Compute Derrida parameters
            print("Computing Derrida parameters...")
            derrida_results = compute_derrida_parameters(steps, results, pdb_file, rates_file)

            if 'chain' in derrida_results:
                V, D, avg_spacing, subunit_length, full_length, n_steps, chain_str, geometries = derrida_results['chain']
                
                # Convert D to physical units using average spacing
                D_cm2_s = D * (avg_spacing * 1E-8)**2
                
                print(f"  Velocity: {V:.2E} s⁻¹")
                print(f"  Diffusion coefficient: {D_cm2_s:.2E} cm²/s")
                print(f"  Average Fe-Fe spacing: {avg_spacing:.2f} Å")
                print(f"  Subunit length: {subunit_length:.2f} Å")
                print(f"  Full chain length: {full_length:.2f} Å")

                # Compute current using the subunit and full chain lengths
                print("Computing currents...")
                current_results = compute_current(structure, pdb_file, D_cm2_s, 
                                               subunit_length, full_length)

                # Save results
                output_file = f"{args.output_prefix}_{structure}_analysis.txt"
                print(f"Saving analysis results to {output_file}")
                save_analysis_results(structure, steps, results, current_results, 
                                    V, D_cm2_s, avg_spacing, subunit_length,
                                    full_length, output_file)

                # Compute and save length-dependent current
                print("Computing length-dependent current...")
                length_current_results = compute_current_vs_length(
                    D_cm2_s=D_cm2_s,
                    n_hemes=current_results['n_hemes'],
                    wire_length_ang=full_length,  # Use full length for charge density
                    start_nm=args.length_start,
                    end_nm=args.length_end,
                    steps=args.length_steps
                )
                
                # Save length-dependent results
                length_output_file = f"{args.output_prefix}_{structure}_length_current.txt"
                print(f"Saving length-dependent results to {length_output_file}")
                save_current_vs_length(
                    structure=structure,
                    results=length_current_results,
                    D_cm2_s=D_cm2_s,
                    n_hemes=current_results['n_hemes'],
                    wire_length_ang=full_length,
                    output_file=length_output_file
                )
            else:
                print("Error: Could not compute Derrida parameters")

        except Exception as e:
            print(f"Error processing structure {structure}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
