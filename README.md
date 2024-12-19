# Electron Transfer Analysis Package

A comprehensive Python package for analyzing electron transfer in multi-heme systems.

## Overview

This package provides tools for analyzing electron transfer pathways, flux calculations, heme cooperativity, and parameter space exploration in multi-heme systems. It includes modules for:

- Derrida analysis of electron transfer kinetics
- Flux and current analysis
- Heme cooperativity analysis
- Parameter space exploration
- Parameter visualization
- Redox titration analysis

## Installation

```bash
pip install et-analysis
```

## Command-Line Tools

The package provides several command-line tools:

- `et-derrida`: Run Derrida analysis
- `et-flux`: Run flux analysis  
- `et-coop`: Run cooperativity analysis
- `et-explore`: Run parameter exploration
- `et-visualize`: Run parameter visualization
- `et-redox`: Run redox titration analysis

## Module Documentation

### AnalyzeHemeCooperativity

Analyzes cooperativity in multi-heme systems by calculating:
- Independent oxidation energies
- Sequential oxidation pathways
- Geometric oxidation sequences
- Flux calculations
- Delta G landscapes

#### Usage Example
```bash
et-coop plot_option adjacent_only make_periodic biodc_mat biodc_model biodc_eng_shift biodc_int_scale [qm_mat qm_model qm_eng_shift qm_int_scale] [exp_mat exp_model exp_eng_shift exp_int_scale]
```

Arguments:
- `plot_option`: Choose 'ox', 'red', or 'both' for oxidized/reduced fraction plots
- `adjacent_only`: Calculate only adjacent hemes ('true'/'false')
- `make_periodic`: Include step from last heme back to first ('true'/'false')
- `biodc_mat`: BioDC matrix file
- `biodc_model`: Analysis model type ('geo', 'seq', or 'both')
- `biodc_eng_shift`: Energy shift for BioDC analysis
- `biodc_int_scale`: Interaction scale for BioDC analysis

### FluxCurrentAnalysis 

Calculates electron transfer rates and fluxes through multi-heme chains by:
- Computing Marcus rates with/without optimization
- Analyzing flux pathways
- Generating rate distributions
- Calculating redox potentials and populations

#### Usage Example
```bash
et-flux --structures STRUCT1 [STRUCT2 ...] [--input-dir DIR] [--output-prefix PREFIX] [--shift-diagonal SHIFT] [--scale-offdiagonal SCALE] [--replicate N]
```

Arguments:
- `--structures`: List of structure names to analyze
- `--input-dir`: Directory containing input files (default: current)
- `--output-prefix`: Prefix for output files (default: "flux")
- `--shift-diagonal`: Shift for diagonal matrix elements
- `--scale-offdiagonal`: Scale factor for off-diagonal elements
- `--replicate`: Number of times to replicate the system

### DerridaCurrentAnalysis

Analyzes electron transport using Derrida's method by:
- Calculating velocity and diffusion coefficients
- Analyzing pathways and currents 
- Comparing geometric vs thermodynamic paths
- Generating potential progression plots

#### Usage Example
```bash
et-derrida --structures STRUCT1 [STRUCT2 ...] [--input-dir DIR] [--output-prefix PREFIX] [--length-start START] [--length-end END] [--length-steps STEPS]
```

Arguments:
- `--structures`: List of structures to analyze
- `--input-dir`: Directory containing input files
- `--output-prefix`: Prefix for output files (default: "derrida")
- `--length-start`: Starting length for length-dependent analysis (nm)
- `--length-end`: Ending length (nm)
- `--length-steps`: Number of length points

### Parameter Exploration

Explores parameter space for electron transfer systems by:
- Generating random parameter sets within physical ranges
- Filtering results based on target diffusion coefficients
- Parallelizing computation across multiple cores
- Removing redundant parameter sets

#### Usage Example 
```bash
et-explore --num-tot N --num-s S --num-t T --seq SEQ --num-sets SETS --free-eng-opt BOOL --pdbname FILE --seqids IDS [--d-target D] [--d-tolerance TOL] [--max-attempts MAX] [--num-processes P]
```

Required Arguments:
- `--num-tot`: Total number of hemes
- `--num-s`: Number of S-type pairs
- `--num-t`: Number of T-type pairs 
- `--seq`: Sequence of S/T pairs (e.g., 'STSTST')
- `--num-sets`: Number of parameter sets to generate
- `--free-eng-opt`: Use free energy optimization ('true'/'false')
- `--pdbname`: Path to PDB file
- `--seqids`: Comma-separated heme residue IDs

Optional Arguments:
- `--d-target`: Target diffusion coefficient
- `--d-tolerance`: Tolerance for matching target D
- `--max-attempts`: Maximum attempts to find matching sets
- `--num-processes`: Number of CPU processes to use

### Parameter Visualization 

Visualizes parameter evolution and distributions by creating:
- Parameter trajectory plots
- Heatmap ribbon plots
- Parameter envelope plots
- Statistical analysis visualizations

#### Usage Example
```bash
et-visualize results_files [results_files ...] [--output-prefix PREFIX] [--n-trajectories N] [--n-bins BINS] [--hide-trajectories] [--plots TYPES]
```

Arguments:
- `results_files`: One or more results files to visualize
- `--output-prefix`: Prefix for output files
- `--n-trajectories`: Number of trajectories to show
- `--n-bins`: Number of bins for histograms
- `--hide-trajectories`: Don't show individual trajectories
- `--plots`: Plot types to generate ('all', 'trajectories', 'heatmap', 'envelope')

## File Formats

### Input Files

#### Energy Matrix Files
- Format: Square matrix in bracketed rows
- Values in meV
- Example:
```
[E11, E12, E13]
[E21, E22, E23]
[E31, E32, E33]
```

#### Rates Files
Format: `HEM-X -> HEM-Y; kf = X.XXE+XX s^-1; kb = X.XXE+XX s^-1; geometry = S/T/U`

#### DG Files
Format: `(HEM-X = E1 eV) -> (HEM-Y = E2 eV); DG = E3 eV`

#### Lambda Files
Contains reorganization energies for each heme pair

#### Hda Files
Contains electronic coupling values in meV

### Output Files

The package generates various output files including:
- Analysis summaries (TXT)
- Rate distribution plots (PNG)
- Potential progression plots (PNG)
- Flux landscapes (PNG)
- Parameter statistics (TXT)
- Visualization plots (PNG)

## Dependencies

- NumPy
- SciPy 
- Matplotlib
- Pandas
- Tabulate
- Scikit-learn

## Development

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests (`pytest`)
5. Submit pull request

## License

MIT License

## Citation

If you use this package in your research, please cite:

[Citation information will be added here]
