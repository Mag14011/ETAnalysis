import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.patheffects import withStroke
from matplotlib import animation
import scipy.interpolate as interp
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde, pearsonr
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import os
import argparse
from datetime import datetime
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

class PlotType(Enum):
    TRAJECTORIES = "trajectories"
    HEATMAP = "heatmap"
    ENVELOPE = "envelope"

class SelectionMethod(Enum):
    SORT = "sort"
    RANDOM = "random"
    KMEANS = "kmeans"
    MEAN = "mean"

class ParameterEvolutionVisualizer:
    def __init__(self, results_files: List[str]):
        """Initialize visualizer with multiple results files."""
        # Verify sequence consistency before loading data
        if not self._verify_sequence_consistency(results_files):
            raise ValueError("Input files have different sequences! All files must have the same sequence.")
        
        self.results = self.load_multiple_results(results_files)
        # Use sequence from first file since we verified they're all the same
        self.sequence = self._extract_sequence_from_header(results_files[0])
        self.num_pairs = len(self.sequence)
        self.num_hemes = self.num_pairs + 1

    def load_multiple_results(self, filenames: List[str]) -> Dict:
        """Load and combine results from multiple files."""
        combined_data = {
            'couplings': [],
            'lambdas': [],
            'deltaG': [],
            'D': []
        }

        total_trajectories = 0
        for filename in filenames:
            print(f"\nLoading data from {filename}...")
            try:
                file_data = self._load_single_file(filename)
                
                # Append data from this file to combined results
                for key in combined_data:
                    if isinstance(file_data[key], np.ndarray):
                        combined_data[key].extend(file_data[key])
                    else:
                        combined_data[key].extend(file_data[key])
                
                file_trajectories = len(file_data['D'])
                total_trajectories += file_trajectories
                print(f"Loaded {file_trajectories} trajectories from {filename}")
                
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                print("Skipping this file and continuing with others...")
                continue

        # Convert lists to numpy arrays
        for key in combined_data:
            combined_data[key] = np.array(combined_data[key])
            print(f"{key} final shape: {combined_data[key].shape}")

        print(f"\nTotal trajectories loaded: {total_trajectories}")
        
        if total_trajectories == 0:
            raise ValueError("No valid data was loaded from any of the input files!")
            
        return combined_data

    def _load_single_file(self, filename: str) -> Dict:
        """Load results from a single file."""
        data = {
            'couplings': [],
            'lambdas': [],
            'deltaG': [],
            'D': []
        }

        with open(filename, 'r') as f:
            header_found = False
            coupling_count = None
            lambda_count = None
            deltaG_count = None

            for line_num, line in enumerate(f, 1):
                if line.startswith('#') or not line.strip():
                    continue

                if 'Index' in line:
                    header_found = True
                    headers = line.strip().split()
                    coupling_count = sum(1 for h in headers if 'V' in h and '(' in h)
                    lambda_count = sum(1 for h in headers if 'L' in h and '-' in h)
                    deltaG_count = sum(1 for h in headers if 'dG' in h)
                    print(f"Found columns: {coupling_count} couplings, {lambda_count} lambdas, {deltaG_count} deltaGs")
                    continue

                if not header_found or '----' in line:
                    continue

                try:
                    values = line.strip().split()
                    if not values:
                        continue

                    # Verify we have enough values
                    expected_values = 1 + coupling_count + lambda_count + deltaG_count + 1  # Index + params + D
                    if len(values) < expected_values:
                        raise ValueError(f"Line has {len(values)} values, expected {expected_values}")

                    # Extract values
                    couplings = [float(values[i]) for i in range(1, coupling_count + 1)]
                    start_idx = coupling_count + 1
                    lambdas = [float(values[i]) for i in range(start_idx, start_idx + lambda_count)]
                    start_idx = start_idx + lambda_count
                    deltaGs = [float(values[i]) for i in range(start_idx, start_idx + deltaG_count)]
                    D_value = float(values[-1])

                    # Basic validation
                    if any(v <= 0 for v in couplings) or any(v <= 0 for v in lambdas) or D_value <= 0:
                        print(f"Warning: Line {line_num} contains suspicious zero or negative values")

                    data['couplings'].append(couplings)
                    data['lambdas'].append(lambdas)
                    data['deltaG'].append(deltaGs)
                    data['D'].append(D_value)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing line {line_num} in {filename}: {line.strip()}")
                    print(f"Error details: {str(e)}")
                    continue

        # Verify we loaded some data
        if not data['D']:
            raise ValueError(f"No valid data was loaded from {filename}")

        # Convert to numpy arrays and verify shapes
        n_trajectories = len(data['D'])
        for key in data:
            data[key] = np.array(data[key])
            if key != 'D' and data[key].shape[0] != n_trajectories:
                raise ValueError(f"Inconsistent number of trajectories in {key} data")

        return data

    def _extract_sequence_from_header(self, filename: str) -> str:
        """Extract sequence from file header."""
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        for param in line.split():
                            if param.startswith('seq='):
                                return param.split('=')[1]
            raise ValueError(f"No sequence found in header of {filename}")
        except Exception as e:
            raise ValueError(f"Error reading sequence from {filename}: {str(e)}")

    def _verify_sequence_consistency(self, files: List[str]) -> bool:
        """Verify that all input files have the same sequence."""
        try:
            first_sequence = self._extract_sequence_from_header(files[0])
            for file in files[1:]:
                current_sequence = self._extract_sequence_from_header(file)
                if current_sequence != first_sequence:
                    print(f"Sequence mismatch!")
                    print(f"First file sequence: {first_sequence}")
                    print(f"Sequence in {file}: {current_sequence}")
                    return False
            return True
        except Exception as e:
            raise ValueError(f"Error verifying sequence consistency: {str(e)}")

    def select_trajectories(self, 
                        method: SelectionMethod,
                        n_trajectories: int = 50,
                        n_bins: int = 10) -> List[int]:
        """Select trajectory indices using specified method."""
        d_values = np.array(self.results['D'])
        
        # If n_trajectories is -1 or equals total trajectories, return all indices
        if n_trajectories == -1 or n_trajectories >= len(d_values):
            if method == SelectionMethod.SORT:
                # Return all indices sorted by D value
                return list(np.argsort(d_values))
            else:
                # Return all indices in original order
                return list(range(len(d_values)))
            
#       if method == SelectionMethod.SORT:
#           indices = np.argsort(d_values)[::-1]
#           return list(indices[:n_trajectories])

        if method == SelectionMethod.SORT:
            indices = np.argsort(d_values)
            # Return both lowest and highest D trajectories
            return [indices[0], indices[-1]]  # lowest and highest

        # For histogram-based methods
        log_d = np.log10(d_values)
        bin_edges = np.linspace(log_d.min(), log_d.max(), n_bins + 1)
        bin_indices = np.digitize(log_d, bin_edges) - 1
        
        trajectories_per_bin = max(1, n_trajectories // n_bins)
        selected_indices = []
        
        for bin_idx in range(n_bins):
            bin_mask = (bin_indices == bin_idx)
            bin_trajectories = np.where(bin_mask)[0]
            
            if len(bin_trajectories) == 0:
                continue
                
            if method == SelectionMethod.RANDOM:
                n_samples = min(trajectories_per_bin, len(bin_trajectories))
                selected = np.random.choice(bin_trajectories, size=n_samples, replace=False)
                
            elif method == SelectionMethod.KMEANS:
                n_samples = min(trajectories_per_bin, len(bin_trajectories))
                if n_samples == 1:
                    selected = [bin_trajectories[0]]
                else:
                    features = []
                    for idx in bin_trajectories:
                        trajectory_features = []
                        trajectory_features.extend(self.results['couplings'][idx])
                        trajectory_features.extend(self.results['lambdas'][idx])
                        trajectory_features.extend(self.results['deltaG'][idx])
                        features.append(trajectory_features)
                    
                    features = np.array(features)
                    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)
                    
                    kmeans = KMeans(n_clusters=n_samples, n_init=10)
                    kmeans.fit(features)
                    
                    selected = []
                    for center in kmeans.cluster_centers_:
                        distances = np.linalg.norm(features - center, axis=1)
                        closest_idx = bin_trajectories[distances.argmin()]
                        selected.append(closest_idx)
                
            elif method == SelectionMethod.MEAN:
                n_samples = min(trajectories_per_bin, len(bin_trajectories))
                
                mean_params = {}
                for param in ['couplings', 'lambdas', 'deltaG']:
                    bin_params = [self.results[param][idx] for idx in bin_trajectories]
                    mean_params[param] = np.mean(bin_params, axis=0)
                
                distances = []
                for idx in bin_trajectories:
                    dist = 0
                    for param in ['couplings', 'lambdas', 'deltaG']:
                        param_dist = np.linalg.norm(
                            np.array(self.results[param][idx]) - mean_params[param]
                        )
                        dist += param_dist
                    distances.append(dist)
                
                closest_indices = np.argsort(distances)[:n_samples]
                selected = bin_trajectories[closest_indices]
            
            selected_indices.extend(selected)
        
        return selected_indices[:n_trajectories]

    def calculate_activation_energies(self, lambda_vals: List[float],
                                    deltaG_vals: List[float]) -> List[float]:
        """Calculate activation energies."""
        return [(l + dg)**2 / (4 * l) for l, dg in zip(lambda_vals, deltaG_vals)]

    def _get_param_key(self, param: str) -> str:
        """Get the correct parameter key for accessing data."""
        param_map = {
            'coupling': 'couplings',
            'lambda': 'lambdas',
            'deltaG': 'deltaG',
            'activation': 'activation'
        }
        return param_map.get(param, param)

    def _create_x_labels(self):
        """Create x-axis labels showing pair relationships."""
        labels = []
        for i in range(self.num_pairs):
            labels.append(f'{i+1}-{i+2}\n({self.sequence[i]})')
        return labels

    def plot_parameter_heatmap_ribbon(self,
                                    param: str,
                                    show_trajectories: bool = False,
                                    n_trajectories: int = 20,
                                    selection_method: SelectionMethod = SelectionMethod.SORT,
                                    n_bins: int = 10,
                                    discrete: bool = False,
                                    figsize: Tuple[int, int] = (10, 10)) -> Tuple[plt.Figure, plt.Axes]:
        """Create heatmap ribbon plot with aligned distribution panel and horizontal colorbars."""
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create gridspec with proper spacing
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1])
        gs.update(top=0.85, bottom=0.1, left=0.12, right=0.88, hspace=0.05)
        
        # Create axes
        ax_heat = fig.add_subplot(gs[0])
        ax_dist = fig.add_subplot(gs[1], sharex=ax_heat)
        
        # Get parameter values
        if param == 'activation':
            all_values = [
                self.calculate_activation_energies(
                    self.results['lambdas'][i],
                    self.results['deltaG'][i]
                )
                for i in range(len(self.results['D']))
            ]
        else:
            param_key = self._get_param_key(param)
            all_values = self.results[param_key]
                
        all_values = np.array(all_values)
        
        if discrete:
            # Discrete version - calculate density at each pair position
            x_positions = np.arange(self.num_pairs)
            y_grid = np.linspace(np.min(all_values), np.max(all_values), 100)
            density = np.zeros((len(y_grid), len(x_positions)))
            
            # Calculate density at each discrete position
            for i, pos in enumerate(x_positions):
                values = all_values[:, pos]
                try:
                    kde = gaussian_kde(values)
                    density[:, i] = kde(y_grid)
                except:
                    continue
            
            # Create edges for pcolormesh
            x_edges = np.arange(-0.5, self.num_pairs + 0.5)
            y_edges = np.linspace(np.min(all_values), np.max(all_values), 101)
            
            # Plot heatmap with discrete blocks
            im = ax_heat.pcolormesh(x_edges, y_edges, density, cmap='YlOrBr', alpha=0.7)
            
        else:
            # Interpolated version - smooth transitions between positions
            x_grid = np.linspace(-0.5, self.num_pairs - 0.5, 100)
            y_grid = np.linspace(np.min(all_values), np.max(all_values), 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            density = np.zeros_like(X)
            
            for i, x in enumerate(x_grid):
                pos = int(round(x))
                if pos < 0:
                    pos = 0
                if pos >= self.num_pairs:
                    pos = self.num_pairs - 1
                values = all_values[:, pos]
                try:
                    kde = gaussian_kde(values)
                    density[:, i] = kde(y_grid)
                except:
                    continue
            
            # Smooth density
            density = gaussian_filter1d(density, sigma=1, axis=1)
            
            # Plot interpolated heatmap
            im = ax_heat.pcolormesh(X, Y, density, cmap='YlOrBr', alpha=0.7)
        
        # Create horizontal colorbar for density
        cax1 = fig.add_axes([0.15, 0.92, 0.3, 0.02])
        cbar1 = fig.colorbar(im, cax=cax1, orientation='horizontal')
        cbar1.set_label('Density')
        
        # Handle trajectories if requested
        if show_trajectories:
            selected_indices = self.select_trajectories(
                selection_method, n_trajectories, n_bins
            )
            
            d_vals = np.array([self.results['D'][i] for i in selected_indices])
            
            # Find appropriate power of 10 for normalization
            max_d = np.max(d_vals)
            power = int(np.floor(np.log10(max_d)))
            scale_factor = 10**power
            
            # Normalize the values directly
            d_vals_normalized = d_vals/scale_factor
            
            # Create normalized colormap without LogNorm
            norm = plt.Normalize(vmin=min(d_vals_normalized), vmax=max(d_vals_normalized))
            
            for idx in selected_indices:
                if param == 'activation':
                    y_vals = self.calculate_activation_energies(
                        self.results['lambdas'][idx],
                        self.results['deltaG'][idx]
                    )
                else:
                    y_vals = self.results[self._get_param_key(param)][idx]
                
                coupling_vals = self.results['couplings'][idx]
                
                x = np.linspace(0, self.num_pairs-1, 100)
                y_interp = interp.interp1d(range(self.num_pairs), y_vals, kind='cubic')
                coupling_interp = interp.interp1d(range(self.num_pairs), coupling_vals, kind='cubic')
                
                y = y_interp(x)
                coupling = coupling_interp(x)
                thickness = 0.5 + (coupling - 1) * (2.5 - 0.5) / (13 - 1)
                
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                lc = LineCollection(segments, 
                                linewidths=thickness[:-1],
                                colors=[plt.cm.PuBu_r(norm(self.results['D'][idx]/scale_factor))],
                                alpha=0.7)
                ax_heat.add_collection(lc)
            
            # Add second horizontal colorbar for diffusion coefficient
            cax2 = fig.add_axes([0.55, 0.92, 0.3, 0.02])
            sm = plt.cm.ScalarMappable(norm=norm, cmap='PuBu_r')
            cbar2 = fig.colorbar(sm, cax=cax2, orientation='horizontal')
            cbar2.set_label(f'Diffusion Coefficient (10^{power} cm²/s)')
            
            # Format tick labels to show just the decimal numbers
            def fmt(x, p):
                return f'{x:.1f}'
            cbar2.formatter = plt.FuncFormatter(fmt)
            cbar2.update_ticks()
        
        # Add violin plots in bottom panel
        violin_parts = ax_dist.violinplot(
            all_values, 
            positions=range(self.num_pairs),
            showmeans=True,
            showmedians=True,
            widths=0.8
        )
        
        # Customize violin plots
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightgray')
            pc.set_alpha(0.7)
        violin_parts['cmeans'].set_color('black')
        violin_parts['cmedians'].set_color('red')
        
        # Add vertical lines to both panels
        for i in range(self.num_pairs):
            ax_heat.axvline(i, color='gray', linestyle='--', alpha=0.3)
            ax_dist.axvline(i, color='gray', linestyle='--', alpha=0.3)
        
        # Labels
        param_labels = {
            'coupling': 'Electronic Coupling [meV]',
            'lambda': 'Reorganization Energy [eV]',
            'deltaG': 'Reaction Free Energy [eV]',
            'activation': 'Activation Energy [eV]'
        }
        
        ax_heat.set_ylabel(param_labels[param])
        ax_dist.set_xlabel('Heme Pair')
        ax_dist.set_ylabel('Distribution')
        
        # Set x-axis ticks and labels
        ax_heat.set_xticks([])
        ax_dist.set_xticks(range(self.num_pairs))
        ax_dist.set_xticklabels(self._create_x_labels())
        
        # Add legend for violin plot
        ax_dist.plot([], [], color='black', label='Mean', linestyle='-')
        ax_dist.plot([], [], color='red', label='Median', linestyle='-')
        ax_dist.legend(loc='upper right')
        
        return fig, (ax_heat, ax_dist)

    def calculate_parameter_statistics(self, param: str) -> Dict:
            """Calculate comprehensive statistics for parameter values at each position."""
            stats = {}
            
            # Get parameter values
            if param == 'activation':
                all_values = [
                    self.calculate_activation_energies(
                        self.results['lambdas'][i],
                        self.results['deltaG'][i]
                    )
                    for i in range(len(self.results['D']))
                ]
            else:
                param_key = self._get_param_key(param)
                all_values = self.results[param_key]
            
            # Convert to numpy array for easier manipulation
            all_values = np.array(all_values)
            d_values = np.array(self.results['D'])
            
            # For pairs instead of hemes
            num_positions = self.num_pairs
            
            for pos in range(num_positions):
                pos_values = all_values[:, pos]
                
                # Calculate basic statistics
                stats[pos] = {
                    'mean': np.mean(pos_values),
                    'median': np.median(pos_values),
                    'std': np.std(pos_values),
                    'quartiles': np.percentile(pos_values, [25, 50, 75]),
                    'min': np.min(pos_values),
                    'max': np.max(pos_values),
                }
                
                # Calculate correlation with D
                corr, p_value = pearsonr(pos_values, d_values)
                stats[pos]['correlation_with_D'] = {
                    'coefficient': corr,
                    'p_value': p_value
                }
            
            return stats

    def plot_parameter_evolution(self, 
                            param: str,
                            selection_method: SelectionMethod = SelectionMethod.SORT,
                            n_trajectories: int = 50,
                            n_bins: int = 10,
                            figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """Create parameter evolution plot with varying line thickness."""
        fig, ax = plt.subplots(figsize=figsize)
        
        selected_indices = self.select_trajectories(selection_method, n_trajectories, n_bins)
        
        d_vals = np.array([self.results['D'][i] for i in selected_indices])
        norm = LogNorm(vmin=min(self.results['D']), vmax=max(self.results['D']))
        
        x = np.linspace(0, self.num_pairs-1, 100)
        
        for idx in selected_indices:
            if param == 'activation':
                y_vals = self.calculate_activation_energies(
                    self.results['lambdas'][idx],
                    self.results['deltaG'][idx]
                )
            else:
                param_key = self._get_param_key(param)
                y_vals = self.results[param_key][idx]
            
            coupling_vals = self.results['couplings'][idx]
            
            y_interp = interp.interp1d(range(self.num_pairs), y_vals, kind='cubic')
            coupling_interp = interp.interp1d(range(self.num_pairs), coupling_vals, kind='cubic')
            
            y = y_interp(x)
            coupling = coupling_interp(x)
            thickness = 0.5 + (coupling - 1) * (2.5 - 0.5) / (13 - 1)
            
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            lc = LineCollection(segments, linewidths=thickness[:-1],
                            colors=[plt.cm.rainbow(norm(self.results['D'][idx]))],
                            alpha=0.7)
            ax.add_collection(lc)
        
        # Add vertical lines
        for i in range(self.num_pairs):
            ax.axvline(i, color='gray', linestyle='--', alpha=0.3)
        
        # Labels and title
        param_labels = {
            'coupling': 'Electronic Coupling [meV]',
            'lambda': 'Reorganization Energy [eV]',
            'deltaG': 'Reaction Free Energy [eV]',
            'activation': 'Activation Energy [eV]'
        }
        
        ax.set_xlabel('Heme Pair')
        ax.set_ylabel(param_labels[param])
        title = f'Parameter Evolution: {param_labels[param]}\nSequence: {self.sequence}'
        title += f'\nSelection Method: {selection_method.value}'
        ax.set_title(title)
        
        ax.set_xticks(range(self.num_pairs))
        ax.set_xticklabels(self._create_x_labels())
        
        sm = plt.cm.ScalarMappable(norm=norm, cmap='rainbow')
        plt.colorbar(sm, ax=ax, label='Diffusion Coefficient [cm²/s]')
        
        ax.set_xlim(-0.2, self.num_pairs - 0.8)
        ax.autoscale_view()
        
        plt.tight_layout()
        return fig, ax

    def plot_parameter_heatmap_ribbon(self,
                                    param: str,
                                    show_trajectories: bool = False,
                                    n_trajectories: int = 20,
                                    selection_method: SelectionMethod = SelectionMethod.SORT,
                                    n_bins: int = 10,
                                    discrete: bool = False,
                                    figsize: Tuple[int, int] = (10, 10)) -> Tuple[plt.Figure, plt.Axes]:
        """Create heatmap ribbon plot with aligned distribution panel and horizontal colorbars."""
        # Create figure
        fig = plt.figure(figsize=figsize)

        # Create gridspec with proper spacing
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1])
        gs.update(top=0.85, bottom=0.1, left=0.12, right=0.88, hspace=0.05)

        # Create axes
        ax_heat = fig.add_subplot(gs[0])
        ax_dist = fig.add_subplot(gs[1], sharex=ax_heat)

        # Get parameter values
        if param == 'activation':
            all_values = [
                self.calculate_activation_energies(
                    self.results['lambdas'][i],
                    self.results['deltaG'][i]
                )
                for i in range(len(self.results['D']))
            ]
        else:
            param_key = param
            all_values = self.results[param_key]

        all_values = np.array(all_values)

        if discrete:
            # Discrete version - calculate density at each pair position
            x_positions = np.arange(self.num_pairs)
            y_grid = np.linspace(np.min(all_values), np.max(all_values), 100)
            density = np.zeros((len(y_grid), len(x_positions)))

            # Calculate density at each discrete position
            for i, pos in enumerate(x_positions):
                values = all_values[:, pos]
                try:
                    kde = gaussian_kde(values)
                    density[:, i] = kde(y_grid)
                except:
                    continue

            # Create edges for pcolormesh
            x_edges = np.arange(-0.5, self.num_pairs + 0.5)
            y_edges = np.linspace(np.min(all_values), np.max(all_values), 101)

            # Plot heatmap with discrete blocks
            im = ax_heat.pcolormesh(x_edges, y_edges, density, cmap='YlOrBr', alpha=0.7)

        else:
            # Interpolated version - smooth transitions between positions
            x_grid = np.linspace(-0.5, self.num_pairs - 0.5, 100)
            y_grid = np.linspace(np.min(all_values), np.max(all_values), 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            density = np.zeros_like(X)

            for i, x in enumerate(x_grid):
                pos = int(round(x))
                if pos < 0:
                    pos = 0
                if pos >= self.num_pairs:
                    pos = self.num_pairs - 1
                values = all_values[:, pos]
                try:
                    kde = gaussian_kde(values)
                    density[:, i] = kde(y_grid)
                except:
                    continue

            # Smooth density
            density = gaussian_filter1d(density, sigma=1, axis=1)

            # Plot interpolated heatmap
            im = ax_heat.pcolormesh(X, Y, density, cmap='YlOrBr', alpha=0.7)

        # Create horizontal colorbar for density
        cax1 = fig.add_axes([0.15, 0.92, 0.3, 0.02])
        cbar1 = fig.colorbar(im, cax=cax1, orientation='horizontal')
        cbar1.set_label('Density')

        # Handle trajectories if requested
        if show_trajectories:
            selected_indices = self.select_trajectories(
                selection_method, n_trajectories, n_bins
            )

            d_vals = np.array([self.results['D'][i] for i in selected_indices])

            # Find appropriate power of 10 for normalization
            max_d = np.max(d_vals)
            power = int(np.floor(np.log10(max_d)))
            scale_factor = 10**power

            # Normalize the values directly
            d_vals_normalized = d_vals/scale_factor

            # Create normalized colormap without LogNorm
            norm = plt.Normalize(vmin=min(d_vals_normalized), vmax=max(d_vals_normalized))

            for idx in selected_indices:
                if param == 'activation':
                    y_vals = self.calculate_activation_energies(
                        self.results['lambdas'][idx],
                        self.results['deltaG'][idx]
                    )
                else:
                    param_key = param
                    y_vals = self.results[param_key][idx]

                coupling_vals = self.results['couplings'][idx]

                x = np.linspace(0, self.num_pairs-1, 100)
                y_interp = interp.interp1d(range(self.num_pairs), y_vals, kind='cubic')
                coupling_interp = interp.interp1d(range(self.num_pairs), coupling_vals, kind='cubic')

                y = y_interp(x)
                coupling = coupling_interp(x)
                thickness = 0.5 + (coupling - 1) * (2.5 - 0.5) / (13 - 1)

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = LineCollection(segments,
                                linewidths=thickness[:-1],
                                colors=[plt.cm.PuBu_r(norm(self.results['D'][idx]/scale_factor))],
                                alpha=0.7)
                ax_heat.add_collection(lc)

            # Add second horizontal colorbar for diffusion coefficient
            cax2 = fig.add_axes([0.55, 0.92, 0.3, 0.02])
            sm = plt.cm.ScalarMappable(norm=norm, cmap='PuBu_r')
            cbar2 = fig.colorbar(sm, cax=cax2, orientation='horizontal')
            cbar2.set_label(f'Diffusion Coefficient (10^{power} cm²/s)')

            # Format tick labels to show just the decimal numbers
            def fmt(x, p):
                return f'{x:.1f}'
            cbar2.formatter = plt.FuncFormatter(fmt)
            cbar2.update_ticks()

        # Add violin plots in bottom panel
        violin_parts = ax_dist.violinplot(
            all_values,
            positions=range(self.num_pairs),
            showmeans=True,
            showmedians=True,
            widths=0.8
        )

        # Customize violin plots
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightgray')
            pc.set_alpha(0.7)
        violin_parts['cmeans'].set_color('black')
        violin_parts['cmedians'].set_color('red')

        # Add vertical lines to both panels
        for i in range(self.num_pairs):
            ax_heat.axvline(i, color='gray', linestyle='--', alpha=0.3)
            ax_dist.axvline(i, color='gray', linestyle='--', alpha=0.3)

        # Labels
        param_labels = {
            'coupling': 'Electronic Coupling [meV]',
            'lambda': 'Reorganization Energy [eV]',
            'deltaG': 'Reaction Free Energy [eV]',
            'activation': 'Activation Energy [eV]'
        }

        ax_heat.set_ylabel(param_labels[param])
        ax_dist.set_xlabel('Heme Pair')
        ax_dist.set_ylabel('Distribution')

        # Set x-axis ticks and labels
        ax_heat.set_xticks([])
        ax_dist.set_xticks(range(self.num_pairs))
        ax_dist.set_xticklabels(self._create_x_labels())

        # Add legend for violin plot
        ax_dist.plot([], [], color='black', label='Mean', linestyle='-')
        ax_dist.plot([], [], color='red', label='Median', linestyle='-')
        ax_dist.legend(loc='upper right')

        return fig, (ax_heat, ax_dist)

#                # Create line collection with path effects
#                lc = LineCollection(segments, 
#                                linewidths=thickness[:-1],
#                                colors=[plt.cm.cool(norm(self.results['D'][idx]/scale_factor))],
#                                alpha=0.7,
#                                path_effects=[withStroke(linewidth=base_width+1, 
#                                                        foreground='black', 
#                                                        alpha=0.3)])
#                ax_heat.add_collection(lc)

    def plot_parameter_envelope(self,
                            param: str,
                            show_trajectories: bool = False,
                            n_trajectories: int = 20,
                            selection_method: SelectionMethod = SelectionMethod.SORT,
                            n_bins: int = 10,
                            percentiles: List[Tuple[float, float]] = [(25, 75), (10, 90), (5, 95)],
                            figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
        """Create parameter envelope plot showing distribution ranges."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get parameter values
        if param == 'activation':
            all_values = [
                self.calculate_activation_energies(
                    self.results['lambdas'][i],
                    self.results['deltaG'][i]
                )
                for i in range(len(self.results['D']))
            ]
        else:
            param_key = self._get_param_key(param)
            all_values = self.results[param_key]
                
        all_values = np.array(all_values)
        
        # Calculate median line and percentile bands
        x_positions = range(self.num_pairs)
        median_values = np.median(all_values, axis=0)
        
        # Plot percentile bands with grayscale
        # Reverse percentiles to plot wider bands first
        for (low_p, high_p), alpha in zip(reversed(percentiles), 
                                        np.linspace(0.1, 0.3, len(percentiles))):
            low_values = np.percentile(all_values, low_p, axis=0)
            high_values = np.percentile(all_values, high_p, axis=0)
            
            ax.fill_between(x_positions, low_values, high_values, 
                        color='gray', alpha=alpha,
                        label=f'{low_p}-{high_p} percentile')
        
        # Plot median values with markers
        ax.plot(x_positions, median_values, 'o', color='black', 
                markersize=8, label='Median', zorder=3)
        
        # Add trajectories if requested
        if show_trajectories:
            selected_indices = self.select_trajectories(
                selection_method, n_trajectories, n_bins
            )
            
            d_vals = np.array([self.results['D'][i] for i in selected_indices])
            norm = LogNorm(vmin=min(self.results['D']), vmax=max(self.results['D']))
            
            for idx in selected_indices:
                if param == 'activation':
                    y_vals = self.calculate_activation_energies(
                        self.results['lambdas'][idx],
                        self.results['deltaG'][idx]
                    )
                else:
                    param_key = self._get_param_key(param)
                    y_vals = self.results[param_key][idx]
                
                coupling_vals = self.results['couplings'][idx]
                
                # Scale marker sizes based on coupling values
                marker_sizes = 20 + (coupling_vals - 1) * (100 - 20) / (13 - 1)
                
                # Create color array with same length as data
                colors = np.full_like(x_positions, self.results['D'][idx], dtype=float)
                
                # Plot markers for each point
                scatter = ax.scatter(x_positions, y_vals, 
                                s=marker_sizes, 
                                c=colors,
                                norm=norm,
                                cmap='viridis',
                                alpha=0.7, 
                                zorder=4)
            
            # Add colorbar for D values
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Diffusion Coefficient [cm²/s]', rotation=270, labelpad=15)
        
        # Add vertical lines
        for i in x_positions:
            ax.axvline(i, color='gray', linestyle='--', alpha=0.3, zorder=1)
        
        # Set axis labels
        param_labels = {
            'coupling': 'Electronic Coupling [meV]',
            'lambda': 'Reorganization Energy [eV]',
            'deltaG': 'Reaction Free Energy [eV]',
            'activation': 'Activation Energy [eV]'
        }
        
        ax.set_xlabel('Heme Pair')
        ax.set_ylabel(param_labels[param])
        
        # Set x-axis ticks and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(self._create_x_labels())
        
        # Style tick marks
        ax.tick_params(direction='in', which='both')
        for spine in ax.spines.values():
            spine.set_position(('outward', 5))
        
        # Add legend within plot area
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        return fig, ax

    def plot_multi_parameter_envelope(self,
                                    show_trajectories: bool = True,
                                    n_trajectories: int = 15,
                                    selection_method: SelectionMethod = SelectionMethod.KMEANS,
                                    n_bins: int = 10,
                                    percentiles: List[Tuple[float, float]] = [(25, 75), (10, 90), (5, 95)],
                                    figsize: Tuple[int, int] = (3.3, 7)) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Create three-panel parameter envelope plot showing deltaG, lambda, and activation energy."""
        
        # Create figure with three vertically stacked subplots
        fig, (ax_dg, ax_lambda, ax_act) = plt.subplots(3, 1, figsize=figsize)
        
        # Define colors for percentile bands
        band_colors = ['#F2F0F7', '#E1E0F0', '#CECEE5']
        
        # Parameters to plot with their labels
        params = ['deltaG', 'lambda', 'activation']
        axes = [ax_dg, ax_lambda, ax_act]
        y_labels = [r'$\mathrm{\Delta G^{\circ}}$ (eV)', 
                    r'$\mathrm{\lambda}$ (eV)',
                    r'$\mathrm{E_{a}}$ (eV)']
        
        # Select trajectories and set up coloring
        if show_trajectories:
            selected_indices = self.select_trajectories(
                selection_method, n_trajectories, n_bins
            )
            
            # Calculate appropriate power of 10 for normalization
            d_vals = np.array([self.results['D'][i] for i in selected_indices])
            max_d = np.max(d_vals)
            power = int(np.floor(np.log10(max_d)))
            scale_factor = 10**power
            
            # Create normalized colormap
            norm = plt.Normalize(vmin=np.min(d_vals/scale_factor), 
                            vmax=np.max(d_vals/scale_factor))

            # Print trajectory information
            print("\nSelected trajectories:")
            print(f"{'Line #':<8}{'D value':<12}{'Norm D':<12}{'V_pairs':<40}{'Lambdas':<40}{'DeltaGs':<40}")
            print("-" * 140)
            for i, idx in enumerate(selected_indices):
                v_str = ' '.join([f"{v:.1f}" for v in self.results['couplings'][idx]])
                l_str = ' '.join([f"{l:.3f}" for l in self.results['lambdas'][idx]])
                dg_str = ' '.join([f"{dg:.3f}" for dg in self.results['deltaG'][idx]])
                print(f"{idx:<8}{self.results['D'][idx]:.2e}{d_vals[i]/scale_factor:.3f}   {v_str:<40}{l_str:<40}{dg_str:<40}")
        
        # Plot each parameter
        for param, ax, ylabel in zip(params, axes, y_labels):
            # Get parameter values
            if param == 'activation':
                all_values = [
                    self.calculate_activation_energies(
                        self.results['lambdas'][i],
                        self.results['deltaG'][i]
                    )
                    for i in range(len(self.results['D']))
                ]
            else:
                param_key = self._get_param_key(param)
                all_values = self.results[param_key]
            
            all_values = np.array(all_values)
            x_positions = range(self.num_pairs)
            
            # Plot percentile bands
            for (low_p, high_p), color in zip(reversed(percentiles), band_colors):
                low_values = np.percentile(all_values, low_p, axis=0)
                high_values = np.percentile(all_values, high_p, axis=0)
                ax.fill_between(x_positions, low_values, high_values,
                            color=color, alpha=0.7,
                            edgecolor='black', linewidth=0.5,
                            label=f'{low_p}–{high_p}%' if ax == ax_dg else "")
            
            # Plot median as dotted line
            median_values = np.median(all_values, axis=0)
            ax.plot(x_positions, median_values, ':', color='black', linewidth=1.5,
                    label='Median' if ax == ax_dg else "", zorder=3)
            
            # Add trajectories if requested
            if show_trajectories:
                for idx in selected_indices:
                    if param == 'activation':
                        y_vals = self.calculate_activation_energies(
                            self.results['lambdas'][idx],
                            self.results['deltaG'][idx]
                        )
                    else:
                        y_vals = self.results[param_key][idx]
                    
                    # Get color based on D value
                    trajectory_color = plt.cm.Oranges(norm(self.results['D'][idx]/scale_factor))
                    
                    # Plot dotted lines and points
                    ax.plot(x_positions, y_vals, ':', color=trajectory_color,
                        alpha=0.3, zorder=2)
                    
                    # Scale marker sizes based on coupling values
                    coupling_vals = self.results['couplings'][idx]
                    marker_sizes = 20 + (coupling_vals - coupling_vals.min()) * (80 - 20) / (coupling_vals.max() - coupling_vals.min())
                    ax.scatter(x_positions, y_vals, 
                            s=marker_sizes,
                            color=trajectory_color,
                            edgecolor='black',  # Add black edges
                            linewidth=0.5,      # Control edge thickness
                            alpha=1.0,
                            zorder=4)
                                
            # Add vertical lines
            for i in x_positions:
                ax.axvline(i, color='gray', linestyle='--', alpha=0.3, zorder=1)
            
            # Set y-axis label and limits
            ax.set_ylabel(ylabel)
            y_data_min = np.percentile(all_values, 1)
            y_data_max = np.percentile(all_values, 99)
            y_range = y_data_max - y_data_min
            ax.set_ylim(y_data_min - 0.1 * y_range, y_data_max + 0.1 * y_range)
            
            # Direct tick marks inward
            ax.tick_params(direction='in', which='both')
            
            # Only show x-axis labels for bottom plot
            if ax != ax_act:
                ax.set_xticks(x_positions)
                ax.set_xticklabels([])
            else:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(self._create_x_labels())
                ax.set_xlabel('Heme Pair')
        
        # Add colorbar for diffusion coefficients if showing trajectories
        if show_trajectories:
            cax = fig.add_axes([0.2, 0.95, 0.78, 0.02])
            sm = plt.cm.ScalarMappable(norm=norm, cmap='Oranges')
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal',
                            ticks=np.linspace(np.min(d_vals/scale_factor), 
                                            np.max(d_vals/scale_factor), 3))
            
            # Format the ticks to 1 decimal place
            cbar.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            
            cbar.set_label(rf'Diffusion Coefficient (10$^{{{power}}}$ cm²/s)', labelpad=5)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
        
        # Add legend to top subplot
        ax_dg.legend(bbox_to_anchor=(0.99, 1.02), loc='upper right',
                    ncol=2,
                    fontsize=8,
                    frameon=False,
                    labelspacing=0.1,
                    columnspacing=0.7,
                    handlelength=1.0)
        
        # Adjust spacing
        plt.subplots_adjust(left=0.2, right=0.98, bottom=0.1, top=0.92, hspace=0)
        
        return fig, [ax_dg, ax_lambda, ax_act]

    def create_trajectory_animation(self,
                                output_file: str = 'trajectory_animation.mp4',
                                n_trajectories: int = 50,
                                selection_method: SelectionMethod = SelectionMethod.KMEANS,
                                n_bins: int = 10,
                                fps: int = 2) -> None:
        """Create animation showing one trajectory at a time in multi-panel plot."""
        
        # Set up the figure like in multi-panel plot
        fig, (ax_dg, ax_lambda, ax_act) = plt.subplots(3, 1, figsize=(3.3, 7))
        
        # Define colors for percentile bands
        band_colors = ['#F2F0F7', '#E1E0F0', '#CECEE5']
        trajectory_color = '#FD8D3C'  # Fixed orange color for all trajectories
        
        # Parameters to plot with their labels
        params = ['deltaG', 'lambda', 'activation']
        axes = [ax_dg, ax_lambda, ax_act]
        y_labels = [r'$\Delta G^{\circ}$ (eV)', 
                    r'$\lambda$ (eV)',
                    r'$E_{a}$ (eV)']
        
        # Select trajectories
        selected_indices = self.select_trajectories(
            selection_method, n_trajectories, n_bins
        )
        
        def update(frame_idx):
            """Update function for animation - called for each frame"""
            # Clear previous frame
            for ax in axes:
                ax.clear()
            
            # Current trajectory index
            idx = selected_indices[frame_idx]
            
            # Calculate D value and text at start
            d_value = self.results['D'][idx]
            power = int(np.floor(np.log10(d_value)))
            normalized_d = d_value / (10**power)
            
            # Plot each parameter
            for param, ax, ylabel in zip(params, axes, y_labels):
                # Get parameter values
                if param == 'activation':
                    all_values = [
                        self.calculate_activation_energies(
                            self.results['lambdas'][i],
                            self.results['deltaG'][i]
                        )
                        for i in range(len(self.results['D']))
                    ]
                else:
                    param_key = self._get_param_key(param)
                    all_values = self.results[param_key]
                
                all_values = np.array(all_values)
                x_positions = range(self.num_pairs)
                
                # Plot percentile bands
                percentiles = [(25, 75), (10, 90), (5, 95)]
                for (low_p, high_p), color in zip(reversed(percentiles), band_colors):
                    low_values = np.percentile(all_values, low_p, axis=0)
                    high_values = np.percentile(all_values, high_p, axis=0)
                    ax.fill_between(x_positions, low_values, high_values,
                            color=color, alpha=0.7,
                            edgecolor='black', linewidth=0.5,
                            label=f'{low_p}–{high_p}%' if ax == ax_dg else "")
                
                # Plot median as dotted line
                median_values = np.median(all_values, axis=0)
                ax.plot(x_positions, median_values, ':', color='black', linewidth=1.5,
                    label='Median' if ax == ax_dg else "", zorder=3)
                
                # Plot current trajectory
                if param == 'activation':
                    y_vals = self.calculate_activation_energies(
                        self.results['lambdas'][idx],
                        self.results['deltaG'][idx]
                    )
                else:
                    y_vals = self.results[param_key][idx]
                
                # Plot dotted lines and points
                ax.plot(x_positions, y_vals, ':', color=trajectory_color,
                    alpha=0.3, zorder=2)
                
                coupling_vals = self.results['couplings'][idx]
                marker_sizes = 20 + (coupling_vals - coupling_vals.min()) * (80 - 20) / (coupling_vals.max() - coupling_vals.min())
                ax.scatter(x_positions, y_vals, 
                        s=marker_sizes,  # Now using variable sizes
                        color=trajectory_color,
                        alpha=1.0,
                        zorder=4)
                
                # Add vertical lines
                for i in x_positions:
                    ax.axvline(i, color='gray', linestyle='--', alpha=0.3, zorder=1)
                
                # Set y-axis label and limits
                ax.set_ylabel(ylabel)
                y_data_min = np.percentile(all_values, 1)
                y_data_max = np.percentile(all_values, 99)
                y_range = y_data_max - y_data_min
                ax.set_ylim(y_data_min - 0.1 * y_range, y_data_max + 0.1 * y_range)
                
                # Direct tick marks inward
                ax.tick_params(direction='in', which='both')
                
                # Only show x-axis labels for bottom plot
                if ax != ax_act:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels([])
                else:
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels(self._create_x_labels())
                    ax.set_xlabel('Heme Pair')
            
            # Add D value annotation at very top
            # Calculate the center point of the plot area for the D value annotation
            center_pos = 0.2 + (0.98 - 0.2) / 2  # midpoint between left and right margins

            # Center D value annotation over plot area
            fig.suptitle(f'D = {normalized_d:.1f} × 10$^{{{power}}}$ cm²/s', 
                        x=center_pos, y=0.95, fontsize='medium')
            
            # Add legend to top subplot            
            ax_dg.legend(bbox_to_anchor=(0.99, 1.02), loc='upper right',
                        ncol=2,  # Change from 4 to 2 columns
                        fontsize=8,  # Explicit 8pt
                        frameon=False,
                        labelspacing=0.1,
                        columnspacing=0.7,
                        handlelength=1.0) 

            # Adjust spacing
            plt.subplots_adjust(left=0.2, right=0.98, bottom=0.1, top=0.92, hspace=0)
        
        # Create animation
        anim = animation.FuncAnimation(fig, update,
                                    frames=len(selected_indices),
                                    interval=1000/fps)
        
        # Save animation
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(output_file, writer=writer)
        plt.close()
        
        print(f"Animation saved to {output_file}")

def write_statistics_summary(stats: Dict, param: str, output_file: str):
    """Write parameter statistics to file."""
    param_labels = {
        'coupling': 'Electronic Coupling [meV]',
        'lambda': 'Reorganization Energy [eV]',
        'deltaG': 'Reaction Free Energy [eV]',
        'activation': 'Activation Energy [eV]'
    }
    
    with open(output_file, 'a') as f:
        f.write(f"\nStatistics for {param_labels[param]}\n")
        f.write("=" * 50 + "\n")
        
        for pos in range(len(stats)):
            pos_stats = stats[pos]
            f.write(f"\nPair {pos+1}-{pos+2}:\n")
            f.write(f"  Mean: {pos_stats['mean']:.3f}\n")
            f.write(f"  Median: {pos_stats['median']:.3f}\n")
            f.write(f"  Std Dev: {pos_stats['std']:.3f}\n")
            f.write(f"  Quartiles: {pos_stats['quartiles'][0]:.3f}, "
                   f"{pos_stats['quartiles'][1]:.3f}, {pos_stats['quartiles'][2]:.3f}\n")
            f.write(f"  Range: {pos_stats['min']:.3f} to {pos_stats['max']:.3f}\n")
            f.write(f"  Correlation with D: {pos_stats['correlation_with_D']['coefficient']:.3f} "
                   f"(p={pos_stats['correlation_with_D']['p_value']:.3e})\n")

def visualize_parameter_evolution(results_files: List[str], 
                                output_prefix: str,
                                n_trajectories: int = 50,
                                n_bins: int = 10,
                                show_trajectories: bool = True,
                                plot_types: Set[PlotType] = None):
    """Generate parameter evolution plots and statistics from multiple input files."""
    # Use all plot types if none specified
    if plot_types is None:
        plot_types = {PlotType.TRAJECTORIES, PlotType.HEATMAP, PlotType.ENVELOPE}
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_prefix}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Initialize visualizer with multiple files
    try:
        vis = ParameterEvolutionVisualizer(results_files)
    except Exception as e:
        print(f"Error initializing visualizer: {str(e)}")
        return
    
    # Create summary file
    summary_file = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Parameter Evolution Analysis\n")
        f.write(f"=========================\n")
        f.write("Input files:\n")
        for file in results_files:
            f.write(f"  - {file}\n")
        f.write(f"\nSequence: {vis.sequence}\n")
        f.write(f"Total trajectories loaded: {len(vis.results['D'])}\n")
        f.write(f"Number of trajectories plotted: {n_trajectories}\n")
        f.write(f"Number of bins: {n_bins}\n")
        f.write(f"Plot types: {', '.join(pt.value for pt in plot_types)}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add basic statistics about the combined dataset
        f.write("\nCombined Dataset Statistics:\n")
        f.write("-------------------------\n")
        f.write(f"Total number of trajectories: {len(vis.results['D'])}\n")
        f.write(f"D value range: {vis.results['D'].min():.2e} to {vis.results['D'].max():.2e}\n")
        f.write(f"Number of heme pairs: {vis.num_pairs}\n\n")

    print("Generating multi-panel envelope plot...")
    try:
        fig, axes = vis.plot_multi_parameter_envelope(
            show_trajectories=True,
            n_trajectories=2,
            selection_method=SelectionMethod.SORT,
            n_bins=100
        )
        output_file = os.path.join(output_dir, "multi_panel_envelope.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved multi-panel plot: {output_file}")
    except Exception as e:
        print(f"Error generating multi-panel plot: {str(e)}")

    # Generate trajectory animation
    print("Generating trajectory animation...")
    try:
        animation_file = os.path.join(output_dir, "trajectory_animation.mp4")
        n_all = len(vis.results['D'])
        fps = max(2, n_all // 30)
        vis.create_trajectory_animation(
            output_file=animation_file,
            n_trajectories=250,
            selection_method=SelectionMethod.KMEANS,
            fps=2
        )
    except Exception as e:
        print(f"Error generating animation: {str(e)}")

    # Process each parameter
    params = ['activation', 'lambda', 'deltaG']
    for param in params:
        try:
            print(f"\nProcessing {param}...")
            
            # Calculate and write statistics
            print(f"Calculating statistics for {param}...")
            stats = vis.calculate_parameter_statistics(param)
            write_statistics_summary(stats, param, summary_file)
            
            # Generate plots based on selected types
            if PlotType.TRAJECTORIES in plot_types:
                print(f"Generating trajectory plots for {param}...")
                for method in SelectionMethod:
                    try:
                        fig, ax = vis.plot_parameter_evolution(
                            param,
                            selection_method=method,
                            n_trajectories=n_trajectories,
                            n_bins=n_bins
                        )
                        output_file = os.path.join(output_dir, 
                                                f"evolution_{param}_{method.value}.png")
                        fig.savefig(output_file, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        print(f"  Saved trajectory plot: {output_file}")
                    except Exception as e:
                        print(f"Error generating trajectory plot for {param} with {method.value}: {str(e)}")
            
            if PlotType.HEATMAP in plot_types:
                print(f"Generating heatmap ribbon plots for {param}...")
                for method in SelectionMethod:
                    try:
                        # Interpolated version
                        print(f"  Creating interpolated heatmap with {method.value} selection...")
                        fig, ax = vis.plot_parameter_heatmap_ribbon(
                            param,
                            show_trajectories=show_trajectories,
                            n_trajectories=n_trajectories//2,
                            selection_method=method,
                            discrete=False
                        )
                        output_file = os.path.join(output_dir, f"heatmap_{param}_{method.value}.png")
                        fig.savefig(output_file, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        print(f"  Saved heatmap plot: {output_file}")
                        
                        # Discrete version
                        print(f"  Creating discrete heatmap with {method.value} selection...")
                        fig, ax = vis.plot_parameter_heatmap_ribbon(
                            param,
                            show_trajectories=show_trajectories,
                            n_trajectories=n_trajectories//2,
                            selection_method=method,
                            discrete=True
                        )
                        output_file = os.path.join(output_dir, f"heatmap_{param}_{method.value}_discrete.png")
                        fig.savefig(output_file, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        print(f"  Saved discrete heatmap plot: {output_file}")
                    except Exception as e:
                        print(f"Error generating heatmap for {param} with {method.value}: {str(e)}")
            
            if PlotType.ENVELOPE in plot_types:
                print(f"Generating envelope plots for {param}...")
                for method in SelectionMethod:
                    try:
                        print(f"  Creating envelope plot with {method.value} selection...")
                        fig, ax = vis.plot_parameter_envelope(
                            param,
                            show_trajectories=show_trajectories,
                            n_trajectories=n_trajectories//2,
                            selection_method=method,
                            percentiles=[(25,75), (10,90), (5,95)]
                        )
                        output_file = os.path.join(output_dir, f"envelope_{param}_{method.value}.png")
                        fig.savefig(output_file, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        print(f"  Saved envelope plot: {output_file}")
                    except Exception as e:
                        print(f"Error generating envelope plot for {param} with {method.value}: {str(e)}")
                        
        except Exception as e:
            print(f"Error processing parameter {param}: {str(e)}")
            continue

    # Update file count summary to reflect only generated plot types
    n_files = 1  # Start with analysis summary
    n_plots = len(params)
    if PlotType.ENVELOPE in plot_types:
        n_files += n_plots * len(SelectionMethod)
    if PlotType.HEATMAP in plot_types:
        n_files += n_plots * len(SelectionMethod) * 2  # Both discrete and interpolated
    if PlotType.TRAJECTORIES in plot_types:
        n_files += n_plots * len(SelectionMethod)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Generated {n_files} files:")
    print(f"  - 1 analysis summary")
    if PlotType.ENVELOPE in plot_types:
        print(f"  - {n_plots * len(SelectionMethod)} envelope plots")
    if PlotType.HEATMAP in plot_types:
        print(f"  - {n_plots * len(SelectionMethod)} interpolated heatmaps")
        print(f"  - {n_plots * len(SelectionMethod)} discrete heatmaps")
    if PlotType.TRAJECTORIES in plot_types:
        print(f"  - {n_plots * len(SelectionMethod)} trajectory plots")

def main():
    """Main function for parameter visualization. Handles command line arguments and runs the visualization."""
    parser = argparse.ArgumentParser(description="Visualize parameter evolution from multiple files")
    parser.add_argument("results_files", nargs='+',
                       help="Paths to results files")
    parser.add_argument("--output-prefix", default="param_evolution",
                       help="Prefix for output directory")
    parser.add_argument("--n-trajectories", type=int, default=50,
                       help="Number of trajectories to plot")
    parser.add_argument("--n-bins", type=int, default=10,
                       help="Number of bins for histogram methods")
    parser.add_argument("--hide-trajectories", action="store_true",
                       help="Don't show individual trajectories in heatmap and envelope plots")
    parser.add_argument("--plots", choices=['all', 'trajectories', 'heatmap', 'envelope'],
                       nargs='+', default=['all'],
                       help="Types of plots to generate. Can specify multiple types.")

    args = parser.parse_args()

    try:
        # Convert plot choices to set of plot types
        plot_types = set()
        if 'all' in args.plots:
            plot_types = {PlotType.TRAJECTORIES, PlotType.HEATMAP, PlotType.ENVELOPE}
        else:
            type_map = {
                'trajectories': PlotType.TRAJECTORIES,
                'heatmap': PlotType.HEATMAP,
                'envelope': PlotType.ENVELOPE
            }
            plot_types = {type_map[plot] for plot in args.plots}

        # Run visualization
        visualize_parameter_evolution(
            results_files=args.results_files,
            output_prefix=args.output_prefix,
            n_trajectories=args.n_trajectories,
            n_bins=args.n_bins,
            show_trajectories=not args.hide_trajectories,
            plot_types=plot_types
        )

        return 0  # Success

    except Exception as e:
        print(f"Error during visualization: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
