import sys
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import pandas as pd
import argparse
from scipy import stats

# Constants
R = 8.314  # J/(mol*K)
T = 300    # K
F = 96485  # C/mol

def calculate_K_sequential(E_values: List[float]) -> List[float]:
    """Calculate K values for sequential model."""
    K = [1]
    cumulative_sum = 0
    for E in E_values:
        cumulative_sum += E
        K.append(np.exp(-cumulative_sum * F / (R * T)))
    return K

def calculate_f_ox_sequential(E: float, E_values: List[float]) -> float:
    """Calculate oxidized fraction for sequential model at given potential."""
    n = len(E_values)
    X = np.exp(E * F / (R * T))
    K = calculate_K_sequential(E_values)
    
    numerator = sum(i * K[i] * X**i for i in range(n+1))
    denominator = sum(K[i] * X**i for i in range(n+1))
    
    return numerator / (n * denominator)

def calculate_f_red_sequential(E: float, E_values: List[float]) -> float:
    """Calculate reduced fraction for sequential model at given potential."""
    return 1 - calculate_f_ox_sequential(E, E_values)

def calculate_b_independent(E_values: List[float]) -> List[float]:
    """Calculate b values for independent model."""
    return [np.exp(-E_i * F / (R * T)) for E_i in E_values]

def calculate_f_ox_independent(E: float, E_values: List[float]) -> float:
    """Calculate oxidized fraction for independent model at given potential."""
    n = len(E_values)
    X = np.exp(E * F / (R * T))
    b = calculate_b_independent(E_values)
    
    return (1/n) * sum((b_i * X) / (b_i * X + 1) for b_i in b)

def calculate_f_red_independent(E: float, E_values: List[float]) -> float:
    """Calculate reduced fraction for independent model at given potential."""
    return 1 - calculate_f_ox_independent(E, E_values)

def find_column_positions(header_line: str) -> List[tuple]:
    """Find the start and end positions of each column in a fixed-width format.
    Returns list of (start_pos, end_pos, column_name) tuples."""
    
    # Remove leading/trailing whitespace
    header = header_line.strip()
    
    # Find column positions by looking for blocks of non-whitespace
    columns = []
    start = 0
    in_parentheses = False
    current_start = None
    
    i = 0
    while i < len(header):
        char = header[i]
        
        # Track if we're inside parentheses
        if char == '(':
            in_parentheses = True
        elif char == ')':
            in_parentheses = False
            i += 1
            continue
            
        # If we find a space and we're not in parentheses
        if char.isspace() and not in_parentheses:
            # If we were in a column, this is the end
            if current_start is not None:
                columns.append((current_start, i, header[current_start:i].strip()))
                current_start = None
        # If we find a non-space and we're not currently in a column
        elif not char.isspace() and current_start is None:
            current_start = i
            
        i += 1
    
    # Don't forget the last column
    if current_start is not None:
        columns.append((current_start, len(header), header[current_start:].strip()))
    
    return columns

def load_experimental_data(filename: str, potential_col: int = 0, fraction_col: int = 1, 
                         verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load experimental data from file."""
    try:
        # Read data using pandas, skipping comments
        data = pd.read_csv(filename, delim_whitespace=True, comment='#')
        
        # Get headers
        headers = list(data.columns)
        
        if verbose:
            print("\nAvailable columns in data file:")
            for i, col in enumerate(headers):
                print(f"Column {i}: {col}")
            
            print(f"\nUsing column {potential_col} ({headers[potential_col]}) for potentials")
            print(f"Using column {fraction_col} ({headers[fraction_col]}) for fractions")
        
        # Convert data to numeric, forcing conversion from string
        potentials = pd.to_numeric(data.iloc[:, potential_col], errors='coerce').values
        fractions = pd.to_numeric(data.iloc[:, fraction_col], errors='coerce').values
        
        # Check for any NaN values that would indicate failed conversion
        if np.any(np.isnan(potentials)) or np.any(np.isnan(fractions)):
            raise ValueError("Some data values could not be converted to numbers")
        
        return potentials, fractions, headers
        
    except Exception as e:
        print(f"Error reading data file: {str(e)}")
        print("\nFile format should be space-separated columns with headers.")
        print("Lines starting with # are treated as comments.")
        raise

def objective_function(E_values: List[float], potentials: np.ndarray, 
                      experimental_data: np.ndarray, model: str, 
                      data_type: str) -> float:
    """Generic objective function for fitting models."""
    if model == 'sequential':
        if data_type == 'ox':
            calc_func = calculate_f_ox_sequential
        else:  # red
            calc_func = calculate_f_red_sequential
    else:  # independent
        if data_type == 'ox':
            calc_func = calculate_f_ox_independent
        else:  # red
            calc_func = calculate_f_red_independent
            
    predicted = np.array([calc_func(E, E_values) for E in potentials])
    return np.sum((predicted - experimental_data) ** 2)

def fit_models(potentials: np.ndarray, fractions: np.ndarray, 
              num_hemes: int, data_type: str) -> Tuple[List[float], List[float]]:
    """Fit both sequential and independent models to experimental data."""
    # Initial guess: evenly spaced potentials around the midpoint
    midpoint_idx = np.argmin(abs(fractions - 0.5))
    midpoint_potential = potentials[midpoint_idx]
    spread = 0.1  # Initial spread of potentials around midpoint (V)
    initial_guess = np.linspace(midpoint_potential - spread/2, 
                              midpoint_potential + spread/2, 
                              num_hemes)
    
    # Define bounds for the potentials (V vs. SHE)
    bounds = [(-0.4, 0.4) for _ in range(num_hemes)]
    
    # Fit sequential model
    print("Fitting sequential model...")
    result_seq = minimize(objective_function, initial_guess,
                         args=(potentials, fractions, 'sequential', data_type),
                         method='L-BFGS-B',  # Change to bounded optimization
                         bounds=bounds)      # Add bounds
    
    if not result_seq.success:
        print("Warning: Sequential model fit may not have converged")
        print(f"Optimization message: {result_seq.message}")
    
    fitted_E_values_seq = result_seq.x
    
    # Fit independent model
    print("Fitting independent model...")
    result_ind = minimize(objective_function, initial_guess,
                         args=(potentials, fractions, 'independent', data_type),
                         method='L-BFGS-B',  # Change to bounded optimization
                         bounds=bounds)      # Add bounds
    
    if not result_ind.success:
        print("Warning: Independent model fit may not have converged")
        print(f"Optimization message: {result_ind.message}")
        
    fitted_E_values_ind = result_ind.x
    
    return fitted_E_values_seq, fitted_E_values_ind

def plot_results(potentials: np.ndarray, exp_data: np.ndarray,
                fitted_E_values_seq: List[float],
                fitted_E_values_ind: List[float], 
                output_file: str, data_type: str):
    """Plot experimental data and fitted curves."""
    # Generate smooth curves for fitted models
    E_smooth = np.linspace(min(potentials), max(potentials), 200)
    
    # Select appropriate functions based on data type
    if data_type == 'ox':
        calc_func_seq = calculate_f_ox_sequential
        calc_func_ind = calculate_f_ox_independent
        ylabel = 'Oxidized Fraction'
    else:  # red
        calc_func_seq = calculate_f_red_sequential
        calc_func_ind = calculate_f_red_independent
        ylabel = 'Reduced Fraction'
    
    # Calculate model predictions
    f_seq = [calc_func_seq(E, fitted_E_values_seq) for E in E_smooth]
    f_ind = [calc_func_ind(E, fitted_E_values_ind) for E in E_smooth]
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(potentials, exp_data, 'ko', label='Experimental')
    plt.plot(E_smooth, f_seq, 'r-', label='Sequential Model')
    plt.plot(E_smooth, f_ind, 'b--', label='Independent Model')
    
    plt.xlabel('Potential (V vs. SHE)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    # Print fitted potentials and RMSE
    print("\nFitted Potentials (V vs. SHE):")
    print("Sequential Model:", ", ".join(f"{E:.3f}" for E in fitted_E_values_seq))
    print("Independent Model:", ", ".join(f"{E:.3f}" for E in fitted_E_values_ind))
    
    rmse_seq = np.sqrt(objective_function(fitted_E_values_seq, potentials, exp_data, 
                                        'sequential', data_type) / len(potentials))
    rmse_ind = np.sqrt(objective_function(fitted_E_values_ind, potentials, exp_data, 
                                        'independent', data_type) / len(potentials))
    print(f"\nRMSE Sequential: {rmse_seq:.4f}")
    print(f"RMSE Independent: {rmse_ind:.4f}")
    
    plt.savefig(output_file)
    plt.close()

def save_fitted_curves(potentials: np.ndarray, fitted_E_seq: List[float], 
                      fitted_E_ind: List[float], data_type: str, 
                      output_file: str):
    """Save fitted curves data to a text file."""
    # Generate smooth potential range
    E_smooth = np.linspace(min(potentials), max(potentials), 200)
    
    # Calculate fractions for both models
    if data_type == 'ox':
        f_seq = [calculate_f_ox_sequential(E, fitted_E_seq) for E in E_smooth]
        f_ind = [calculate_f_ox_independent(E, fitted_E_ind) for E in E_smooth]
        f_seq_complement = [1 - f for f in f_seq]
        f_ind_complement = [1 - f for f in f_ind]
    else:  # red
        f_seq = [calculate_f_red_sequential(E, fitted_E_seq) for E in E_smooth]
        f_ind = [calculate_f_red_independent(E, fitted_E_ind) for E in E_smooth]
        f_seq_complement = [1 - f for f in f_seq]
        f_ind_complement = [1 - f for f in f_ind]

    # Create DataFrame with simple column names (no spaces or parentheses)
    df = pd.DataFrame({
        'E': E_smooth,
        'Fox_Ind': f_ind if data_type == 'ox' else f_ind_complement,
        'Fred_Ind': f_ind_complement if data_type == 'ox' else f_ind,
        'Fox_Seq': f_seq if data_type == 'ox' else f_seq_complement,
        'Fred_Seq': f_seq_complement if data_type == 'ox' else f_seq
    })
    
    # Add header with fitted potentials
    header = [
        "# Fitted Potentials (V vs. SHE):",
        "# Independent model: " + ", ".join(f"{E:.3f}" for E in fitted_E_ind),
        "# Sequential model: " + ", ".join(f"{E:.3f}" for E in fitted_E_seq),
        "#"
    ]
    
    # Save to file
    with open(output_file, 'w') as f:
        # Write header comments
        for line in header:
            f.write(line + '\n')
        
        # Write data with simple formatting
        df.to_csv(f, sep='\t', index=False, float_format='%.6f')

def generate_initial_guesses(potentials: np.ndarray, fractions: np.ndarray, 
                           num_hemes: int, num_guesses: int = 5) -> List[np.ndarray]:
    """Generate multiple initial guesses using different strategies."""
    guesses = []
    
    # 1. Midpoint-based guess
    midpoint_idx = np.argmin(abs(fractions - 0.5))
    midpoint_potential = potentials[midpoint_idx]
    spread = 0.1
    guess1 = np.linspace(midpoint_potential - spread/2, 
                        midpoint_potential + spread/2, 
                        num_hemes)
    guesses.append(guess1)
    
    # 2. Derivative-based guess
    diff = np.gradient(fractions, potentials)
    peaks = np.sort(potentials[np.argsort(abs(diff))[-num_hemes:]])
    guesses.append(peaks)
    
    # 3. Equal spacing across experimental range
    guess3 = np.linspace(potentials.min(), potentials.max(), num_hemes)
    guesses.append(guess3)
    
    # 4. Random perturbations of midpoint-based guess
    for _ in range(2):
        perturbation = np.random.normal(0, 0.05, num_hemes)
        guesses.append(guess1 + perturbation)
    
    return guesses

def calculate_correlation_matrix(potentials: np.ndarray, fractions: np.ndarray, 
                               fitted_E_values: List[float], model_type: str,
                               data_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate correlation matrix with improved numerical stability."""
    def residual_vector(E_values):
        if model_type == 'sequential':
            calc_func = calculate_f_ox_sequential if data_type == 'ox' else calculate_f_red_sequential
        else:
            calc_func = calculate_f_ox_independent if data_type == 'ox' else calculate_f_red_independent
        predicted = np.array([calc_func(E, E_values) for E in potentials])
        return fractions - predicted
    
    try:
        # Calculate Jacobian matrix numerically with improved stability
        epsilon = 1e-6
        n_params = len(fitted_E_values)
        jacobian = np.zeros((len(potentials), n_params))
        
        for i in range(n_params):
            E_values_plus = fitted_E_values.copy()
            E_values_plus[i] += epsilon
            E_values_minus = fitted_E_values.copy()
            E_values_minus[i] -= epsilon
            
            # Central difference with error checking
            res_plus = residual_vector(E_values_plus)
            res_minus = residual_vector(E_values_minus)
            if not (np.any(np.isnan(res_plus)) or np.any(np.isnan(res_minus))):
                jacobian[:, i] = (res_plus - res_minus) / (2 * epsilon)
        
        # Calculate covariance matrix with conditioning
        JTJ = np.dot(jacobian.T, jacobian)
        # Add small value to diagonal for numerical stability
        conditioning_factor = 1e-10 * np.trace(JTJ) / len(JTJ)
        JTJ += np.eye(len(JTJ)) * conditioning_factor
        
        try:
            covariance = np.linalg.inv(JTJ)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if regular inverse fails
            print("Warning: Using pseudo-inverse for covariance calculation")
            covariance = np.linalg.pinv(JTJ)
        
        # Calculate correlation matrix
        D = np.sqrt(np.maximum(np.diag(covariance), 0))  # Ensure positive values
        correlation_matrix = covariance / np.outer(D, D)
        
        # Ensure correlation matrix is properly conditioned
        correlation_matrix = np.clip(correlation_matrix, -1, 1)
        
        # Calculate standard errors
        std_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
        
        return correlation_matrix, std_errors
        
    except Exception as e:
        print(f"Warning: Error calculating correlation matrix: {str(e)}")
        n_params = len(fitted_E_values)
        return np.eye(n_params), np.zeros(n_params)

def plot_correlation_heatmap(correlation_matrix: np.ndarray, model_type: str, 
                           output_prefix: str):
    """Create a heatmap visualization of the correlation matrix."""
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    
    # Create heatmap
    sns.heatmap(correlation_matrix, 
                mask=mask,
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                center=0,
                square=True,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation'})
    
    plt.title(f'{model_type.capitalize()} Model Parameter Correlations')
    plt.xlabel('Heme Number')
    plt.ylabel('Heme Number')
    
    # Save plot
    plt.savefig(f'{output_prefix}_correlation_{model_type.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def calculate_fit_diagnostics(result, potentials: np.ndarray, fractions: np.ndarray, 
                            fitted_E_values: List[float], model_type: str, 
                            data_type: str) -> Dict:
    """Calculate comprehensive diagnostics for the fit."""
    if model_type == 'sequential':
        calc_func = calculate_f_ox_sequential if data_type == 'ox' else calculate_f_red_sequential
    else:
        calc_func = calculate_f_ox_independent if data_type == 'ox' else calculate_f_red_independent
            
    predicted = np.array([calc_func(E, fitted_E_values) for E in potentials])
    residuals = fractions - predicted
    
    # Calculate correlation matrix and standard errors
    correlation_matrix, std_errors = calculate_correlation_matrix(
        potentials, fractions, fitted_E_values, model_type, data_type
    )
    
    # Calculate confidence intervals (95%)
    confidence_intervals = []
    for i, (value, std_err) in enumerate(zip(fitted_E_values, std_errors)):
        ci = stats.norm.interval(0.95, loc=value, scale=std_err)
        confidence_intervals.append(ci)
    
    # Basic diagnostics
    diagnostics = {
        'success': result.success,
        'message': result.message,
        'num_iterations': result.nit,
        'num_function_evals': result.nfev,
        'rmse': np.sqrt(np.mean(residuals**2)),
        'max_residual': np.max(abs(residuals)),
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'correlation_matrix': correlation_matrix,
        'standard_errors': std_errors,
        'confidence_intervals': confidence_intervals
    }
    
    # Calculate R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((fractions - np.mean(fractions))**2)
    diagnostics['r_squared'] = 1 - (ss_res / ss_tot)
    
    # Calculate AIC
    n = len(potentials)
    k = len(fitted_E_values)
    diagnostics['aic'] = n * np.log(ss_res/n) + 2*k
    
    return diagnostics

def print_fit_results(fitted_E_seq: List[float], fitted_E_ind: List[float], 
                     diagnostics: Dict, bootstrap_results: dict = None):
    """Print comprehensive fitting results with bootstrap-based confidence intervals."""
    for model_type, fitted_E in [('Sequential', fitted_E_seq), ('Independent', fitted_E_ind)]:
        print(f"\n{model_type} Model Results:")
        print("=" * 50)
        
        # Print fitted potentials with confidence intervals
        print("\nFitted Potentials (V vs. SHE) with 95% Confidence Intervals:")
        if bootstrap_results is not None and len(bootstrap_results[model_type.lower()]) > 0:
            results_array = bootstrap_results[model_type.lower()]
            for i, E in enumerate(fitted_E):
                # Calculate confidence intervals from bootstrap results
                ci = np.percentile(results_array[:, i], [2.5, 97.5])
                print(f"Heme {i+1}: {E:.3f} V  [{ci[0]:.3f}, {ci[1]:.3f}]")
        else:
            # If no bootstrap results, just print fitted values
            for i, E in enumerate(fitted_E):
                print(f"Heme {i+1}: {E:.3f} V")
        
        # Print other diagnostics
        model_key = model_type.lower()
        if model_key in diagnostics:
            diag = diagnostics[model_key]
            print(f"\nRMSE: {diag['rmse']:.6f}")
            print(f"R-squared: {diag['r_squared']:.6f}")
            print(f"AIC: {diag['aic']:.2f}")
            
            print("\nOptimization Details:")
            print(f"Convergence: {diag['success']}")
            print(f"Message: {diag['message']}")
            print(f"Number of iterations: {diag['num_iterations']}")
            print(f"Number of function evaluations: {diag['num_function_evals']}")

def fit_models_with_diagnostics(potentials: np.ndarray, fractions: np.ndarray, 
                              num_hemes: int, data_type: str, 
                              output_prefix: str) -> Tuple[List[float], List[float], Dict]:
    """Enhanced fitting function with multiple initial guesses and comprehensive diagnostics."""
    
    # Generate multiple initial guesses
    initial_guesses = generate_initial_guesses(potentials, fractions, num_hemes)
    bounds = [(-0.4, 0.4) for _ in range(num_hemes)]
    
    # Fit both models
    best_results = {}
    best_scores = {'sequential': np.inf, 'independent': np.inf}
    best_fits = {}
    
    for model_type in ['sequential', 'independent']:
        print(f"\nFitting {model_type} model...")
        
        for i, guess in enumerate(initial_guesses):
            print(f"Trying initial guess {i+1}/{len(initial_guesses)}...")
            result = minimize(objective_function, guess,
                            args=(potentials, fractions, model_type, data_type),
                            method='L-BFGS-B',
                            bounds=bounds)
            
            if result.fun < best_scores[model_type]:
                best_scores[model_type] = result.fun
                best_results[model_type] = result
                best_fits[model_type] = result.x
    
    # Calculate diagnostics for both models
    diagnostics = {}
    for model_type in ['sequential', 'independent']:
        diagnostics[model_type] = calculate_fit_diagnostics(
            best_results[model_type], potentials, fractions,
            best_fits[model_type], model_type, data_type
        )
        
        # Create correlation heatmap
        plot_correlation_heatmap(
            diagnostics[model_type]['correlation_matrix'],
            model_type,
            output_prefix
        )
    
    return best_fits['sequential'], best_fits['independent'], diagnostics

def compare_models_AIC_BIC(potentials: np.ndarray, fractions: np.ndarray, 
                          max_hemes: int, data_type: str) -> dict:
    """Compare models with different numbers of hemes using AICc and BIC."""
    results = []
    
    for n_hemes in range(2, max_hemes + 1):
        print(f"\nTesting model with {n_hemes} hemes...")
        
        # Fit both sequential and independent models
        fitted_E_seq, fitted_E_ind, diagnostics = fit_models_with_diagnostics(
            potentials, fractions, n_hemes, data_type, output_prefix=None
        )
        
        # Calculate AICc and BIC
        n_points = len(potentials)
        for model_type in ['sequential', 'independent']:
            mse = diagnostics[model_type]['rmse']**2
            
            # Calculate standard AIC
            aic = n_points * np.log(mse) + 2 * n_hemes
            
            # Calculate correction term for AICc
            # Correction is (2k² + 2k)/(n-k-1) where k is number of parameters
            correction = (2 * n_hemes**2 + 2 * n_hemes) / (n_points - n_hemes - 1)
            
            # Calculate AICc
            aicc = aic + correction
            
            # Calculate BIC
            bic = n_points * np.log(mse) + n_hemes * np.log(n_points)
            
            results.append({
                'n_hemes': n_hemes,
                'model_type': model_type,
                'rmse': diagnostics[model_type]['rmse'],
                'aic': aic,
                'aicc': aicc,
                'bic': bic
            })
    
    return results

def plot_model_comparison(comparison_results: list, output_prefix: str):
    """Plot AICc and BIC comparison results."""
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(comparison_results)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot AICc
    for model in ['sequential', 'independent']:
        mask = df['model_type'] == model
        ax1.plot(df[mask]['n_hemes'], df[mask]['aicc'], 
                marker='o', label=f'{model.capitalize()} Model')
    ax1.set_xlabel('Number of Hemes')
    ax1.set_ylabel('AICc')
    ax1.set_title('Model Comparison - AICc')
    ax1.legend()
    ax1.grid(True)
    
    # Plot BIC
    for model in ['sequential', 'independent']:
        mask = df['model_type'] == model
        ax2.plot(df[mask]['n_hemes'], df[mask]['bic'],
                marker='o', label=f'{model.capitalize()} Model')
    ax2.set_xlabel('Number of Hemes')
    ax2.set_ylabel('BIC')
    ax2.set_title('Model Comparison - BIC')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed results
    print("\nDetailed Model Comparison Results:")
    print("=" * 60)
    for model in ['sequential', 'independent']:
        print(f"\n{model.capitalize()} Model:")
        mask = df['model_type'] == model
        model_df = df[mask].sort_values('n_hemes')
        for _, row in model_df.iterrows():
            print(f"Hemes: {row['n_hemes']}")
            print(f"  RMSE: {row['rmse']:.6f}")
            print(f"  AIC:  {row['aic']:.2f}")
            print(f"  AICc: {row['aicc']:.2f}")
            print(f"  BIC:  {row['bic']:.2f}")

def parameter_sensitivity(potentials: np.ndarray, fractions: np.ndarray,
                        fitted_E: np.ndarray, model_type: str,
                        data_type: str, delta: float = 0.01) -> np.ndarray:
    """Calculate sensitivity of fit to each parameter using simple finite differences.
    
    Parameters:
    -----------
    potentials : np.ndarray
        Experimental potential values
    fractions : np.ndarray
        Experimental fraction values
    fitted_E : np.ndarray
        Fitted potential values
    model_type : str
        'sequential' or 'independent'
    data_type : str
        'ox' or 'red'
    delta : float
        Step size for finite difference (default: 0.01V)
        
    Returns:
    --------
    np.ndarray
        Sensitivity values for each parameter
    """
    def calculate_residual(E_values: np.ndarray) -> float:
        """Calculate sum of squared residuals for given potentials."""
        try:
            if model_type == 'sequential':
                calc_func = calculate_f_ox_sequential if data_type == 'ox' else calculate_f_red_sequential
            else:
                calc_func = calculate_f_ox_independent if data_type == 'ox' else calculate_f_red_independent
                
            predicted = np.array([calc_func(E, E_values) for E in potentials])
            return np.sum((predicted - fractions) ** 2)
        except:
            return np.inf
    
    base_residual = calculate_residual(fitted_E)
    sensitivities = []
    
    for i in range(len(fitted_E)):
        try:
            # Calculate residual for parameter + delta
            E_plus = fitted_E.copy()
            E_plus[i] += delta
            residual_plus = calculate_residual(E_plus)
            
            # Calculate residual for parameter - delta
            E_minus = fitted_E.copy()
            E_minus[i] -= delta
            residual_minus = calculate_residual(E_minus)
            
            # Calculate sensitivity as average absolute change in residual
            if np.isinf(residual_plus) or np.isinf(residual_minus):
                print(f"Warning: Infinite residual encountered for heme {i+1}")
                sensitivity = 0.0
            else:
                sensitivity = (abs(residual_plus - base_residual) + 
                             abs(residual_minus - base_residual)) / (2 * delta)
                
        except Exception as e:
            print(f"Warning: Error calculating sensitivity for heme {i+1}: {str(e)}")
            sensitivity = 0.0
            
        sensitivities.append(sensitivity)
    
    # Normalize sensitivities to make them comparable
    sensitivities = np.array(sensitivities)
    if np.sum(sensitivities) > 0:
        sensitivities = sensitivities / np.sum(sensitivities)
    
    return sensitivities

def plot_parameter_sensitivity(sensitivities: dict, output_prefix: str):
    """Plot parameter sensitivities with improved visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (model_type, sens) in zip([ax1, ax2], sensitivities.items()):
        # Create bar plot
        heme_numbers = np.arange(1, len(sens) + 1)
        bars = ax.bar(heme_numbers, sens)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
        
        # Customize plot
        ax.set_xlabel('Heme Number')
        ax.set_ylabel('Normalized Sensitivity')
        ax.set_title(f'{model_type.capitalize()} Model Parameter Sensitivity')
        ax.set_xticks(heme_numbers)
        ax.grid(True, alpha=0.3)
        
        # Add threshold line for significant sensitivity
        threshold = 1.0 / len(sens)  # Equal sensitivity line
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5,
                  label='Equal sensitivity')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_parameter_sensitivity.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def bootstrap_analysis(potentials: np.ndarray, fractions: np.ndarray,
                      n_hemes: int, data_type: str,
                      n_bootstrap: int = 1000) -> dict:
    """Perform bootstrap analysis with improved numerical stability."""
    bootstrap_results = {'sequential': [], 'independent': []}
    n_points = len(potentials)
    successful_iterations = 0
    
    # Generate indices for all bootstrap samples in advance
    try:
        # Use a different random seed for each run
        np.random.seed()
        bootstrap_indices = [
            np.random.choice(n_points, size=n_points, replace=True)
            for _ in range(n_bootstrap)
        ]
    except Exception as e:
        print(f"Error generating bootstrap samples: {str(e)}")
        return None
    
    for i, indices in enumerate(bootstrap_indices):
        if i % 100 == 0:
            print(f"Bootstrap iteration {i}/{n_bootstrap}")
            
        try:
            # Create bootstrap sample
            boot_potentials = potentials[indices]
            boot_fractions = fractions[indices]
            
            # Sort by potential to ensure consistent ordering
            sort_idx = np.argsort(boot_potentials)
            boot_potentials = boot_potentials[sort_idx]
            boot_fractions = boot_fractions[sort_idx]
            
            # Initial guess: evenly spaced potentials around the midpoint
            midpoint_idx = np.argmin(abs(boot_fractions - 0.5))
            midpoint_potential = boot_potentials[midpoint_idx]
            spread = 0.1
            initial_guess = np.linspace(midpoint_potential - spread/2, 
                                      midpoint_potential + spread/2, 
                                      n_hemes)
            
            # Define bounds for the optimization
            bounds = [(-0.4, 0.4) for _ in range(n_hemes)]
            
            # Fit sequential model
            result_seq = minimize(objective_function, initial_guess,
                                args=(boot_potentials, boot_fractions, 
                                     'sequential', data_type),
                                method='L-BFGS-B',
                                bounds=bounds)
            
            # Fit independent model
            result_ind = minimize(objective_function, initial_guess,
                                args=(boot_potentials, boot_fractions, 
                                     'independent', data_type),
                                method='L-BFGS-B',
                                bounds=bounds)
            
            # Check convergence and store results
            if (result_seq.success and result_ind.success and
                not np.any(np.isnan(result_seq.x)) and 
                not np.any(np.isnan(result_ind.x))):
                bootstrap_results['sequential'].append(result_seq.x)
                bootstrap_results['independent'].append(result_ind.x)
                successful_iterations += 1
                
        except Exception as e:
            print(f"Warning: Bootstrap iteration {i} failed: {str(e)}")
            continue
    
    print(f"\nBootstrap analysis completed with {successful_iterations} successful iterations")
    
    if successful_iterations < n_bootstrap * 0.1:  # Less than 10% success rate
        print("Error: Too few successful bootstrap iterations!")
        return None
    
    # Convert to arrays
    try:
        results = {
            'sequential': np.array(bootstrap_results['sequential']),
            'independent': np.array(bootstrap_results['independent'])
        }
        
        # Calculate confidence intervals
        confidence_intervals = {model_type: [] for model_type in ['sequential', 'independent']}
        for model_type in ['sequential', 'independent']:
            for i in range(n_hemes):
                param_values = results[model_type][:, i]
                ci = np.percentile(param_values, [2.5, 97.5])  # 95% confidence interval
                confidence_intervals[model_type].append(ci)
        
        results['confidence_intervals'] = confidence_intervals
        
        return results
        
    except Exception as e:
        print(f"Error processing bootstrap results: {str(e)}")
        return None

def plot_bootstrap_results(bootstrap_results: dict, output_prefix: str):
    """Plot bootstrap analysis results with improved visualization."""
    if bootstrap_results is None:
        print("Warning: Cannot plot bootstrap results - no successful iterations")
        return
        
    for model_type in ['sequential', 'independent']:
        results = bootstrap_results[model_type]
        if len(results) == 0:
            print(f"Warning: No results to plot for {model_type} model")
            continue
            
        n_hemes = results.shape[1]
        confidence_intervals = bootstrap_results['confidence_intervals'][model_type]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create violin plot
        violin_parts = ax.violinplot(results, points=100)
        
        # Add median and confidence intervals
        positions = range(1, n_hemes + 1)
        medians = np.median(results, axis=0)
        ax.plot(positions, medians, 'k_', ms=10, label='Median')
        
        # Plot confidence intervals
        for i, (ci_low, ci_high) in enumerate(confidence_intervals):
            ax.plot([i+1, i+1], [ci_low, ci_high], 'r-', alpha=0.5)
        
        # Customize plot
        ax.set_xlabel('Heme Number')
        ax.set_ylabel('Fitted Potential (V vs. SHE)')
        ax.set_title(f'{model_type.capitalize()} Model - Bootstrap Distribution\n'
                    f'(Based on {len(results)} iterations)')
        ax.set_xticks(positions)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.plot([], [], 'r-', alpha=0.5, label='95% CI')
        ax.legend()
        
        plt.savefig(f'{output_prefix}_bootstrap_{model_type}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

def add_model_analysis_arguments(parser):
    """Add model analysis arguments to argument parser."""
    analysis_group = parser.add_argument_group('Model Analysis')
    analysis_group.add_argument('--analyze', action='store_true',
                              help='Perform detailed model analysis')
    analysis_group.add_argument('--max-hemes', type=int, default=8,
                              help='Maximum number of hemes to test in model comparison')
    analysis_group.add_argument('--n-bootstrap', type=int, default=1000,
                              help='Number of bootstrap iterations')

def perform_model_analysis(potentials: np.ndarray, fractions: np.ndarray,
                         args, fitted_E_seq: np.ndarray,
                         fitted_E_ind: np.ndarray) -> dict:
    """Perform comprehensive model analysis."""
    analysis_results = {}
    
    # Model comparison
    print("\nPerforming model comparison...")
    comparison_results = compare_models_AIC_BIC(
        potentials, fractions, args.max_hemes, args.type
    )
    plot_model_comparison(comparison_results, args.output_plot.rsplit('.', 1)[0])
    analysis_results['comparison'] = comparison_results
    
    # Parameter sensitivity
    print("\nCalculating parameter sensitivities...")
    sensitivities = {
        'sequential': parameter_sensitivity(potentials, fractions, fitted_E_seq,
                                         'sequential', args.type),
        'independent': parameter_sensitivity(potentials, fractions, fitted_E_ind,
                                          'independent', args.type)
    }
    plot_parameter_sensitivity(sensitivities, args.output_plot.rsplit('.', 1)[0])
    analysis_results['sensitivities'] = sensitivities
    
    # Bootstrap analysis
    print("\nPerforming bootstrap analysis...")
    bootstrap_results = bootstrap_analysis(
        potentials, fractions, args.hemes, args.type, args.n_bootstrap
    )
    if bootstrap_results is not None:
        plot_bootstrap_results(bootstrap_results, args.output_plot.rsplit('.', 1)[0])
        analysis_results['bootstrap'] = bootstrap_results
    
    return analysis_results, bootstrap_results

def add_validation_arguments(parser):
    """Add validation-related arguments to parser."""
    validation_group = parser.add_argument_group('Validation Options')
    validation_group.add_argument(
        '--validation-method',
        choices=['bootstrap', 'kfold', 'auto', 'both'],
        default='auto',
        help='Validation method to use (default: auto)'
    )
    validation_group.add_argument(
        '--k-folds',
        type=int,
        default=5,
        help='Number of folds for k-fold validation (default: 5)'
    )
    validation_group.add_argument(
        '--bootstrap-samples',
        type=int,
        default=1000,
        help='Number of bootstrap samples (default: 1000)'
    )
    validation_group.add_argument(
        '--fold-size',
        type=int,
        help='Override automatic fold size calculation'
    )
    validation_group.add_argument(
        '--leave-out',
        type=int,
        help='Use leave-N-out cross-validation instead of k-fold'
    )

def choose_validation_method(n_points: int, method: str = 'auto', 
                           leave_out: int = None) -> str:
    """Choose appropriate validation method."""
    if leave_out is not None:
        print(f"Using leave-{leave_out}-out cross-validation as specified")
        return 'leave-out'
        
    if method != 'auto':
        return method
        
    if n_points < 30:
        print(f"Small dataset detected (n={n_points}). Using k-fold cross-validation.")
        return 'kfold'
    elif n_points >= 100:
        print(f"Large dataset detected (n={n_points}). Using bootstrap validation.")
        return 'bootstrap'
    else:
        print(f"Medium dataset detected (n={n_points}). Using both methods for comparison.")
        return 'both'

def perform_kfold_validation(potentials: np.ndarray, fractions: np.ndarray,
                           n_hemes: int, data_type: str, k_folds: int,
                           fold_size: int = None) -> dict:
    """Perform k-fold cross-validation.
    
    Parameters:
    -----------
    potentials : np.ndarray
        Potential values
    fractions : np.ndarray
        Fraction values
    n_hemes : int
        Number of hemes to fit
    data_type : str
        'ox' or 'red'
    k_folds : int
        Number of folds
    fold_size : int, optional
        Override automatic fold size calculation
        
    Returns:
    --------
    dict
        Cross-validation results
    """
    n_points = len(potentials)
    if fold_size is None:
        fold_size = n_points // k_folds
    
    # Create folds
    indices = np.arange(n_points)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    results = {
        'sequential': {'parameters': [], 'val_errors': []},
        'independent': {'parameters': [], 'val_errors': []}
    }
    
    for i, val_fold in enumerate(folds):
        print(f"Processing fold {i+1}/{k_folds}")
        # Create training set
        train_indices = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        train_potentials = potentials[train_indices]
        train_fractions = fractions[train_indices]
        
        # Fit models
        fitted_E_seq, fitted_E_ind, _ = fit_models_with_diagnostics(
            train_potentials, train_fractions, n_hemes, data_type,
            output_prefix=None
        )
        
        # Validate
        val_potentials = potentials[val_fold]
        val_fractions = fractions[val_fold]
        
        for model_type, fitted_E in [('sequential', fitted_E_seq), 
                                   ('independent', fitted_E_ind)]:
            results[model_type]['parameters'].append(fitted_E)
            
            if model_type == 'sequential':
                calc_func = calculate_f_ox_sequential if data_type == 'ox' else calculate_f_red_sequential
            else:
                calc_func = calculate_f_ox_independent if data_type == 'ox' else calculate_f_red_independent
            
            predicted = np.array([calc_func(E, fitted_E) for E in val_potentials])
            val_error = np.sqrt(np.mean((predicted - val_fractions) ** 2))
            results[model_type]['val_errors'].append(val_error)
    
    # Calculate statistics
    for model_type in ['sequential', 'independent']:
        params = np.array(results[model_type]['parameters'])
        errors = np.array(results[model_type]['val_errors'])
        
        results[model_type].update({
            'mean_parameters': np.mean(params, axis=0),
            'std_parameters': np.std(params, axis=0),
            'mean_val_error': np.mean(errors),
            'std_val_error': np.std(errors)
        })
    
    return results

def plot_validation_comparison(kfold_results: dict, bootstrap_results: dict,
                             output_prefix: str):
    """Plot comparison of validation methods.
    
    Parameters:
    -----------
    kfold_results : dict
        Results from k-fold validation
    bootstrap_results : dict
        Results from bootstrap validation
    output_prefix : str
        Prefix for output files
    """
    for model_type in ['sequential', 'independent']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot parameter distributions
        n_hemes = len(kfold_results[model_type]['mean_parameters'])
        x = np.arange(1, n_hemes + 1)
        
        # K-fold results
        mean_kfold = kfold_results[model_type]['mean_parameters']
        std_kfold = kfold_results[model_type]['std_parameters']
        ax1.errorbar(x - 0.1, mean_kfold, yerr=std_kfold, fmt='o',
                    label='K-fold', capsize=5)
        
        # Bootstrap results
        mean_boot = np.mean(bootstrap_results[model_type], axis=0)
        std_boot = np.std(bootstrap_results[model_type], axis=0)
        ax1.errorbar(x + 0.1, mean_boot, yerr=std_boot, fmt='s',
                    label='Bootstrap', capsize=5)
        
        ax1.set_xlabel('Heme Number')
        ax1.set_ylabel('Fitted Potential (V vs. SHE)')
        ax1.set_title(f'{model_type.capitalize()} Model Parameter Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot error distributions
        if 'val_errors' in kfold_results[model_type]:
            ax2.hist(kfold_results[model_type]['val_errors'],
                    bins=10, alpha=0.5, label='K-fold')
        if hasattr(bootstrap_results[model_type], 'rmse'):
            ax2.hist(bootstrap_results[model_type]['rmse'],
                    bins=10, alpha=0.5, label='Bootstrap')
        
        ax2.set_xlabel('Validation Error (RMSE)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution Comparison')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_validation_comparison_{model_type}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

def print_validation_results(kfold_results: dict = None, 
                           bootstrap_results: dict = None):
    """Print comprehensive validation results."""
    if kfold_results:
        print("\nK-fold Cross-validation Results:")
        print("=" * 50)
        for model_type in ['sequential', 'independent']:
            print(f"\n{model_type.capitalize()} Model:")
            print(f"Mean validation error: {kfold_results[model_type]['mean_val_error']:.6f}")
            print(f"Std validation error: {kfold_results[model_type]['std_val_error']:.6f}")
            print("\nMean fitted potentials with uncertainties:")
            mean_params = kfold_results[model_type]['mean_parameters']
            std_params = kfold_results[model_type]['std_parameters']
            for i, (mean, std) in enumerate(zip(mean_params, std_params)):
                print(f"Heme {i+1}: {mean:.3f} ± {std:.3f} V")
    
    if bootstrap_results:
        print("\nBootstrap Validation Results:")
        print("=" * 50)
        for model_type in ['sequential', 'independent']:
            results = bootstrap_results[model_type]
            means = np.mean(results, axis=0)
            stds = np.std(results, axis=0)
            ci = [np.percentile(results[:, i], [2.5, 97.5]) 
                 for i in range(results.shape[1])]
            
            print(f"\n{model_type.capitalize()} Model:")
            print("Fitted potentials with 95% confidence intervals:")
            for i, (mean, std, (ci_low, ci_high)) in enumerate(zip(means, stds, ci)):
                print(f"Heme {i+1}: {mean:.3f} ± {std:.3f} V  [{ci_low:.3f}, {ci_high:.3f}]")

def perform_leave_n_out_validation(potentials: np.ndarray, fractions: np.ndarray,
                                 n_hemes: int, data_type: str, 
                                 leave_out: int = 1) -> dict:
    """Perform leave-N-out cross-validation.
    
    Parameters:
    -----------
    potentials : np.ndarray
        Potential values
    fractions : np.ndarray
        Fraction values
    n_hemes : int
        Number of hemes to fit
    data_type : str
        'ox' or 'red'
    leave_out : int
        Number of points to leave out in each iteration
    
    Returns:
    --------
    dict
        Cross-validation results
    """
    from itertools import combinations
    
    n_points = len(potentials)
    if leave_out >= n_points:
        raise ValueError(f"Cannot leave out {leave_out} points from dataset of size {n_points}")
    
    # Generate all possible combinations of indices to leave out
    all_indices = range(n_points)
    validation_sets = list(combinations(all_indices, leave_out))
    
    print(f"Performing leave-{leave_out}-out cross-validation")
    print(f"Total number of combinations: {len(validation_sets)}")
    
    results = {
        'sequential': {'parameters': [], 'val_errors': []},
        'independent': {'parameters': [], 'val_errors': []}
    }
    
    for i, val_indices in enumerate(validation_sets):
        if i % 100 == 0:
            print(f"Processing combination {i+1}/{len(validation_sets)}")
            
        # Create training set (all points not in validation set)
        train_indices = np.array([i for i in all_indices if i not in val_indices])
        train_potentials = potentials[train_indices]
        train_fractions = fractions[train_indices]
        
        # Create validation set
        val_potentials = potentials[list(val_indices)]
        val_fractions = fractions[list(val_indices)]
        
        try:
            # Fit models on training data
            fitted_E_seq, fitted_E_ind, _ = fit_models_with_diagnostics(
                train_potentials, train_fractions, n_hemes, data_type,
                output_prefix=None
            )
            
            # Validate fits
            for model_type, fitted_E in [('sequential', fitted_E_seq), 
                                       ('independent', fitted_E_ind)]:
                results[model_type]['parameters'].append(fitted_E)
                
                if model_type == 'sequential':
                    calc_func = calculate_f_ox_sequential if data_type == 'ox' else calculate_f_red_sequential
                else:
                    calc_func = calculate_f_ox_independent if data_type == 'ox' else calculate_f_red_independent
                
                predicted = np.array([calc_func(E, fitted_E) for E in val_potentials])
                val_error = np.sqrt(np.mean((predicted - val_fractions) ** 2))
                results[model_type]['val_errors'].append(val_error)
                
        except Exception as e:
            print(f"Warning: Failed to process combination {i+1}: {str(e)}")
            continue
    
    # Calculate statistics
    for model_type in ['sequential', 'independent']:
        params = np.array(results[model_type]['parameters'])
        errors = np.array(results[model_type]['val_errors'])
        
        results[model_type].update({
            'mean_parameters': np.mean(params, axis=0),
            'std_parameters': np.std(params, axis=0),
            'mean_val_error': np.mean(errors),
            'std_val_error': np.std(errors),
            'num_combinations': len(validation_sets)
        })
    
    return results

def main():
    """Main function for redox titration analysis."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fit and analyze redox models')
    
    # Basic arguments
    parser.add_argument('data_file', 
                       help='Path to experimental data file (multi-column format)')
    parser.add_argument('--hemes', type=int,
                       help='Number of hemes in the system')
    parser.add_argument('--type', choices=['ox', 'red'],
                       help='Type of data: oxidized (ox) or reduced (red) fraction')
    parser.add_argument('--potential-col', type=int, default=0,
                       help='Index of column containing potential values (default: 0)')
    parser.add_argument('--fraction-col', type=int,
                       help='Index of column containing fraction values')
    parser.add_argument('--output-plot', default='fit_results.png',
                       help='Output plot filename (default: fit_results.png)')
    parser.add_argument('--output-data', default='fit_results.txt',
                       help='Output data filename (default: fit_results.txt)')
    
    # Analysis arguments
    parser.add_argument('--analyze', action='store_true',
                       help='Perform detailed model analysis')
    parser.add_argument('--max-hemes', type=int, default=8,
                       help='Maximum number of hemes to test in model comparison')
    parser.add_argument('--list-col', action='store_true',
                       help='List available columns in the data file and exit')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                       help='Number of bootstrap iterations (default: 1000)')

    # Add validation arguments
    validation_group = parser.add_argument_group('Validation Options')
    validation_group.add_argument(
        '--validation-method',
        choices=['bootstrap', 'kfold', 'auto', 'both'],
        default='auto',
        help='Validation method to use (default: auto)'
    )
    validation_group.add_argument(
        '--k-folds',
        type=int,
        default=5,
        help='Number of folds for k-fold validation (default: 5)'
    )
    validation_group.add_argument(
        '--bootstrap-samples',
        type=int,
        default=1000,
        help='Number of bootstrap samples (default: 1000)'
    )
    validation_group.add_argument(
        '--fold-size',
        type=int,
        help='Override automatic fold size calculation'
    )
    validation_group.add_argument(
        '--leave-out',
        type=int,
        help='Use leave-N-out cross-validation instead of k-fold'
    )
    
    args = parser.parse_args()

    try:
        # If --list-col is specified, just show columns and exit
        if args.list_col:
            data = pd.read_csv(args.data_file, sep='\s+', comment='#')
            print("\nAvailable columns in data file:")
            for i, col in enumerate(data.columns):
                print(f"Column {i}: {col}")
            return 0
        
        # Validate required arguments when not using --list-col
        if not all([args.hemes, args.type, args.fraction_col]):
            parser.error("--hemes, --type, and --fraction-col are required when not using --list-col")
        
        # Load experimental data
        print(f"\nFitting models for {args.hemes} hemes using {args.type} fraction data from {args.data_file}")
        potentials, fractions, _ = load_experimental_data(args.data_file, 
                                                        args.potential_col, 
                                                        args.fraction_col,
                                                        verbose=True)
   
        # Initial model fitting
        fitted_E_seq, fitted_E_ind, diagnostics = fit_models_with_diagnostics(
            potentials, fractions, args.hemes, args.type,
            args.output_plot.rsplit('.', 1)[0]
        )
        
        # Initialize validation results
        validation_results = {}
        
        # Choose validation method if analyzing
        if args.analyze or args.leave_out:
            validation_method = choose_validation_method(
                len(potentials), 
                args.validation_method,
                args.leave_out
            )
            
            # Perform validation(s)
            if args.leave_out:
                print(f"\nPerforming leave-{args.leave_out}-out cross-validation...")
                validation_results['leave-out'] = perform_leave_n_out_validation(
                    potentials, fractions, args.hemes, args.type,
                    args.leave_out
                )
            elif validation_method in ['kfold', 'both']:
                print("\nPerforming k-fold cross-validation...")
                validation_results['kfold'] = perform_kfold_validation(
                    potentials, fractions, args.hemes, args.type,
                    args.k_folds, args.fold_size
                )
            
            if validation_method in ['bootstrap', 'both']:
                print("\nPerforming bootstrap validation...")
                validation_results['bootstrap'] = bootstrap_analysis(
                    potentials, fractions, args.hemes, args.type,
                    args.bootstrap_samples
                )
            
            # Perform model analysis and comparison
            analysis_results, bootstrap_results = perform_model_analysis(
                potentials, fractions, args, fitted_E_seq, fitted_E_ind
            )
            
            # Plot validation comparison if both methods used
            if validation_method == 'both':
                plot_validation_comparison(
                    validation_results['kfold'],
                    validation_results['bootstrap'],
                    args.output_plot.rsplit('.', 1)[0]
                )
            
            # Print comprehensive results
            print_validation_results(
                validation_results.get('kfold'),
                validation_results.get('bootstrap')
            )
        else:
            # Print basic results without validation
            print_fit_results(fitted_E_seq, fitted_E_ind, diagnostics)
        
        # Plot and save basic fit results
    #   plot_results(potentials, fractions, fitted_E_seq, fitted_E_ind,
    #               args.output_plot, args.type, args.potential_col, args.fraction_col)

        # Plot and save basic fit results
        plot_results(potentials, fractions, fitted_E_seq, fitted_E_ind,
                    args.output_plot, args.type)  # Remove the extra arguments
        
        # Save fitted curves data
        save_fitted_curves(potentials, fitted_E_seq, fitted_E_ind,
                        args.type, args.output_data)
        print(f"\nFitted curves data saved to: {args.output_data}")
        
        # Print final overall summary
        print("\nAnalysis Summary:")
        print("=" * 50)
        print(f"Data points analyzed: {len(potentials)}")
        print(f"Number of hemes fitted: {args.hemes}")
        if args.analyze or args.leave_out:
            print(f"Validation method used: {validation_method}")
            if 'kfold' in validation_results:
                print(f"K-fold cross-validation folds: {args.k_folds}")
            if 'bootstrap' in validation_results:
                print(f"Bootstrap samples: {args.bootstrap_samples}")
            if 'leave-out' in validation_results:
                print(f"Leave-out validation points: {args.leave_out}")
        print("\nOutput files:")
        print(f"- Plot: {args.output_plot}")
        print(f"- Data: {args.output_data}")
        if args.analyze:
            print(f"- Additional analysis plots saved with prefix: {args.output_plot.rsplit('.', 1)[0]}")

        return 0  # Success

    except Exception as e:
        print(f"Error during redox titration analysis: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())

