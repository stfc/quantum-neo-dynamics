import argparse
import pandas as pd
from scipy.optimize import curve_fit
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from quantum_neo_dynamics.paths import HARDWARE_EXPERIMENTS_JOBDATA, STATEVECTOR_RESULTS_FILE, ZNE_HARDWARE_FIGURES_DIR, HARDWARE_EXPERIMENTS_RESULTSFILE, MPLSTYLE_FILE

def extract_job_metadata(job_id):
    """Extract job metadata from the experiments CSV file."""
    df = pd.read_csv(HARDWARE_EXPERIMENTS_JOBDATA, dtype={'state': str})
    
    job_row = df[df['job_id'] == job_id]
    if job_row.empty:
        raise ValueError(f"Job ID '{job_id}' not found in {HARDWARE_EXPERIMENTS_JOBDATA}")
    
    return job_row.iloc[0]

def get_runtime_zne_job_results(job_id, channel=None, instance=None):
    """Retrieve and extract data from Qiskit Runtime job."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    
    service_kwargs = {}
    if channel is not None:
        service_kwargs['channel'] = channel
    if instance is not None:
        service_kwargs['instance'] = instance

    service = QiskitRuntimeService(**service_kwargs)
    job = service.job(job_id)
    primitive_result = job.result()
    
    # Extract data
    pub_result = primitive_result[0]
    observable = 0
    data = pub_result.data
    
    # Get noise factors and raw results
    noise_factors = primitive_result.metadata['resilience']['zne']['noise_factors']
    evs_raw = data.evs_noise_factors[observable]
    stds_raw = data.stds_noise_factors[observable]
    zne_automatic = data.evs[observable]
    
    return noise_factors, evs_raw, stds_raw, zne_automatic

def save_noise_data_to_csv(job_id, job_metadata, noise_factors, evs_raw, stds_raw):
    """Save noise factor data to results CSV file."""
    # Create list of dictionaries for new rows
    new_rows = []
    for i, noise_factor in enumerate(noise_factors):
        row = {
            'job_id': job_id,
            'method': job_metadata['method'],
            'approximation': job_metadata['approximation'],
            'state': job_metadata['state'],
            'shots': job_metadata['shots'],
            'noisescalefactor': noise_factor,
            'mean_expectation_value': evs_raw[i],
            'std_expectation_value': stds_raw[i]
        }
        new_rows.append(row)
    
    # Convert to DataFrame
    new_df = pd.DataFrame(new_rows)
    
    # Check if results file exists
    if os.path.exists(HARDWARE_EXPERIMENTS_RESULTSFILE):
        # Load existing data
        existing_df = pd.read_csv(HARDWARE_EXPERIMENTS_RESULTSFILE)
        
        # Check if this job_id already exists
        if job_id in existing_df['job_id'].values:
            print(f"Warning: Job ID '{job_id}' already exists in results file. Skipping CSV save...")
            return new_df
        
        # Append new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Create new file
        combined_df = new_df
        print(f"Creating new results file: {HARDWARE_EXPERIMENTS_RESULTSFILE}")
    
    # Save to CSV
    combined_df.to_csv(HARDWARE_EXPERIMENTS_RESULTSFILE, index=False)
    print(f"Successfully added {len(new_rows)} rows for job_id '{job_id}' to {HARDWARE_EXPERIMENTS_RESULTSFILE}")
    
    return new_df

def load_reference_energies(state):
    """Load reference energies from JSON file."""
    with open(STATEVECTOR_RESULTS_FILE, 'r') as f:
        reference_data = json.load(f)
    
    return {
        'aqc_low': reference_data["aqc-low"][str(state)],
        'casci': reference_data["casci"][str(state)]
    }

def perform_fitting_analysis(noise_factors, evs_raw, stds_raw):
    """Perform linear, quadratic, and exponential fits on the data."""
    x = np.array(noise_factors)
    y_mean = np.array(evs_raw)
    y_std = np.array(stds_raw)
    
    # Define fitting functions
    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c

    # Perform fits
    linear_coeffs = np.polyfit(x, y_mean, 1, w=1/y_std)
    linear_poly = np.poly1d(linear_coeffs)

    quad_coeffs = np.polyfit(x, y_mean, 2, w=1/y_std)
    quad_poly = np.poly1d(quad_coeffs)

    # Exponential fit with fallback
    exp_fit_success = False
    popt_exp = None
    
    try:
        popt_exp, pcov_exp = curve_fit(exponential_func, x, y_mean, 
                                       sigma=y_std, 
                                       p0=[y_mean[0], -1, y_mean[-1]],
                                       maxfev=5000)
        exp_fit_success = True
    except:
        print("Warning: Exponential fit failed, using alternative approach")
        try:
            y_shifted = y_mean - np.min(y_mean) + 1e-10
            log_coeffs = np.polyfit(x, np.log(y_shifted), 1, w=1/y_std)
            popt_exp = [np.exp(log_coeffs[1]), log_coeffs[0], np.min(y_mean)]
            exp_fit_success = True
        except:
            exp_fit_success = False
            print("Exponential fit completely failed")

    # Calculate y-intercepts and R-squared values
    y_intercepts = {
        'Linear': linear_poly(0),
        'Quadratic': quad_poly(0)
    }
    
    if exp_fit_success:
        y_intercepts['Exponential'] = exponential_func(0, *popt_exp)
    
    # R-squared calculations
    def r_squared(y_actual, y_predicted):
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        return 1 - (ss_res / ss_tot)

    r_squared_values = {
        'Linear': r_squared(y_mean, linear_poly(x)),
        'Quadratic': r_squared(y_mean, quad_poly(x))
    }
    
    if exp_fit_success:
        r_squared_values['Exponential'] = r_squared(y_mean, exponential_func(x, *popt_exp))
    
    return {
        'linear_coeffs': linear_coeffs,
        'linear_poly': linear_poly,
        'quad_coeffs': quad_coeffs,
        'quad_poly': quad_poly,
        'exp_coeffs': popt_exp if exp_fit_success else None,
        'exp_func': exponential_func if exp_fit_success else None,
        'exp_fit_success': exp_fit_success,
        'y_intercepts': y_intercepts,
        'r_squared_values': r_squared_values,
        'x': x,
        'y_mean': y_mean,
        'y_std': y_std
    }

def create_zne_plot(job_id, job_metadata, fitting_results, reference_energies, zne_automatic):
    """Create and save the ZNE analysis plot."""
    # Apply matplotlib style
    plt.style.use(MPLSTYLE_FILE)
    
    # Temporarily increase font sizes for this specific plot
    plt.rcParams.update({
        'font.size': 12,
        'legend.fontsize': 10,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    x = fitting_results['x']
    y_mean = fitting_results['y_mean']
    y_std = fitting_results['y_std']
    
    # Create fine grid for smooth plotting
    x_fine = np.linspace(0, np.max(x), 100)
    
    # Generate fitted curves
    y_linear = fitting_results['linear_poly'](x_fine)
    y_quad = fitting_results['quad_poly'](x_fine)
    
    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot raw data with error bars
    plt.errorbar(x, y_mean, yerr=y_std, fmt='ko', markersize=8, 
                 capsize=5, capthick=2, label='Raw Data', zorder=3)

    # Plot fits
    plt.plot(x_fine, y_linear, 'b-', linewidth=2, label='Linear Fit')
    plt.plot(x_fine, y_quad, 'r-', linewidth=2, label='Quadratic Fit')
    
    if fitting_results['exp_fit_success']:
        y_exp = fitting_results['exp_func'](x_fine, *fitting_results['exp_coeffs'])
        plt.plot(x_fine, y_exp, 'g-', linewidth=2, label='Exponential Fit')

    # Add vertical line at x=0
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Zero Noise')

    # Add Qiskit automatic ZNE result as marker at x=0
    plt.plot(0, zne_automatic, marker='*', markersize=15, color='red', 
             label='Qiskit ZNE', zorder=4)

    # Add reference energy lines
    plt.axhline(y=reference_energies['aqc_low'], color='purple', linestyle='-', 
                alpha=0.8, label='aqc-low (statevector)')
    plt.axhline(y=reference_energies['casci'], color='orange', linestyle='-', 
                alpha=0.8, label='casci')

    # Formatting
    plt.xlabel('Noise Factor')
    plt.ylabel('Expectation Value')
    plt.title(f'Zero Noise Extrapolation - {job_metadata["method"]}-{job_metadata["approximation"]} | '
              f'State: {job_metadata["state"]} | Shots: {job_metadata["shots"]} | '
              f'Backend: {job_metadata["backend_str"]}')
    plt.legend()
    plt.grid(True)

    # Add text box with extrapolated values and references
    intercept_text = "Y-intercepts (Zero Noise):\n"
    for method_name, value in fitting_results['y_intercepts'].items():
        intercept_text += f"{method_name}: {value:.6f}\n"
    intercept_text += f"Qiskit ZNE: {zne_automatic:.6f}\n\n"
    intercept_text += f"Reference Values:\n"
    intercept_text += f"AQC-Low: {reference_energies['aqc_low']:.6f}\n"
    intercept_text += f"CASCI: {reference_energies['casci']:.6f}"

    plt.text(0.02, 0.98, intercept_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    
    # Save the figure
    os.makedirs(ZNE_HARDWARE_FIGURES_DIR, exist_ok=True)
    output_path = os.path.join(ZNE_HARDWARE_FIGURES_DIR, f"{job_id}.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Reset rcParams to avoid affecting other plots
    plt.rcParams.update(plt.rcParamsDefault)
    
    return output_path

def print_analysis_summary(job_id, fitting_results, reference_energies):
    """Print detailed analysis results."""
    print("\n" + "="*50)
    print(f"FITTING RESULTS for Job ID: {job_id}")
    print("="*50)

    linear_coeffs = fitting_results['linear_coeffs']
    quad_coeffs = fitting_results['quad_coeffs']
    
    print(f"\nLinear Fit: y = {linear_coeffs[0]:.6f}x + {linear_coeffs[1]:.6f}")
    print(f"Linear Y-intercept: {fitting_results['y_intercepts']['Linear']:.6f}")

    print(f"\nQuadratic Fit: y = {quad_coeffs[0]:.6f}x² + {quad_coeffs[1]:.6f}x + {quad_coeffs[2]:.6f}")
    print(f"Quadratic Y-intercept: {fitting_results['y_intercepts']['Quadratic']:.6f}")

    if fitting_results['exp_fit_success']:
        exp_coeffs = fitting_results['exp_coeffs']
        print(f"\nExponential Fit: y = {exp_coeffs[0]:.6f}*exp({exp_coeffs[1]:.6f}*x) + {exp_coeffs[2]:.6f}")
        print(f"Exponential Y-intercept: {fitting_results['y_intercepts']['Exponential']:.6f}")

    print("\n" + "="*50)
    print("SUMMARY OF Y-INTERCEPTS")
    print("="*50)
    for method_name, value in fitting_results['y_intercepts'].items():
        print(f"{method_name:>12}: {value:.6f}")

    print(f"\nReference Values:")
    print(f"AQC-Low (statevector): {reference_energies['aqc_low']:.6f}")
    print(f"CASCI: {reference_energies['casci']:.6f}")

    print(f"\nFit Quality (R-squared):")
    for method_name, r2_value in fitting_results['r_squared_values'].items():
        print(f"{method_name} R²: {r2_value:.4f}")

def analyze_qem_job(job_id, channel=None, instance=None):
    """
    Main function to analyze quantum error mitigation results for a given job ID.
    
    Parameters:
    -----------
    job_id : str
        The job ID to analyze
    channel : str, optional
        IBM Quantum channel
    instance : str, optional
        IBM Quantum instance
    """
    # Extract job metadata
    job_metadata = extract_job_metadata(job_id)
    
    # Get quantum job results
    noise_factors, evs_raw, stds_raw, zne_automatic = get_runtime_zne_job_results(job_id, channel, instance)
    
    # Save data to CSV
    save_noise_data_to_csv(job_id, job_metadata, noise_factors, evs_raw, stds_raw)
    
    # Load reference energies
    reference_energies = load_reference_energies(job_metadata['state'])
    
    # Perform fitting analysis
    fitting_results = perform_fitting_analysis(noise_factors, evs_raw, stds_raw)
    
    # Create and save plot
    output_path = create_zne_plot(job_id, job_metadata, fitting_results, reference_energies, zne_automatic)
    
    # Print summary
    print_analysis_summary(job_id, fitting_results, reference_energies)
    print(f"\nFigure saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse hardware experiment results.")

    parser.add_argument('-id', '--job-id', required=True, type=str, help=(
        "The Qiskit Runtime job ID to analyze for quantum error mitigation results"
    ))

    parser.add_argument('-c', '--channel', type=str, default=None, help=(
        "IBM Quantum channel (e.g., 'ibm_quantum_platform')"
    ))
    
    parser.add_argument('-i', '--instance', type=str, default=None, help=(
        "IBM Quantum instance CRN identifier"
    ))

    args = parser.parse_args()
    analyze_qem_job(job_id=args.job_id, channel=args.channel, instance=args.instance)