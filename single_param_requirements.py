import sys
import csv
import os
import argparse
import numpy as np
from argparse import ArgumentParser
import yaml
from yaml.loader import SafeLoader
import subprocess
import math
from datetime import datetime
from multiprocessing import Pool

def convert_seconds(total_seconds, factor=1):
    """factor = 1e0 for seconds, 1e3 for milliseconds, 1e6 for microseconds, etc."""
    days = (total_seconds/factor) // (24 * 3600)  # Calculate the number of days
    total_seconds %= (24 * 3600)         # Update total_seconds to the remainder
    hours = total_seconds // 3600        # Calculate the number of hours
    total_seconds %= 3600                # Update total_seconds to the remainder
    minutes = total_seconds // 60        # Calculate the number of minutes
    seconds = total_seconds % 60         # Calculate the number of remaining seconds
    return f"{int(days)}d{int(hours)}h{int(minutes)}m{round(seconds, 2)}s"

steady_param_yaml = "/home/timalbers/CODE/Measurement-Only-BQC/steady_params.yaml"  # Path to yaml file containing the paramters that are not varied over

with open(steady_param_yaml) as f:  # Find parameters as stored in the yaml file
    steady_params = yaml.load(f, Loader=SafeLoader)

# Default parameters that must be in opt_params
param_base_dict = {
    "p_loss_init": 0.8847,
    "coherence_time": 62000000,
    "single_qubit_depolar_prob": 0.02,
    "ms_depolar_prob": 0.1,
    "emission_fidelity": 0.947
}

def find_error_prob(num_runs, run_amount, opt_params, script_path):
    outcomes = []
    runtimes = []
    attempts = []
    iterations = math.floor(num_runs / run_amount)
    if iterations == 0:
        raise IterationError()
    last_bit = num_runs - iterations * run_amount
    command = ['python', str(script_path), "--opt_params", str(opt_params), "--run_amount", str(run_amount)]
    for k in range(iterations):
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                meas_outcome, runtime, num_attempts = stdout.decode().split(',')
                outcomes.append(float(meas_outcome))
                runtimes.append(float(runtime))
                attempts.append(float(num_attempts))
            else:
                error_mes = stderr.decode()
                print(f"Error running simulation script:\n {error_mes}")
        except subprocess.CalledProcessError as e:
            print(f"Error running the script: {e}")
        print(f"Iteration {k+1} of {iterations} complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if last_bit != 0:
        command = ['python', str(script_path), "--opt_params", str(opt_params), "--run_amount", str(last_bit)]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                meas_outcome, runtime, num_attempts = stdout.decode().split(',')
                outcomes.append(float(meas_outcome))
                runtimes.append(float(runtime))
                attempts.append(float(num_attempts))
            else:
                error_mes = stderr.decode()
                print(f"Error running simulation script:\n {error_mes}")
        except subprocess.CalledProcessError as e:
            print(f"Error running the script: {e}")
    avg_outcome = sum(outcomes) / len(outcomes)
    avg_runtime = sum(runtimes) / len(runtimes)
    avg_attempts = sum(attempts) / len(attempts)
    print(f"successprob: {avg_outcome}, avg attempts: {avg_attempts}, avg simulated time: {convert_seconds(avg_runtime, 1e6)} for {num_runs} runs")
    if avg_outcome is not None:
        return avg_outcome, avg_runtime, avg_attempts
    else:
        print('No valid values found in for finding average outcome')

def run_simulation(param):
    script_path = '/home/timalbers/CODE/Measurement-Only-BQC/Simulationscript.py'
    # Ensure all required parameters are present in opt_params
    opt_params = param_base_dict.copy()
    #opt_params['p_loss_init'] = float(p_loss)
    opt_params['single_qubit_depolar_prob'] = float(param)
    avg_outcome, avg_runtime, avg_attempts = find_error_prob(15000, 7500, opt_params, script_path)
    return param, avg_outcome, avg_runtime, avg_attempts

p_loss_init_values = np.linspace(0.01, 0.8846, 70)
coherence_time_values = np.linspace(10000000, 62000000, 40)
single_qubit_depolar_prob_values = np.linspace(0.02, 0.821, 40)
ms_depolar_prob_values = np.linspace(0.1, 0.754, 40)
emission_fidelity_values = np.linspace(0.794, 0.947, 40)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate JSON file with metadata.')
    #parser.add_argument('--parameter', type=str, help='name of the parameter', required=True)
    parser.add_argument('--uid',type=str, help='UID for the folder and JSON file', required=True)
    args = parser.parse_args()

    # if args.parameter == 'p_loss_init':
    #     param_values = p_loss_init_values
    # if args.parameter == 'coherence_time':
    #     param_values = coherence_time_values
    # if args.parameter == 'single_qubit_depolar_prob':
    #     param_values = single_qubit_depolar_prob_values
    # if args.parameter == 'ms_depolar_prob':
    #     param_values = ms_depolar_prob_values
    # if args.parameter == 'emission_fidelity':
    #     param_values = emission_fidelity_values
    # else:
    #     print(f"Parameter {args.parameter} not recognized")

    print(f"Starting simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    num_runs = 15000
    run_amount = 7500
    confidence = np.sqrt(np.log(2/0.05)/(2*num_runs)) # 95% confidence interval
    # Create a pool of workers equal to the number of available cores
    with Pool(processes=80) as pool:
        results = pool.map(run_simulation, single_qubit_depolar_prob_values) #EDIT THIS LINE TO CHANGE THE PARAMETER

    # Print results to console
    for param, avg_outcome, avg_runtime, avg_attempts in results:
        print(f"single_qubit_depolar_prob: {param}, successprob: {avg_outcome} +/- {confidence}, avg attempts: {avg_attempts},  avg simulated time: {convert_seconds(avg_runtime, 1e6)} for {num_runs} runs")
    print(f"Simulation finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure the output directory exists
    output_dir = os.path.join(args.uid, 'output')
    ensure_directory_exists(output_dir)

    # Save results to a CSV file
    csv_file_path = os.path.join(output_dir, 'results.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['single_qubit_depolar_prob', 'avg_outcome', 'avg_runtime', 'avg_attempts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for param, avg_outcome, avg_runtime, avg_attempts in results:
            writer.writerow({'single_qubit_depolar_prob': param, 'avg_outcome': avg_outcome, 'avg_runtime': avg_runtime, 'avg_attempts': avg_attempts})
    print(f"Results saved to {csv_file_path}")