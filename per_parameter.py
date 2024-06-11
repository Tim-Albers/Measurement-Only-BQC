import sys
import csv
import numpy as np
from argparse import ArgumentParser
import yaml
from yaml.loader import SafeLoader
import subprocess
import math
from multiprocessing import Pool

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

class IterationError(Exception):
    """Custom exception for iteration errors."""
    def __init__(self, msg="Iterations should not be 0: decrease run_amount"):
        super().__init__(msg)

def find_error_prob(num_runs, run_amount, opt_params, script_path):
    outcomes = []
    runtimes = []
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
                meas_outcome, runtime = stdout.decode().split(',')
                outcomes.append(float(meas_outcome))
                runtimes.append(float(runtime))
            else:
                error_mes = stderr.decode()
                print(f"Error running simulation script:\n {error_mes}")
        except subprocess.CalledProcessError as e:
            print(f"Error running the script: {e}")
        print(f"Iteration {k+1} of {iterations} complete")
    if last_bit != 0:
        command = ['python', str(script_path), "--opt_params", str(opt_params), "--run_amount", str(last_bit)]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                meas_outcome, runtime = stdout.decode().split(',')
                outcomes.append(float(meas_outcome))
                runtimes.append(float(runtime))
            else:
                error_mes = stderr.decode()
                print(f"Error running simulation script:\n {error_mes}")
        except subprocess.CalledProcessError as e:
            print(f"Error running the script: {e}")
    avg_outcome = sum(outcomes) / len(outcomes)
    avg_runtime = sum(runtimes) / len(runtimes)
    return avg_outcome, avg_runtime

def run_simulation(p_loss, script_path):
    # Ensure all required parameters are present in opt_params
    opt_params = param_base_dict.copy()
    opt_params['p_loss_init'] = float(p_loss)
    
    num_runs = 70000
    run_amount = 10000
    chunk_size = 10000
    num_chunks = num_runs // chunk_size
    
    results = []
    for _ in range(num_chunks):
        avg_outcome, avg_runtime = find_error_prob(chunk_size, chunk_size, opt_params, script_path)
        results.append((avg_outcome, avg_runtime))
    
    avg_outcomes = [result[0] for result in results]
    avg_runtimes = [result[1] for result in results]
    
    overall_avg_outcome = sum(avg_outcomes) / len(avg_outcomes)
    overall_avg_runtime = sum(avg_runtimes) / len(avg_runtimes)
    
    return p_loss, overall_avg_outcome, overall_avg_runtime

p_loss_init_values = np.linspace(0.8846, 0.95, 10)

if __name__ == '__main__':
    script_path = '/home/timalbers/CODE/Measurement-Only-BQC/Simulationscript.py'
    
    # Create a pool of workers equal to the number of available cores
    with Pool(processes=70) as pool:
        results = pool.starmap(run_simulation, [(p_loss, script_path) for p_loss in p_loss_init_values])
    
    for p_loss, avg_outcome, avg_runtime in results:
        print(f"p_loss_init: {p_loss}, successprob: {avg_outcome}, avg runtime: {avg_runtime} ms")
