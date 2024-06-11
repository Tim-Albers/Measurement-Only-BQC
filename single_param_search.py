import sys
import csv
import numpy as np
from argparse import ArgumentParser
import yaml
from yaml.loader import SafeLoader
import subprocess
import math
from multiprocessing import Pool

steady_param_yaml = "/home/timalbers/CODE/Measurement-Only-BQC/steady_params.yaml"  # Path to yaml file containing the parameters that are not varied over

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

def run_single_iteration(args):
    command, k = args
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            meas_outcome, runtime = stdout.decode().split(',')
            return float(meas_outcome), float(runtime)
        else:
            error_mes = stderr.decode()
            print(f"Error running simulation script:\n {error_mes}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running the script: {e}")
        return None
    print(f"Iteration {k+1} complete")

def find_error_prob(num_runs, run_amount, opt_params, script_path):
    outcomes = []
    runtimes = []
    iterations = math.floor(num_runs / run_amount)
    if iterations == 0:
        raise IterationError()
    last_bit = num_runs - iterations * run_amount
    command = ['python', str(script_path), "--opt_params", str(opt_params), "--run_amount", str(run_amount)]
    
    # Prepare arguments for parallel execution
    args = [(command, k) for k in range(iterations)]
    
    with Pool(processes=min(70, iterations)) as pool:
        results = pool.map(run_single_iteration, args)
    
    for result in results:
        if result:
            meas_outcome, runtime = result
            outcomes.append(meas_outcome)
            runtimes.append(runtime)

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
    print("successprob: ", avg_outcome)
    if avg_outcome is not None:
        return avg_outcome, avg_runtime
    else:
        print('No valid values found in for finding average outcome')

def run_simulation(p_loss, script_path):
    # Ensure all required parameters are present in opt_params
    opt_params = param_base_dict.copy()
    opt_params['p_loss_init'] = float(p_loss)
    
    num_runs = 70000
    run_amount = 10000
    
    avg_outcome, avg_runtime = find_error_prob(num_runs, run_amount, opt_params, script_path)
    
    return p_loss, avg_outcome, avg_runtime

p_loss_init_values = np.linspace(0.8846, 0.95, 10)

if __name__ == '__main__':
    script_path = '/home/timalbers/CODE/Measurement-Only-BQC/Simulationscript.py'
    
    # Create a pool of workers equal to the number of available cores
    with Pool(processes=10) as pool:
        results = pool.starmap(run_simulation, [(p_loss, script_path) for p_loss in p_loss_init_values])
    
    for p_loss, avg_outcome, avg_runtime in results:
        print(f"p_loss_init: {p_loss}, successprob: {avg_outcome}, avg runtime: {avg_runtime} ms")
