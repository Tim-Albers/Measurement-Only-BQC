import sys
import csv
import numpy as np
from argparse import ArgumentParser
import yaml
from yaml.loader import SafeLoader
import subprocess
import math

steady_param_yaml = "/home/timalbers/CODE/Measurement-Only-BQC/steady_params.yaml" # Path to yaml file containing the paramters that are not varied over

with open(steady_param_yaml) as f: #find parameters as stored in the yaml file
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
    iterations = math.floor(num_runs/run_amount)
    if iterations == 0:
        raise IterationError()
    last_bit = num_runs - iterations*run_amount
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
        print(f"Iteration {k} of {iterations} complete")
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
    avg_outcome = sum(outcomes)/len(outcomes)
    avg_runtime = sum(runtimes)/len(runtimes)
    print("succesprob: ", avg_outcome)
    if avg_outcome is not None:
        return avg_outcome, avg_runtime
    else:
        print('No valid values found in for finding average outcome')

p_loss_init_values = np.linspace(0.8846, 0.95, 10)

if __name__ == '__main__':
    script_path = '/home/timalbers/CODE/Measurement-Only-BQC/Simulationscript.py'
    for p_loss in p_loss_init_values:
        opt_params = steady_params.copy()
        # Ensure all required parameters are present in opt_params
        for key, value in param_base_dict.items():
            if key not in opt_params:
                opt_params[key] = value
        opt_params['p_loss_init'] = float(p_loss)
        avg_outcome, avg_runtime = find_error_prob(70000, 10000, opt_params, script_path)
        print(f"p_loss_init: {p_loss}, succesprob: {avg_outcome}, avg runtime: {avg_runtime} ms")