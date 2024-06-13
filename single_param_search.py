import sys
import csv
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
        print(f"Iteration {k+1} of {iterations} complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

def run_simulation(param):
    script_path = '/home/timalbers/CODE/Measurement-Only-BQC/Simulationscript.py'
    # Ensure all required parameters are present in opt_params
    opt_params = param_base_dict.copy()
    #opt_params['p_loss_init'] = float(p_loss)
    opt_params['coherence_time'] = float(param)
    avg_outcome, avg_runtime = find_error_prob(15, 15, opt_params, script_path)
    return param, avg_outcome, avg_runtime

p_loss_init_values = np.linspace(0.01, 0.8846, 70)
coherence_time_values = np.linspace(30000000, 62000000, 30)

if __name__ == '__main__':
    print(f"Starting simulation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    num_runs = 15
    run_amount = 15
    confidence = np.sqrt(np.log(2/0.05)/(2*run_amount)) # 95% confidence interval
    # Create a pool of workers equal to the number of available cores
    with Pool(processes=80) as pool:
        results = pool.map(run_simulation, coherence_time_values)
    
    for param, avg_outcome, avg_runtime in results:
        print(f"coherence_time: {param}, successprob: {avg_outcome} +/- {confidence}, avg simulated time: {convert_seconds(avg_runtime, 1e6)} for {num_runs} runs")
    print(f"Simulation finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")