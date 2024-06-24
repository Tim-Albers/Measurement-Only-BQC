import sys
import csv
import numpy as np
from argparse import ArgumentParser
import yaml
from yaml.loader import SafeLoader
import subprocess
import math

def find_error_prob(num_runs, run_amount, opt_params, script_path):
    outcomes = []
    runtimes = []
    attempts = []
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
                meas_outcome, runtime, attempt = stdout.decode().split(',')
                outcomes.append(float(meas_outcome))
                runtimes.append(float(runtime))
                attempts.append(float(attempt))

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
                meas_outcome, runtime, attempt = stdout.decode().split(',')
                outcomes.append(float(meas_outcome))
                runtimes.append(float(runtime))
                attempts.append(float(attempt))
            else:
                error_mes = stderr.decode()
                print(f"Error running simulation script:\n {error_mes}")
        except subprocess.CalledProcessError as e:
            print(f"Error running the script: {e}")
    avg_outcome = sum(outcomes)/len(outcomes)
    avg_runtime = sum(runtimes)/len(runtimes)
    avg_attempts = sum(attempts)/len(attempts)
    print("succesprob: ", avg_outcome)
    if avg_outcome is not None:
        return avg_outcome, avg_runtime, avg_attempts
    else:
        print('No valid values found in for finding average outcome')


if __name__ == "__main__":
    # Parse the input argument
    parser = ArgumentParser()
    parser.add_argument('--num_runs', type=int, help="Number of runs to be executed", required=True)
    parser.add_argument('--run_amount', type=int, help="Number of iterations the simulation is repeater for (=number of test rounds)", required=True)
    args = parser.parse_args()
    num_runs = args.num_runs
    run_amount = args.run_amount
    # Load the optimization parameters
    with open('baseline.yaml') as file:
        opt_params = yaml.load(file, Loader=SafeLoader)
    script_path = 'Simulation_script.py'
    find_error_prob(num_runs, run_amount, opt_params, script_path)
    