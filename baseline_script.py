import sys
import csv
import numpy as np
from argparse import ArgumentParser
import yaml
from yaml.loader import SafeLoader
import subprocess
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def convert_seconds(total_seconds):
    days = total_seconds // (24 * 3600)  # Calculate the number of days
    total_seconds %= (24 * 3600)         # Update total_seconds to the remainder
    hours = total_seconds // 3600        # Calculate the number of hours
    total_seconds %= 3600                # Update total_seconds to the remainder
    minutes = total_seconds // 60        # Calculate the number of minutes
    seconds = total_seconds % 60         # Calculate the number of remaining seconds
    return f"Runtime: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {round(seconds, 2)} seconds"

def run_simulation(command):
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            meas_outcome, runtime, attempt = stdout.decode().strip().split(',')
            return float(meas_outcome), float(runtime), float(attempt)
        else:
            error_mes = stderr.decode()
            print(f"Error running simulation script:\n {error_mes}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running the script: {e}")
        return None

def find_error_prob(num_runs, run_amount, opt_params, script_path):
    outcomes = []
    runtimes = []
    attempts = []
    iterations = math.floor(num_runs / run_amount)
    if iterations == 0:
        raise IterationError()
    last_bit = num_runs - iterations * run_amount

    command = ['python', str(script_path), "--opt_params", str(opt_params), "--run_amount", str(run_amount)]
    commands = [command] * iterations

    if last_bit != 0:
        last_command = ['python', str(script_path), "--opt_params", str(opt_params), "--run_amount", str(last_bit)]
        commands.append(last_command)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_simulation, cmd): i for i, cmd in enumerate(commands)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
                if result:
                    meas_outcome, runtime, attempt = result
                    outcomes.append(meas_outcome)
                    runtimes.append(runtime)
                    attempts.append(attempt)
                print(f"Iteration {i} complete")
            except Exception as e:
                print(f"Iteration {i} generated an exception: {e}")

    avg_outcome = sum(outcomes) / len(outcomes) if outcomes else None
    avg_runtime = sum(runtimes) / len(runtimes) if runtimes else None
    avg_attempts = sum(attempts) / len(attempts) if attempts else None

    print("successprob: ", avg_outcome)
    if avg_outcome is not None:
        return avg_outcome, avg_runtime, avg_attempts
    else:
        print('No valid values found in for finding average outcome')
        return None, None, None

if __name__ == "__main__":
    t1 = time.time()
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
    script_path = 'Simulationscript.py'
    find_error_prob(num_runs, run_amount, opt_params, script_path)
    t2 = time.time()
    print(f"number of runs: {num_runs}, run amount: {run_amount}")
    print(convert_seconds(t2-t1))
    print(f"Average time per run: {(t2-t1)/num_runs} seconds")
