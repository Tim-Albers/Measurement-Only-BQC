"""
This file contains the costfunction for the 'minimal improvements'/YOTSE workflow. 
BQC_main.py calls on this file (costfunction.py) to find the cost of a given set of parameters.
Two files need to be directed to: the simulation script ('script_path') and a yaml file with baseline parameters ('baseline_yaml')
Some parameters should be defined at the start of this file:

num_runs: (int)----------------------------------------------------------------------------------------------------
    Times the simulation is repeated - determines confidence interval of average outcome based on Hoeffding's bound
run_amount: (int)
    The simulation is run in batches of run_amount to limit memory usage
I: (int)
    Number of nodes in graph state
G: ([[int, int], [int, int], ...])
    List of interger pairs (refering to nodes) defining the edges of the graph state
mbqc_bases: ([float, float, ...])
    Angles wrt the x-axis in xy plane of Bloch sphere defining the measurement bases for MBQC - adapted based on measurement outcomes automatically.
    len(mbqc_bases) should equal I (so all qubits get measured) 
    The last angle in this list is not part of the computation, but is a final measuremet on the output qubit. The code could be adapted to include 
    the final measurement to not be on the equator of the Bloch sphere, or to not have a final measurement at all (i.e., have a qubit state as outcome)
   but this is not currently the case. 

"""
import sys
import csv
import numpy as np
from argparse import ArgumentParser
import yaml
from yaml.loader import SafeLoader
import subprocess
import math

class IterationError(Exception):
    def __init__(self, msg="Iterations should not be 0: decrease run_amount"):
        super().__init__(msg)

TO_PROB_NO_ERROR_FUNCTION = {"p_loss_init": lambda x: 1.0 - x,
                             "single_qubit_depolar_prob": lambda x: 1.0 - x,
                             "ms_depolar_prob": lambda x: 1.0 - x,
                             "coherence_time": lambda x: np.exp(-1.0 * (100000.0 / x)**2),
                             "emission_fidelity": lambda x: (4 * x - 1) / 3
                             }

def append_data_to_file(data_filename, value):
    try:
        with open(data_filename, 'a') as datafile:
            datafile.write(f'{value}\n')
    except Exception as e:
        print(f"An error occurred while appending data: {e}")

def calculate_average_from_file(data_filename):
    try:
        with open(data_filename, 'r') as datafile:
            data = datafile.readlines()

            total = 0
            count = 0

            for value in data:
                total += float(value)
                count += 1

            if count == 0:
                return None  # Return None if no valid values found
            else:
                average = total / count
                return average
    except FileNotFoundError:
        print("File not found.")
        return None

def delete_data(data_filename):
    try:
        with open(data_filename, 'w') as datafile:
            datafile.write('')
    except Exception as e:
        print(f"An error occurred while deleting data: {e}")

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
    print("errorprob: ", avg_outcome)
    if avg_outcome is not None:
        return avg_outcome, avg_runtime
    else:
        print('No valid values found in for finding average outcome')

def costfunction(p_loss_init, coherence_time, single_qubit_depolar_prob, ms_depolar_prob, emission_fidelity, script_path, baseline_path, num_runs, run_amount):
    """Returns cost associated with a given set of hardware parameters."""
    # Weights associate with cost function (w1>>w2 to ensure requirement being met)
    w1 = 10**5
    w2 = -1
    input_value_dict = {"p_loss_init": p_loss_init,
                        "coherence_time": coherence_time,
                        "single_qubit_depolar_prob": single_qubit_depolar_prob,
                        "ms_depolar_prob": ms_depolar_prob,
                        "emission_fidelity": emission_fidelity
                        }
    # Hardware costs as determined by the improvement factor k compared to the baseline
    def Hc(function_dict, **kwargs):
        with open(baseline_path) as f: 
            baseline = yaml.load(f, Loader=SafeLoader)
        pni_dict = {}
        pni_base_dict = {}
        for param_name, func in function_dict.items():
            if param_name not in kwargs:
                raise ValueError(f"Parameter '{param_name}' not provided in the input.")
            if param_name not in baseline:
                raise ValueError(f"Parameter '{param_name}' not provided in baseline.")
            value = kwargs[param_name]
            value_base = baseline[param_name]
            pni_dict[param_name] = func(value)
            pni_base_dict[param_name] = func(value_base)

        Hc = 0.0
        for param_name in set(pni_dict.keys()) | set(pni_base_dict.keys()):
            pni = pni_dict.get(param_name, 0)
            pni_base = pni_base_dict.get(param_name, 0)
            param_cost = 1/(np.log(pni)/np.log(pni_base))
            Hc += param_cost
            print(f"{param_name} cost = {param_cost}")
        return Hc
    error_prob, avg_runtime = find_error_prob(num_runs, run_amount, input_value_dict, script_path) # Average error probability as returned from simulation script
    hardware_cost = Hc(TO_PROB_NO_ERROR_FUNCTION, **input_value_dict) # Hardware cost
    cost = w1*(1 + (error_prob - 0.25)**2)*np.heaviside(error_prob - 0.25, 0) + w2*hardware_cost # Total cost
    print("cost calculated: ", cost)
    return cost, error_prob, avg_runtime


if __name__ == "__main__":
    print("RUNNING COSTFUNCTION.PY")
    # Parse the input argument
    parser = ArgumentParser()
    parser.add_argument('--filebasename', type=str)
    parser.add_argument('--p_loss_init', type=float)
    parser.add_argument('--coherence_time', type=float)
    parser.add_argument('--single_qubit_depolar_prob', type=float)
    parser.add_argument('--ms_depolar_prob', type=float)
    parser.add_argument('--emission_fidelity', type=float)
    args = parser.parse_args()
    parameter_values = [args.p_loss_init, args.coherence_time, args.single_qubit_depolar_prob, args.ms_depolar_prob, args.emission_fidelity]
    num_runs = 100 #20000  #18444 #73777 # Times the simulation is repeated - determines confidence interval of average outcome based on Hoeffding's bound
    run_amount = 10 #10000 # The simulation is run in batches of run_amount to limit memory usage
    script_path = '/home/timalbers/CODE/Measurement-Only-BQC/Simulationscript.py'
    baseline_path = '/home/timalbers/CODE/Measurement-Only-BQC/baseline.yaml'
    # Run the "simulation"
    print(f"Input: {args.p_loss_init},{args.coherence_time}, {args.single_qubit_depolar_prob}, {args.ms_depolar_prob}, {args.emission_fidelity}.\n")
    output_value, error_prob, avg_runtime = costfunction(p_loss_init=args.p_loss_init, coherence_time=args.coherence_time, single_qubit_depolar_prob=args.single_qubit_depolar_prob,
                                ms_depolar_prob=args.ms_depolar_prob, emission_fidelity=args.emission_fidelity, script_path=script_path, baseline_path=baseline_path,
                                num_runs=num_runs, run_amount=run_amount)
    print(f"Output of costfunction is {output_value}.")

    # Store the output value of the costfunction in a file together with respective input values
    csv_filename = args.filebasename + ".csv"
    with open(csv_filename, mode='w') as csv_file:
        print('writing to csv...', csv_filename)
        csv_writer = csv.writer(csv_file, delimiter=' ')
        csv_writer.writerow(['C', 'p_loss_init', 'coherence_time', 'single_qubit_depolar_prob', 'ms_depolar_prob', 'emission_fidelity', 'error_prob', 'num_runs', 'average_runtime [ns]'])
        csv_writer.writerow([output_value, args.p_loss_init, args.coherence_time, args.single_qubit_depolar_prob, args.ms_depolar_prob, args.emission_fidelity, error_prob, num_runs, avg_runtime])
