import json
from datetime import datetime
import subprocess
import time
import argparse
import uuid
import socket
import os
import shutil
import sys

class SubprocessError(Exception):
    pass

def run_subprocess(command):
    # Use subprocess.Popen to run the subprocess
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,  # line-buffered
    )

    # Read and print the output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            if output.strip().startswith("Optimization parameters:"):
                opt_params = output.strip()
            sys.stdout.flush()

    # Wait for the subprocess to complete and get its return code
    return_code = process.wait()

    # Check the return code and raise an exception if non-zero
    if return_code != 0:
        error_message = process.stderr.read().strip()  # Read any error messages
        raise SubprocessError(f"Subprocess failed with return code {return_code}. Error: {error_message}")

    return return_code

def get_git_info():
    try:
        # Get the repository URL
        repo_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode('utf-8').strip()
        
        # Get the latest commit information
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        
        return repo_url, commit_hash
    except subprocess.CalledProcessError:
        return None, None

def run_python_file(file_name, uid, sim_args):
    start_time = time.time()
    if file_name == 'finder.py':
        command = ['python3', file_name, '--parameter', sim_args[0], '--num_runs', sim_args[1], '--lower_est', sim_args[2], '--upper_est', sim_args[3]]
        opt_params = None
    else:
        command = ['python', file_name]
    try:
        print(f"Running command: {command}")
        return_code = run_subprocess(command)
        print(f"{file_name} exited with return code: {return_code}")
    except SubprocessError as e:
        print(f"Error running {file_name}: {e}")
    
    end_time = time.time()

    # Calculate elapsed time in seconds
    elapsed_seconds = end_time - start_time

    # Convert elapsed time to hours and minutes
    elapsed_hours = int(elapsed_seconds // 3600)
    elapsed_minutes = int((elapsed_seconds % 3600) // 60)

    # Check if the "output" folder exists
    output_folder = os.path.join(os.path.dirname(__file__), 'output')
    new_output_folder = os.path.join(os.path.dirname(__file__), f"{datetime.now().strftime('%Y-%m-%d')}{uid}", 'output')

    if os.path.exists(output_folder):
        # Move the "output" folder into the "YYYY-MM-DDuid" folder
        os.makedirs(new_output_folder, exist_ok=True)
        shutil.move(output_folder, new_output_folder)
        return f"{elapsed_hours}h{elapsed_minutes}m"

    # Check if the "output.csv" file exists
    output_csv_file = os.path.join(os.path.dirname(__file__), 'output.csv')
    new_output_csv_file = os.path.join(os.path.dirname(__file__), f"{datetime.now().strftime('%Y-%m-%d')}{uid}", 'output.csv')

    if os.path.exists(output_csv_file):
        # Move the "output.csv" file into the "YYYY-MM-DDuid" folder
        os.makedirs(os.path.join(os.path.dirname(__file__), f"{datetime.now().strftime('%Y-%m-%d')}{uid}"), exist_ok=True)
        shutil.move(output_csv_file, new_output_csv_file)
        return f"{elapsed_hours}h{elapsed_minutes}m"

    print("Warning: Neither 'output' folder nor 'output.csv' file found. No data moved.")
    return None

def generate_json(file_name, uid, sim_args):
    opt_params = None
    current_time = datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')
    
    repo_url, commit_hash = get_git_info()

    # Get the hostname (device name)
    device_name = socket.gethostname()

    # Run the Python file and get elapsed time
    elapsed_time = run_python_file(file_name, uid, sim_args)

    if elapsed_time is None:
        return None, None

    # Format the date and uid for the folder name
    folder_name = f"{datetime.now().strftime('%Y-%m-%d')}{uid}"

    # Specify the output file path
    output_file_path = os.path.join(os.path.dirname(__file__), folder_name, 'output.json')

    data = {
        "author": "Janice van Dam & Tim Albers",
        "dataset_type": "simulation",
        "ran": current_time,
        "uid": uid,
        "uuid": str(uuid.uuid4()),
        "ranking": 2,
        "metadata": {
            "setup": device_name,
            "simulatedVariables": ["error_prob", "simulated_time"],
            "Code": {
                "name": file_name,
                "source_repo": repo_url,
                "version": "testround",
                "commit": commit_hash
            },
            "optional_metadata": {
                "elapsed_time": elapsed_time,
                "optimization_parameters": opt_params
            }
        }
    }

    return data, output_file_path

def save_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Generate JSON file with metadata.')
    parser.add_argument('--file', help='Name of the Python file to run and include in metadata', required=True)
    parser.add_argument('--uid', help='UID for the folder and JSON file', required=True)

    args = parser.parse_args()

    additional_args = []

    # Prompt for additional arguments if the script name is 'finder.py'
    if args.file == 'finder.py':
        additional_args.append(input("Enter parameter (str) for finder.py (e.g., coherence_time): "))
        additional_args.append(input("Enter num_runs (int) for finder.py: "))
        additional_args.append(input("Enter lower_est (float) for finder.py: "))
        additional_args.append(input("Enter upper_est (float) for finder.py: "))

    # Generate JSON data and get the output file path
    json_data, output_file_path = generate_json(args.file, args.uid, additional_args)

    if json_data is not None and output_file_path is not None:
        # Save to the specified file path
        save_json(json_data, output_file_path)
        print("DONE! \nJSON file saved to:", output_file_path)

if __name__ == "__main__":
    main()
