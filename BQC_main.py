"""Example script for execution of a wobbly_function.py experiment."""
import os
# import matplotlib
import shutil
from yotse.pre import Experiment, SystemSetup, Parameter, OptimizationInfo
from yotse.execution import Executor
from datetime import datetime
import cProfile

"""
This file contains the initialization of the 'minimal improvements' genetic alogrithm (GA) costfunction minimization as per the YOTSE workflow. 
Here, one can specify the parameters to be minimized over, the ranges from which the initial values are picked (param_range/constraints) 
and how many points are sampled (number_points).
In opt_info_list, the GA parameters are defined.
"""

def BQC_pre():
    BQC_experiment = Experiment(
        experiment_name="BQC_measonly",
        system_setup=SystemSetup(source_directory=os.getcwd(),
                                 venv="/home/timalbers/CODE/venv", #This venv should contain all dependencies for the simulation script (which can be installed using the requirements.txt)
                                 slurm_venv="/home/timalbers/venv3.10", #This venv should contain all dependencies for YOTSE (which can be installed with Poetry)
                                 program_name='costfunction.py',
                                 command_line_arguments={"--filebasename": 'output'},
                                 analysis_script="analyse_function_output.py",
                                 executor="python",
                                 #files_needed=["steady_params.yaml"] # todo not implemented
                                 ),
        parameters=[
            Parameter(
                name="p_loss_init",
                param_range=[0.8675, 0.9],
                number_points=3,
                distribution="uniform",
                constraints={'low': 0.8675, 'high': 0.9},
                weights=None,   # todo not implemented
                parameter_active=True,
                param_type="continuous"
            ),
            Parameter(
                name="coherence_time",
                param_range=[30000000, 62000000],
                number_points=3,
                distribution="uniform",
                constraints={'low': 30000000, 'high': 62000000},
                weights=None,
                parameter_active=True,
                param_type="continuous"
            ),
            Parameter(
                name="single_qubit_depolar_prob",
                param_range=[0.0199, 0.05],
                number_points=2,
                distribution="uniform",
                constraints={'low': 0.0199, 'high': 0.05},
                weights=None,
                parameter_active=True,
                param_type="continuous"
            ),
            Parameter(
                name="ms_depolar_prob",
                param_range=[0.09, 0.5],
                number_points=3,
                distribution="uniform",
                constraints={'low': 0.09, 'high': 0.5},
                weights=None,
                parameter_active=True,
                param_type="continuous"
            ),
            Parameter(
                name="emission_fidelity",
                param_range=[0.5, 0.947],
                number_points=2,
                distribution="uniform",
                constraints={'low': 0.5, 'high': 0.947},
                weights=None,
                parameter_active=True,
                param_type="continuous"
            )
        ],
        opt_info_list=[
            OptimizationInfo(
                name="GA",                      # name of the optimization algorithm (Genetic Algorithm)
                opt_parameters={
                    "num_generations": 20,       # number of iterations of the algorithm
                    #"num_points": 10,           # number of points per param to re-create , now determined by initial
                    "num_parents_mating": 8,    # number of points taken of each generation to go onto the next generation
                    "mutation_probability": .2, # probability of mutation
                    "refinement_factors": [.5, .5],
                    "logging_level": 1,         # logging level of the algorithm
                },
                blackbox_optimization=True,     # Defined the blackbox optimization to the optimization algorithm
                is_active=True)]
    )
    with open('optimization_outcomes', 'a') as file:
        file.write(f"Optimization parameters: {BQC_experiment.opt_info_list[0].opt_parameters}\n")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        file.write(f"Run on: {dt_string}\n")
    return BQC_experiment


def remove_files_after_run():
    # remove files and directories
    shutil.rmtree('output')
    dirs = [f for f in os.listdir(os.getcwd()) if (f.startswith(".qcg"))]
    for d in dirs:
        shutil.rmtree(os.path.join(os.getcwd(), d))

def main():
    print("RUNNING BQC_MAIN.PY")
    experiment = BQC_pre()
    BQC_executor = Executor(experiment=experiment)

    experiment.parse_slurm_arg("BQC_main.py")

    for i in range(experiment.opt_info_list[0].opt_parameters["num_generations"]):
#        assert BQC_executor.optimization_alg.ga_instance.generations_completed == i   # sanity check
        # todo : the grid based point generation is still somehow bugged
        print(f"START STEP {i} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        BQC_executor.run(step_number=i, evolutionary_point_generation=True)
        print(f"END STEP {i} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    solution = BQC_executor.optimizer.suggest_best_solution()
    print(f"Optimization parameters: {experiment.opt_info_list[0].opt_parameters}\n")
    print("Solution: ", solution)
    solution_file_path = os.path.join('output', 'solution.txt')
    with open(solution_file_path, 'w') as file:
        file.write(f"Solution: {solution}\n")
   # remove_files_after_run()


if __name__ == "__main__":
    main()
