""" Prepare a benchmark folder for a task. 

Requirements:
1. This should only download the expected data from Kaggle and all files necessary should be in workspace/. There should not be any hard-coded information like a pre-filled train.py file
"""

import os
import subprocess
import sys
import json
import pandas as pd
import random
import shutil

benchmarks_dir = os.path.dirname(os.path.realpath(__file__)) + "/benchmarks"

# def get_task_info(task):
#     """Get research problem and benchmark folder name for task"""
#     research_problem = task
#     benchmark_folder_name= task

#     # Retrieve task from benchmarks
#     tasks = json.load(open(os.path.join(benchmarks_dir, "tasks.json")))
#     if task in tasks:
#         research_problem = tasks[task].get("research_problem", None)
#         benchmark_folder_name = tasks[task].get("benchmark_folder_name", None)

#     elif task in os.listdir(benchmarks_dir) and os.path.isdir(os.path.join(benchmarks_dir, task, "env")):
#         # default benchmarks
#         benchmark_folder_name = task 
    
#     else:
#         raise ValueError(f"task {task} not supported in benchmarks")

#     if research_problem is None:
#         research_problem_file = os.path.join(benchmarks_dir, benchmark_folder_name, "scripts", "research_problem.txt")
#         if os.path.exists(research_problem_file):
#             # Load default research problem from file
#             with open(research_problem_file, "r") as f:
#                 research_problem = f.read()

#     return benchmark_folder_name, research_problem


def prepare_task(work_dir, task_name, task_type, python="python"):
    print("Preparing task", task_name, ", of type: ", task_type)
    work_dir = os.path.join(work_dir, task_name)

    try:
        if not os.path.exists(work_dir):
            # Create an empty workspace directory if none exists for the custom task
            os.makedirs(work_dir)
            if task_type == "kaggle":
                prepare_kaggle(work_dir, task_name)

        # Set the updated workspace directory to called the branch and overwrite if necessary
        new_dir = f"{work_dir}_branch"
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        shutil.copytree(work_dir, new_dir)

        # Delete answer.csv for Kaggle if evaluating
        if os.path.isfile(os.path.join(new_dir, "answer.csv")):
            os.remove(os.path.join(new_dir, "answer.csv"))
    except:
        print("Failed to prepare task", task_name, ", of type: ", task_type, ", at work directory: ", work_dir)
        print("If you used a custom task, please make sure that you have created a folder with the same name as the task in the workspace directory.")
        raise
    
    return new_dir


# ex. taskname = "home-data-for-ml-course"
def prepare_kaggle(work_dir, taskname):
    input(f"Consent to the competition at https://www.kaggle.com/competitions/{taskname}/data. Once completed, press any key: ")

    subprocess.run(["kaggle", "competitions", "download", "-c", taskname], cwd=work_dir) 
    subprocess.run(["unzip", "-n", f"{taskname}.zip"], cwd=work_dir) 
    subprocess.run(["rm", f"{taskname}.zip"], cwd=work_dir)

    # TODO: I believe this is just to split the data into train and test sets where we know the full test set and don't use the entire targets
    # trainset = pd.read_csv(f"{work_dir}/train.csv")
    # trainset = trainset.reset_index(drop=True)
    # trainset.iloc[:int(len(trainset)*0.8)].to_csv(f"{work_dir}/train.csv", index=False)
    # testset = trainset.iloc[int(len(trainset)*0.8):]


    # testset.drop(list(trainset.keys())[1:-1], axis=1).to_csv(f"{work_dir}/answer.csv", index=False)
    # testset = testset.drop(['SalePrice'], axis=1).to_csv(f"{work_dir}/test.csv", index=False)



if __name__ == "__main__":

    task = sys.argv[1]
    if len(sys.argv) > 2:
        python = sys.argv[2]
    else:
        python = "python"
    benchmark_name, _ = get_task_info(task)
    print("1benchmark_name", benchmark_name, "benchmark_dir", benchmarks_dir)
    benchmark_dir = os.path.join(benchmarks_dir, benchmark_name)
    print("benchmark_name", benchmark_name, "benchmark_dir", benchmark_dir)
    prepare_task(task, python=python)