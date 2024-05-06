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
import zipfile

benchmarks_dir = os.path.dirname(os.path.realpath(__file__)) + "/benchmarks"

def prepare_task(work_dir, task_name, task_type, python="python"):
    print("Preparing task", task_name, ", of type: ", task_type)
    work_dir = os.path.join(work_dir, task_name)

    try:
        if not os.path.exists(work_dir):
            # Create an empty workspace directory if none exists for the custom task
            os.makedirs(work_dir)
            if task_type == "kaggle":
                prepare_kaggle(work_dir, task_name)

                # Ensure that kaggle directory is not without files
                assert(len(os.listdir(work_dir)) != 0)

        # Set the updated workspace directory to called the branch and overwrite if necessary
        new_dir = f"{work_dir}_branch"
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir) # Start from scratch if it already exists, or clean out to be replaced by an existing log file
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

    # Download the data
    subprocess.run(["kaggle", "competitions", "download", "-c", taskname], cwd=work_dir) 

    # Unzip the data
    zip_path = os.path.join(work_dir, f"{taskname}.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(work_dir)
    os.remove(zip_path)


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