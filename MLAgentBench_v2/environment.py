"""
This file contains the Environment class, which prepares the environment for the research agent to run in.

Requirements:
1. This should have access to the workspace and be clear about what the research problem is.
2. This should have access to the actions.py file.

Add-ons:
- maybe give a list of libraries that are already installed?
"""

import json
import os
import sys
import subprocess
import selectors
import shutil
import copy
import time
import fnmatch
import signal
import requests
from traceback import format_exception
from multiprocessing import active_children
# import readline # to make sure input() works properly # Not on Windows
# from dacite import from_dict # Not on Windows
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import functools
from openai import OpenAI
import openai
from dotenv import load_dotenv
from .LLM import complete_text_fast, complete_text
load_dotenv()

import MLAgentBench_v2.high_level_actions as high_level_actions
from .schema import Step, Trace, EnvException, TooLongPromptError, LLMError, EnhancedJSONEncoder 
from .LLM import complete_text_claude
from .prepare_task import prepare_task
from MLAgentBench_v2.actions import TOOL_DESCRIPTIONS
from MLAgentBench_v2.high_level_actions import understand_file, append_to_research_log, inspect_script_lines, edit_script, edit_script_lines, reflection, retrieval_from_research_log
from MLAgentBench_v2.low_level_actions import list_files, read_file, write_file, append_file, copy_file, undo_edit_script, execute_script, python_repl, request_help

class Environment:
    def __init__(self, args):
        # Note: main environment variables that the agent can use
        print("Initializing environment...")
        self._args = args # Might be able to be deleted, more for other potentially deletable environment functions to use like signal alarm
        print("args", args)

        # Set up workspace and research problem.
        with open('MLAgentBench_v2/research_problem.txt', 'r') as f:
            self._research_problem = f.read() # self.R(s) = reward function of current state
        self._benchmark_folder_name = args.task
        self._work_dir = prepare_task(
            work_dir = args.work_dir, 
            task_name = args.task, 
            task_type = args.task_type
        )
        self.files = [os.path.relpath(os.path.join(root, file), self.work_dir) for root, dirs, files in os.walk(self.work_dir) for file in files] # Include skill library files for now
        self.files_no_skill_lib = os.listdir(self.work_dir) # temporary to not give the curriculum agent the skill library to see if it can come up with better ideas
        self.max_states = 32
        self.max_history = 32
        self.answer_states = [{
            "attempted_task": "None",
            "plan": "None",
            "result": "None",
            "files": self.files,
            "answer_state": "None"
        }] # High level state representation at answer_state level. State representation: s_t * self.max_states
        self.files_action_result_history = [{
            "action": "None",
            "result": "None",
            "files": self.files,
        }] # Low level state representation at action level. Like a memory.
        self.completed_tasks = [] # for curriculum agent
        self.failed_tasks = [] # for curriculum agent

        # Set up actions
        self._tool_descriptions = TOOL_DESCRIPTIONS # Formatted for OpenAI function calling
        self._available_actions = {
                'reflection': self.reflection,
                'readFile': self.read_file,
                'writeFile': self.write_file,
                'executeScript': self.execute_script,
            }
        self.request_session = requests.Session()

        # Assistants API specific instantiation
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai.api_key)
        self.model = args.llm_name
        self.MAX_PROMPT_TOKENS = 6000
        self.MAX_TOKENS_TO_SAMPLE = 2000

        # Set up logging, overwrite existing logs if they exist
        self._log_dir = args.log_dir
        self.main_log_path = os.path.join(self.log_dir, "main_log.txt")
        self.env_trace_path = os.path.join(self.log_dir, "latest_env_trace.json")
        self.num_steps = 0
        self.num_tasks = 0
        self._start_time = time.time()
        self.execution_runs = 0
        self.eval_path = self.log_dir + '/eval'
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)

        # Other variables in a partially observable Markov Decision Process
        # self.transition = None # Transition probabilities between states. Problem, how do you operate when you don't even know what s' is until you take action a from state s?
        # self.reward = S x A = reward function. # LLM. The agent is the reward modeler based on the Eureka paper. 

        # Load state if environment trace exists to restore at latest checkpoint, otherwise, create new logged file
        if os.path.isfile(self.env_trace_path):
            self.log(f"\n\n--- RESTORING ENVIRONMENT CHECKPOINT HERE ---\n")
            with open(self.env_trace_path, 'r') as f:
                state = json.load(f)
            self.files = state['files']
            self.answer_states = state['answer_states']
            self.files_action_result_history = state['files_action_result_history']
            self.num_steps = state['num_steps'] + 1 # Offset so there's no overlap
            self.num_tasks = state['num_tasks'] + 1
            self.execution_runs = state['execution_runs'] if 'execution_runs' in state else 0
            self._start_time = state['start_time']

            if 'completed_tasks' in state:
                self.completed_tasks = state['completed_tasks']
            if 'failed_tasks' in state:
                self.failed_tasks = state['failed_tasks']

            # Restore work_dir
            source_dir = os.path.join(self.log_dir, f"latest_work_dir")
            shutil.copytree(source_dir, self.work_dir, dirs_exist_ok=True)
        elif not os.path.exists(self.log_dir):
            os.makedirs(self._log_dir)

        # Checks
        assert(self.research_problem is not None)
        assert("workspace" in self.work_dir and "branch" in self.work_dir) # we should only list files in the workspace and branch
        assert(len(self.tool_descriptions) == len(list(self.available_actions.keys()))) # action descriptions should be the same as action functions


    ############## for logging ##############

    # Logging decorator
    def log_decorator(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Update files
            self.files = [os.listdir(self.work_dir)] # Include skill library files for now # TODO: Just for Eureka so it doesn't see everything. Old used to be self.files in init()
            kwargs['work_dir'] = self.work_dir # Update to actual work_dir
            assert(kwargs['work_dir'] == self.work_dir) # Ensure that the work_dir sent into any function is the work directory and nothing else
            update_history = kwargs.get('update_files_action_result_history', True)
            kwargs.pop('update_files_action_result_history', None) # Remove update_files_action_result_history from kwargs so that it doesn't get passed into the function

            # Update research log
            # TODO: Maybe the except errors aren't logging the full errors / information into "result"? 
            try:
                self.log(f"\n\n--- LOGGING NEW ACTION ---\nStep: {self.num_steps}\nCalling function {func.__name__}(args = {args}, kwargs = {kwargs})\n")

                # Perform the actual function
                result = func(*args, **kwargs)

                self.log("--- TOOL SUCCESS ---")
            except TooLongPromptError:
                result = f"EnvError: too long input for the tool.\n{e}" 
                self.log("--- TOOL ERROR ---", e)
            except LLMError as e:
                result = "LLMError: " + str(e)
                self.log("--- TOOL ERROR ---", e)
            except EnvException as e:
                result = "EnvError: " + str(e).replace(self.work_dir, '.')
                self.log("--- TOOL ERROR ---", str(e))
            except TypeError as e:
                invalid_action_error = f"The arguments needs to have proper entries. You may have missed some entries or used inappropriate ones. Please use the correct format and try again.\n{e}"
                result = "EnvError: " + invalid_action_error
                self.log("--- TOOL ERROR ---", e)
            # except TimeoutException as e:
            #     raise e
            #     self.log("--- TOOL ERROR ---", e)
            except Exception as e:
                result = f"EnvError: Error executing {func.__name__}.\n{e}"
                self.log("--- TOOL ERROR ---", e)

            # Copy work_dir if it exists
            if self.work_dir and os.path.exists(self.work_dir):
                # Save what the work dir was at the time of the action
                dest_dir = os.path.join(self.log_dir, f"{self.num_steps}_work_dir")
                shutil.copytree(self.work_dir, dest_dir, dirs_exist_ok=True)

                # Update latest work dir to restore checkpoint
                dest_dir = os.path.join(self.log_dir, f"latest_work_dir")
                shutil.copytree(self.work_dir, dest_dir, dirs_exist_ok=True)

                # Save env variables to restore checkpoint
                self.save_env_state()

            # Update states
            self.num_steps += 1
            kwargs['work_dir'] = "." # replace work_dir for the agent to stay in its workspace when it reads the answer states

            # Update history to have newest action and result to oldest, only if update_files_action_result_history is not False
            if update_history:
                # TODO: temporary fix is to summarize the action to not deal with a recursive history loop. TODO: Given that I'm summarizing, the true action and result should be logged in a memory folder in the workspace so the ground truth can be accessed somewhere. Step indexes should also be available.
                summarize_action_system_prompt = '''You are a helpful assistant. Your goal is to re-iterate the action taken without the long history section. Therefore, the output should be much shorter and only contain the function name and the most important and relevant arguments and their values. If the history argument is important, than please just summarize it. Do not answer or respond to the action, only re-iterate what the action is and the important parts of the argument values without the history argument.
                
                You will be given this information:
                Action: ...'''
                summarized_action = self.summarize_without_logging(system_prompt=summarize_action_system_prompt, user_prompt=f"Action: Calling function {func.__name__}(args = {args}, kwargs = {kwargs}) \nPlease respond with only the concise re-iteration of the action taken and the most important and relevant arguments and their values.")
                self.log("\n--- ORIGINAL ACTION ---\n", f"Calling function {func.__name__}(args = {args}, kwargs = {kwargs})", "\n--- ACTION SUMMARY ---\n", summarized_action)

                self.files_action_result_history.insert(0, {
                    "files": self.files,
                    "action": summarized_action, # Chat completion will automatically truncate when history is added
                    "result": result,
                })
                while len(self.files_action_result_history) > self.max_history:
                    self.files_action_result_history.pop()
            
                # Log most recent state
                self.log(f"\nStep: {self.num_steps}\nfiles_action_result_history latest addition:\n{json.dumps(self.files_action_result_history[0], indent=4)}\n")
            else:
                # Log most recent action and result for debugging
                self.log(f"\n\n--- Step: {self.num_steps} not recorded in history\n\n--- Step Action: Calling function {func.__name__}(args = {args}, kwargs = {kwargs})\n\n--- Step Result: {result}\n")
            return result
        return wrapper
    
    # Update answer state for curriculum agent
    def update_answer_state(self, attempted_task, plan, result):
        """Update the states of the agent based on attempted_task, plan, result, and files.
        TODO: Extra 1: Break up the state into 1) problem 2) current best answer 3) metric 4) problem to solve 5) next step / plan to solve the problem -- some kind of structure like that.

        TODO: If you don't use Assistants API, then you can have one action at a time and then the update state should only use the current action, result, and state to be the new state instead of the entire history. 2) Then you should have a MCTS to plan what is the next move. Or the updated state should just say what is missing, and not say how to fix it.
        """

        # Update files
        self.files = [os.path.relpath(os.path.join(root, file), self.work_dir) for root, dirs, files in os.walk(self.work_dir) for file in files] # Include skill library files for now

        system_prompt = '''You are a helpful assistant that tells me the next immediate task to do. My ultimate goal is to achieve the research goal as quick as possible and produce answers that are better than myself and anyone else -- effectively becoming the best researcher in the world in solving this research goal. Given a research goal, tools, files, most recently attempted task, the plan, the result, and previous answer states, your goal is to update my answer state so that it best positions me to continually get the best answer possible. You can choose what states is best to keep track of all in your answer state that you think would be most useful (ex. state of best approaches and results so far, state of plans, state of problems, etc.)

You will be given this information:
Research Goal: ...
Tools / functions: ...
Attempted Task: Task to accomplish to better achieve the research goal
Plan: Plan to accomplish the task
Result: Evaluation after executing plan
Files: Files after executing plan
Most recent answer states (newest to oldest): ...

You should then respond to me with only your updated answer state.
'''

        user_prompt = f'''Research Goal: {self.research_problem}
Tools / functions: {list(self.available_actions.keys())}
a) Attempted Task: {attempted_task}
b) Plan: {plan}
c) Result: {result}
d) Files: {self.files}
Most recent answer states (newest to oldest): 
{self.formatted_answer_states()}
'''

        # Raw Chat Completion API to avoid logging loop
        # Truncate user prompt if too long and add a note
        if len(user_prompt) > self.MAX_PROMPT_TOKENS * 4:
            user_prompt = user_prompt[:self.MAX_PROMPT_TOKENS * 4] + f"... The rest was truncated because it was too long (over {self.MAX_PROMPT_TOKENS * 4} chars). Please use the information given above, or if you're reading a file, please use or write a script to read chunks of the file."
            self.log(f"\n(update_answer_states) Truncated user prompt: {user_prompt}\n")

        raw_request = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.MAX_TOKENS_TO_SAMPLE,
            "stop": None,  # API doesn't like empty list
        }
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = self.client.chat.completions.create(**{"messages": messages, **raw_request})
        new_answer_state = response.choices[0].message.content
        self.log(f"\n(update_answer_states) New answer state: {new_answer_state}\n")

        # Update answer state to have newest action and result to oldest
        self.answer_states.insert(0, {
            "attempted_task": attempted_task,
            "plan": plan,
            "result": result,
            "files": self.files,
            "answer_state": new_answer_state,
        })

        while len(self.answer_states) > self.max_states:
            self.answer_states.pop()

    def save_env_state(self):
        """Save the environment state to work_dir for the agent to read."""
        state = {
            'files': self.files,
            'answer_states': self.answer_states,
            'files_action_result_history': self.files_action_result_history,
            'num_steps': self.num_steps,
            'start_time': self._start_time,
            'execution_runs': self.execution_runs,
            'num_tasks': self.num_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
        }
        with open(self.env_trace_path, 'w') as f:
            json.dump(state, f, indent=4)

    def summarize_without_logging(self, system_prompt, user_prompt):
        # Raw Chat Completion API to avoid logging loop
        # Truncate user prompt if too long and add a note
        if len(user_prompt) > self.MAX_PROMPT_TOKENS * 4:
            user_prompt = user_prompt[:self.MAX_PROMPT_TOKENS * 4] + f"... The rest was truncated because it was too long (over {self.MAX_PROMPT_TOKENS * 4} chars). Please use the information given above, or if you're reading a file, please use or write a script to read chunks of the file."
            self.log(f"\n(summarize_without_logging) Truncated user prompt: {user_prompt}\n")

        try:
            raw_request = {
                "model": self.model,
                "temperature": 0,
                "max_tokens": self.MAX_TOKENS_TO_SAMPLE,
                "stop": None,  # API doesn't like empty list
            }
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            response = self.client.chat.completions.create(**{"messages": messages, **raw_request})
            summarization = response.choices[0].message.content
            self.log(f"\n(summarize_without_logging) New summarization: {summarization}\n")
        except TooLongPromptError:
            summarization = f"EnvError: too long input.\n{e}" 
            self.log("--- SUMMARIZATION ERROR ---", e)
        except LLMError as e:
            summarization = "LLMError: " + str(e)
            self.log("--- SUMMARIZATION ERROR ---", e)
        except EnvException as e:
            summarization = "EnvError: " + str(e)
            try:
                self.log("--- SUMMARIZATION ERROR ---", str(e).replace(self.work_dir, '.'))
            except:
                self.log("--- SUMMARIZATION ERROR ---", e)
        except TypeError as e:
            invalid_action_error = f"The arguments needs to have proper entries. You may have missed some entries or used inappropriate ones. Please use the correct format and try again.\n{e}"
            summarization = "EnvError: " + invalid_action_error
            self.log("--- SUMMARIZATION ERROR ---", e)
        # except TimeoutException as e:
        #     raise e
        #     self.log("--- TOOL ERROR ---", e)
        except Exception as e:
            summarization = f"EnvError: \n{e}"
            self.log("--- SUMMARIZATION ERROR ---", e)

        return summarization

    ############## for actions ##############
    
    def reflection(self, **kwargs):
        @self.log_decorator
        def wrapped_reflection(things_to_reflect_on="", work_dir = ".", **kwargs):
            formatted_history = ""
            for idx, files_action_result_history in enumerate(self.files_action_result_history):
                formatted_history += "\nStep: " + str(idx) 
                formatted_history += "\nFiles: " + str(files_action_result_history['files']) 
                formatted_history += "\nAction: " + files_action_result_history['action'] 
                formatted_history += "\nResult: " + files_action_result_history['result']

            prompt = f"""We are trying to solve this research problem: {self.research_problem}

            Your current research log:
            ```
            {formatted_history}
            ```

            Reflect on this: {things_to_reflect_on} 
            
            Give an answer in natural language paragraphs as truthfully as possible. 
            """

            reflection = self.complete_text_openai(user_prompt=prompt)
            return f"Reflection: {reflection}\n"
        return wrapped_reflection(**kwargs)

    def list_files(self, **kwargs):
        @self.log_decorator
        def wrapped_list_files(**kwargs):
            return list_files(**kwargs)
        return wrapped_list_files(**kwargs)

    def read_file(self, **kwargs):
        @self.log_decorator
        def wrapped_read_file(file_name, work_dir = '.', max_char_read = 5000, **kwargs):
            assert("workspace" in self.work_dir and "branch" in self.work_dir) # we should only list files in the workspace and branch
            try:
                observation = open(os.path.join(work_dir, file_name)).read()
                if len(observation) > self.MAX_PROMPT_TOKENS * 4: # Auto-truncate if too long
                    observation = observation[:self.MAX_PROMPT_TOKENS * 4] + f"... The rest was truncated because it was too long (over {self.MAX_PROMPT_TOKENS * 4} chars). Please use the information given above, or if you're reading a file, please use or write a script to read chunks of the file."
                return observation
            except Exception as e:
                try:
                    raise EnvException(f"cannot read file {file_name}: {e.replace(self.work_dir, '.')}")
                except:
                    raise EnvException(f"cannot read file {file_name}: {e}")
        return wrapped_read_file(**kwargs)

    def write_file(self, **kwargs):
        @self.log_decorator
        def wrapped_write_file(file_name='', content='', **kwargs):
            try:
                # Original files should be read only
                read_only_directory = self.work_dir.split("_branch")[0]
                if os.path.isfile(os.path.join(read_only_directory, file_name)):
                    raise EnvException(f"File {file_name} is given and read only. Please try again with a different file name.")

                # Extract the directory path from the full file path and create directory if necessary
                directory = os.path.dirname(os.path.join(self.work_dir, file_name))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # Write the file
                with open(os.path.join(self.work_dir, file_name), "w") as f:
                    f.write(content)
                observation = f"File {file_name} written successfully."
                return observation
            except Exception as e:
                try:
                    raise EnvException(f"cannot write file {file_name}: {e.replace(self.work_dir, '.')}")
                except:
                    raise EnvException(f"cannot write file {file_name}: {e}")
        return wrapped_write_file(**kwargs)

    # helper function for extract_scores to visualize the scores
    def visualize_scores(self, scores, score_type):

        def visualize(viz_type="both"):
            # Assuming 'scores' is your list of dictionaries
            id_list = [item['id'] for item in scores]
            train_scores = [item['train_score'] for item in scores]
            val_scores = [item['val_score'] for item in scores]
            min_val_scores = [min(val_scores[:i+1]) for i in range(len(val_scores))]

            # to measure number of calls before each successful execution as a measure of efficiency
            num_calls = [scores[0]['step_id']]
            for i in range(1, len(scores)):
                num_calls.append(scores[i]['step_id'] - scores[i-1]['step_id'])

            # Now, plot id vs train_score and val_score and num_calls
            y_label = score_type
            plt.figure(figsize=(10, 6))
            if viz_type == 'both':
                train_scores = [item['train_score'] for item in scores]
                plt.scatter(id_list, train_scores, color='gray', alpha=0.25, label='Train Score')
                plt.scatter(id_list, val_scores, color='blue', alpha=0.25, label='Validation Score')
                plt.plot(id_list, min_val_scores, color='#1f77b4', label='Best Validation Score', marker='o')
                plot_path = os.path.join(self.eval_path, "scores.png")
            elif viz_type == 'val':
                plt.scatter(id_list, val_scores, color='blue', alpha=0.25, label='Validation Score')
                plt.plot(id_list, min_val_scores, color='#1f77b4', label='Best Validation Score', marker='o')
                plot_path = os.path.join(self.eval_path, "val_scores.png")
            elif viz_type == 'num_steps':
                y_label = '# of Steps Before Executing Script'
                plt.scatter(id_list, num_calls, color='green', alpha=0.5, label='# of Steps')
                plot_path = os.path.join(self.eval_path, "num_steps.png")
                plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            # Adding labels and legend
            plt.xlabel('Execution Run')
            plt.ylabel(f'{y_label}')
            plt.title(f'Run vs {y_label}')
            plt.legend()

            # Save the plot
            plt.savefig(plot_path)
            plt.clf()

        visualize('both') # train and val scores in log/<log>/scores.png
        visualize('val') # val scores in log/<log>/val_scores.png
        visualize('num_steps') # number of steps in log/<log>/num_steps.png

    # helper function for execute_script
    def extract_scores(self, python_code, result):
        extract_score_args = {
            'system_prompt': '''You are a helpful assistant. Your goal is to check if the code outputs the train and validation score, specifically not from log values, but normal values, and make sure that the code for calculating validation score is actually from a validation set. If so, then extract the train and validation score value from the result. If the code doesn't output the train and validation score or its not for normal values or the code doesn't actually calculate and print validation score from a validation set, then please write "inf" as the traing and validation score. Please also write down what type of validation score it is (ex. Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), accuracy, etc.)
            
            Example:
            ```json
            {
                "observations": "<string>",
                "score_type": "<string>",
                "train_score": <float>,
                "val_score": <float>
            }''',
            'json_required': True,
            'user_prompt': "Code: " + python_code + "\nResult after executing code: " + result,
            'max_tokens': 4096,
            'temperature': 0.0,
            'top_p': 0.0,
            'update_files_action_result_history': False,
        }
        scores_json = self.complete_text_openai(**extract_score_args)
        
        # try updating scores.json
        try:
            scores = json.loads(scores_json)
            score_type = scores["score_type"]
            train_score = float(scores['train_score'])
            val_score = float(scores['val_score'])

            if score_type != "N/A": # ensure correct parsing
                scores_file_path = os.path.join(self.eval_path, "scores.json")

                # Check if the file exists.
                if not os.path.exists(scores_file_path):
                    # If the file does not exist, create an empty list and save it to the file.
                    with open(scores_file_path, 'w') as file:
                        json.dump([], file)
                    scores = []
                else:
                    # If the file exists, load its contents.
                    with open(scores_file_path, 'r') as file:
                        scores = json.load(file)

                # Now, add a new entry to the scores list.
                new_score = {
                    'id': self.execution_runs,
                    'step_id': self.num_steps,
                    'train_score': train_score,
                    'val_score': val_score
                }
                scores.append(new_score)
                self.execution_runs += 1

                # Save the updated scores list back to the file.
                with open(scores_file_path, 'w') as file:
                    json.dump(scores, file, indent=4)

                # Update eval.img
                self.visualize_scores(scores, score_type)

        except:
            score_type = "N/A"
            train_score = 'inf'
            val_score = 'inf'
        return

    # TODO: add the "check_file_in_work_dir" function from before
    def execute_script(self, **kwargs):
        @self.log_decorator
        def wrapped_execute_script(script_name, work_dir = ".", **kwargs):
            assert("workspace" in self.work_dir and "branch" in self.work_dir) # we should only list files in the workspace and branch

            if not os.path.exists(os.path.join(work_dir, script_name)):
                raise EnvException(f"The file {script_name} does not exist.")
            
            # Trying to get execute script to work on Windows
            device = kwargs.get("device", "0")  # Default device is "0"
            python_executable = kwargs.get("python", "python")  # Default Python command is "python"

            # Set environment variables
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = device

            # Execute the script
            try:
                process = subprocess.Popen(
                    [python_executable, "-u", script_name],  # script_name is used directly
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    cwd=self.work_dir  # self.work_dir is used as the current working directory
                )
                stdout, stderr = process.communicate()  # This waits for the process to finish and gets the output

                # Sometimes there is no output, therefore, we append the code as well as the message to the output for more information
                observation = open(os.path.join(work_dir, script_name)).read()

                if len(observation) > self.MAX_PROMPT_TOKENS * 4: # Auto-truncate if too long
                    observation = observation[:self.MAX_PROMPT_TOKENS * 4] + f"... The rest was truncated because it was too long (over {self.MAX_PROMPT_TOKENS * 4} chars). Please use the information given above, or if you're reading a file, please use or write a script to read chunks of the file."
                if process.returncode != 0:
                    # Handle error
                    error_message = "Error executing the script: " + stderr
                    self.log(error_message)
                    return error_message
                    # + "\nExecuted file contents: \n" + observation
                else:
                    # Successful execution
                    success_message = "Script output: " + stdout
                    self.log(success_message)

                    # Evaluation
                    self.extract_scores(python_code=observation, result=stdout)

                    return success_message
                    #  + "\nExecuted file contents: \n" + observation
            except Exception as e:
                try:
                    raise EnvException(f"Something went wrong in executing {script_name}: {e.replace(self.work_dir, '.')}. Please check if it is ready to be executed.")
                except:
                    raise EnvException(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")



            # Below works on non-Windows environments
            # try:
            #     script_path = script_name
            #     device = kwargs.get("device", "0")  # Default device is "0"
            #     python = kwargs.get("python", "python")  # Default Python command is "python"

            #     cmd = f"CUDA_VISIBLE_DEVICES={device} {python} -u {script_path}"
            #     process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=work_dir) # this sets the path for execution!

            #     stdout_lines = []
            #     stderr_lines = []

            #     selector = selectors.DefaultSelector()
            #     selector.register(process.stdout, selectors.EVENT_READ)
            #     selector.register(process.stderr, selectors.EVENT_READ)

            #     while process.poll() is None and selector.get_map():
            #         events = selector.select(timeout=1)

            #         for key, _ in events:
            #             line = key.fileobj.readline()
            #             if key.fileobj == process.stdout:
            #                 self.log("STDOUT:", line, end =" ")
            #                 stdout_lines.append(line)
            #             else:
            #                 self.log("STDERR:", line, end =" ")
            #                 stderr_lines.append(line)

            #     for line in process.stdout:
            #         line = line
            #         self.log("STDOUT:", line, end =" ")
            #         stdout_lines.append(line)
            #     for line in process.stderr:
            #         line = line
            #         self.log("STDERR:", line, end =" ")
            #         stderr_lines.append(line)

            #     return_code = process.returncode

            #     if return_code != 0:
            #         observation = "".join(stderr_lines)
            #     else:
            #         observation = "".join(stdout_lines)
            #     if observation == "" and return_code == 0:
            #         # self.loged to stderr only
            #         observation = "".join(stderr_lines)

            #     return "The script has been executed. Here is the output:\n" + observation + "\nSTDOUT:\n" + "".join(stdout_lines) + "\nSTDERR:\n" + "".join(stderr_lines)
            # except Exception as e:
            #     raise EnvException(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")
        return wrapped_execute_script(**kwargs)

    def request_help(self, **kwargs):
        @self.log_decorator
        def wrapped_request_help(**kwargs):
            return request_help(**kwargs)
        return wrapped_request_help(**kwargs)

    def final_answer(self, **kwargs):
        @self.log_decorator
        def wrapped_final_answer(**kwargs):
            self.final_answer = kwargs.get('final_answer', "No final answer was submitted as an argument.")
            return "You have successfully submitted your final answer. No more actions necessary."
        return wrapped_final_answer(**kwargs)
    
    def web_search(self, **kwargs):
        @self.log_decorator
        def wrapped_web_search(query = '', **kwargs):
            try:
                web_search_res = input(f"Query: {query} | Result: ") # temporary quick way for web searching
                return web_search_res
            except:
                raise EnvException(f"Web search failed.")
        return wrapped_web_search(**kwargs)
    
    # Adding code completion here so that it's easier to log
    def complete_text_openai(self, **kwargs):
        @self.log_decorator
        def wrapped_complete_text_openai(system_prompt="You are a helpful assistant.", user_prompt="", stop_sequences=[], model=self.model, max_tokens_to_sample=2000, temperature=0.2, json_required=False, tools=None, available_functions=None, max_prompt_tokens=self.MAX_PROMPT_TOKENS, **kwargs):
            """ Call the OpenAI API to complete a prompt."""
            # Truncate user prompt if too long and add a note
            if len(user_prompt) > self.MAX_PROMPT_TOKENS * 4:
                user_prompt = user_prompt[:self.MAX_PROMPT_TOKENS * 4] + f"... The rest was truncated because it was too long (over {self.MAX_PROMPT_TOKENS * 4} chars). Please use the information given above, or if you're reading a file, please use or write a script to read chunks of the file."
                self.log(f"\n(complete_text_openai) Truncated user prompt: {user_prompt}\n")
                self.log("Truncated user prompt: ", user_prompt)
                self.log("# of input tokens start: ", len(system_prompt + user_prompt) // 4)
            
            kwargs.pop('work_dir', None) # Chat completions can't take work_dir as an arg
            raw_request = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens_to_sample,
                "stop": stop_sequences or None,  # API doesn't like empty list
                **kwargs
            }

            # Add additional parameters if necessary (JSON or function calling)
            if json_required and (model == "gpt-3.5-turbo-1106" or model == "gpt-4-1106-preview"):
                raw_request["response_format"] = {"type": "json_object"}
                user_prompt += '\nEnsure the response can be parsed by Python "json.loads", e.g.: no trailing commas, no single quotes, etc. This is important.'
            if tools and available_functions:
                raw_request["tools"] = tools
                raw_request["tool_choice"] = "auto"
                
            # Call the API
            try:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                response = self.client.chat.completions.create(**{"messages": messages,**raw_request})
                completion = response.choices[0].message.content
                tool_calls = response.choices[0].message.tool_calls

                # Ensure that the completion is JSON parsable. If it isn't, ask GPT to make it JSON parsable by increasing max tokens
                if json_required and (model == "gpt-3.5-turbo-1106" or model == "gpt-4-1106-preview"):
                    try:
                        completion_json = json.loads(completion)
                    except:
                        convert_to_json_prompt = f'''Close this incomplete JSON so that it's in proper JSON format: {completion}'''
                        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": convert_to_json_prompt}]
                        raw_request["max_tokens"] = max_tokens_to_sample + 100
                        response = self.client.chat.completions.create(**{"messages": messages,**raw_request})
                        completion = response.choices[0].message.content

                # Handle function calling
                if tool_calls:
                    messages.append(response.choices[0].message)
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_to_call = available_functions[function_name]
                        function_args = json.loads(tool_call.function.arguments)
                        function_response = function_to_call(**function_args)
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )
                    return "Function calling complete! Action and result should be saved to history."
            except Exception as e:
                completion = f"EnvError: Chat Completions API call failed: {e}. Please try again or edit your files or prompt based on the error message to prevent the error from happening again."
            
            return completion
        return wrapped_complete_text_openai(**kwargs)
    
    # Function to run the OpenAI Assistants API
    def run_assistant(self, **kwargs):
        @self.log_decorator
        def wrapped_run_assistant(system_prompt, user_prompt, tool_descriptions=None, **kwargs):
            # Truncate user prompt if too long and add a note
            if len(user_prompt) > self.MAX_PROMPT_TOKENS * 4:
                user_prompt = user_prompt[:self.MAX_PROMPT_TOKENS * 4]
                user_prompt += f"... The rest was truncated because it was too long (over {self.MAX_PROMPT_TOKENS * 4} chars). Please use the information given above, or if you're reading a file, please use or write a script to read chunks of the file."
                self.log(f"\n(run_assistant) Truncated user prompt: {user_prompt}\n")
                self.log("Assistants Truncated user prompt: ", user_prompt)
                self.log("# of input tokens start: ", len(system_prompt + user_prompt) // 4)
                
            # Default tool_descriptions is the initialized one
            if tool_descriptions is None:
                tool_descriptions = self.tool_descriptions

            try:
                # Instantiate an Assistant
                self.assistant = self.client.beta.assistants.create(
                    name="Research Agent",
                    instructions=system_prompt,
                    tools=tool_descriptions,
                    model=self.model
                )
                self.thread = self.client.beta.threads.create()

                # Invoke the Assistants API to answer
                self.client.beta.threads.messages.create(
                    thread_id=self.thread.id,
                    role="user",
                    content=user_prompt
                )
                run = self.client.beta.threads.runs.create(
                    thread_id=self.thread.id,
                    assistant_id=self.assistant.id,
                )

                # Wait until the run has looped
                run_complete = False
                num_tries = 100
                while not run_complete:
                    # Check if there's an update on the run
                    run = self.client.beta.threads.runs.retrieve(
                        thread_id=self.thread.id,
                        run_id=run.id
                    )
                    run_complete = run.status == "completed"
                    self.log("\nrun.status: ", run.status)

                    # Call the tools if the run status is requires action
                    if run.status == "requires_action":
                        tool_outputs = []
                        for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                            tool_id, tool_function, tool_type = tool_call.id, tool_call.function, tool_call.type
                            self.log(f"Run required action: \ntool_id: {tool_id}, \ntool_function.arguments: {tool_function.arguments}, \ntool_function.name: {tool_function.name}, \ntool_type: {tool_type}")

                            # Call the function directly if `tool_function` is a callable object
                            # and `arguments` is a dictionary of arguments to pass to the function.
                            try:
                                arguments = json.loads(tool_function.arguments)
                                function_output = self.available_actions[tool_function.name](**arguments)
                            except Exception as e:
                                function_output = f"Tool function {tool_function.name} for tool_id {tool_id} does not exist and is not callable with arguments {tool_function.arguments}. Make sure you are using only tools listed here: {self.available_actions.keys()} with the right arguments."

                            tool_outputs.append({
                                "tool_call_id": tool_id,
                                "output": function_output
                            })

                        # Submit tool outputs as a new run
                        run = self.client.beta.threads.runs.submit_tool_outputs(
                            thread_id=self.thread.id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )
                    elif run.status == "failed":
                        self.log("Run failed: ", run)
                        completion = "Assistants API call run failed. Please try again."
                        break

                    time.sleep(1)
                    num_tries -= 1
                    if num_tries == 0:
                        self.log("Run timed out, cancelling...")
                        run = self.client.beta.threads.runs.cancel(thread_id=self.thread.id, run_id=run.id)
                        while run.status == "in_progress":
                            run = self.client.beta.threads.runs.retrieve(
                                thread_id=self.thread.id,
                                run_id=run.id
                            )
                            # TODO: Looks like there's an error here where the status doesn't turn to cancelled?
                        completion = "Execution timed out. Please try again."
                        self.log("Run cancelled!")
                        break
                messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
                completion = messages.data[0].content[0].text.value
            except Exception as e:
                completion = f"EnvError: Assistants API call failed: {e}. Please try again or edit your files or prompt based on the error message to prevent the error from happening again."
            return completion
        return wrapped_run_assistant(**kwargs)

    ############################## internal functions ########################################

    def __enter__(self):
        # set time out
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.args.max_time)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):  
        # save error message
        active = active_children()
        self.log(f'Active Children: {len(active)}')
        # terminate all active children
        for child in active:
            child.terminate()
        # block until all children have closed
        for child in active:
            child.join()
        # report active children
        active = active_children()
        self.log(f'Active Children: {len(active)}')
            
        if traceback is not None:
            self.log("Error message saved in error.txt")
            open(os.path.join(self.log_dir, "error.txt"), "w").write(''.join(format_exception(exc_type, exc_value, traceback)))
        open(os.path.join(self.log_dir, "overall_time.txt"), "w").write(str(time.time() - self.start_time))
           
    
    ############################## getters ########################################

    @property
    def research_problem(self):
        return self._research_problem

    @property
    def benchmark_folder_name(self):
        return self._benchmark_folder_name

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def work_dir(self):
        return self._work_dir

    @property
    def tool_descriptions(self):
        return self._tool_descriptions

    @property
    def available_actions(self):
        return self._available_actions
    
    @property
    def args(self):
        return self._args

    @property
    def start_time(self):
        return self._start_time
     
    ################################# public functions ########################################

    def is_final(self):
        """Check if the task has reached a final state, either by reaching the maximum steps or time, or because the agent has submitted a final answer. """
        if self.num_steps >= self.args.max_steps or time.time() - self.start_time > self.args.max_time:
            return True, None
            
        if self.final_answer:
            final_answer_evaluation = input(f"\nFinal answer submitted: {self.final_answer} Did the agent submit a valid final answer? If yes, respond with 'yes'. If not, provide feedback. ")
            if final_answer_evaluation == "yes":
                return True, None
            return False, final_answer_evaluation
        
        return False, None
    
    # Formatting answer states for rapid experimentation
    def formatted_answer_states(self):
        assert('files' in self.answer_states[0].keys() and 'attempted_task' in self.answer_states[0].keys() and 'plan' in self.answer_states[0].keys() and 'result' in self.answer_states[0].keys() and 'answer_state' in self.answer_states[0].keys())
        formatted_answer_states = ""
        for idx, answer_state in enumerate(self.answer_states):
            formatted_answer_states += "\nStep: " + str(len(self.answer_states) - 1 - idx) 
            # formatted_answer_states += "\na) Attempted Task: " + str(answer_state['attempted_task']) 
            # formatted_answer_states += "\nb) Plan: " + str(answer_state['plan']) 
            # formatted_answer_states += "\nc) Result: " + str(answer_state['result']) 
            # formatted_answer_states += "\nd) Files: " + str(answer_state['files']) 
            formatted_answer_states += "\nAnswer State: " + str(answer_state['answer_state']) 
        return formatted_answer_states
    
    def formatted_action_history(self):
        assert('action' in self.files_action_result_history[0].keys() and 'result' in self.files_action_result_history[0].keys() and 'files' in self.files_action_result_history[0].keys())
        formatted_history = ""
        for idx, files_action_result_history in enumerate(self.files_action_result_history):
            formatted_history += "\nStep: " + str(len(self.files_action_result_history) - 1 - idx) 
            formatted_history += "\nFiles: " + str(files_action_result_history['files']) 
            formatted_history += "\nAction: " + files_action_result_history['action'] 
            formatted_history += "\nResult: " + files_action_result_history['result']
        return formatted_history
    
    def search_wikipedia(self, concept):
        # Get the page ID for the concept
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": concept
        }
        response = self.request_session.get(url=url, params=params)
        data = response.json()
        page_id = data['query']['search'][0]['pageid']

        # Get the page content for the page ID
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "pageids": page_id,
            "explaintext": True
        }
        response = self.request_session.get(url=url, params=params)
        text = response.json()['query']['pages'][str(page_id)]['extract']

        # Truncate text if necessary. Divide by 2 because the text likely won't be that significant, we'll use only the first half of the text on the page
        max_chars = self.MAX_PROMPT_TOKENS * 4 // 2
        if len(text) > max_chars:
            text = text[:max_chars]
            text += f"... The rest was truncated because it was too long (over {max_chars} chars). Please use the information given above."
            self.log(f"\n(wikipedia) Truncated wikipedia text: {text}\n")
        return text

    def log(self, *args):
        message = ' '.join(str(arg) for arg in args)
        with open(self.main_log_path, "a", encoding='utf-8', buffering=1) as log_file:
            log_file.write(message + '\n')
        print(message)
