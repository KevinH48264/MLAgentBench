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
from traceback import format_exception
from multiprocessing import active_children
import readline # to make sure input() works properly
from dacite import from_dict
import functools

import MLAgentBench_v2.high_level_actions as high_level_actions
from .schema import Step, Trace, EnvException, TooLongPromptError, LLMError, EnhancedJSONEncoder 
from .LLM import complete_text_claude
from .prepare_task import prepare_task
from MLAgentBench_v2.actions import TOOL_DESCRIPTIONS
from MLAgentBench.high_level_actions import understand_file, append_to_research_log, inspect_script_lines, edit_script, edit_script_lines, reflection, retrieval_from_research_log
from MLAgentBench.low_level_actions import list_files, read_file, write_file, append_file, copy_file, undo_edit_script, execute_script, python_repl, request_help

class Environment:
    def __init__(self, args):
        print("Initializing environment...")
        self._args = args

        # Set up workspace and research problem.
        with open('MLAgentBench_v2/research_problem.txt', 'r') as f:
            self._research_problem = f.read()
        self._benchmark_folder_name = args.task
        self._work_dir = prepare_task(
            work_dir = args.work_dir, 
            task_name = args.task, 
            task_type = args.task_type
        )

        # Set up logging, overwrite existing logs if they exist
        self._log_dir = args.log_dir
        if os.path.exists(self._log_dir):
            shutil.rmtree(self._log_dir)
        os.makedirs(self._log_dir)
        self.main_log_path = os.path.join(self._log_dir, "main_log.txt")
        self.num_steps = 0
        self._start_time = time.time()

        # Set up actions and logging
        self._tool_descriptions = TOOL_DESCRIPTIONS # Formatted for OpenAI function calling
        self._available_actions = {
                # 'understandFile': understand_file,
                # 'appendSummaryToResearchLog': append_to_research_log,
                # 'inspectScriptLines': inspect_script_lines,
                # 'editScript': edit_script,
                # 'editScriptSegment': edit_script_lines,
                'reflection': self.reflection,
                # 'retrievalFromResearchLog': retrieval_from_research_log,
                'listFiles': self.list_files,
                'readFile': self.read_file,
                'writeFile': self.write_file,
                # 'appendFile': append_file,
                # 'copyFile': copy_file,
                # 'undoEditScript': undo_edit_script,
                'executeScript': self.execute_script,
                # 'pythonREPL': python_repl,
                # 'requestHelp': self.request_help,
                'finalAnswer': self.final_answer,
                # 'openaiAssistantCreateAssistant': pass,
                # 'openaiAssistantCreateThread': pass,
                # 'openaiAssistantCreateThreadMessage': pass,
                # 'openaiAssistantCreateRun': pass,
                # 'openaiAssistantListThreadMessageCompletion': pass,
            }
        self.final_answer = False

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
    
    # @property
    # def read_only_files(self):
    #     return self._read_only_files

    # @property
    # def action_infos(self):
    #     return self._action_infos
    
    @property
    def args(self):
        return self._args

    # @property
    # def static_kwargs_for_tools(self):
    #     return self._static_kwargs_for_tools
    
    # @property
    # def trace(self):
    #     return copy.deepcopy(self._trace)

    @property
    def start_time(self):
        return self._start_time
    
    ############################## internal functions ########################################
    
    def _setup_log_dir(self):
        # set up log dir
        if os.path.exists(self.args.log_dir):
            print("log_dir {} already exists".format(self.log_dir))
        else:
            os.makedirs(self.log_dir)


    def _initialize_interactive_env(self):
        # set up read only files
        can_modify_files = input("What existing files can Research Assistant modify (relative paths separated by comma)? default is nothing: ").split(",")
        size = 0
        self._read_only_files = []
        for path, subdirs, files in os.walk(os.path.join(self.work_dir)):
            relpath = os.path.relpath(path, self.work_dir)
            # filter out the files that are read only
            filenames = [os.path.join(relpath, filename) for filename in files]
            for not_ignore in can_modify_files:
                ignore_filenames = [n for n in filenames if not fnmatch.fnmatch(n, not_ignore)]
                self.read_only_files.extend(ignore_filenames)
            for f in files:
                size += os.path.getsize(os.path.join(path, f))
                
        # try save this task to a benchmark folder
        os.makedirs(os.path.join(self.log_dir, self.benchmark_folder_name), exist_ok=True)
        if size / 1e6 < 10:
            # save if the size is smaller than 10MB
            shutil.copytree(self.work_dir, os.path.join(self.log_dir, self.benchmark_folder_name, "env"))
        os.makedirs(os.path.join(self.log_dir, self.benchmark_folder_name, "scripts"), exist_ok=True)
        with open(os.path.join(self.log_dir, self.benchmark_folder_name, "scripts", "research_problem.txt"), "w") as f:
            f.write(self.research_problem)
        with open(os.path.join(self.log_dir, self.benchmark_folder_name, "scripts", "read_only_files.txt"), "w") as f:
            f.write("\n".join(self.read_only_files))

        # init backup folder and remove all content if it exists
        if os.path.exists(os.path.join(self.work_dir, "backup")):
            shutil.rmtree(os.path.join(self.work_dir, "backup"))
        os.mkdir(os.path.join(self.work_dir, "backup"))
    
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
        print(f'Active Children: {len(active)}')
        # terminate all active children
        for child in active:
            child.terminate()
        # block until all children have closed
        for child in active:
            child.join()
        # report active children
        active = active_children()
        print(f'Active Children: {len(active)}')
            
        if traceback is not None:
            print("Error message saved in error.txt")
            open(os.path.join(self.log_dir, "error.txt"), "w").write(''.join(format_exception(exc_type, exc_value, traceback)))
        open(os.path.join(self.log_dir, "overall_time.txt"), "w").write(str(time.time() - self.start_time))
            
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

    def save(self, curr_step):
        print("SAVING IN ENVIRONMENT.PY!")
        """ Save the trace and snapshot of the workspace folder """     
        with open(os.path.join(self.log_dir, f"trace.json"), "w") as f:
            json.dump(self.trace, f, indent=4, cls=EnhancedJSONEncoder)

        ##### save a snapshot of the current step
        save_folder = os.path.join(self.log_dir, f"traces/step_{curr_step}_files")
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)

        # save files in the folder that are not read only
        for path, subdirs, files in os.walk(os.path.join(self.work_dir)):

            relpath = os.path.relpath(path, self.work_dir)
            dest = os.path.join(save_folder, relpath)

            for file_name in files:
                file_path = os.path.join(relpath, file_name)
                if file_path not in self.read_only_files:
                    # check wether the file to copy is part of self.log_dir
                    if  os.path.abspath(os.path.join(self.work_dir, file_path)).startswith(os.path.abspath(self.log_dir.split("/env_log")[0])):
                        continue                    
                    if not os.path.exists(dest):
                        os.makedirs(dest)            
                    shutil.copyfile(os.path.join(self.work_dir, file_path), os.path.join(save_folder, file_path))

    ############## for logging ##############

    # Logging decorator
    def log_decorator(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                print(f"\nStep: {self.num_steps}\nCalling function {func.__name__}({args}, {kwargs})\n")

                # Log the function call
                with open(self.main_log_path, "a", 1) as log_file:
                    log_file.write(f"\nStep: {self.num_steps}\nCalling function {func.__name__}({args}, {kwargs})\n")

                # Perform the actual function
                result = func(*args, **kwargs)

                print("--- TOOL SUCCESS ---")
            except TooLongPromptError:
                result = "EnvError: too long input for the tool"
                print("--- TOOL ERROR ---", e)
            except LLMError as e:
                result = "LLMError: " + e.message
                print("--- TOOL ERROR ---", e)
            except EnvException as e:
                result = "EnvError: " + e.message
                print("--- TOOL ERROR ---", e)
            except TypeError as e:
                invalid_action_error = f"The arguments needs to have proper entries. You may have missed some entries or used inappropriate ones. Please use the correct format and try again:\n{self.tool_descriptions}"
                result = "EnvError: " + invalid_action_error
                print("--- TOOL ERROR ---", e)
            except TimeoutException as e:
                raise e
                print("--- TOOL ERROR ---", e)
            except Exception as e:
                result = f"EnvError: Error executing {action_name}."
                print("--- TOOL ERROR ---", e)

            # Log the function output
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\nFunction {func.__name__} returned: \n{result}\n")

            # Copy work_dir if it exists
            if self.work_dir and os.path.exists(self.work_dir):
                dest_dir = os.path.join(self.log_dir, f"{self.num_steps}_work_dir")
                shutil.copytree(self.work_dir, dest_dir, dirs_exist_ok=True)

            self.num_steps += 1
            return result
        return wrapper

    ############## for actions ##############
    
    # TODO: This can likely be fixed by actually putting all functions in here
    def reflection(self, **kwargs):
        @self.log_decorator
        def wrapped_reflection(**kwargs):
            return reflection(research_problem=self.research_problem, log_file=self.main_log_path, **kwargs)
        return wrapped_reflection(work_dir=self.work_dir, **kwargs)

    def list_files(self, **kwargs):
        @self.log_decorator
        def wrapped_list_files(**kwargs):
            return list_files(**kwargs)
        return wrapped_list_files(work_dir = self.work_dir, **kwargs)

    def read_file(self, **kwargs):
        @self.log_decorator
        def wrapped_read_file(file_name, work_dir = '.', max_char_read = 5000, **kwargs):
            print("Reading file!", file_name, work_dir)
            try:
                observation = open(os.path.join(work_dir, file_name)).read()
                return observation[:max_char_read]
            except:
                raise EnvException(f"cannot read file {file_name}")
        return wrapped_read_file(work_dir=self.work_dir, max_char_read = 2000, **kwargs)

    def write_file(self, **kwargs):
        @self.log_decorator
        def wrapped_write_file(**kwargs):
            return write_file(**kwargs)
        return wrapped_write_file(work_dir=self.work_dir, **kwargs)

    # TODO: add the "check_file_in_work_dir" function from before
    def execute_script(self, **kwargs):
        @self.log_decorator
        def wrapped_execute_script(script_name, work_dir = ".", **kwargs):
            print("\nEXECUTE SCRIPT WAS CALLED")
            print("\nRunning execute script! From directory: ", os.getcwd(), " work_dir: ", work_dir)
            print("THIS IS THE SCRIPT WE'RE LOOKING FOR: ", os.path.join(work_dir, script_name))
            if not os.path.exists(os.path.join(work_dir, script_name)):
                raise EnvException(f"The file {script_name} does not exist.")
            try:
                script_path = script_name
                device = kwargs.get("device", "0")  # Default device is "0"
                python = kwargs.get("python", "python")  # Default Python command is "python"

                cmd = f"CUDA_VISIBLE_DEVICES={device} {python} -u {script_path}"
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=work_dir) # this sets the path for execution!

                stdout_lines = []
                stderr_lines = []

                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                selector.register(process.stderr, selectors.EVENT_READ)

                while process.poll() is None and selector.get_map():
                    events = selector.select(timeout=1)

                    for key, _ in events:
                        line = key.fileobj.readline()
                        if key.fileobj == process.stdout:
                            print("STDOUT:", line, end =" ")
                            stdout_lines.append(line)
                        else:
                            print("STDERR:", line, end =" ")
                            stderr_lines.append(line)

                for line in process.stdout:
                    line = line
                    print("STDOUT:", line, end =" ")
                    stdout_lines.append(line)
                for line in process.stderr:
                    line = line
                    print("STDERR:", line, end =" ")
                    stderr_lines.append(line)

                return_code = process.returncode

                if return_code != 0:
                    observation = "".join(stderr_lines)
                else:
                    observation = "".join(stdout_lines)
                if observation == "" and return_code == 0:
                    # printed to stderr only
                    observation = "".join(stderr_lines)

                return "The script has been executed. Here is the output:\n" + observation + "\nSTDOUT:\n" + "".join(stdout_lines) + "\nSTDERR:\n" + "".join(stderr_lines)
            except Exception as e:
                raise EnvException(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")
        return wrapped_execute_script(work_dir=self.work_dir, **kwargs)

    def request_help(self, **kwargs):
        @self.log_decorator
        def wrapped_request_help(**kwargs):
            return request_help(**kwargs)
        return wrapped_request_help(work_dir=self.work_dir, **kwargs)

    def final_answer(self, **kwargs):
        @self.log_decorator
        def wrapped_final_answer(**kwargs):
            self.final_answer = kwargs.get('final_answer', "No final answer was submitted as an argument.")
            return "You have successfully submitted your final answer. No more actions necessary."
        return wrapped_final_answer(work_dir=self.work_dir, **kwargs)