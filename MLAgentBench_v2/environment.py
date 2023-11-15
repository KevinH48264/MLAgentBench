"""
This file contains the Environment class, which prepares the environment for the research agent to run in.

Requirements:
1. This should have access to the workspace and be clear about what the research problem is.
2. This should have access to the actions.py file.
"""

import json
import os
import sys
import subprocess
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
        script_dir = '/MLAgentBench'

        # Set up workspace and research problem.
        with open('MLAgentBench_v2/research_problem.txt', 'r') as f:
            self._research_problem = f.read()
        self._benchmark_folder_name = args.task
        self._work_dir = prepare_task(
            work_dir = os.path.join(script_dir, args.work_dir), 
            task_name = args.task, 
            task_type = args.task_type
        )

        # Set up logging
        self._log_dir = args.log_dir
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        self.main_log_path = os.path.join(self._log_dir, "main.log")
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
                # 'requestHelp': request_help,
                'finalAnswer': print,
                # 'openaiAssistantCreateAssistant': pass,
                # 'openaiAssistantCreateThread': pass,
                # 'openaiAssistantCreateThreadMessage': pass,
                # 'openaiAssistantCreateRun': pass,
                # 'openaiAssistantListThreadMessageCompletion': pass,
            }

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


    # def _initialize_trace(self):
    #     if self.args.resume:
    #         print("Restoring trace from {}".format(self.args.resume))
    #         prev_trace = from_dict(data_class=Trace, data=json.load(open(os.path.join(self.args.resume, "env_log","trace.json"), "r")))
    #         print("Resetting trace to step {}".format(self.args.resume_step))
    #         steps = prev_trace.steps[:self.args.resume_step+1]
    #         t = steps[-1].timestamp
    #         low_level_steps = [s for s in prev_trace.low_level_steps if s.timestamp < t]
    #         trace = Trace(
    #             steps=steps,
    #             low_level_steps=low_level_steps,
    #             action_infos=self.action_infos,
    #             task_description=self.research_problem,
    #         )
    #     else:   
    #         trace = Trace(
    #         steps=[],
    #         low_level_steps=[],
    #         action_infos=self.action_infos,
    #         task_description=self.research_problem,
    #     )
    #     return trace
    
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
        
        curr_step = len(self.trace.steps)
        # check if any step is final answer
        any_final_answer = any([s.action.name == "Final Answer" for s in self.trace.steps])
        return curr_step >= self.args.max_steps or any_final_answer or time.time() - self.start_time > self.args.max_time

    def execute(self, action):
        """Execute an action and return the observation."""

        curr_step = len(trace.steps)
        action_name = action.name
        action_input = action.args

        if action_name == "Final Answer":
            observation = "end"

        elif self.is_final():
            observation = "The environment has shut down because the maximum number of steps or time has been reached. Please submit your final answer."

        elif action_name not in list(self.action_infos.keys()):
            actions = ", ".join(self.action_infos.keys())
            observation = f"Invalid action: {action_name}. Action did not execute. Please use one of the following actions:\n{actions}"

        else:
            # execute the action and get the observation
            log_file = os.path.join(os.path.join(self.log_dir, "tool_logs") , f"step_{curr_step}_tool_log.log")
            usage = ",\n            ".join([f"{k}: [{v}]" for k, v in self.action_infos[action_name].usage.items()])
            usage = f"""{{
            {usage}
            }}"""
            invalid_action_error = f"The action input for {action_name} needs to be a valid json with proper entries. You may have missed the comma between entries. Please use the correct format and try again:\n{usage}"

            if isinstance(action_input, dict):
                try:
                    observation = self.action_infos[action_name].function(**action_input, log_file=log_file, **self.static_kwargs_for_tools)
                except TooLongPromptError:
                    observation="EnvError: too long input for the tool"
                except LLMError as e:
                    observation = "LLMError: " + e.message
                except EnvException as e:
                    observation = "EnvError: " + e.message
                except TypeError as e:
                    print("Step: ", curr_step, file=sys.stderr)
                    print(e, file=sys.stderr)
                    print(action_input, file=sys.stderr)
                    observation = "EnvError: " + invalid_action_error
                except TimeoutException as e:
                    raise e
                except Exception as e:
                    # should not happen
                    print("Step: ", curr_step, file=sys.stderr)
                    print(e, file=sys.stderr)
                    if "Connection aborted." in str(e):
                        raise Exception("Connection aborted for crfm")
                    observation = f"EnvError: Error executing {action_name}."
            else:
                observation = invalid_action_error


        step_time = time.time()

        trace.steps.append(Step(action, observation, step_time))

        self.save(curr_step)
        return observation

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
            # Log the function call
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\nStep: {self.num_steps}\nCalling function: {func.__name__} with args: {args}, kwargs: {kwargs}\n")

            # Perform the actual function
            result = func(*args, **kwargs)

            # Log the function output
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"Function {func.__name__} returned: {result}\n")

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
            return reflection(**kwargs)
        return wrapped_reflection(**kwargs)

    def list_files(self, **kwargs):
        @self.log_decorator
        def wrapped_list_files(**kwargs):
            return list_files(**kwargs)
        return list_files(**kwargs)

    def read_file(self, **kwargs):
        @self.log_decorator
        def wrapped_read_file(**kwargs):
            return read_file(**kwargs)
        return read_file(**kwargs)

    def write_file(self, **kwargs):
        @self.log_decorator
        def wrapped_write_file(**kwargs):
            return write_file(**kwargs)
        return write_file(**kwargs)

    def execute_script(self, **kwargs):
        @self.log_decorator
        def wrapped_execute_script(**kwargs):
            return execute_script(**kwargs)
        return execute_script(**kwargs)