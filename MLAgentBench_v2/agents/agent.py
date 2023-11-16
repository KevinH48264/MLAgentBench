""" This file defines the basic agent class that can be used to implement different agents. 

This should be a lightweight agent class that demos how to solve the research problem via a simple init() and run() function. Perhaps the interface run() function just calls the Assistants API until it works with 1 prompt with the research problem.

The goal is that the class agent interface should be extensible to other agent frameworks like Eureka, Voyager, OPRO, and AutoGen.

Notes:
The Abstract Base Class should just define the structure and exepcted behavior of its subclasses through abstract methods and also provide shared functionality the child classes can use. It should not have everything that's necessary for the agent interface.

Requirements:
1. init()
- Actions
- Logging
- Workspace
2. run()
- Iterative cycle of calling LLM with Actions and Workspace, and logging it all

The actual Child Class must implement all the abstract methods (init(), run())
"""

import json
import sys
import os
import re
import glob
import copy
from argparse import Namespace
import anthropic
import MLAgentBench.high_level_actions as high_level_actions
from MLAgentBench.schema import Action, EnhancedJSONEncoder
from MLAgentBench.LLM import complete_text
from MLAgentBench.high_level_actions import understand_file, append_to_research_log, inspect_script_lines, edit_script, edit_script_lines, reflection, retrieval_from_research_log
from MLAgentBench.low_level_actions import list_files, read_file, write_file, append_file, copy_file, undo_edit_script, execute_script, python_repl, request_help
import openai
from openai import OpenAI
import time
from dotenv import load_dotenv
load_dotenv()

system_prompt = '''You are a helpful and first-rate research assistant and you never give up until you have completed the task.'''

initial_prompt = """You are a helpful research assistant. 

Research Problem: {task_description}

You have access to the following pieces of information in your file directory:
{files_prompt}

You have access to the following tools:
{tools_prompt}

Always respond in this format exactly:
{{
    "Thought": "What you are currently doing, what actions to perform and why",
    "Action": "the action to take, should be one of the names of the tools",
    "Action Input": "the input to the action as a valid JSON string",
}}
"""

# Updated base class to accomadate for Assistants API
class Agent:
    """ Base class for agents. """
    def __init__(self, args, env):
        print("args: ", args)        
        self.args = args
        self.work_dir = env.work_dir
        self.log_dir = env.log_dir
        self.TOOL_DESCRIPTIONS = env.tool_descriptions
        self.AVAILABLE_ACTIONS = env.available_actions
        self.history_steps = []
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def run(self):
        pass

    #     self.history_steps = []
    #     self.initialize_logging(env)


    # def initialize_logging(self, env): 
    #     """ Initialize logging folder for the agent. """

    #     if os.path.exists(self.log_dir):
    #         print("Log dir {} already exists. Overwriting.".format(self.log_dir))
    #     else:
    #         os.makedirs(self.log_dir)

    #     with open(os.path.join(self.log_dir, "main.log"), "w", 1) as f:
    #         f.write("Enabled Tools in Prompt:" + str(list(env.action_infos.keys())) + "\n") 
    #         f.write("================================Start=============================\n")

    #     print("Agent is up! See progress in {}".format(os.path.join(self.log_dir, "main.log")))


    # TODO: look at save and restore to check if they work. Restore likely doesn't work because action_infos was gotten rid of
    # def save(self, file_path):
    #     """ Save the agent state to a file. """
    #     with open(file_path, "w") as f:
    #         json.dump(self.__dict__, f, indent=4,cls=EnhancedJSONEncoder)


    # def restore(self, file_path):
    #     """ Restore the agent state from a file."""
    #     with open(file_path, "r") as f:
    #         agent_state = json.load(f)
    #     agent_state["args"] = Namespace(**agent_state["args"])
    #     for key, value in agent_state.items():
    #         if key == "log_dir":
    #             continue
    #         if key == "action_infos":
    #             continue
    #         setattr(self, key, value)


class SimpleAssistantAgent(Agent):
    """ Agent that takes actions based on the LLM output with the simplest prompt using Assistants API."""

    def __init__(self, args, env):
        super().__init__(args, env)  

    def run(self, env):
        print("Starting to run Simple Assistant Agent")

        # TODO: not surew here to put this
        self.client = OpenAI(api_key=openai.api_key)
        client = self.client
        assistant = client.beta.assistants.create(
            name="Research Agent",
            instructions=system_prompt,
            tools=self.TOOL_DESCRIPTIONS,
            model=self.args.llm_name
        )
        self.tools = self.TOOL_DESCRIPTIONS
        self.available_functions = self.AVAILABLE_ACTIONS
        thread = client.beta.threads.create()
        print("got here")

        # # Just to test functions if they work!
        # # try:
        # log_file = os.path.join(self.log_dir , f"step_1_log.log")
        # tool_output = self.available_functions['editScript'](**{
        #     'edit_instruction': "# Load the data_description.txt file and use it to understand the data and all the features\nfile_path = '/mnt/data/data_description.txt'\nwith open(file_path, 'r') as file:\n    data_description = file.read()\n\n# Print the content of the data_description.txt file\ndata_description", 
        #     'save_name': 'read_data_description.py', 
        #     'script_name': 'load_data_description.py', 
        #     'log_file': log_file,
        #     **env.static_kwargs_for_tools
        # })
        # print("TESTING :", tool_output)
        # # except Exception as e:
        # #     print("EXCEPTION: ", e)

        # print("\nCOMPLETE COOL\n")
        # return "Cool"

        while len(self.history_steps) < self.args.agent_max_steps:
            # Potentially break or get feedback if the agent submitted a final answer
            is_final, final_answer_feedback = env.is_final()
            if is_final:
                break
            if final_answer_feedback:
                self.initial_prompt += "\n You originally submitted a final answer, but it was not acceptable. Here is feedback: " + final_answer_feedback + ".\n"

            # self.history_steps = [{"step_idx": len(env.trace.steps), "action": entries, "observation": observation})]
            curr_step = len(self.history_steps)
            log_file = os.path.join(self.log_dir , f"step_{curr_step}_log.log")
            print("--- Step: ", curr_step, " ---")

            #### call LLM for next action ###

            ###############################################################
            #     construct prompt for LLM based on research log          #
            ###############################################################

            self.system_prompt = system_prompt
            self.initial_prompt = initial_prompt.format(
                task_description=env.research_problem, 
                tools_prompt=self.AVAILABLE_ACTIONS.keys(), 
                files_prompt=os.listdir(self.work_dir)
            )       

            print("iniital prompt: ", self.initial_prompt)
            with open(os.path.join(self.log_dir , "main_log.txt"), "a", 1) as f:
                f.write(self.initial_prompt + "\n")
            prompt = self.initial_prompt

            # Add research log to prompt
            prompt += "Here are the most recent steps you have taken:\n"
            with open(os.path.join(self.log_dir, "main_log.txt"), "r") as f:
                log = f.read()
            prompt += "\n" + log[-2500:]
                # for idx in range(self.args.max_steps_in_context):
                #     if idx < len(self.history_steps):
                #         prompt += str(self.history_steps[idx])

            ###############################################
            #     call LLM until the response is valid    #
            ###############################################
            
            # Log the prompt
            with open(os.path.join(self.log_dir, "main_log.txt"), "a", 1) as f:
                f.write("\n\nPROMPT: " + str(prompt) + "\nPrompt length: " + str(len(prompt)) + "\n")

            # TODO: Need to wait until the run is inactive

            # Invoke the Assistants API to answer
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            thread_messages = client.beta.threads.messages.list(thread.id)
            # print("thread_messages.data", thread_messages.data)
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
            )

            # Wait until the run has looped
            run_complete = False
            num_tries = 50
            while not run_complete:
                # Check if there's an update on the run
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                run_complete = run.status == "completed"
                print("\nrun.status: ", run.status)

                # Call the tools if the run status is requires action
                if run.status == "requires_action":
                    tool_outputs = []
                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        tool_id = tool_call.id
                        tool_function = tool_call.function
                        tool_type = tool_call.type

                        print(f"Run required action: \ntool_id: {tool_id}, \ntool_function.arguments: {tool_function.arguments}, \ntool_function.name: {tool_function.name}, \ntool_type: {tool_type}")

                        # Call the function directly if `tool_function` is a callable object
                        # and `arguments` is a dictionary of arguments to pass to the function.
                        try:
                            arguments = json.loads(tool_function.arguments)
                            # arguments["log_file"] = log_file # Add log file because low and high level defined actions require the log file as an arg
                            print("Arguments in JSON", arguments)
                            function_output = self.available_functions[tool_function.name](**arguments)
                            print("function_output", function_output)
                        except Exception as e:
                            # Perhaps not all arguments were provided
                            desired_function_block = None
                            for tool in self.tools:
                                if "function" in tool:
                                    if tool["function"]["name"] == tool_function.name:
                                        desired_function_block = tool
                                        break

                            if desired_function_block:
                                function_output = f"Tool function {tool_function.name} for tool_id {tool_id} is not callable with arguments {tool_function.arguments}. Make sure you use all properties required: {desired_function_block['function']['parameters']}"
                            else:
                                function_output = f"Tool function {tool_function.name} for tool_id {tool_id} does not exist and is not callable with arguments {tool_function.arguments}. Make sure you are using only tools listed here: {self.available_functions.keys()}"

                        tool_outputs.append({
                            "tool_call_id": tool_id,
                            "output": function_output
                        })
                    print("tool_outputs", tool_outputs[:500]) # Just get an idea of the first few lines, can read full log if necessary

                    # Submit tool outputs as a new run
                    run = client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                elif run.status == "failed":
                    print("Run failed: ", run)
                    break

                time.sleep(1)
                num_tries -= 1
                if num_tries == 0:
                    print("Run timed out, cancelling...")
                    run = client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
                    print("Run.status: ", run.status)
                    while run.status != "cancelled":
                        run = client.beta.threads.runs.retrieve(
                            thread_id=thread.id,
                            run_id=run.id
                        )
                    print("Run cancelled!")
                    break
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            completion = messages.data[0].content[0].text.value
            print("Completion: ", completion) # TODO: will this output JSON in the right format?

            # Log the completion
            with open(os.path.join(self.log_dir , "main_log.txt"), "a", 1) as f:
                f.write("\n\nCOMPLETION: " + str(completion) + "\nCompletion length: " + str(len(str(completion))) + "\n")
                print("\n\nCOMPLETION: " + str(completion) + "\nCompletion length: " + str(len(str(completion))) + "\n")

            #######################################################
            #               update base on observation            #
            #######################################################

            self.history_steps.append({"step_idx": curr_step, "action": completion, "observation": completion})

            with open(os.path.join(self.log_dir , "main_log.txt"), "a", 1) as f:
                f.write("\nObservation:```\n" + self.history_steps[-1]["observation"] + "\n```\n\n")

            # step_idx = len(env.trace.steps) - 1
            # self.save(os.path.join(self.log_dir , f"agent_{step_idx}_{curr_step}.json")) # TODO: why would step index be different from curr_step? Currently not saving because of class JSON serialization error becaues not everything is JSON serializable

        return "Finished successfully"