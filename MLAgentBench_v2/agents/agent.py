""" This file defines the basic agent class that can be used to implement different agents. 

This should be a lightweight agent class that demos how to solve the research problem via a simple init() and run() function. 

The goal is that the class agent interface should be extensible to other agent frameworks like Eureka, Voyager, OPRO, and AutoGen.

Notes:
The Abstract Base Class should just define the structure and exepcted behavior of its subclasses through abstract methods and also provide shared functionality the child classes can use. It should not have everything that's necessary for the agent interface.

Requirements:
1. init()
- Reward / Goal
- States
- Actions
- Workspace
2. run()
- Iterative cycle of calling LLM with Actions and Workspace, and logging it all

The actual Child Class must implement all the abstract methods (init(), run())
"""

import time
import json
# from MLAgentBench_v2.LLM import complete_text_openai

# Updated base class that agents can take and fill in and iterate on
class Agent:
    """ Base class for agents to leverage useful environment variables """
    def __init__(self, env):
        # Goal / Reward
        self.research_problem = env.research_problem

        # States
        self.answer_states = env.answer_states
        self.max_states = env.max_states
        self.update_answer_state = env.update_answer_state

        self.files_action_result_history = env.files_action_result_history
        self.max_history = env.max_history
        self.completed_tasks = [] # for curriculum agent
        self.failed_tasks = [] # for curriculum agent

        # Actions
        self.tool_descriptions = env.tool_descriptions
        self.available_actions = env.available_actions
        self.client = env.client
        self.model = env.model
        self.complete_text_openai = env.complete_text_openai
        self.run_assistant = env.run_assistant
        self.search_wikipedia = env.search_wikipedia

        self.work_dir = env.work_dir
        self.files = env.files
        self.files_no_skill_lib = env.files_no_skill_lib # temporary to not give the curriculum agent the skill library

        # Logging
        # self.log_dir = env.log_dir
        self.main_log_path = env.main_log_path
        self.log = env.log
        self.num_tasks = env.num_tasks
        # self.num_steps = env.num_steps

        # Misc
        # Formatting answer states for rapid experimentation and clearer prompts
        self.formatted_answer_states = env.formatted_answer_states
        self.formatted_action_history = env.formatted_action_history

        # Read tool description for only read action. Maybe put it in env?
        self.read_tool_description = [{
            "type": "function",
            "function": {
                "name": "readFile",
                "description": "Use this to read an existing file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "A valid file name with relative path to current directory if needed"
                        }
                    },
                    "required": ["file_name"]
                }
            }
        }]

    def run(self):
        pass
    

# Function calling allows for greater control than Assistants API
class SimpleFunctionCallingAgent(Agent):
    """ Agent that uses function calling based on a prompt."""
    def __init__(self, env):
        super().__init__(env)  

    def run(self):
        print("Starting to run Simple Function Calling Agent")
        self.system_prompt = '''You are a helpful and first-rate research assistant.'''

        MAX_STEPS = 10
        count = 0
        while True:
            # Create the prompt for function calling
            assert('action' in self.answer_states[0].keys() and 'result' in self.answer_states[0].keys() and 'answer_state' in self.answer_states[0].keys() and 'files' in self.answer_states[0].keys())
            formatted_answer_states = ""
            for idx, answer_state in enumerate(self.answer_states):
                formatted_answer_states += "\nStep: " + str(idx) 
                formatted_answer_states += "\nFiles: " + str(answer_state['files']) 
                formatted_answer_states += "\nAction: " + answer_state['action'] 
                formatted_answer_states += "\nResult: " + answer_state['result'] 
                formatted_answer_states += "\nAnswer: " + answer_state['answer_state'] 
            self.initial_prompt = f"""You are a helpful research assistant. Given a research problem, files, tools, and at most {self.max_states} of your most recent files, action, result, and answer, your goal is to choose and take the next best action and tool that you think could lead to a better answer and get you closer to solving the research problem. 

            Research Problem: {self.research_problem}
            Current Files: {self.files}
            Tools / functions: {list(self.available_actions.keys())}
            Most recent files, action, result, and answer states (oldest to newest):
            {formatted_answer_states}        
            """

            # Log initial prompt
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\nCalling Function Calling API with initial prompt: ")
                log_file.write(self.initial_prompt)
                log_file.write("\n")

            # FUNCTION CALLING AGENT: Call the function calling API by giving tools and available functions
            completion = self.complete_text_openai(system_prompt=self.system_prompt, user_prompt=self.initial_prompt, model=self.model, tools=self.tool_descriptions, available_functions=self.available_actions)

            # Log completion
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\Function Calling API output: ")
                log_file.write(completion)
                log_file.write("\n")

            count += 1
            if count > MAX_STEPS:
                break
        return "Finished successfully! Final message: " + completion

class SimpleAssistantAgent(Agent):
    """ Agent that takes actions based on the LLM output with the simplest prompt using Assistants API."""
    def __init__(self, env):
        super().__init__(env)  

    def run(self):
        print("Starting to run Simple Assistant Agent")
        self.system_prompt = '''You are a helpful and first-rate research assistant.'''

        # Instantiate an Assistant
        self.assistant = self.client.beta.assistants.create(
            name="Research Agent",
            instructions=self.system_prompt,
            tools=self.tool_descriptions,
            model=self.model
        )
        self.thread = self.client.beta.threads.create()

        MAX_STEPS = 10
        count = 0
        while True:   
            # Assistants API
            # Update answer states each round
            assert('action' in self.answer_states[0].keys() and 'result' in self.answer_states[0].keys() and 'answer_state' in self.answer_states[0].keys() and 'files' in self.answer_states[0].keys())
            formatted_answer_states = ""
            for idx, answer_state in enumerate(self.answer_states):
                formatted_answer_states += "\nStep: " + str(idx) 
                formatted_answer_states += "\nFiles: " + str(answer_state['files']) 
                formatted_answer_states += "\nAction: " + answer_state['action'] 
                formatted_answer_states += "\nResult: " + answer_state['result'] 
                formatted_answer_states += "\nAnswer: " + answer_state['answer_state'] 
            self.initial_prompt = f"""You are a helpful research assistant. Given a research problem, files, tools, and at most {self.max_states} of your most recent action, result, and answer, your goal is to choose and take the next best action and tool that you think could lead to a better answer and get you closer to solving the research problem. 

            Research Problem: {self.research_problem}
            Current Files: {self.files}
            Tools / functions: {list(self.available_actions.keys())}
            Most recent files, action, result, and answer states (oldest to newest):
            {formatted_answer_states}        
            """

            completion = self.run_assistant(system_prompt=self.system_prompt, user_prompt=self.initial_prompt)

            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\nCalling Assistants API Completion: ")
                log_file.write(completion)
                log_file.write("\n")
            
            count += 1
            if count > MAX_STEPS:
                break
            # Check if completion is successful
            # continue_res = input(f'This is the final message: {completion}. Do you want them to continue (y/n): ')
            # if continue_res == 'n':
            #     break

        return "Finished successfully! Final message: " + completion