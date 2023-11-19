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
from MLAgentBench_v2.LLM import complete_text_openai

# Updated base class that agents can take and fill in and iterate on
class Agent:
    """ Base class for agents to leverage useful environment variables """
    def __init__(self, env):
        # Goal / Reward
        self.research_problem = env.research_problem

        # States
        self.answer_states = env.answer_states

        # Actions
        self.tool_descriptions = env.tool_descriptions
        self.available_actions = env.available_actions
        self.client = env.client
        self.model = env.model

        self.work_dir = env.work_dir
        self.files = env.files

        # Logging
        # self.log_dir = env.log_dir
        self.main_log_path = env.main_log_path
        # self.num_steps = env.num_steps

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
                # formatted_answer_states += "\nResult: " + answer_state['result'] 
                formatted_answer_states += "\nAnswer: " + answer_state['answer_state'] 
            self.initial_prompt = f"""You are a helpful research assistant. Given a research problem, files, tools, and at most 5 of your most recent files, action, result, and answer, your goal is to choose and take the next best action and tool that you think could lead to a better answer and get you closer to solving the research problem. 

            Research Problem: {self.research_problem}
            Current Files: {self.files}
            Tools / functions: {self.available_actions.keys()}
            Most recent files, action, result, and answer states (oldest to newest):
            {formatted_answer_states}        
            """

            # 1) NEXT STEP AGENT: Ask for a direct next step for helping function calling API
            next_step = complete_text_openai(self.initial_prompt + "\nWhat is the next best action I should take. Be sure to look at the most recent action, result, and answer states because if I failed in completing a step, you should give me an easier next step. Only respond with the action I should take.", system_prompt=self.system_prompt, model=self.model)
            print("\nThis is the next step reported: ", next_step)

            # 2) FUNCTION CALLING AGENT: Call the function calling API by giving tools and available functions
            complete_text_openai(next_step, system_prompt=self.system_prompt, model=self.model, tools=self.tool_descriptions, available_functions=self.available_actions)

            # Optional: Add that information about the next step into the answer_state action column
            self.answer_states[-1]['action'] = "Assigned action: " + next_step + "\nTaken action: " + self.answer_states[-1]['action']
            completion = str(self.answer_states[-1])

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

        while True:   
            # Assistants API
            # Update answer states each round
            assert('action' in self.answer_states[0].keys() and 'result' in self.answer_states[0].keys() and 'answer_state' in self.answer_states[0].keys() and 'files' in self.answer_states[0].keys())
            self.initial_prompt = f"""You are a helpful research assistant. Given a research problem, files, tools, and at most 5 of your most recent action, result, and answer, your goal is to choose and take the next best action and tool that you think could lead to a better answer and get you closer to solving the research problem. 

            Research Problem: {self.research_problem}
            Current Files: {self.files}
            Tools / functions: {self.available_actions.keys()}
            Most recent files, action, result, and answer states (oldest to newest):
            {self.answer_states}        
            """

            # Invoke the Assistants API to answer
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\nCalling Assistants API with initial prompt: ")
                log_file.write(json.dumps(self.initial_prompt, indent=4))
                log_file.write("\n")
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=self.initial_prompt
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
                print("\nrun.status: ", run.status)

                # Call the tools if the run status is requires action
                if run.status == "requires_action":
                    tool_outputs = []
                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        tool_id, tool_function, tool_type = tool_call.id, tool_call.function, tool_call.type
                        print(f"Run required action: \ntool_id: {tool_id}, \ntool_function.arguments: {tool_function.arguments}, \ntool_function.name: {tool_function.name}, \ntool_type: {tool_type}")

                        # Call the function directly if `tool_function` is a callable object
                        # and `arguments` is a dictionary of arguments to pass to the function.
                        try:
                            tool_function.arguments = tool_function.arguments.replace('\n', '\\n') # To support multi-lines in JSON.loads
                            arguments = json.loads(tool_function.arguments)
                            print("Arguments was JSON parsed successfully")
                            function_output = self.available_actions[tool_function.name](**arguments)
                        except Exception as e:
                            function_output = f"Tool function {tool_function.name} for tool_id {tool_id} does not exist and is not callable with arguments {tool_function.arguments}. Make sure you are using only tools listed here: {self.available_actions.keys()} with the right arguments."

                        print("Function output: ", function_output)
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
                    print("Run failed: ", run)
                    # Retry run if failed
                    run = self.client.beta.threads.runs.create(
                        thread_id=self.thread.id,
                        assistant_id=self.assistant.id,
                    )

                time.sleep(1)
                num_tries -= 1
                if num_tries == 0:
                    print("Run timed out, cancelling...")
                    run = self.client.beta.threads.runs.cancel(thread_id=self.thread.id, run_id=run.id)
                    while run.status != "cancelled":
                        run = self.client.beta.threads.runs.retrieve(
                            thread_id=self.thread.id,
                            run_id=run.id
                        )
                    print("Run cancelled!")
                    break
            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
            completion = messages.data[0].content[0].text.value

            # Check if completion is successful
            # continue_res = input(f'This is the final message: {completion}. Do you want them to continue (y/n): ')
            # if continue_res == 'n':
            #     break

        return "Finished successfully! Final message: " + completion