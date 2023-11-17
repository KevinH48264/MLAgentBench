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

import time
import json

# Updated base class that agents can take and fill in and iterate on
class Agent:
    """ Base class for agents to leverage useful environment variables """
    def __init__(self, env):
        self.research_problem = env.research_problem
        self.work_dir = env.work_dir
        self.files = env.files
        self.tool_descriptions = env.tool_descriptions
        self.available_actions = env.available_actions
        self.client = env.client
        self.model = env.model
        self.log_dir = env.log_dir
        self.main_log_path = env.main_log_path
        self.num_steps = env.num_steps

    def run(self):
        pass

# Goal: ultimately be able to write an agent that 
# 1. Instantiates an OpenAI Assistant using the environment's client key
# 2. Makes calls to available actions
# 2.5 Extra: the Assistants API calls are in the available actions too

# Questions: 
# 1. I don't know how Eureka handled when the reward code was generated with different arguments and it still executed it properly and automatically so only the useful environment variables are used.
class SimpleAssistantAgent(Agent):
    """ Agent that takes actions based on the LLM output with the simplest prompt using Assistants API."""
    def __init__(self, env):
        super().__init__(env)  

    def run(self):
        print("Starting to run Simple Assistant Agent")
        self.system_prompt = '''You are a helpful and first-rate research assistant and you never give up until you have completed the task. You always respond in proper JSON format so that json.loads will accept your response.'''
        self.initial_prompt = f"""You are a helpful research assistant. You always respond in proper JSON format so that json.loads will accept your response.

        Research Problem: {self.research_problem}

        You have access to the following pieces of information in your file directory:
        {self.files}

        You have access to the following tools:
        {self.available_actions}

        Always respond in this format exactly:
        {{
            "Thought": "What you are currently doing, what actions to perform and why",
            "Action": "the action to take, should be one of the names of the tools",
            "Action Input": "the input to the action as a valid JSON string",
        }}
        """
        # Instantiate an Assistant
        self.assistant = self.client.beta.assistants.create(
            name="Research Agent",
            instructions=self.system_prompt,
            tools=self.tool_descriptions,
            model=self.model
        )
        self.thread = self.client.beta.threads.create()

        while True:
            # Add research log to prompt
            self.initial_prompt += "Here are the most recent steps you have taken:\n"
            with open(self.main_log_path, "r") as f:
                log = f.read()
            self.initial_prompt += "\n" + log[-2500:]
            print("Recent steps in log: ", log[-2500:])

            # Invoke the Assistants API to answer
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
            finished = input(f'This is the final message: {completion}. Are they finished? (y/n): ')
            if finished == 'y':
                break

        return "Finished successfully! Final message: " + completion