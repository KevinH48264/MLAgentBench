""" This class defines a simple agent that uses the old function calling SDK from OpenAI to avoid the automatic action process of the current Assistants API
"""

from MLAgentBench_v2.agents.agent import Agent
import time
import json
from MLAgentBench_v2.LLM import complete_text_openai

class SingleOldFunctionCallingAgent(Agent):
    """ Agent that takes actions based on the LLM output with the simplest prompt using Assistants API."""
    def __init__(self, env):
        super().__init__(env)  

    def run(self):
        print("Starting to run Single Old Function Calling Agent")
        self.system_prompt = '''You are a helpful and first-rate research assistant.'''

        while True:
            self.initial_prompt = f"""You are a helpful research assistant. Given a research problem, files, tools, and at most 5 of your most recent action, result, and answer, your goal is to choose and take the next best action and tool that you think could lead to a better answer and get you closer to solving the research problem. 

            Research Problem: {self.research_problem}
            Current Files: {self.files}
            Tools / functions: {self.available_actions.keys()}
            Most recent files, action, result, and answer states (oldest to newest):
            {self.answer_states}        
            """

            # Invoke the Assistants API to answer
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\nCalling Function Calling API with initial prompt: ")
                log_file.write(json.dumps(self.initial_prompt, indent=4))
                log_file.write("\n")

            # Ask for a direct next step for helping function calling API
            next_step = complete_text_openai(self.initial_prompt + "\nWhat is the next best action I should take. Be sure to look at the most recent action, result, and answer states because if I failed in completing a step, you should give me an easier next step. Only respond with the action I should take.", system_prompt=self.system_prompt, model=self.model)
            print("\nThis is the next step reported: ", next_step)

            # Call the function calling API
            complete_text_openai(next_step, system_prompt=self.system_prompt, model=self.model, tools=self.tool_descriptions, available_functions=self.available_actions)

            # Add that information about the next step into the answer_state action column

            self.answer_states[-1]['action'] += "\nAssigned action: " + next_step

            completion = str(self.answer_states[-1])

            # Check if completion is successful
            finished = input(f'This is the final message: {completion}. Are they finished? (y/n): ')
            if finished == 'y':
                break

        return "Finished successfully! Final message: " + completion