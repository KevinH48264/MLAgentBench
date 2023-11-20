'''
This agent is supposed to be an adaptation of the Minecraft Voyager agent, but in the knowledge discovery and research goal.

Agent system components:
1. Curriculum Agent to propose a next task
2. Skill manager to retrieve from memory most relevant information
3. Action agent to figure out how to execute the task
4. Execution agent to execute the task
5. Critic agent to check if the task was successful

Agent components:
1. Memory -- RAG retrieval

Add-ons:
1. Self-explain (https://arxiv.org/pdf/2311.05997.pdf)
2. Self-check (https://arxiv.org/pdf/2311.05997.pdf)
'''


from MLAgentBench_v2.agents.agent import Agent
import time
import json
from MLAgentBench_v2.LLM import complete_text_openai


# Function calling allows for greater control than Assistants API
class VoyagerAgent(Agent):
    """ Agent that uses function calling based on a prompt."""
    def __init__(self, env):
        super().__init__(env)  

    def run(self):
        print("Starting to run Voyager Agent")
        self.system_prompt = '''You are a helpful and first-rate research assistant.'''

        while True:
            # OLD CODE BELOW
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
            Tools / functions: {self.available_actions.keys()}
            Most recent files, action, result, and answer states (oldest to newest):
            {formatted_answer_states}        
            """

            # Log initial prompt
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\nCalling Function Calling API with initial prompt: ")
                log_file.write(self.initial_prompt)
                log_file.write("\n")

            # FUNCTION CALLING AGENT: Call the function calling API by giving tools and available functions
            completion = complete_text_openai(self.initial_prompt, system_prompt=self.system_prompt, model=self.model, tools=self.tool_descriptions, available_functions=self.available_actions)

            # Log completion
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\Function Calling API output: ")
                log_file.write(completion)
                log_file.write("\n")

        return "Finished successfully! Final message: " + completion