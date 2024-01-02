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

TODO Add-ons:
0. Search memory during planning!! That's probably why planning sucks -- the agent doesn't know what it has already done. Think about Simulacra memory system
1. Self-explain (https://arxiv.org/pdf/2311.05997.pdf)
2. Self-check (https://arxiv.org/pdf/2311.05997.pdf)
'''


from MLAgentBench_v2.agents.agent import Agent
from MLAgentBench_v2.agents.Voyager.curriculum_agent import CurriculumAgent
from MLAgentBench_v2.agents.Voyager.methods_agent import MethodsAgent
from MLAgentBench_v2.agents.Voyager.execution_agent import ExecutionAgent
from MLAgentBench_v2.agents.Voyager.critic_agent import CriticAgent
import time
import json


# Function calling allows for greater control than Assistants API
class VoyagerAgent(Agent):
    """ Agent that uses function calling based on a prompt."""
    def __init__(self, env):
        super().__init__(env)  
        self.curriculum_agent = CurriculumAgent(env)
        self.methods_agent = MethodsAgent(env)
        self.execution_agent = ExecutionAgent(env)
        self.critic_agent = CriticAgent(env)

    def run(self):
        self.log("Starting to run Voyager Agent")

        # Running a cycle of Voyager
        num_rounds = 4 # Just to test if it can give easier tasks too
        max_tasks = 3

        for task_idx in range(self.num_tasks, max_tasks):
            self.log(f"\nTask {task_idx} of {max_tasks}")

            exploration_progress = self.curriculum_agent.get_exploration_progress()
            self.log("Exploration progress: ", exploration_progress)

            next_task = self.curriculum_agent.propose_next_task()
            self.log("next_task", next_task)

            methods_prompt = next_task # First round, methods prompt is the task
            execution_feedback = None
            execution_errors = None
            critique = None
            success = False
            for i in range (num_rounds):
                self.log(f"\nRound {i} Task: ", next_task)

                if i != 0: # Don't need to generate methods prompt for the first round
                    self.log("\nStarting methods agent")
                    methods_prompt = self.methods_agent.generate_function_callable_prompt(task=next_task, methods_prompt=methods_prompt, execution_feedback=execution_feedback, execution_errors=execution_errors, critique=critique)
                    self.log("\nMethods agent output:\n", methods_prompt)

                self.log("\nStarting execution agent")
                execution_feedback = self.execution_agent.function_call(methods_prompt=methods_prompt, task=next_task)
                self.log("\nExecution agent output: ", execution_feedback)

                self.log("\nStarting critic agent")
                success, critique = self.critic_agent.check_task_success(task=next_task, methods_prompt=methods_prompt, execution_feedback=execution_feedback)
                self.log("Critic agent output", "\nOriginal task: ", next_task, "\nSuccess: ", success, "\nEvaluation: ", critique)

                # TODO: Make grading fast and easier to look at
                # if i == len(num_rounds) - 1:
                # create 1) new log that only logs the final critic agent user prompt (so you know the history) and the critic output / evaluation and 2) a collection of all critic outputs & tasks to figure out why it went wrong, and perhaps 3) an automatic initial evaluation of what you think the agent system struggled with based on the critic user prompt and evaluation.

                if success:
                    break
            if success:
                self.curriculum_agent.add_completed_task(next_task, methods_prompt, critique)
            else:
                self.curriculum_agent.add_failed_task(next_task, methods_prompt, critique)

        return "Finished successfully!"
    
