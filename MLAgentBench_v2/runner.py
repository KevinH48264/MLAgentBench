""" 
This file is the entry point for MLAgentBench.

Requirements:
1. Accept user commands to 1) solve a house-price task for kaggle, 2) the research problem, and 3) add configs if necessary.
2. Runs the prepare_task.py script to prepare the 1) workspace in work_dir
"""

import os
import argparse
import sys
from MLAgentBench_v2 import LLM
from MLAgentBench_v2.environment import Environment

# Ensure that you add your agent class that you want to use
from MLAgentBench_v2.agents.agent import Agent, SimpleAssistantAgent
from MLAgentBench_v2.agents.agent_research import ResearchAgent
from MLAgentBench_v2.agents.agent_basic_assistants import BasicAssistantAgent
from MLAgentBench_v2.agents.agent_langchain  import LangChainAgent

def run(agent_cls, args):
    print("Running the run function!", agent_cls, args)
    with Environment(args) as env:
        print("=====================================")
        print("Benchmark folder name: ", env.benchmark_folder_name)
        print("Research problem: ", env.research_problem)
        print("=====================================")  

        agent = agent_cls(args, env) # initial a ResearchAgent with the args and env
        final_message = agent.run(env)

        print("=====================================")
        print("Final message: ", final_message)
    print("Final answer was submitted by the agent system. You can view results and process in logs/<task-name>/main_log.txt")

if __name__ == "__main__":
    # configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="debug", help="task name")
    parser.add_argument("--task-type", type=str, default="kaggle", help="task type (custom, kaggle)")
    parser.add_argument("--log-dir", type=str, default="./logs", help="log dir")
    parser.add_argument("--work-dir", type=str, default="./workspace", help="work dir")
    parser.add_argument("--max-steps", type=int, default=50, help="number of steps")
    parser.add_argument("--max-time", type=int, default=5* 60 * 60, help="max time")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--python", type=str, default="python", help="python command")
    parser.add_argument("--interactive", action="store_true", help="interactive mode")
    parser.add_argument("--resume", type=str, default=None, help="resume from a previous run")
    parser.add_argument("--resume-step", type=int, default=0, help="the step to resume from")

    # general agent configs
    parser.add_argument("--agent-type", type=str, default="ResearchAgent", help="agent type")
    parser.add_argument("--llm-name", type=str, default="gpt-35-turbo", help="llm name")
    parser.add_argument("--fast-llm-name", type=str, default="gpt-35-turbo", help="llm name")
    parser.add_argument("--edit-script-llm-name", type=str, default="gpt-35-turbo", help="llm name")
    parser.add_argument("--edit-script-llm-max-tokens", type=int, default=4000, help="llm max tokens")
    parser.add_argument("--agent-max-steps", type=int, default=50, help="max iterations for agent")

    # research agent configs
    parser.add_argument("--actions-remove-from-prompt", type=str, nargs='+', default=[], help="actions to remove in addition to the default ones: Read File, Write File, Append File, Retrieval from Research Log, Append Summary to Research Log, Python REPL, Edit Script Segment (AI)")
    parser.add_argument("--actions-add-to-prompt", type=str, nargs='+', default=[], help="actions to add")
    parser.add_argument("--no-retrieval", action="store_true", help="disable retrieval")
    parser.add_argument("--valid-format-entires", type=str, nargs='+', default=None, help="valid format entries")
    parser.add_argument("--max-steps-in-context", type=int, default=1, help="max steps in context") # how many steps should you retrieve in context
    parser.add_argument("--max-observation-steps-in-context", type=int, default=3, help="max observation steps in context")
    parser.add_argument("--max-retries", type=int, default=4, help="max retries")

    # langchain configs
    parser.add_argument("--langchain-agent", type=str, default="zero-shot-react-description", help="langchain agent")

    args = parser.parse_args()
    args.log_dir = os.path.join('logs/' + args.log_dir)
    print("\nArgs: ", args, file=sys.stderr)
    run(getattr(sys.modules[__name__], args.agent_type), args) # run <AgentName (ex. ResearchAgent)>(args)