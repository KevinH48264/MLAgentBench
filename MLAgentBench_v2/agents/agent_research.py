""" This file contains the agent class for our AI research agent. This should work with the old Agent class and be able to evaluate MAE for a submission.csv file that is produced from test.csv file."""
import os
import sys
import anthropic
from MLAgentBench.LLM import complete_text_fast, complete_text
from MLAgentBench.schema import Action
from .agent import Agent
import json

# problem: it's just learnings, no WW/DD to max-min!
# problem: this information not be stored in the research assistant but rather stored in a variable.
initial_prompt = """You are a helpful research assistant designed to output JSON. You have access to the following tools:
{tools_prompt}

Research Problem: {task_description}

You do not know anything about this problem so far. 

Follow these instructions and do not forget them:
- First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
- Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. 
- Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation. 
- Performance numbers and estimates can only be confirmed and included in the status by running the code and observing the output.
- You should come up with a good experiment design that addresses the problem, and whenever applicable, define and measure the baseline performance of the relevant system or model before attempting any improvements.
- Follow the plan and try to achieve the goal as straightforwardly as possible.
- Highlight the supporting experiment results and reasoning before drawing any conclusions. 
- Do not try installing any new packages or libraries.
- If you believe you have solved the problem, you can use the Final Answer action to submit your answer. You can only submit once, so double check that you have achieved the goal before submitting.

Always respond in JSON format with all keys and values.
```json
{format_prompt}
```

Ensure the response can be parsed by Python "json.loads", e.g.: no trailing commas, no single quotes, etc. Always include the Action and Action Input. This is important.
"""


format_prompt_dict = '''{
    "Reflection": "What does the observation mean? If there is an error, what caused the error and how to debug?",
    "Research Plan and Status": "The full high level research plan, with current status and confirmed results of each step briefly annotated. It must only include progress that has been made by previous steps. If there is any update, enclose the new update text in double asterisks **like this**. If there is no update, just copy the previous step Research Plan and Status. The high level plan from the previous step should be fully retained, unless it is intentionally revised.",
    "Fact Check": "List all objective statements in the updates to Research Plan and Status one by one and point out whether it is guessed versus directly confirmed by the previous observation directly above. Performance numbers can only be confirmed by running the code and observing the output.",
    "Thought": "What you are currently doing, what actions to perform and why",
    "Action": "the action to take, should be one of the names of the tools",
    "Action Input": "the input to the action as a valid JSON string"
}'''


class ResearchAgent(Agent):
    """This class implements AI research agent with different configurations."""

    def __init__(self, args, env):
        # pretty much just initializes the prompt in ReAcT format
        print(f"STARTING RESEARCH AGENT: \n{args} \nENV: \n {env}")
        super().__init__(args, env)
        self.valid_format_entires = ["Reflection",  "Research Plan and Status","Fact Check", "Thought","Action", "Action Input"] # use all entries by default
        if args.valid_format_entires:
            self.valid_format_entires = args.valid_format_entires
        self.initial_prompt = initial_prompt.format(tools_prompt=self.tools_prompt, tool_names=self.prompt_tool_names,  task_description=env.research_problem, format_prompt=format_prompt_dict)

    # main code for running
    def run(self, env):
        last_steps = self.args.max_steps_in_context # this can be improved with a rating system and cosine similarity for recency + relevance + importance instead of just how many steps earlier

        last_observation_step = self.args.max_observation_steps_in_context # I mean yeah this is a good limit

        # agent_max_steps is important to be wary of cost!
        while not env.is_final() and len(self.history_steps) < self.args.agent_max_steps:
            # 1. Mark what step it is
            curr_step = len(self.history_steps)
            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write("Step " + str(curr_step) + ":\n")
                print("Step " + str(curr_step) + ":\n")

            #### call LLM for next action ###

            ###########################################################
            #     construct prompt for LLM based on previous steps    #
            ###########################################################

            # 2. Retrieve previous steps
            print("\nRetrieval step!\n")
            prompt = self.initial_prompt
            if curr_step >= last_steps:
                if not self.args.no_retrieval:

                    # retrieval action
                    relevant_history = env.execute(Action("Retrieval from Research Log", {"current_plan": ""}))

                    prompt += f"""
        Here is a summary of relevant actions and observations you have done:
        ```
        {relevant_history}
        ```
        Here are the exact step you have done most recently (up to 1 step):
        """
            else:
                prompt += "\nNow let's start!\n\n"

            for idx in range(max(curr_step - last_steps, 0), curr_step):
                action_string = ""
                action_string = self.print_action(self.history_steps[idx]["action"], self.valid_format_entires)

                prompt += anthropic.AI_PROMPT + "\n"+ action_string + "\nObservation:"
                if curr_step - idx > last_observation_step:
                    prompt += "<Done>\n\n"
                else:
                    try:
                        prompt += "\n```\n" + self.history_steps[idx]["observation"] + "\n```\n\n"
                    except:
                        import pdb; pdb.set_trace()
                

            ###############################################
            #     call LLM until the response is valid    #
            ###############################################

            entries = None
            valid_response = False
            for _ in range(self.args.max_retries):
                log_file = os.path.join(self.log_dir , f"step_{curr_step}_log.log")
                # PROBLEM: observation is sometimes not outputted
                with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                    f.write("\n\nPROMPT: " + str(prompt) + "\nPrompt length: " + str(len(prompt)) + "\n")
                    print("\n\nPROMPT: ", prompt, "\n", str(len(prompt)))


                completion = json.loads(complete_text(prompt, log_file, self.args.llm_name, json=True))

                with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                    f.write("\n\nCOMPLETION: " + str(completion) + "\nCompletion length: " + str(len(str(completion))) + "\n")
                    print("\n\nCOMPLETION NON-JSON: ", completion, type(completion))

                # # Ensure that the completion is JSON parsable. If it isn't, ask GPT to make it JSON parsable by doubling the max tokens.
                # try:
                #     completion = json.loads(completion)
                # except:
                #     print("COMPLETION NOT IN JSON")
                #     convert_to_json_prompt = f'''Close this incomplete JSON so that it's in proper JSON format: {completion}'''
                #     completion = complete_text(convert_to_json_prompt, log_file, self.args.llm_name, json=True, max_tokens_to_sample=1000)
                #     print("NEW COMPLETION: ", completion)
                #     completion = json.loads(completion)

                #     with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                #         f.write("\n\nNEW COMPLETION NONJSON: " + str(completion) + "\nNew completion length: " + str(len(str(completion))) + "\n")
                #         print("NEW COMPLETION JSON: ", completion, type(completion), str(len(str(completion))))


                try:
                    # this is some weird parsing going on, likely should shift to JSON controls
                    # entries = self.parse_entries(completion, self.valid_format_entires)
                    entries = completion # now that it's a JSON, we don't need to parse to turn it into a JSON
                    assert entries["Action"].strip() in self.all_tool_names
                    valid_response = True
                except:
                    print("ERROR")
                    print("Step", curr_step, file=sys.stderr)
                    print("Entries: ", entries, file=sys.stderr)
                    print("self.all_tool_names", self.all_tool_names)
                    print(anthropic.AI_PROMPT + "\n" + str(completion) + "\nObservation:\n", file=sys.stderr)
                    print("Response is invalid and discarded", file=sys.stderr)
                    prompt += "\n\n Your response was in incorrect format. Please provide a valid response with all entries: " + ", ".join(self.valid_format_entires) + "\n\n"
                else:
                    break
            if not valid_response:
                return "No valid response after max_retries"

            ########################################################
            #     postprocess LLM output and parse to env actions  #
            ########################################################

            # Extract from entries all the keys. 
            rg = str(entries["Research Plan and Status"])
            action = entries["Action"].strip()

            # Parse as JSON if necessary
            action_input = entries["Action Input"]
            if type(entries["Action Input"]) == str:
                try:
                    action_input = json.loads(entries["Action Input"])
                except:
                    action_input = entries["Action Input"]

            new_research_plan_content = rg.strip("```") + "\n\n" 
            entries["Research Plan and Status"] = new_research_plan_content
            entries["Research Plan and Status"]= new_research_plan_content.replace("**", "")

            
            # parse the action input if we can ; other wise just return the original input and wait env to throw back an error
            # parsing_error = ""
            # try:
            #     action_input = self.parse_action_input(raw_action_input, self.action_infos[action])
            # except Exception as e:
            #     print("\nERROR: \nraw_action_input: ", raw_action_input)
            #     action_input = raw_action_input
            #     parsing_error = str(e)

            ########################################
            #         execute action in env        #
            ########################################

            print("\nExecuting action: ", action, " | with input: ", action_input)
            if type(action_input) == dict:
                observation = env.execute(Action(action, action_input))
            else:
                # parsing failed, give agent parsing error
                usage = ",\n            ".join([f"{k}: [{v}]" for k, v in self.action_infos[action].usage.items()])
                usage = f"""{{
            {usage}
}}"""
                invalid_action_error = f"The action input for {action} needs to be a valid json with proper entries. You may have missed the comma between entries or used triple quotes (json does not recognizes triple quotes). Please use the correct format and try again:\n{usage}"

                observation = "ActionInputParsingError: \n" + invalid_action_error


            #######################################################
            #               update history_steps                  #
            #######################################################

            # if observation is too long, we need to summarize it
            if len(observation) > 5000:
                log_file = os.path.join(self.log_dir , f"step_{curr_step}_summarize_observation_log.log")

                print("Observation is too long. Summarizing...", file=sys.stderr)
                observation = self.summarize_observation(self.print_action(entries, self.valid_format_entires), observation, log_file)

            self.history_steps.append({"step_idx": len(env.trace.steps), "action": entries, "observation": observation})

            ## filter out ActionInputParsingError if last step is not action input parsing error
            if not observation.startswith("ActionInputParsingError"):
                self.history_steps = [step for step in self.history_steps if not step["observation"].startswith("ActionInputParsingError")]

            # This is where the Observation is written to the main log
            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                print("\nObservation after executing action: ```\n" + self.history_steps[-1]["observation"] + "\n```\n\n")
                f.write("\nObservation after executing action: ```\n" + self.history_steps[-1]["observation"] + "\n```\n\n")


            #######################################################
            #      write to research log for retrieval            #
            #######################################################
            # You always do this action of adding to memory, not the agent
            print("\n\nAdding to research log... self.args.no_retrieval: ", str(self.args.no_retrieval))
            if not self.args.no_retrieval:
                summary_of_last_step = "Too long to summarize."
                for _ in range(self.args.max_retries):
                    try:
                        log_file = os.path.join(self.log_dir , f"step_{curr_step}_summary_log.log")
                        summary_of_last_step = self.summarize_action_and_observation(self.print_action(self.history_steps[-1]["action"], self.valid_format_entires), self.history_steps[-1]["observation"], log_file = log_file)
                        print("\n\nsummary_of_last_step", summary_of_last_step)
                        break
                    except Exception as e:
                        print(e)
                        print("Trying again.")


                action = "Append Summary to Research Log"
                action_input = { "content": "\n\nStep " + str(curr_step) + ":\n" + summary_of_last_step + "\n"}
                print("\n\nTrying to append summary to research log with this action input: ", str(action_input))
                env.execute(Action(action, action_input))

            step_idx = len(env.trace.steps) - 1
            self.save(os.path.join(self.log_dir , f"agent_{step_idx}_{curr_step}.json"))

            print("END OF STEP ", str(curr_step))

        if env.is_final():
            return "Finished due to env.is_final() == True"
        else:
            return "Finished due to agent max steps reached"


    ################### Helper functions #####################

    def summarize_observation(self, action, observation, log_file, bs = 10000):
        """ Summarize the observation if it is too long with a sliding window of size bs """

        bs = 10000
        blocks = [observation[i:i+bs] for i in range(0, len(observation), bs)]
        descriptions = []
        for idx, b in enumerate(blocks):
            start_line_number = bs*idx+1
            end_line_number = bs*idx+1 + len(b)
            prompt = f"""
{action}

The full observation is too long. Given this (partial) observation from character {start_line_number} to character {end_line_number}: 
``` 
{b}
```
Summarize the observation concisely in this format:
[Observation]: Summarize all relevant details in the observation objectively

Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""

            completion = complete_text_fast(prompt, log_file=log_file +f"_{idx}")
            descriptions.append(completion)
        if len(descriptions) == 1:
            completion = descriptions[0]
        else:
            descriptions = "\n\n".join(["Segment {idx}: \n\n" + s for s in descriptions])
            prompt = f"""
{action}

The full observation is too long. 
Given summaries for each segments of the whole observation, summarize to get a cohesive description of the entire observation.
{descriptions}

Summarize the observation concisely in this format:
[Observation]: Summarize all relevant details in the observation objectively

Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""

            completion = complete_text_fast(prompt, log_file=log_file)
        try:
            return completion.split("[Observation]:")[1]
        except:
            return completion

    @staticmethod
    def summarize_action_and_observation(action, observation, **kwargs):
        """ Summarize the action and observation to an entry in the research log """

        prompt = f"""Given your action and the observation: 
        {action} 
        [Observation]:
        ```
        {observation}
        ```
        Summarize your action and the observation in this format:
        [Reasoning]: Summarize the reasoning behind the action
        [Action]: Summarize all relevant details of the action objectively
        [Observation]: Summarize all relevant details in the observation objectively
        Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
        """

        summary = "[Reasoning]:" + complete_text_fast(prompt, log_file=kwargs["log_file"]).split("[Reasoning]:")[1]
        return summary
    
