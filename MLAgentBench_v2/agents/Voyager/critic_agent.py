from MLAgentBench_v2.agents.agent import Agent
import json

class CriticAgent(Agent):
    # TODO: Perhaps include the skills to the critic so the critic knows the facts to check if this makes sense or not
    def __init__(self, env):
        super().__init__(env)

    def check_task_success(self, task, methods_prompt, execution_feedback):
        # took out "Approach: My plan and reasoning to achieve the task." because critic agent would say Success to a good plan but incomplete reuslts
        system_prompt = '''You are a first-rate researcher that assesses my progress of research and provides useful guidance. 
        
Based on the final files, actions, and results, you are required to evaluate if I have already completed and satisfied all the task requirements. Exceeding the task requirements is also considered a success while failing to complete any of them requires you to provide critique to help me improve and mark my success as False. There must be evidence to show that all the task requirements are already and fully completed for it to be counted as a success. This is important.

I will give you the following information:
Task: The objective I need to accomplish.
Skills: these are skills that I can take action with.
Files: these are my current files that I have in my working directory.
History of files, action, and result (newest to oldest): After following the plan, this is my history of files, action, and result I had and took at that point in time.

You should only respond in JSON format as described below:
```json
{
    "task": "task",
    "evidence": "potential evidence of success",
    "counter_evidence": "potential evidence of failure",
    "reasoning": "reasoning",
    "success": boolean,
    "critique": "critique",
}
```
Ensure the response can be parsed by Python "json.loads", e.g.: no trailing commas, no single quotes, etc.
'''
# Commenting out the example because GPT3.5 just inappropriately uses it verbatim it sometimes
# RESPONSE:
# {
#     "reasoning": "The reasoning to get to the answer makes sense, but there's no direct answer for what the actual distribution of the sale price is.",
#     "success": False,
#     "critique": "The answer only tells us how to get the distribution is, but does not tell us what the actual distribution. Please tell us what the actual distribution is.",
# }

        # Commented out: "You can only read files to help check if task has been fully completed." because it didn't read all the files necessary sometimes
        user_prompt = f'''Task: {task}
Skills: {list(self.available_actions.keys())}
Files: {self.files}
History of files, action, and result: {self.formatted_action_history()}''' # Execution feedback should be logged in self.formatted_action_history()

        # 1. Employing a read assistant first to log files to be checked into file_action_result_history for the critic agent
        response_message = self.run_assistant(system_prompt=system_prompt, user_prompt=user_prompt, tool_descriptions=self.read_tool_description)

        # 2. Employing a chat completion based on the updated file_action_reuslt_history to make a final judgement
        user_prompt = f'''Task: {task}
Skills: {list(self.available_actions.keys())}
Files: {self.files}
History of files, action, and result: {self.formatted_action_history()}''' # Execution feedback should be logged in self.formatted_action_history()
        
        self.log("Critic system prompt: ", system_prompt, "\n\nCritic user prompt: ", user_prompt, "\n\nTask: " + task + "\n\nCritic response: ", response_message)

        response_message = self.complete_text_openai(system_prompt=system_prompt, user_prompt=response_message, json_required=True)

        try:
            response_json = json.loads(response_message)
            task = str(response_json['task'])
            success = response_json['success'] # Must be bool
            evidence = str(response_json['evidence'])
            opposition = str(response_json['counter_evidence'])
            reasoning = str(response_json['reasoning'])
            critique = str(response_json['critique'])
        except Exception as e:
            return False, response_message + " JSON parsing error: " + str(e)

        # Handle null values
        if not reasoning:
            reasoning = ""
        if not critique:
            critique = ""

        return success, "\nEvidence: " + evidence + "\nCounter evidence: " + opposition + "\nReasoning: " + reasoning + "\nCritique: " + critique