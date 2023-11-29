from MLAgentBench_v2.agents.agent import Agent
import json

class ExecutionAgent(Agent):
    # Ultimately the answer will go into Description or be "returned" with the description being condensed into a tldr and the methods_prompt will be added to history

    # TODO: Perhaps the execution agent can get info about the files too, but maybe that's the action agent's responsibility.
    def __init__(self, env):
        super().__init__(env)

    def function_call(self, task, methods_prompt):
        try:
            num_tries = 2
            for _ in range(num_tries):
                system_prompt = f'''You are a helpful assistant. Your goal is to execute the given instructions and output the complete answer to the question. If the instructions don't seem reasonable or you cannot get to the complete answer, then you should give feedback on why you couldn't do it and what you tried. 

You will be given this information:
Skills: these are skills that I can take action with.
Files: these are my current files that I have in my working directory.
Task: ...
Instructions: ...
History of files, action, and result (newest to oldest): By following the plan, this is my history of files, actions, and results I had and took at that point in time.'''
                execute_prompt = f'''Skills: {list(self.available_actions.keys())}
Files: {self.files}
Task: {task}
Instructions: {methods_prompt}
History of files, actions, and results:
{self.formatted_action_history()}'''

                # complete_text_openai(system_prompt=system_prompt, prompt=execute_prompt, tools=self.tool_descriptions, available_functions=self.available_actions) # Normal function calling
                self.run_assistant(system_prompt=system_prompt, user_prompt=execute_prompt) # OpenAI Assistants API

                # Currently the run_assistant API will not complete the instructions entirely, so this will check for task completion and then continue the assistants by running the assistants API one more time, ideally continuing the instructions.
                check_task_completion_system_prompt = '''You are a helpful assistant. Your goal is to check if the instructions have been completed based on the history of files, actions, and results (newest to oldest) that I will give you. If based on the the recent files, actions, and results, it seems like the execution of the instructions is complete, then respond with True. Otherwise, respond with False

    You will be given this information:
    Instructions: ...
    History of files, actions, and results: By following the plan, this is my history of files, actions, and results I had and took at that point in time.

    You should only respond in JSON format as described below:
    ```json
    {
        "completed": boolean,
    }
    ```
    Ensure the response can be parsed by Python "json.loads", e.g.: no trailing commas, no single quotes, etc.
                '''
                check_task_completion_prompt = f'''Instructions: {methods_prompt}\nHistory of files, actions, and results:\n{self.formatted_action_history()}'''

                task_completion_response_message = self.complete_text_openai(system_prompt=check_task_completion_system_prompt, user_prompt=check_task_completion_prompt, json_required=True, update_files_action_result_history=False)
                try:
                    task_completion_response_json = json.loads(task_completion_response_message)
                    completed = task_completion_response_json['completed']
                    if completed: # Break if completed, otherwise, continue the execution agent
                        break
                except Exception as e:
                    break # Just return the existing execution agent history
            pass
        except Exception as e:
            return ""
        
        return self.formatted_action_history() # Difficult to manage a start index for only answers because some actions get popped or repeated