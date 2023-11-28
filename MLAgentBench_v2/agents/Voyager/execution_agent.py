from MLAgentBench_v2.agents.agent import Agent

class ExecutionAgent(Agent):
    # Ultimately the answer will go into Description or be "returned" with the description being condensed into a tldr and the methods_prompt will be added to history

    # TODO: Perhaps the execution agent can get info about the files too, but maybe that's the action agent's responsibility.
    def __init__(self, env):
        super().__init__(env)

    def function_call(self, task, methods_prompt):
        system_prompt = f'''You are a helpful assistant. Your goal is to execute the given instructions and output the complete answer to the question. If the instructions don't seem reasonable or you cannot get to the complete answer, then you should give feedback on why you couldn't do it and what you tried. 

You will be given this information:
Skills: these are skills that I can take action with.
Files: these are my current files that I have in my working directory.
Task: ...
Instructions: ...
History of files, action, and result (newest to oldest): By following the plan, this is my history of files, action, and result I had and took at that point in time.'''
        execute_prompt = f'''Skills: {list(self.available_actions.keys())}
Files: {self.files}
Task: {task}
Instructions: {methods_prompt}
History of files, action, and result:
{self.formatted_action_history()}'''

        try:
            # complete_text_openai(system_prompt=system_prompt, prompt=execute_prompt, tools=self.tool_descriptions, available_functions=self.available_actions) # Normal function calling
            self.run_assistant(system_prompt=system_prompt, user_prompt=execute_prompt) # OpenAI Assistants API
            pass
        except Exception as e:
            return ""
        
        return self.formatted_action_history() # Difficult to manage a start index for only answers because some actions get popped or repeated