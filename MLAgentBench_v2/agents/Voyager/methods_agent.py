from MLAgentBench_v2.agents.agent import Agent

class MethodsAgent(Agent):
    # TODO: does there need to be an action agent generating the steps? Or does there need to be a separate execution agent running the prompt? Or can the action agent be the execution agent?
    # TODO: can a critic agent really check if the output is correct? Or can they only check if that aligns with expectation? Otherwise, the critic will have to check the line of content values to make sure the reasoning is sound, which is still doable, but the extent that another critic can check is limited. I guess it's just to make sure the reasoning is sound.

    def __init__(self, env):
        super().__init__(env)

    def generate_function_callable_prompt(self, task,
                methods_prompt,
                execution_feedback,
                execution_errors,
                critique):
        generate_plan_system_prompt = '''You are a helpful assistant and a first-rate problem solver. Given a task or question, your goal is to list out the steps to solve that task given your skills and reasoning. Ultimately, your output should be able to be followed by a human limited by the skills and knowledge given, and another human should be able to check that human's output to see if it's correct and reasonable. Note that the functions asked for may sometimes already be called and the information from the function that you need is already in the prompt, so read carefully. Note that you DO NOT have the ability to see, you can only read, think, write, and execute scripts using the existing skills and knowledge.

You will be given this information:
Task or question: ...
Skills: these are skills that I can take action with.
Files: these are my current files that I have in my working directory.  
Current state plan: ...
Current state output after executing steps: ...
Execution errors: ...
Critique: ...
History of files, action, and result (newest to oldest): By following the plan, this is my history of files, action, and result I had and took at that point in time. 

You should then respond to me with
Explain (if applicable): Are there any steps missing in your plan? Why do the current state steps not complete the task? What do the current state output, execution errors, and critique imply?
Plan: How to complete the task step by step. You should pay attention and read Files because it tells you what information you have. The task completeness check is also based on your final action results and final files.
Steps: 
1) Write how to complete the task step by step. 
2) Reuse the above useful files as much as possible.
3) Your task completion and plan will be reused to achieving more complex tasks. Thereofre, you should make it generic and reusable. You should not make strong assumptions about the files (as it may be changed at a later time), and therefore you should always check whether you have the required files before using them. If not, you should first create the required files, get the necessary information, and reuse the above useful actions.
'''

        user_prompt = f'''Task: {task}
Files: {self.files}
Skills: {list(self.available_actions.keys())}  
Current state plan: {methods_prompt}
Current state output after executing steps: {execution_feedback}
Execution errors: {execution_errors}
Critique: {critique}
History of files, action, and result:
{self.formatted_action_history()}'''

        methods_agent_feedback = self.run_assistant(system_prompt=generate_plan_system_prompt, user_prompt=user_prompt, tool_descriptions=self.read_tool_description)

        return methods_agent_feedback