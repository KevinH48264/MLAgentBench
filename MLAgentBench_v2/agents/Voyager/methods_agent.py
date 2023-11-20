class MethodsAgent():
    # TODO: does there need to be an action agent generating the steps? Or does there need to be a separate execution agent running the prompt? Or can the action agent be the execution agent?
    # TODO: can a critic agent really check if the output is correct? Or can they only check if that aligns with expectation? Otherwise, the critic will have to check the line of content values to make sure the reasoning is sound, which is still doable, but the extent that another critic can check is limited. I guess it's just to make sure the reasoning is sound.

    def __init__(self):
        pass

    def generate_function_callable_prompt(self, task,
                methods_prompt,
                execution_feedback,
                execution_errors,
                critique,
                skills,
                info_blocks,
                skill_manager):
        generate_methods_system_prompt = '''You are a helpful assistant and a first-rate problem solver. Given a task or question, your goal is to list out the steps to solve that task or question given your skills and reasoning. Ultimately, your output should be able to be followed by a human limited by the skills and knowledge given, and another human should be able to check that human's output to see if it's correct and reasonable. Note that the functions asked for may sometimes already be called and the information from the function that you need is already in the prompt, so read carefully. Note that you DO NOT have access to run any code, you can only read, think, and write about the existing skills and knowledge.
        
        You will be given this information:
        Task or question: ...
        Skills: You only have these skills to help you write the steps to achieve this task or answer this question.
        Knowledge: You only have these information blocks as files that you can read to get more information about the details of the information block's answer and the reasoning and methods used to arrive at the answer.
        Current state steps: ...
        Current state output after executing steps: ...
        Execution errors: ...
        Critique: ...
        
        You should then respond to me with
        Explain (if applicable): Are there any steps missing in your plan? Why do the current state steps not lead to the answer? What do the current state output, execution errors, and critique imply?
        Steps: 
        1) Write how to answer the question and complete the task step by step. 
        2) Reuse the above useful knowledge as much as possible. The question answering check is ultimately based on how reasonable your steps are and if the final answer after following your steps is a complete answer to the question.
        3) Your steps will be reused to trace back the reasoning behind the answer.
        4) The final answer will be reused for building more complex information blocks. Therefore, you should make the final answer reusable.
        5) You should not make strong assumptions about knowledge and information, and therefore you should always check whether you have the required information before using them in reasoning. If not, you should first collect the required information and reuse the above useful knowledge.
        5) Do not propose steps that the above skills cannot accomplish.
        '''

        user_prompt = f'''Task or question: {task}
        Skills: {skills}
        Knowledge: {info_blocks}
        Current state solution: {methods_prompt}
        Current state output after executing steps: {execution_feedback}
        Execution errors: {execution_errors}
        Critique: {critique}'''

        agent_methods_feedback, agent_methods_errors = self.function_call(generate_methods_system_prompt, user_prompt, skill_manager)
        # TODO: parse out just the steps to save
        # agent_methods_feedback_json = json.loads(agent_methods_feedback['content'])

        # TODO: Add error handling for if the agent_methods_feedback is empty and agent_methods_errors exists
        print("agent_methods_feedback", agent_methods_feedback, "agent_methods_errors", agent_methods_errors)

        # steps_idx = agent_methods_feedback['content'].find("Steps: ")
        # steps = agent_methods_feedback['content'][steps_idx + len("Steps: "):].strip()

        return f'Task or question: {task} \nInstructions: ' + agent_methods_feedback['content']
    
    def function_call(self, system_prompt, methods_prompt, skill_manager):
        print()
        try:
            response_message, messages = chat_openai(system_prompt=system_prompt, prompt=methods_prompt, functions=skill_manager.functions, available_functions=skill_manager.available_functions,verbose=True)
        except Exception as e:
            return "", e
        return response_message, None