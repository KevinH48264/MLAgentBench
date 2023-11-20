class ExecutionAgent():
    # Ultimately the answer will go into Description or be "returned" with the description being condensed into a tldr and the methods_prompt will be added to history

    # TODO: Perhaps the execution agent can get info about the files too, but maybe that's the action agent's responsibility.

    def function_call(self, methods_prompt, skills, info_blocks, skill_manager):
        system_prompt = f'''You are a helpful assistant. Your goal is to execute the given instructions and output the complete answer to the question. If the instructions don't seem reasonable or you cannot get to the complete answer, then you should give feedback on why you couldn't do it and what you tried. 

        You will be given this information:
        Skills: You only have these skills to help you execute the instructions.
        Knowledge: You only have these information blocks as files that you can read to get more information about the details of the information block's answer and the reasoning and methods used to arrive at the answer.
        Instructions: ...
'''
        execute_prompt = f'''Skills: {skills}
        Knowledge: {info_blocks}
        Instructions: {methods_prompt}'''

        try:
            response_message, messages = chat_openai(system_prompt=system_prompt, prompt=execute_prompt, functions=skill_manager.functions, available_functions=skill_manager.available_functions,verbose=True)
            conclusion = response_message['content'] 

            # # Assuming response_message['content'] is a string that contains 'Conclusion:'
            # match = re.search(r'Conclusion:(.*)', response_message['content'], re.DOTALL)
            # if match:
            #     conclusion = match.group(1).strip()
            # else:
            #     conclusion = response_message['content']  # or some default value, in case "Conclusion:" isn't found
        except Exception as e:
            return "", e
        return conclusion, None