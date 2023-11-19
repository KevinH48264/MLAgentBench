from MLAgentBench_v2.agents.agent import Agent
import time
import json
from MLAgentBench_v2.LLM import complete_text_openai

class CurriculumAndAssistantAgent(Agent):
    """ Agent that takes actions based on the LLM output with the simplest prompt using Assistants API."""
    def __init__(self, env):
        super().__init__(env)  

    def run(self):
        print("Starting to run Curriculum and Assistant Agent")
        self.system_prompt = '''You are a helpful and first-rate research assistant.'''

        while True:

            # Start the curriculum agent. Taken from the LLM.py file, but a curriculum agent will just come up with a plan for now (alternative is to give one step at a time) 
            formatted_answer_states = ""
            for idx, answer_state in enumerate(self.answer_states):
                formatted_answer_states += "\nStep: " + str(idx) 
                formatted_answer_states += "\nFiles: " + str(answer_state['files']) 
                formatted_answer_states += "\nAction: " + answer_state['action'] 
                formatted_answer_states += "\nResult: " + answer_state['result'] 
                formatted_answer_states += "\nAnswer: " + answer_state['answer_state'] 
            self.initial_prompt = f"""You are a helpful research assistant. Given a research problem, files, tools, and at most 5 of your most recent files, action, result, and answer, your goal is to make a concrete plan using the tools and functions available to lead to a better answer and get you closer to solving the research problem. 

            Research Problem: {self.research_problem}
            Current Files: {self.files}
            Tools / functions: {self.available_actions.keys()}
            Most recent files, action, result, and answer states (oldest to newest):
            {formatted_answer_states}

            You should only respond with a clear step-by-step plan.       
            """
            
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\n\nPLANNING AGENT with answer states: {formatted_answer_states}")
            self.curriculum = complete_text_openai(self.initial_prompt, system_prompt=self.system_prompt, model=self.model, max_tokens_to_sample=500)
            print("Here is the curriculum: ", self.curriculum)
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\n\nPLANNING AGENT output: {self.curriculum}")

            # Start the assistants agent
            self.system_prompt = '''You are a helpful and first-rate research assistant.'''
            self.initial_prompt = f"""You are a helpful research assistant. 

            Research Problem: {self.research_problem}

            Plan: {self.curriculum}

            You have access to the following pieces of information in your file directory:
            {self.files}

            You have access to the following tools:
            {self.available_actions.keys()}

            Respond with your best answer to the research problem.
            """
            # Instantiate an Assistant
            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\n\nASSISTANTS AGENT prompt: {self.initial_prompt}")
            self.assistant = self.client.beta.assistants.create(
                name="Research Agent",
                instructions=self.system_prompt,
                tools=self.tool_descriptions,
                model=self.model
            )
            self.thread = self.client.beta.threads.create()

            # Invoke the Assistants API to answer
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=self.initial_prompt
            )
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
            )

            # Wait until the run has looped
            run_complete = False
            num_tries = 100
            while not run_complete:
                # Check if there's an update on the run
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=run.id
                )
                run_complete = run.status == "completed"
                print("\nrun.status: ", run.status)

                # Call the tools if the run status is requires action
                if run.status == "requires_action":
                    tool_outputs = []
                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        tool_id, tool_function, tool_type = tool_call.id, tool_call.function, tool_call.type
                        print(f"Run required action: \ntool_id: {tool_id}, \ntool_function.arguments: {tool_function.arguments}, \ntool_function.name: {tool_function.name}, \ntool_type: {tool_type}")

                        # Call the function directly if `tool_function` is a callable object
                        # and `arguments` is a dictionary of arguments to pass to the function.
                        try:
                            arguments = json.loads(tool_function.arguments)
                            function_output = self.available_actions[tool_function.name](**arguments)
                        except Exception as e:
                            function_output = f"Tool function {tool_function.name} for tool_id {tool_id} does not exist and is not callable with arguments {tool_function.arguments}. Make sure you are using only tools listed here: {self.available_actions.keys()} with the right arguments."

                        tool_outputs.append({
                            "tool_call_id": tool_id,
                            "output": function_output
                        })

                    # Submit tool outputs as a new run
                    run = self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=self.thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                elif run.status == "failed":
                    print("Run failed: ", run)
                    break

                time.sleep(1)
                num_tries -= 1
                if num_tries == 0:
                    print("Run timed out, cancelling...")
                    run = self.client.beta.threads.runs.cancel(thread_id=self.thread.id, run_id=run.id)
                    while run.status != "cancelled":
                        run = self.client.beta.threads.runs.retrieve(
                            thread_id=self.thread.id,
                            run_id=run.id
                        )
                    print("Run cancelled!")
                    break
            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
            completion = messages.data[0].content[0].text.value

            with open(self.main_log_path, "a", 1) as log_file:
                log_file.write(f"\n\nASSISTANTS AGENT output: {completion}")

            ans = input("\Continue? (y/n): ")
            if ans == "n":
                break

        return "Finished successfully! Final message: " + completion