from MLAgentBench_v2.agents.agent import Agent
import time
import json

class CurriculumAndAssistantAgent(Agent):
    """ Agent that takes actions based on the LLM output with the simplest prompt using Assistants API."""
    def __init__(self, env):
        super().__init__(env)  

    def run(self):
        print("Starting to run Curriculum and Assistant Agent")

        # Start the curriculum agent. Taken from the LLM.py file, but a curriculum agent will just come up with a plan for now (alternative is to give one step at a time) 
        # TODO: Chat completions should be an action / function so we have automatic logging in place
        plan_prompt = f'''Please create a simple step-by-step plan for a data scientist who is just starting this competition to achieve the goal. \n Competition: \n{self.research_problem}'''
        response = self.client.chat.completions.create(**{
            "model": self.model,
            "temperature": 0,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": plan_prompt}]
        })
        self.curriculum = response.choices[0].message.content
        print("Here is the curriculum: ", self.curriculum)

        # Start the assistants agent
        self.system_prompt = '''You are a helpful and first-rate research assistant and you never give up until you have completed the task.'''
        self.initial_prompt = f"""You are a helpful research assistant. 

        Research Problem: {self.research_problem}

        Plan: {self.curriculum}

        You have access to the following pieces of information in your file directory:
        {self.files}

        You have access to the following tools:
        {self.available_actions}

        Always respond in this format exactly:
        {{
            "Thought": "What you are currently doing, what actions to perform and why",
            "Action": "the action to take, should be one of the names of the tools",
            "Action Input": "the input to the action as a valid JSON string",
        }}
        """
        # Instantiate an Assistant
        self.assistant = self.client.beta.assistants.create(
            name="Research Agent",
            instructions=self.system_prompt,
            tools=self.tool_descriptions,
            model=self.model
        )
        self.thread = self.client.beta.threads.create()

        # Add research log to prompt
        self.initial_prompt += "Here are the most recent steps you have taken:\n"
        with open(self.main_log_path, "r") as f:
            log = f.read()
        self.initial_prompt += "\n" + log[-2500:]

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

        return "Finished successfully! Final message: " + completion