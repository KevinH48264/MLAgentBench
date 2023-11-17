'''
Here's the theory for this agent system's approach:
Doing research has two parts
1) Assume you have a goal
2) step 2 is to test if you have already achieved the goal or the answer or you know you have achieved the goal already
3) Before you combine, you need to know what you’re missing from your top ideas
Multi agents are necessary for testing top ideas, not just improving SOTA because other approaches may be more promising
Identify failure modes BY testing. Testing informs agents of what’s missing
Figuring out what to ask is probably the hardest question / bottleneck
You need to know what’s interesting — rate facts for how interesting of a capability it is
4) Then once you know what you’re missing, then you have to go look for existing ideas that might help solve it or generate some ideas that could help solve it
5) Combining existing ideas via imagination / asking a question -- LLMs can probably do this
6) Validating in reality to build a library of high certainty expectation — easy for kaggle research
7) Return to step 2

Modeling it formally as a Partially Observable Markow Decision Process
1. R(s') = R(s, a) --> R(a')  Reward model -- the agent is given a research goal. The agent is the reward model that given a log of observations (from past states and action and feedback), it will do an internal calculation of what the next best action is. 
'''

from MLAgentBench_v2.agents.agent import Agent
import time
import json

class InformationLibraryAgent(Agent):
    """ Agent that takes actions based on the LLM output with the simplest prompt using Assistants API."""
    def __init__(self, env):
        super().__init__(env)  

    def run(self):
        print("Starting to run Curriculum and Assistant Agent")
        MAX_SIZE = 5000

        # 1) COMPLETE - Assume you have a goal
        # self.research_problem

        critic_agent_completion = ""
        while True:
            # 2) step 2 is to test if you have already achieved the goal or the answer or you know you have achieved the goal already
            with open(self.main_log_path, "r") as f:
                research_log = f.read()
            critic_agent_system_prompt = '''You are a helpful and first-rate research assistant.'''
            critic_agent_user_prompt = f'''You are an assistant that assesses my progress of research and provides useful guidance. 
            
            You are required to evaluate if I have achieved the research goal. Providing more information and exceeding the task requirements is also considered a success while failing to meet them or not actually answering the question requires you to provide critique to help me improve. You must require evidence showing that I have achieved the research goal.

            I will give you the following information:
            Research Goal: The research goal I need to achieve.
            State: My current state memory & research log.
            Files: Existing files I have.
            Actions: The actions that both you and I can take.
            Previous Feedback: The feedback you gave last time. If I failed to follow it, you should give me an easier step by step plan.

            Research Goal: {self.research_problem}
            State: {research_log[-MAX_SIZE:]}
            Files: {self.files}
            Actions: {self.available_actions.keys()}
            Previous Feedback: {critic_agent_completion}

            If I have achieved the research goal, respond with "Success". If I have not achieved the research goal, please explain why I have not achieved the research goal. Possible reasons include that I haven't provided the evidence showing that I achieved the research goal or you tested the file I submitted and saw that the evidence didn't meet the research goal requirements. Then, please suggest a step by step plan for how to achieve the research goal based on my current research log and state.
            '''
            critic_agent_completion = self.run_assistant(critic_agent_system_prompt, critic_agent_user_prompt, "Critic")
            if critic_agent_completion == "Success":
                print("\nCritic Agent said Success!\n")
                break
            print("Critic agent completion: ", critic_agent_completion)

            # 3) Before you combine, you need to know what you’re missing from your top ideas
            # Multi agents are necessary for testing top ideas, not just improving SOTA because other approaches may be more promising
            # Identify failure modes BY testing. Testing informs agents of what’s missing
            # Figuring out what to ask is probably the hardest question / bottleneck
            # You need to know what’s interesting — rate facts for how interesting of a capability it is

            # NOTE: Skipping for now to let the LLM do the internal calculus
            # analysis_agent_system_prompt = '''You are a helpful and first-rate research assistant.'''
            # analysis_agent_user_prompt = f'''You are an assistant that assesses my progress of research and provides useful guidance. 
            
            # You are required to evaluate if I have achieved the research goal. Providing more information and exceeding the task requirements is also considered a success while failing to meet them or not actually answering the question requires you to provide critique to help me improve. You must require evidence showing that I have achieved the research goal.

            # I will give you the following information:
            # Research Goal: The research goal I need to achieve.
            # State: My current state memory & research log.
            # Actions: The actions that both you and I can take.

            # Research Goal: {self.research_problem}
            # State: {research_log}
            # Actions: {self.available_actions.keys()}

            # If I have achieved the research goal, respond with "Success". If I have not achieved the research goal, please explain why I have not achieved the research goal. Possible reasons include that I haven't provided the evidence showing that I achieved the research goal or you tested the file I submitted and saw that the evidence didn't meet the research goal requirements.
            # '''
            # analysis_agent_completion = self.run_assistant(analysis_agent_system_prompt, analysis_agent_user_prompt)
            # if analysis_agent_completion == "Success":
            #     break

            # 4) Then once you know what you’re missing, then you have to go look for existing ideas that might help solve it or generate some ideas that could help solve it


            # 5) Combining existing ideas via imagination / asking a question -- LLMs can probably do this
            with open(self.main_log_path, "r") as f:
                research_log = f.read()
            action_agent_system_prompt = '''You are a helpful and first-rate research assistant.'''
            action_agent_user_prompt = f'''You are a helpful assistant.

            Please carefully analyze the feedback and provide a new, improved answer that can better solve the research problem.

            Please think step by 

            I will give you the following information:
            Research Goal: The research goal I need to achieve.
            State: My current state memory & research log.
            Files: Existing files I have.
            Actions: The actions that both you and I can take.
            Feedback: The feedback for why your research log has not shown that you achieved the research goal.

            Research Goal: {self.research_problem}
            State: {research_log[-MAX_SIZE:]}
            Files: {self.files}
            Actions: {self.available_actions.keys()}
            Feedback: {critic_agent_completion}

            Please analyze your research log step by step first, and then take actions to provide a new answer. The actions you take are automatically logged in the research log.
            '''
            action_agent_completion = self.run_assistant(action_agent_system_prompt, action_agent_user_prompt, "Action")

            # 6) COMPLETE -- letting step 2 do it. Validating in reality to build a library of high certainty expectation — easy for kaggle research

            # 7) COMPLETE Return to step 2

        return "Finished successfully!"
    

    def run_assistant(self, system_prompt, user_prompt, agent_type):
        # Instantiate an Assistant
            self.assistant = self.client.beta.assistants.create(
                name="Research Agent",
                instructions=system_prompt,
                tools=self.tool_descriptions,
                model=self.model
            )
            self.thread = self.client.beta.threads.create()

            # Invoke the Assistants API to answer
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=user_prompt
            )
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
            )
            if agent_type != "Critic":
                with open(self.main_log_path, "a", 1) as f:
                    f.write(f"\n\n {agent_type} Assistant was called! \n Prompt: {user_prompt}")

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
            if agent_type != "Critic":
                with open(self.main_log_path, "a", 1) as f:
                    f.write(f"\n\n {agent_type} Assistant Completion: {completion}")
            return completion