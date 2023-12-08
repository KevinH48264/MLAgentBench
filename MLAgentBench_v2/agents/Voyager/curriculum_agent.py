# Note: incomplete update of answer_state in wiki

from MLAgentBench_v2.agents.agent import Agent
import json
import re

class CurriculumAgent(Agent):
    def __init__(self, env, completed_tasks=[], failed_tasks=[]):
        super().__init__(env)
        self.completed_tasks = env.completed_tasks
        self.failed_tasks = env.failed_tasks

        # The crux is a Q&A process
        # Problem with this approach is you still have to deal with searching multiple times, and continuing to search or not. Approach: Or maybe if you search and you don't have the answer, that's a bad thing to search and you need to go more specific / ask a different question!

# Commenting out for now to speed up execution
# Question 1: ...
# Answer: ...
# Question 2: ...
# Answer: ...
# Question 3: ...
# Answer: ...
# ...
        self.system_prompt_automatic_curriculum = f'''You are a helpful assistant that tells me the next immediate task to do. My ultimate goal is to achieve the research goal as quick as possible and produce answers that are better than myself and anyone else -- effectively becoming the best researcher in the world in solving this research goal.

Research Goal: {self.research_problem}

I will give you the following information:
Files: these are my current files and skills that I have in my working directory.
Skills: these are skills that I can take action with.
Completed tasks so far (most recent to least): ...
Failed tasks that are too hard (most recent to least): ...
Most recent answer states (newest to oldest): Answer states are the report of states that I think is best to track so far to best achieve the research goal, given the attempted task, plan, and result that I had at that point in time.

1) You should act as a mentor and guide me to the next task based on my current learning progress. Always give me the task that will help me learn the most and reach my research goal the fastest.
2) Please be very specific about what information or actions I need to take and what expected results I need to achieve. Always include a brief acceptance criteria and rejection criteria for the task.
3) The next task should follow a clear format, such as "Write [file]", "Reflect on why I'm seeing [error]", "Read [file]", "Brainstorm if ideas from [topic 1] be applied to [topic 2]", "Analyze what are the similarities between [topic 1] for success and [topic 2]" , "Reflect on what's significant about this paper: [paper]?", "Reflect on what's currently missing or preventing me from achieving [goal] better", etc. It should be a single task to collect useful information on. Do not propose multiple tasks at the same time. Do not mention anything else. Please include a brief acceptance criteria and rejection criteria for the task.
4) The next task should not be too hard since the internet and I may not contain the full answer in a single article or have learned enough information to complete it yet. 
5) The next task should be novel and interesting based on my current learning progress. I should look for rare and potentially useful pieces of information, upgrade my current answer using better information, and discover new things. I should not be doing the same thing over and over again.
6) I may sometimes need to repeat some tasks or variations of the task if I need to collect more information to answer more difficult tasks. Only repeat tasks if necessary. 
7) I want to explore the world and discover new things. I don’t want to stay with my current answer for too long. 
8) Tasks that require information beyond another reader's ability to theoretically verify and reason if completed or correct should be avoided. For instance, "what else is there on the website?" and "what images and tables are on the website" are not ideal since they require visual confirmation from the screen. All the testing, coding, and asking other people questions should be avoided. Do not propose a task  with these keywords. You should only respond in the format as described below:

RESPONSE FORMAT: 
```json
{{ 
    "research_goal": "<re-iterate the research goal so you understand the problem.>",
    "observations": "<observations about anything that might be useful.>",
    "reasoning": "<reasons about why the observations might be useful.>",
    "complete_plan": "<the best complete plan to increase the likelihood of me achieving the research goal quickly and better than anyone else.>"
    "task": "<the next task, acceptance criteria, and rejection criteria.>"
}}
```

Here’s an example response: 
```json
{{ 
    "research_goal": "To learn as much as possible in the world of Minecraft",
    "observations": "You have acquired most of the existing known items. I also see that there's lava on the ground, and there's a sword in my inventory, that could be interesting.",
    "reasoning": "Because I'm on the cutting edge of what's known, I can try and experiment with potentially new things to try discovering new things. We know that we have a sword and we know there's fire, and fire lights things on fire. Therefore, we could try to make a firesword.",
    "complete_plan": "1. First try to make a firesword. 2. If that works, then come up with ideas about what you can do with your new firesword. 3. If that doesn't work, then come up with ideas about what else you can do that would be new. (Ex. What else might you be able to do with lava? What else might you be able to do with a sword?)"
    "task": "Try to make a firesword and record what happens. Acceptance criteria: the sword is on fire. Rejection criteria: the sword is not on fire."
}}
```

Ensure the response can be parsed by Python "json.loads", e.g.: no trailing commas, no single quotes, etc. This is important.
'''

        # TODO: This is optional, might be useful, but to focus on a system prompt of asking questions and answering questions.
        # System 2: this is a more scoped down version where we have the focus be on only answering questions -- reading and analyzing information & asking questions. No action items. 
        # The current above system 1 is better for self-driving labs type of work where there are going to be more tasks.

    def get_exploration_progress(self):
        # TODO: this should contain inventory of where we're at now and what files we have / memory stream
        return f'''Completed tasks: {self.completed_tasks}, Failed tasks: {self.failed_tasks}'''

    def retrieve_from_wiki(self):
        # This function should solicit 5 questions from the agent and retrieve information from Wikipedia pages about it
        # The answer should be returned as a string in Q: ... A: ... ... format

        # First ask for questions
        asking_questions_system_prompt = f'''You are a helpful assistant that asks questions to help me decide the next immediate task to do in research. My ultimate goal is to discover as many useful pieces of information as possible to better achieve the research goal, answer as many questions as possible to get the best answer, and become the best researcher in the world in solving this research goal.

Research Goal: {self.research_problem}

I will give you the following information:
Files: these are my current files and skills that I have in my working directory.
Skills: these are skills that I can take action with.
Completed tasks so far (most recent to least): ...
Failed tasks that are too hard (most recent to least): ...
Most recent answer states (newest to oldest): Answer states are the report of states that I think is best to track so far to best achieve the research goal, given the attempted task, plan, and result that I had at that point in time.

You must follow the following critiera:
1) You should ask at least 5 questions (but no more than 10 questions) to help me decide the next immediate task to do. Each question should be followed by the concept that the question is about.
2) You question should be specific to a concept in Wikipedia. The question should not be too general.
Bad example (the question is too general):
Question: What is the best way to achieve the research goal?
Concept: unknown
Good example:
Question: What are some predictive models that can be used to predict the SalePrice of a house?
Concept: housing price predictive model
3) Your questions should be self-contained and not require any context.
Bad example (the question requires the context of my current files):
Question: Have you checked 'submission.csv' to ensure that the predicted SalePrice values are in a reasonable range compared to the distribution of SalePrice in 'train.csv'?
Concept: unknown
Bad example (the question requires the context of my current files):
Question: Does the 'model_training_script.py' include a cross-validation process to ensure the model's performance is robust and not overfitting?
Concept: unknown
Good example: 
Question: What are feature engineering techniques that are good to use for predicting the SalePrice of a house?
Concept: Housing price predictive model features

4) Do not ask questions about tasks that are beyond the scope of my skills because they are too hard for me to do.

RESPONSE FORMAT: 
```json
{{ 
    "reasoning": "<reasoning>",
    "1" : {{
        "question": "<question>",
        "concept": "<concept>"
    }},
    "2" : {{
        "question": "<question>",
        "concept": "<concept>"
    }},
    "3" : {{
        "question": "<question>",
        "concept": "<concept>"
    }}
    ...
}}
```

Ensure the response can be parsed by Python "json.loads", e.g.: no trailing commas, no single quotes, etc. This is important.
'''

        asking_questions_user_prompt = f'''Files: {self.files_no_skill_lib}
Skills: {list(self.available_actions.keys())}    
Completed tasks so far: {self.completed_tasks}
Failed tasks that are too hard: {self.failed_tasks}
Most recent answer states (newest to oldest)
{self.formatted_answer_states()}'''
        
        questions_and_concepts = self.complete_text_openai(system_prompt=asking_questions_system_prompt, user_prompt=asking_questions_user_prompt, json_required=True, update_files_action_result_history=False)
        question_and_concepts_json = json.loads(questions_and_concepts) # TODO: potentially add a try and except
        self.log("\nquestion_and_concepts_json: ", question_and_concepts_json, "\n")

        # Answer questions
        answer_question_system_prompt = f'''You are a helpful assistant that answers my question.
        
I will give you the following information:
Question: ...

You will answer the question based on the context (only if available and helpful) and your own knowledge.
1) Start your answer with "Answer: ".
2) Answer "Answer: Unknown" if you don't know the answer.'''
        
        # Iterate through question and concepts
        question_answer_string = ""
        for key, value in question_and_concepts_json.items():
            if key.isdigit():
                context = self.search_wikipedia(value['concept'].encode('utf-8').decode('utf-8'))
                answer_question_user_prompt = f'''Question: {value['question']} \nContext: {context}'''
                answer = self.complete_text_openai(system_prompt=answer_question_system_prompt, user_prompt=answer_question_user_prompt, update_files_action_result_history=False)

                question_answer_string += f"\nQuestion {str(key)}: {value['question']}\n{answer}"

        return question_answer_string

    def propose_next_task(self):
        '''
        This function decomposes a goal into tasks
        '''        
        # question_answer = self.retrieve_from_wiki() # TODO: commented out for now for speed of testing
        question_answer = ""
        user_prompt = f'''{question_answer}
Files: {self.files}
Skills: {list(self.available_actions.keys())}    
Completed tasks so far: {self.completed_tasks}
Failed tasks that are too hard: {self.failed_tasks}
Most recent answer states (newest to oldest):
{self.formatted_answer_states()}''' # TODO: Should I add formatted_action_history which includes tactical steps that were taken?
        
        self.log("System prompt for generating curriculum: \n", self.system_prompt_automatic_curriculum, "\n User prompt: ", user_prompt)
        next_task_response = self.complete_text_openai(system_prompt=self.system_prompt_automatic_curriculum, user_prompt=user_prompt, json_required=True)
        self.log("Response: ", next_task_response)
        next_task = json.loads(next_task_response)["task"]
        return next_task

    def get_completed_tasks():
        pass

    def get_failed_tasks():
        pass

    def add_completed_task(self, task, methods_prompt, result):
        # TODO: probably we should record the entire answer state of files, action, output, and answer state? Or just action and output?
        self.completed_tasks.insert(0, task + result)

        # Experimenting with adding the task to a living skill library in workspace so the methods prompt can build off of the skills library. 

        # Asking GPT to write a short file name, and then write the task + methods prompt to the file.
        res = self.complete_text_openai(system_prompt="You are a helpful assistant that writes a file name of the given task where the file contains a plan for potentially how to achieve that task. The file name should be less than 50 chars. Do not include the extension for the file name, .txt will be automatically added to the end. Your response should be only the file name.", user_prompt=f"Task: {task}", update_files_action_result_history=False)
        sanitized_file_name = self.sanitize_filename(res)
        with open(self.work_dir + "/skill_library/" + sanitized_file_name, "w") as f:
            f.write(f"Task: {task}\n")
            f.write(f"\nInstructions: {methods_prompt}")

        # Considering maintaining a running skill library, but adding files is likely not the way to build a wiki otherwise there will likely be a lot of overlapping information? Or not unless they're actually used in the skill library?
        with open(self.work_dir.split("_branch")[0] + "/skill_library/" + sanitized_file_name, "w") as f:
            f.write(f"Task: {task}\n")
            f.write(f"\nInstructions: {methods_prompt}")

        # Update answer state
        self.update_answer_state(task, methods_prompt, result)

    def add_failed_task(self, task, methods_prompt, result):
        self.failed_tasks.insert(0, task + result) #  + " \nCritique for why it failed: " + critique -- commented this out for now to allow for all tasks to be considered by the curriculum agent without truncation

        # Update answer state
        self.update_answer_state(task, methods_prompt, result)

    def sanitize_filename(self, text):
        # Remove invalid file name characters, replace spaces with underscores, lowercase and trim to 50 characters
        sanitized = re.sub(r'[^\w\s-]', '', text)  # Remove non-word characters except for spaces and hyphens
        sanitized = re.sub(r'\s+', '_', sanitized).strip()[:50].lower()  # Replace spaces, trim, and lower case
        return sanitized + '.txt'