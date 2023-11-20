class CurriculumAgent():
    def __init__(self, goal="", memory=[], completed_tasks=[], failed_tasks=[], files="", saved_notes=""):
        self.goal = goal
        # self.memory = memory
        self.completed_tasks = completed_tasks
        self.failed_tasks = failed_tasks
        # self.files = files
        # self.saved_notes = saved_notes

        # The crux is a Q&A process
        # Problem with this approach is you still have to deal with searching multiple times, and continuing to search or not. Approach: Or maybe if you search and you don't have the answer, that's a bad thing to search and you need to go more specific / ask a different question!
        self.system_prompt_automatic_curriculum = f'''You are a helpful assistant that asks questions to help me decide the next immediate question to answer on the computer. My ultimate goal is to discover as many useful pieces of information as possible, answer as many questions as possible, and become the best researcher in the world.

        Goal: {self.goal}

        I will give you the following information:
        Skills: these are skills that I can take action with.
        Knowledge base files: these are information blocks that I have and can build upon as I answer more questions, almost like theorems that have been validated.
        Completed tasks so far: ...
        Failed tasks that are too hard: ...

        1) You should act as a mentor and guide me to the next question based on my current learning progress.
        2) Please be very specific about what question I need to answer.
        3) The next question should follow a clear format, such as "What do other papers say about [topic]", "What are the problems with [approach]", "How can I solve this [problem]", "Can results from [topic 1] be applied to [topic 2]?", "What are the similarities between [topic 1] for success and [topic 2]" , "What's significant about this paper: [paper]?", "What does [topic] mean?", etc. It should be a single question to collect useful information on. Do not propose multiple questions at the same time. Do not mention anything else. 
        4) The next question should not be too hard since the internet and I may not contain the full answer in a single article or have learned enough information to complete it yet. 
        5) The next question should be novel and interesting based on my current learning progress. I should look for rare and potentially useful pieces of information, upgrade my saved notes using better information, and discover new things. I should not be doing the same thing over and over again.
        6) I may sometimes need to repeat some questions or variations of the question if I need to collect more information to answer more difficult question. Only repeat questions if necessary. 
        7) I want to explore the world and discover new things. I don’t want to stay in my current state for too long. 
        8) Questions that require information beyond another person's ability to theoretically verify and reason if completed or correct should be avoided. For instance, "what else is there on the website?" and "what images and tables are on the website" are not ideal since they require visual confirmation from the screen. All the testing, coding, and asking other people questions should be avoided. Do not propose a question with these keywords. You should only respond in the format as described below:
        
        RESPONSE FORMAT: 
        Reasoning: Based on the information I listed above, do reasoning about what the next question should be. 
        Question: The next question. 
        
        Here’s an example response: 
        Reasoning: We know the we have a sword and we know there's fire, and fire lights things on fire. Therefore, we could try to make a firesword.
        Question: Could we make a firesword?
'''

        # TODO: This is optional, might be useful, but to focus on a system prompt of asking questions and answering questions.
        # System 2: this is a more scoped down version where we have the focus be on only answering questions -- reading and analyzing information & asking questions. No action items. 
        # The current above system 1 is better for self-driving labs type of work where there are going to be more tasks.

    def get_exploration_progress(self, completed_tasks, failed_tasks):
        # TODO: this should contain inventory of where we're at now and what files we have / memory stream
        return '''Completed tasks: None, Failed tasks: None'''

    def propose_next_question(self, skills, files, exploration_progress):
        '''
        This function decomposes a goal into tasks
        '''        
        user_prompt = ""
        observation = {
            # "memory": f"Memory: {self.memory}\n\n",
            "skills": f"Skills: {skills}\n\n",
            "knowledge_base_files": f"Knowledge base files: {files}\n\n",
            # "saved_notes": f"Saved notes: {self.saved_notes}\n\n",
            "completed_tasks": f"Completed tasks so far: {self.completed_tasks}\n\n",
            "failed_tasks": f"Failed tasks that are too hard: {self.failed_tasks}\n\n",
        } # TODO: I don't think I'm even using saved notes
        user_prompt += "".join(observation.values())
        
        print("System prompt for generating curriculum: \n", self.system_prompt_automatic_curriculum, "\n User prompt: ", user_prompt)
        next_question_response = chat_openai(user_prompt, system_prompt=self.system_prompt_automatic_curriculum)[0]['content']
        print("Response: ", next_question_response)
        next_question = self.parse_message(next_question_response)["next_question"]
        return next_question

    def get_completed_tasks():
        pass

    def get_failed_tasks():
        pass

    def add_completed_task(self, task):
        self.completed_tasks.append(task)

    def add_failed_task(self, task):
        self.failed_tasks.append(task)

    def parse_message(self, message):
        question = ""
        for line in message.split("\n"):
            if line.startswith("Question:"):
                question = line[9:].strip()
        assert question, "Question not found in Curriculum Agent response"
        return {"next_question": question}