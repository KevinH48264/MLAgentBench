class CriticAgent():
    # TODO: Perhaps include the skills to the critic so the critic knows the facts to check if this makes sense or not

    def check_task_success(self, task, methods_prompt, execution_feedback, skills, info_blocks, skill_manager):
        system_prompt = '''You are an assistant that assesses my progress of research and provides useful guidance. 
        
        You are required to evaluate if I have provided a correct and direct answer to the question. Providing more information and exceeding the task requirements is also considered a success while failing to meet them or not actually answering the question requires you to provide critique to help me improve. Note that me providing steps is not enough, my final answer must be provided and directly answer the question.

        I will give you the following information:
        Skills: My skills. 
        Knowledge: My information blocks as files.
        Task or question: The question I need to answer.
        Answer: My current answer.
        Approach: My reasoning of how I got to my answer.

        You should only respond in JSON format as described below:
        {
            "biggest_risk", "biggest risk for why the answer might not answer the question",
            "reasoning": "reasoning for if the answer successfully answers the question or not",
            "success": boolean,
            "critique": "critique",
        }
        Ensure the response can be parsed by Python "json.loads", e.g.: no trailing commas, no single quotes, etc.

        Here are some examples:
        INPUT:
        Task or question: What is the distribution of the sale prices in the dataset?
        Answer: To determine the distribution of the sale prices in the dataset, we can follow these steps:\n\n1. Read the dataset file "train.csv" using the `read_file` function.\n2. Extract the column containing the sale prices from the dataset.\n3. Calculate the frequency of each unique sale price in the dataset.\n4. Sort the unique sale prices in ascending order.\n5. Create a histogram or bar chart to visualize the distribution of the sale prices.\n6. Optionally, you can also calculate summary statistics such as mean, median, and standard deviation of the sale prices.\n\nLet\'s start by reading the dataset file "train.csv".
        Approach: Task or question: What is the distribution of the sale prices in the dataset? \nInstructions: To determine the distribution of the sale prices in the dataset, you can follow these steps:\n\n1. Read the dataset file "train.csv" using the `read_file` function.\n2. Extract the column containing the sale prices from the dataset.\n3. Calculate the frequency of each unique sale price in the dataset.\n4. Sort the unique sale prices in ascending order.\n5. Create a histogram or bar chart to visualize the distribution of the sale prices.\n6. Optionally, you can also calculate summary statistics such as mean, median, and standard deviation of the sale prices.\n\nPlease note that the specific implementation details may vary depending on the programming language and libraries you are using.

        RESPONSE:
        {
            "biggest_risk", "There is no actual distribution in the answer.",
            "reasoning": "The reasoning to get to the answer makes sense, but there's no direct answer for what the actual distribution of the sale price is.",
            "success": False,
            "critique": "The answer only tells us how to get the distribution is, but does not tell us what the actual distribution. Please tell us what the actual distribution is.",
        }
        '''

        user_prompt = f'''Skills: {skills}
        Knowledge: {info_blocks}
        Task or question: {task}
        Answer: {execution_feedback}
        Approach: {methods_prompt}'''

        response_message, messages = chat_openai(system_prompt=system_prompt, prompt=user_prompt, verbose=True)

        response_json = json.loads(response_message['content'])
        success = response_json['success']
        reasoning = response_json['reasoning']
        critique = response_json['critique']

        # lines = response_message['content'].split("\n")
        # for line in lines:
        #     # Strip whitespace for accurate matching
        #     line = line.strip()
            
        #     # Check if the line starts with the known titles and parse accordingly
        #     if line.startswith('Reasoning:'):
        #         critique = line[len('Reasoning:'):].strip()
        #     elif line.startswith('Task is reasonably answered:'):
        #         success = line[len('Task is reasonably answered:'):].strip().lower()

        print("System prompt: ", system_prompt, "\n\nUser prompt: ", user_prompt, "\n\nResponse: ", response_message['content'])
        return success, critique