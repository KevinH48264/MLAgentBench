class SkillManager():
    # Additional skills to be added: await bot.findsProblem, await bot.findsInterestingInformation, await bot.findSolution or something like that

    def __init__(self):
        self.READ_MAX_CHAR_SIZE = 2500
        self.functions = [
            {
                "name": "read_file",
                "description": "Get the text from the file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "the complete file name and extension, e.g. abc.txt"
                        }
                    },
                    "required": ["file_name"],            
                }
            }, 
            {
                "name": "write_file",
                "description": "Write text to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "description": "the complete file name and extension, e.g. abc.txt"
                        },
                        "text": {
                            "type": "string",
                            "description": "the text to write to the file"
                        }
                    },
                    "required": ["file_name", "text"],            
                }
            }, 
            {
                "name": "web_search",
                "description": "Given a prompt, I'll return back my web search information about it",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Tell me what to search about"
                        }
                    },
                    "required": ["query"],            
                }
            },
            {
                "name": "think",
                "description": "Given a prompt, I'll return back my thoughts on it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Tell me what to think about"
                        }
                    },
                    "required": ["query"],            
                }
            },
            # {
            #     "name": "num_children",
            #     "description": "This paper talks about how many girls and boys are in this family",
            #     "parameters": {
            #         "type": "object",
            #         "properties": {},
            #         "required": [],
            #     },
            # },
            
            # {
            #     "name": "content_in_data_txt",
            #     "description": "I know the information in data.txt",
            #     "parameters": {
            #         "type": "object",
            #         "properties": {},
            #         "required": [],
            #     },
            # }
            # {'name': 'columns_and_descriptions', 'description': "The question was what are the columns and their descriptions in the training set and the test set. The answer is that the columns and their descriptions can be found in the 'data_description.txt' file using the 'read_file' function. The columns and their descriptions in both the training set and the test set include information about the type of dwelling, zoning classification, lot size, road and alley access, property shape and flatness, utilities available, and lot configuration.", 'parameters': {'type': 'object', 'properties': {}, 'required': []}}
        ]

        self.available_functions = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "web_search": self.web_search,
            "think": self.think,
        }

        # Currently not much utility, but if you needed to trace back to a function or fact it got wrong, this can help.
        # TODO: honestly, not really sure what to do with this. I was thinking seeing the entire message history would be a good way to trace back, but we already have the original message prompt
        self.function_history = [
            "given",
            "given",
            "given",
            "given",
        ]

        self.files = [
            ("train.csv", "the training set"),
            ("test.csv", "the test set"),
            ("data_description.txt", "full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here"),
            ("sample_submission.csv", "a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms"),
            ("data_fields.txt", "a brief version of what you'll find in the data description file."),
        ]

    def retrieve_skills(self, task, execution_feedback):
        # For sake of simplicity, use recency for now (later relevancy and importance can be added)
        func_name_description_list = [str((func["name"], func["description"])) + '\n' for func in self.functions]
        return func_name_description_list
    
    def retrieve_info_blocks(self, task, execution_feedback):
        # retrieving file names and description
        return [name + " - " + description for name, description in self.files]

    # Helper function: be able to create a function with a dynamic name and return value
    # def create_skill_function(self, function_name, return_value):
    #     def dynamic_method(self):
    #         return return_value
    #     # Bind the function to the instance as a method
    #     bound_method = types.MethodType(dynamic_method, self)
    #     setattr(self, function_name, bound_method)
    #     # Add the method to available functions
    #     self.available_functions[function_name] = bound_method

    # Core function: adding a new skill requires an original task, a validated answer, and a message history
    def add_skill(self, task, validated_answer, methods_prompt):
        # TODO: wait until the action agent generates a function because maybe you only need to write a description of the input function instead of task and validated answer.
        # create_function_description_system_prompt = f'''You are a helpful assistant that writes a description of the given '''

        print("Adding skills! ", task, validated_answer, methods_prompt)

        create_skill_system_prompt = f'''You are a helpful assistant. Your goal is to write a short file name and a short description of the question and answer. 
        
        You will receive this information:
        Original task or question: ...
        Answer: ...

        Do not use any of these file names: {[name for name, _ in self.files]}

        Your output should be in the following format if function requires arguments:
        ```json
        {{
            "name": "<file_name>",
            "description": "<insert question and answer>"
        }}
        ```

        Good example output:
        ```json
        {{
            "name": "num_dogs_in_bens_family",
            "description": "The question was how many dogs are in the family. Ben said that he has 2 dogs in his family."
        }}
        ```

        Ensure the response can be parsed by Python "json.loads", e.g.: no trailing commas, no single quotes, etc. This is important.
        '''

        create_function_description_prompt = f'''
        Original task or question: {task}
        Answer: {validated_answer}
        '''
        res, messages = chat_openai(prompt=create_function_description_prompt, system_prompt=create_skill_system_prompt, verbose=True)
        res

        try:
            # Load the function description
            file_name_description = json.loads(res['content'])
            print("file_name_description: ", file_name_description)

            # Create the function as a method of skill_manager
            self.write_file(file_name_description['name'], f"Question: {task}\nAnswer: {validated_answer}\nReasoning and Methods: {methods_prompt}")

            # Add function to function description list
            self.files.append((file_name_description['name'], file_name_description['description']))

            print("COMPLETE!")
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return

        return

    # Below are given functions or dummy functions
    def read_file(self, file_name):
        '''Get the text from the file'''
        file_path = f'{root_file_directory}/{file_name}'
        with open(file_path, 'r') as file:
            text = file.read()[:self.READ_MAX_CHAR_SIZE]
        return text
    
    def write_file(self, file_name, text):
        '''Write the given text to the file.'''
        file_path = f'{root_file_directory}/{file_name}.txt'
        with open(file_path, 'w') as file:  # 'w' is the mode for writing to a file
            file.write(text)  # Write the entire text to the file
        return text

    def web_search(self, query):
        # Temporarily just return search info from the user
        return input(f"Search for {query} on the web: ")
        
    def think(self, query):
        return chat_openai(query + " Think step by step.", verbose=True)[0]['content']
    
    # Demo of how to add plain just fact / info skills that gets added as just natural language?
    def content_in_data_txt(self):
        return '''The data.txt file contains the following information: File descriptions
train.csv - the training set
test.csv - the test set
data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms
data_fields.txt - a brief version of what you'll find in the data desc'ription file.'''

    def num_children(self):
        return '''There are 2 boys, and 8 girls'''
    
    # MANUALLY ADDED FUNCTIONS FOR TESTING
    def columns_and_descriptions():
        return '''The question was what are the columns and their descriptions in the training set and the test set. The answer is that the columns and their descriptions can be found in the 'data_description.txt' file using the 'read_file' function. The columns and their descriptions in both the training set and the test set include information about the type of dwelling, zoning classification, lot size, road and alley access, property shape and flatness, utilities available, and lot configuration.'''