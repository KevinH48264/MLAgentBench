import json
import os
import requests
import time


def get_token():
    # This requires having your token in key_path.txt. Otherwise, it'll throw an error.
    try:
        with open('key_path.txt', encoding='utf8') as f:
            key_path = f.read().strip()
        with open(key_path, 'r', encoding='utf-8') as fi:
            line = fi.read().strip()
            return line[1:]
    except:
        input('Failed to get token. Please check model.py and check if the path to your token is correct. Press Enter to continue.')

class DeepLeo:
    def __init__(self, debug=False) -> None:
        self.debug = debug
        self.temperature = 0.1
        self.max_tokens = 4096
        self.top_p = 1.0
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.stop = ["<|im_end|>", "[assistant](#", "[user](#", "[planner](#"]
        self.url = "https://prom-sahara-eval.centralus.inference.ml.azure.com/v1/engines/davinci/completions"
        self.url = "http://10.184.245.201:5000/internal_raw_ep"

    def set_params(self, **kwargs) -> None:
        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
        if "max_tokens" in kwargs:
            self.max_tokens = kwargs["max_tokens"]
        if "top_p" in kwargs:
            self.top_p = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            self.frequency_penalty = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            self.presence_penalty = kwargs["presence_penalty"]
        if "stop" in kwargs:
            self.stop = kwargs["stop"]
        if "url" in kwargs:
            self.url = kwargs["url"]

    def get_completions(self, prompt: str, state="", tools=None, available_functions=None, json_required=False) -> str:
        data = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
        }

        if tools: # allow for function calling
            data["tools"] = tools
            data["tool_choices"] = "auto"
        # if json_required: # json mode
        #     data['response_format'] = { "type": "json_object" }

        if state == "reasoning":
            data["temperature"] = 0
            data["top_p"] = 0
            data["max_tokens"] = 4096
            data["stop"] = ["```"]
        
        elif state == "planning":
            data["temperature"] = 0
            data["top_p"] = 0
            data["max_tokens"] = 4096
            data["stop"] = ["```"]

        else:
            data["temperature"] = 0
            data["top_p"] = 0
            data["max_tokens"] = 4096
            data["stop"] = None
        
        headers = dict()

        # Get access token
        # stream = os.popen(
        #     "az account get-access-token --resource https://ml.azure.com --query accessToken")
        # access_token = stream.read().strip('\n').strip('"')
        for _ in range(3):
            try:
                f = open(r"\\TURINGQNA-SIYU\token\dv3_token.txt", "r") # raw string for no escape for windows
                access_token = f.read().strip()
            except Exception as e:
                access_token = get_token()
                print("Failed to get token. Please ensure that you're connected to VPN and have access to the token.")
        headers["Authorization"] = "Bearer " + access_token
        headers["Content-Type"] = "application/json"
        
        
        #url = "http://10.184.245.201:5000/internal_raw_ep"
        #url = "https://prom-d-nd40.centralus.inference.ml.azure.com/v1/engines/davinci/completions"
        url = "https://prom-d-nd40.southcentralus.inference.ml.azure.com/v1/engines/davinci/completions" # GPT3.5
        success = False
        tries = 0
        error = ValueError("Failed to get response from DeepLeo")
        while not success:
            try:
                # if self.debug:
                #     print("\ndata: ", data)
                response = requests.post(
                    url, headers=headers, data=json.dumps(data))
                                
                if self.debug and response.status_code == 401:
                    error = ValueError("401 Error in model.py")
                    break
                
                response = response.json()
                # if self.debug:
                #     print("response: ", response)
                if "error" in response:
                    if "Please reduce your prompt" in response["error"]["message"]:
                        error = ValueError(response["error"]["message"])
                        break
                
                    if self.debug and response['error']['message'].find('The server is currently overloaded ') > -1:
                        error = ValueError(response["error"]["message"])
                        print('Waiting 5s to retry due to server-overloaded error')
                        time.sleep(5)
                        continue
                
                result = response["choices"][0]["text"].strip()
                success = True
                break
            except Exception as e:
                error = e
                print(e)
                print(response)
                time.sleep(5)
            tries += 1
            if tries >=3:
                break
        if not success:
            raise error
        return result
