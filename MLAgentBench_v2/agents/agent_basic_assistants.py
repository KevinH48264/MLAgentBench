""" This file contains the agent class for our AI research agent."""
import os
import sys
import anthropic
from MLAgentBench.LLM import complete_text_fast, complete_text
import json


class BasicAssistantAgent():
    """This class implements the most basic AI research agent that uses the OpenAI Assistants API."""

    def __init__(self, args, env):
        print(f"STARTING BASIC ASSISTANT RESEARCH AGENT: \n{args} \nENV: \n {env}")
        
    def run(self, env):
        pass
        