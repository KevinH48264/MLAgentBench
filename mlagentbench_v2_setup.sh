#!/bin/bash

# Export the Kaggle configuration directory
export KAGGLE_CONFIG_DIR=/MLAgentBench/.kaggle

# Install Kaggle API client
pip install kaggle

# Install unzip utility
sudo apt-get install unzip

# Upgrade OpenAI SDK
pip install --upgrade openai

# Install python-dotenv to handle .env files
pip install python-dotenv
