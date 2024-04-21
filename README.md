# LLM Chat Agent 

Creating a Chat tool leveraging OpenAI, LlamaIndex & Streamlit

Below is the demo video

![LLM_Demo](https://github.com/shariqfarhan/LLM_Frontend/assets/57046534/91321dc3-7c43-4cda-affb-db28547a97ca)

## Introduction

Since the introduction of ChatGPT in Nov 2022, LLMs have been in limelight for various language related tasks. Though there have been limitations, the potential of LLMs is immense. In this project, I have created a Chat Agent leveraging OpenAI's ChatGPT, LlamaIndex and Streamlit. The Chat Agent leverages Agents on top of GPT to provide a more interactive and engaging experience.

## Features

1. **GPT**: The Chat Agent uses OpenAI's GPT API in the backend to generate responses to user queries.
2. **LlamaIndex**: The Chat Agent uses LlamaIndex to search for relevant information and provide it to the user. We leverage ReAct Agents to enhance the LLM chat experience.
3. **Streamlit**: The front-end of Chat Agent is built using Streamlit, which provides a simple and interactive interface for the user to interact with the back-end API.

## How to run

1. Clone the repository
2. Install the requirements using `pip install -r requirements.txt`
3. Run the Streamlit app using `streamlit run main.py`

## Methodology

The Chat Agent is built using Streamlit for the front-end and OpenAI's GPT API and LlamaIndex for the back-end. The user interacts with the Chat Agent through the Streamlit app, which sends the user queries to the back-end API. The back-end API generates responses to the user queries using GPT and searches for relevant information using LlamaIndex. The responses are then sent back to the front-end and displayed to the user.


## Future Work

1. **Multi-turn Conversations**: Currently, the Chat Agent only supports single-turn conversations. In the future, I plan to extend it to support multi-turn conversations
2. **Better UI**: The current UI is very basic. I plan to improve it by adding more features and making it more interactive
3. **More Agents**: I plan to add more Agents to the Chat Agent to provide a more engaging experience to the user
4. **Latency Reduction**: The current implementation has some latency issues. I plan to optimize the back-end API to reduce the latency and provide a smoother experience to the user

## Conclusion

In this project, I have created a Chat Agent leveraging OpenAI's GPT API, LlamaIndex and Streamlit. The Chat Agent provides an interactive and engaging experience to the user by generating responses to user queries using GPT and searching for relevant information using LlamaIndex. I plan to extend the Chat Agent in the future to support multi-turn conversations, improve the UI and add more Agents to provide a more engaging experience to the user.
