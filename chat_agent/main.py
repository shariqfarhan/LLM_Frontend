# Works on streamlit version 1.24.0 and greater
# Libraries for Backend
import os
import openai
from openai import OpenAI
import json
import streamlit as st

# Libraries for LLM Agent
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI, ChatMessage
from llama_index.tools import BaseTool, FunctionTool
from llama_index.prompts import PromptTemplate
from transformers import pipeline
from memory.memory_manager import MemoryManager
import torch
import requests

# Libraries for Python Execution Tool
from LLM_Agent_Utils import execute_python_code, multiply, add, duckduckgo_search, get_weather, generate_image, save_image, read_image, display_image
memory_manager = MemoryManager()

react_system_header_str = """\

You are designed to help with a variety of tasks, from answering questions \
    to providing summaries to other types of analyses.

## Tools
You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.

You have access to the following tools:
{tool_desc}

## Output Format
To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

## Additional Rules
- The answer MUST contain a sequence of bullet points that explain how you arrived at the answer. This can include aspects of the previous conversation history.
- You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.

## Current Conversation
Below is the current conversation consisting of interleaving human and assistant messages.

"""
react_system_prompt = PromptTemplate(react_system_header_str)


# Set weather API key in environment
os.environ['OPENAI_API_KEY'] = openai.api_key
os.environ['OPENAI_API_BASE'] = openai.api_base
client = OpenAI(base_url=openai.api_base, api_key=openai.api_key)

# Create a tool for executing Python code
python_execution_tool = FunctionTool.from_defaults(fn=execute_python_code)
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
search_tool = FunctionTool.from_defaults(fn=duckduckgo_search)
weather_tool = FunctionTool.from_defaults(fn=get_weather)
image_generation_tool = FunctionTool.from_defaults(fn=generate_image)
save_image_tool = FunctionTool.from_defaults(fn=save_image)
read_image_tool = FunctionTool.from_defaults(fn=read_image)
display_image_tool = FunctionTool.from_defaults(fn=display_image)

llm = OpenAI(model="gpt-4-32k-0314", api_key=openai.api_key, api_base=openai.api_base)

def call_llm(prompt: str, model: str, llm=llm, max_tokens: int = 100,  temperature: float = 0.7, top_p: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, stop: str = None) -> str:
    response = llm.complete(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, stop=stop)
    return response

generic_tool = FunctionTool.from_defaults(fn=call_llm)

agent = ReActAgent.from_tools([multiply_tool, add_tool, search_tool, weather_tool, python_execution_tool, save_image_tool, read_image_tool, image_generation_tool, display_image_tool, generic_tool], llm=llm, verbose=True)
agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})


# Streamlit code
st.title('LLM Agent')
chat_history = []

# Initialize memory manager
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = MemoryManager()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        # Add message to memory manager
        st.session_state.memory_manager.add_to_memory(message)

safety_context = """ You will be passed a LLM based response. Please ensure that the response is safe for work and does not contain any sensitive information. Remember that the LLM model is not perfect and may generate inappropriate content. Please use your best judgement and do not share any sensitive information. Only share the output as a boolean value. """

image_response_context = """ You will be passed a LLM based response. You need to check if the response is an image or not. If the response is an image, you need to respond with a boolean value. """

# User input
if prompt := st.chat_input("How can I help?"):
    context = "This is the beginning of the conversation." if st.session_state.memory_manager.get_memory() == '' else st.session_state.memory_manager.get_memory()

    input_context = json.dumps(context)
    # Limit the context to the most recent 2048 tokens
    input_context = input_context[-2048:]

    # If context is provided in the request, use it; otherwise, use the memory context
    final_context = input_context + ' ' + prompt if context else ''

    # Display user message in chat message container
    update_details = {"role": "user", "content": final_context + ' ' + prompt}
    message_details = {"role": "user", "content":  prompt}
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append( message_details )
    chat_history.append( update_details )

    try:
        response = agent.chat(final_context + ' ' + prompt)
        response_generated = response.response
    except Exception as e:
        response = call_llm(final_context + ' ' + prompt, "gpt-4-32k-0314", llm=llm, max_tokens=1024, temperature=0.3, top_p=0.1, frequency_penalty=0.0, presence_penalty=0.0, stop=None)
        response_generated = response.text
    safety_prompt = f" The response from the LLM model is as below separated by four backticks: ````{response}````. Is the response safe for work and does not contain any sensitive information? Respond only with a boolean value."
    image_prompt = f" The response from the LLM model is as below separated by four backticks: ````{response}````. Is the response an image? Respond only with a boolean value."

    if agent.chat(image_response_context + ' ' + image_prompt) == True:
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown("The response is an image.")
                st.image(response)

    if agent.chat(safety_context + ' ' + safety_prompt) == False:
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown("Sorry, I cannot answer your query. As this a safety concern, I cannot provide an answer. Please ask a different question.")
    else:
        if agent.chat(image_response_context + ' ' + image_prompt) == True:
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown("The response is an image.")
                st.image(response)
        else:
            with st.chat_message("assistant"):
                st.markdown(response)

    response_append_format = {"role": "assistant", "content": response_generated}
    st.session_state.messages.append(response_append_format)
    chat_history.append(response_append_format)
    st.session_state.memory_manager.add_to_memory({'Query' : update_details , 'response': response_append_format})
