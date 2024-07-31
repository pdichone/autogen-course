import os
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

model = "gpt-3.5-turbo"
llm_config = {
    "model": model,
    "temperature": 0.9,
    "api_key": os.environ["OPENAI_API_KEY"],
}

traveler_agent = ConversableAgent(
    name="Traveler_Agent",
    system_message="You are a traveler planning a vacation.",
    llm_config=llm_config,
)

guide_agent = ConversableAgent(
    name="Guide_Agent",
    system_message="You are a travel guide with extensive knowledge about popular destinations.",
    llm_config=llm_config,
)

chat_result = traveler_agent.initiate_chat(
    guide_agent,
    message="What are the must-see attractions in Tokyo?",
    summary_method="reflection_with_llm",  # reflection_with_llm, reflection, llm -- see above explanations
    max_turns=2,
)

# print(chat_result)

print(" \n ***Chat Summary***: \n")
# summary is a property of the chat result
print(chat_result.summary)

print(" \nDefault Input Prompt: \n")
# The input prompt for the LLM is the following default prompt:
print(ConversableAgent.DEFAULT_SUMMARY_PROMPT)

# Get the chat history.
import pprint

print(" \nChat history: \n")
pprint.pprint(chat_result.chat_history)

print(" \n**Chat Cost**: \n")
# Get the cost of the chat.
pprint.pprint(chat_result.cost)
