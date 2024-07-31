import os
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

load_dotenv()

model = "gpt-3.5-turbo"
llm_config = {
    "model": model,
    "api_key": os.environ.get("OPENAI_API_KEY"),
}

assistant = AssistantAgent("assistant", llm_config)
user_proxy = UserProxyAgent(
    "user_proxy",
    llm_config=llm_config,
    code_execution_config={
        "workd_dir": "code_execution",
        "use_docker": False,
    },
    human_input_mode="NEVER",
)

# start the agents
user_proxy.initiate_chat(
    assistant,
    message="What is the capital of France?",
)
