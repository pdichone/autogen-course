import os
from autogen import ConversableAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv

load_dotenv()

model = "gpt-3.5-turbo"
llm_config = {
    "model": model,
    "temperature": 0.9,
    "api_key": os.environ["OPENAI_API_KEY"],
}

# Define travel planning agents
flight_agent = ConversableAgent(
    name="Flight_Agent",
    system_message="You provide the best flight options for the given destination and dates.",
    llm_config=llm_config,
    description="Provides flight options.",
)

hotel_agent = ConversableAgent(
    name="Hotel_Agent",
    system_message="You suggest the best hotels for the given destination and dates.",
    llm_config=llm_config,
    description="Suggests hotel options.",
)

activity_agent = ConversableAgent(
    name="Activity_Agent",
    system_message="You recommend activities and attractions to visit at the destination.",
    llm_config=llm_config,
    description="Recommends activities and attractions.",
)

restaurant_agent = ConversableAgent(
    name="Restaurant_Agent",
    system_message="You suggest the best restaurants to dine at in the destination.",
    llm_config=llm_config,
    description="Recommends restaurants.",
)


weather_agent = ConversableAgent(
    name="Weather_Agent",
    system_message="You provide the weather forecast for the travel dates.",
    llm_config=llm_config,
    description="Provides weather forecast.",
)

# Create a Group Chat
group_chat = GroupChat(
    agents=[flight_agent, hotel_agent, activity_agent, restaurant_agent, weather_agent],
    messages=[],
    max_round=6,
)

# Create a Group Chat Manager
group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

# Initiate the chat with an initial message
chat_result = weather_agent.initiate_chat(
    group_chat_manager,
    message="I'm planning a trip to Paris for the first week of September. Can you help me plan? I will be departuring from Miami",
    summary_method="reflection_with_llm",
)
