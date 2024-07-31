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


# Define travel planner functions
def calculate_travel_time(
    distance: Annotated[int, "Distance in kilometers"],
    speed: Annotated[int, "Speed in km/h"],
) -> str:
    travel_time = distance / speed
    return f"At a speed of {speed} km/h, it will take approximately {travel_time:.2f} hours to travel {distance} kilometers."


def convert_currency(
    amount: Annotated[float, "Amount in USD"],
    rate: Annotated[float, "Exchange rate to EUR"],
) -> str:
    converted_amount = amount * rate
    return f"${amount} USD is approximately €{converted_amount:.2f} EUR."


def suggest_activity(location: Annotated[str, "Location"]) -> str:
    activities = {
        "Paris": "Visit the Eiffel Tower and the Louvre Museum.",
        "New York": "See Times Square and Central Park.",
        "Tokyo": "Explore the Shibuya Crossing and the Senso-ji Temple.",
    }
    return activities.get(location, f"No specific activities found for {location}.")


# Define the assistant agent that suggests tool calls.
assistant = AssistantAgent(
    name="TravelPlannerAssistant",
    system_message="You are a helpful AI travel planner. Return 'TERMINATE' when the task is done.",
    llm_config=llm_config,
)

# The user proxy agent is used for interacting with the assistant agent and executes tool calls.
user_proxy = ConversableAgent(
    name="User",
    is_termination_msg=lambda msg: msg.get("content") is not None
    and "TERMINATE" in msg["content"],
    human_input_mode="TERMINATE",
)

# Register the tool signatures with the assistant agent.
assistant.register_for_llm(
    name="calculate_travel_time",
    description="Calculate travel time based on distance and speed",
)(calculate_travel_time)
assistant.register_for_llm(
    name="convert_currency", description="Convert USD to EUR based on exchange rate"
)(convert_currency)
assistant.register_for_llm(
    name="suggest_activity", description="Suggest activities for a specific location"
)(suggest_activity)


# Register the tool functions with the user proxy agent.
user_proxy.register_for_execution(name="calculate_travel_time")(calculate_travel_time)
user_proxy.register_for_execution(name="convert_currency")(convert_currency)
user_proxy.register_for_execution(name="suggest_activity")(suggest_activity)

# Example conversation with the assistant
user_proxy.initiate_chat(
    assistant, message="I am planning a trip to Paris. What should I do there?"
)
