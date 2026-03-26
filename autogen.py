import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
import logging

logging.basicConfig(
    level=logging.INFO,   # or DEBUG
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


load_dotenv()

async def main() -> None:
    # Initializing the model client for 2026's latest models
    # The client now handles specific model IDs and API versions natively
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Creating specialized agents with distinct domain expertise
    # The triage agent acts as a router using its internal intelligence
    triage_agent = AssistantAgent(
        name="triage_specialist",
        model_client=model_client,
        system_message="""You are an expert technical triage agent. 
        Evaluate the user's issue and identify if it is a 'Hardware' or 'Software' problem.
        Once identified, call upon the respective specialist. 
        Do not solve the issue yourself."""
    )

    hardware_specialist = AssistantAgent(
        name="hardware_specialist",
        model_client=model_client,
        system_message="You are a hardware expert. You solve problems related to physical components."
    )

    software_specialist = AssistantAgent(
        name="software_specialist",
        model_client=model_client,
        system_message="You are a software engineer. You solve problems related to code, OS, and apps."
    )

    # UserProxy represents the human participant and can execute code if needed
    user_proxy = UserProxyAgent(name="user_proxy")

    # Defining termination conditions to prevent infinite loops
    # The conversation ends if "TERMINATE" is mentioned or after 10 messages
    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)

    # Orchestrating the team using a high-level Team abstraction
    # This replaces the older GroupChatManager from version 0.2
    team = RoundRobinGroupChat(
        [triage_agent, hardware_specialist, software_specialist, user_proxy],
        termination_condition=termination
    )

    # Executing the task asynchronously with a complex, ambiguous prompt
    task = "My screen is flickering when I try to compile C++ code on my laptop."
    
    # Using the Console UI utility to stream the conversation in real-time
    await Console(team.run_stream(task=task))

if __name__ == "__main__":
    asyncio.run(main())