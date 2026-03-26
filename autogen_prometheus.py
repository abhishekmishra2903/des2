#pip install prometheus-client

import asyncio
import os
import time
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from prometheus_client import start_http_server, Counter, Histogram

# =========================
# Metrics (DEFINE AT TOP)
# =========================

REQUEST_COUNT = Counter(
    "agent_requests_total",
    "Total number of agent runs"
)

RUN_DURATION = Histogram(
    "agent_run_duration_seconds",
    "Time spent processing an agent request",
    buckets=(0.5, 1, 2, 3, 5, 8, 13)
)

load_dotenv()

async def main() -> None:
    REQUEST_COUNT.inc()

    start_time = time.time()

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    triage_agent = AssistantAgent(
        name="triage_specialist",
        model_client=model_client,
        system_message="""You are an expert technical triage agent. 
        Evaluate the user's issue and identify if it is a 'Hardware' or 'Software' problem.
        Once identified, call upon the respective specialist. 
        Do not solve the issue yourself.
        Always consider screen issues as software issues"""
    )

    hardware_specialist = AssistantAgent(
        name="hardware_specialist",
        model_client=model_client,
        system_message="You are a hardware expert."
    )

    software_specialist = AssistantAgent(
        name="software_specialist",
        model_client=model_client,
        system_message="You are a software engineer."
    )

    user_proxy = UserProxyAgent(name="user_proxy")

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)

    team = RoundRobinGroupChat(
        [triage_agent, hardware_specialist, software_specialist, user_proxy],
        termination_condition=termination
    )

    task = "My screen is flickering when I try to compile C++ code on my laptop."

    await Console(team.run_stream(task=task))

    duration = time.time() - start_time
    RUN_DURATION.observe(duration)

if __name__ == "__main__":
    # Start Prometheus metrics server on port 8000
    start_http_server(8000)
    asyncio.run(main())