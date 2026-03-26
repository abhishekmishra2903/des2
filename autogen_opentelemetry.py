# autogen_with_opentelemetry.py

import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
#from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

resource = Resource.create({
    "service.name": "autogen-demo"
})

trace.set_tracer_provider(TracerProvider(resource=resource))

load_dotenv()

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318/v1/traces"
)

span_processor = BatchSpanProcessor(otlp_exporter)

#span_processor = BatchSpanProcessor(ConsoleSpanExporter())

trace.get_tracer_provider().add_span_processor(span_processor)


async def main() -> None:
    with tracer.start_as_current_span("autogen_session"):

        model_client = OpenAIChatCompletionClient(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        )

        triage_agent = AssistantAgent(
            name="triage_specialist",
            model_client=model_client,
            system_message="You classify issues as Hardware or Software."
        )

        hardware_specialist = AssistantAgent(
            name="hardware_specialist",
            model_client=model_client,
            system_message="You solve hardware issues."
        )

        software_specialist = AssistantAgent(
            name="software_specialist",
            model_client=model_client,
            system_message="You solve software issues."
        )

        user_proxy = UserProxyAgent(name="user_proxy")

        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)

        team = RoundRobinGroupChat(
            [triage_agent, hardware_specialist, software_specialist, user_proxy],
            termination_condition=termination
        )

        task = "My screen flickers while compiling C++."

        with tracer.start_as_current_span("team_run"):
            await Console(team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())