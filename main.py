import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_trace_processors
from agents.run import RunConfig
from langsmith.wrappers import OpenAIAgentsTracingProcessor

# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Set up LangSmith tracing
set_trace_processors([OpenAIAgentsTracingProcessor()])

# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=False  # Enable tracing
)

async def async_main():
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful Assistant.",
        model=model
    )

    result = await Runner.run(agent, "Tell me about recursion in programming.", run_config=config)
    print("\n🤖 Final Output:\n" + result.final_output)

def main():
    print("✅ Running main()...")
    asyncio.run(async_main())

if __name__ == "__main__":
    main()