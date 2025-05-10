from openai import AzureOpenAI

import asyncio
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from dotenv import load_dotenv
import os
# Set your Azure OpenAI endpoint and API key
# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url="https://oai-bisihtoronto-002.openai.azure.com/openai/deployments/demo/chat/completions?api-version=2025-01-01-preview")'
# openai.api_base = "https://oai-bisihtoronto-002.openai.azure.com/openai/deployments/demo/chat/completions?api-version=2025-01-01-preview"
load_dotenv()  # Load variables from .env

endpoint = os.getenv("ENDPOINT")
model_name = os.getenv("model_name")  # Model name
deployment = os.getenv("deployment")  # Deployment name
subscription_key = os.getenv("SECRET_KEY") 
api_version = os.getenv("api_version")  # API version

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

async def process_prompt_with_openai(prompt):
    """Process the user's prompt using Azure OpenAI's GPT model."""
    try:
        response = client.chat.completions.create(
        messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        }
        ],
        max_tokens=4096,
        top_p=1.0,
        model=deployment
    )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing prompt: {e}"

async def interact_with_mcp_server(prompt):
    """Interact with the MCP server based on the processed prompt."""
    transport = SSETransport("http://127.0.0.1:8080/sse")  # Explicitly use SSETransport
    client = Client(transport)
    async with client:
        print("--- Connected to MCP Server ---")
        # Example: Call the 'greet' tool dynamically based on the prompt
        tool_name = "greet"  # This should be determined dynamically based on the prompt
        tool_args = {"name": prompt}  # Example arguments
        result = await client.call_tool(tool_name, tool_args)
        return result

async def chat_interface():
    """Main chat interface."""
    print("Welcome to the Chat Interface!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat interface.")
            break

        # Process the prompt with Azure OpenAI
        processed_prompt = await process_prompt_with_openai(user_input)
        print(f"Processed Prompt: {processed_prompt}")

        # Interact with MCP server
        mcp_result = await interact_with_mcp_server(processed_prompt)
        print(f"MCP Server Response: {mcp_result}")

if __name__ == "__main__":
    asyncio.run(chat_interface())
