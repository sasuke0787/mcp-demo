from flask import Flask, request, jsonify, render_template
from openai import AzureOpenAI
import asyncio
from fastmcp import Client
from fastmcp.client.transports import SSETransport
from dotenv import load_dotenv
import os
# Set up Flask app
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

load_dotenv()  # Load variables from .env
# Set your Azure OpenAI endpoint and API key
endpoint = os.getenv("ENDPOINT")  # Azure OpenAI endpoint
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
                    "content": "You are a helpful assistant. Here is a list of available tools, resources, and prompts:\n\n"
                               "Tools:\n"
                               "1. greet(name: str) -> str: Returns a greeting message.\n"
                               "2. add(a: int, b: int) -> int: Adds two numbers.\n\n"
                               "Resources:\n"
                               "1. data://config: Provides application configuration.\n"
                               "2. users://{user_id}/profile: Retrieves a user's profile.\n\n"
                               "Prompts:\n"
                               "1. summarize(text: str) -> list[dict]: Generates a summary of the provided text.\n\n"
                               "Based on the user's prompt, determine the appropriate tool, resource, or prompt to use and its arguments. "
                               "Return the result as a valid JSON string. Example: {\"type\": \"tool\", \"name\": \"greet\", \"arguments\": {\"name\": \"John\"}} or {\"type\": \"resource\", \"uri\": \"users://102/profile\"} or {\"type\": \"prompt\", \"name\": \"summarize\", \"arguments\": {\"text\": \"This is a sample text.\"}}",
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
        raw_response = response.choices[0].message.content
        print(f"Raw LLM response: {raw_response}")  # Debugging log
        return raw_response
    except Exception as e:
        return f"Error processing prompt: {e}"

async def interact_with_mcp_server(prompt):
    """Interact with the MCP server based on the processed prompt."""
    transport = SSETransport("http://127.0.0.1:8080/sse")  # Explicitly use SSETransport
    client = Client(transport)
    async with client:
        import json
        
        tool_info = json.loads(prompt)  # Convert JSON string response to dictionary
        if tool_info.get("type") == "tool":
            tool_name = tool_info.get("name", "default_tool")
            tool_args = tool_info.get("arguments", {})
            result = await client.call_tool(tool_name, tool_args)
        elif tool_info.get("type") == "resource":
            resource_uri = tool_info.get("uri", "")
            result = await client.read_resource(resource_uri)
        elif tool_info.get("type") == "prompt":
            prompt_name = tool_info.get("name", "default_prompt")
            prompt_args = tool_info.get("arguments", {})
            result = await client.get_prompt(prompt_name, prompt_args)
        else:
            result = {"error": "Invalid tool/resource/prompt type"}
        return result

@app.route("/process", methods=["POST"])
def process():
    """Process user input and return the response."""
    user_input = request.json.get("prompt", "")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Debugging logs
    print(f"Received user input: {user_input}")
    
    processed_prompt = loop.run_until_complete(process_prompt_with_openai(user_input))
    print(f"Processed prompt: {processed_prompt}")
    
    mcp_result = loop.run_until_complete(interact_with_mcp_server(processed_prompt))
    print(f"MCP server result: {mcp_result}")
    
    # Ensure JSON serialization
    response = {
        "processed_prompt": str(processed_prompt),
        "mcp_result": str(mcp_result)
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)