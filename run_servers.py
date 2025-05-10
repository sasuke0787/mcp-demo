import subprocess

def run_flask_api():
    """Run the Flask API."""
    return subprocess.Popen(["python", "app.py"])

def run_mcp_server():
    """Run the MCP server using fastmcp with SSE transport on port 8080."""
    return subprocess.Popen(["fastmcp", "run", "mcp_calculator_server.py:mcp", "--transport", "sse", "--port", "8080"])

if __name__ == "__main__":
    print("Starting Flask API and MCP server...")
    flask_process = run_flask_api()
    mcp_process = run_mcp_server()

    try:
        flask_process.wait()
        mcp_process.wait()
    except KeyboardInterrupt:
        print("Shutting down servers...")
        flask_process.terminate()
        mcp_process.terminate()