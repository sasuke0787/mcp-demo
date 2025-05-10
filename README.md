# Project Setup Instructions

This project consists of three main components:
1. **Frontend**: Built with Svelte.
2. **Backend**: Flask API.
3. **MCP Server**: FastMCP server.

Follow the steps below to set up and run the project.

---

## Frontend (Svelte)

### Prerequisites
- Node.js installed on your system.

### Setup
1. Navigate to the Svelte project directory:
   ```bash
   cd svelte-chat-ui
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open the application in your browser at the URL provided by the Svelte server (typically `http://localhost:5173`).

---

## Backend (Flask API)

### Prerequisites
- Python installed on your system.

### Setup
1. Create a virtual environment (works for both MCP server and Flask API):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start both the Flask API and MCP server:
   ```bash
   python run_servers.py
   ```
3. The Flask API will be available at `http://127.0.0.1:5000`.

---

## MCP Server

### Prerequisites
- Python installed on your system.

### Setup
1. Run the MCP server:
   ```bash
   python mcp_server.py
   ```
2. Ensure the server is running and accessible.

---

## Integration

1. Ensure both the Flask API and MCP server are running.
2. Use the Svelte frontend to interact with the backend and MCP server.

---

Let me know if you encounter any issues during setup!