<script>
    import { onMount } from "svelte";

    let userInput = "";
    let messages = [];

    async function sendMessage() {
        if (!userInput) return;

        // Display user message
        messages = [...messages, { sender: "You", text: userInput }];

        // Send request to backend
        const response = await fetch("http://127.0.0.1:5000/process", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ prompt: userInput }),
        });

        const data = await response.json();

        // Display processed prompt and MCP server response
        messages = [
            ...messages,
            { sender: "LLM", text: `Processed Prompt: ${data.processed_prompt}` },
            { sender: "MCP Server", text: `Response: ${data.mcp_result}` },
        ];

        // Clear input field
        userInput = "";
    }
</script>

<style>
    body {
        font-family: 'Roboto', sans-serif;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        color: #fff;
    }
    #chat-container {
        width: 60%;
        max-width: 800px;
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        display: flex;
        flex-direction: column;
        align-items: stretch;
    }
    #messages {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
        background-color: #f9f9f9;
    }
    #messages div {
        margin-bottom: 10px;
        padding: 8px;
        border-radius: 6px;
    }
    #messages div:nth-child(odd) {
        background-color: #e3f2fd;
        color: #0d47a1;
    }
    #messages div:nth-child(even) {
        background-color: #ede7f6;
        color: #4a148c;
    }
    #input-container {
        display: flex;
        gap: 10px;
    }
    #input-container input {
        flex: 1;
        padding: 12px;
        border: 1px solid #ddd;
        border-radius: 8px;
        font-size: 16px;
    }
    #input-container button {
        padding: 12px 20px;
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    #input-container button:hover {
        background-color: #0056b3;
    }
</style>

<div id="chat-container">
    <h2>Chat Interface</h2>
    <div id="messages">
        {#each messages as message}
            <div><strong>{message.sender}:</strong> {message.text}</div>
        {/each}
    </div>
    <div id="input-container">
        <input
            type="text"
            bind:value={userInput}
            placeholder="Type your message here..."
        />
        <button on:click={sendMessage}>Send</button>
    </div>
</div>