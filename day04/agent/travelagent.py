# pip install "flights==0.9.0" "mcp[cli]<2" fastmcp pydantic-settings smolagents[mcp] smolagents[openai] ddgs
# MCP - https://github.com/flightclaw/agents
#     - https://github.com/punitarani/fli

import os

from mcp import StdioServerParameters
from smolagents import GradioUI, MCPClient, ToolCallingAgent, OpenAIServerModel

airbnb_params = StdioServerParameters(command="npx", args=[ "-y", "@openbnb/mcp-server-airbnb"])
# Need to install fli_
fli_params = StdioServerParameters(command="./.venv/bin/fli-mcp")

# Start MCP client
airbnb_client = MCPClient(airbnb_params, structured_output=True)
fli_client = MCPClient(fli_params, structured_output=True)
#flightclaw_client = MCPClient(flightclaw_params, structured_output=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = "gpt-5.6-luna"

# DEBUG: list the tools supported by the MCP
for t in airbnb_client.get_tools():
    print(t.name)
for t in fli_client.get_tools():
    print(t.name)

# reasoning_effort = none, low, medium, high, xhigh
# gpt-5.6-luna has reasoning built-in
llm_model = OpenAIServerModel(model_id=model, api_key=OPENAI_API_KEY, reasoning_effort='none')
agent = ToolCallingAgent(
        tools=[ *airbnb_client.get_tools(), *fli_client.get_tools() ],
        model=llm_model, 
        max_tool_threads=2, max_steps=15, 
        add_base_tools=True, 
        planning_interval=5)

ui = GradioUI(agent, reset_agent_memory=False)
ui.launch(share=False, server_name='0.0.0.0', server_port=7890)

airbnb_client.disconnect()
fli_client.disconnect()
#flightclaw_client.disconnect()

