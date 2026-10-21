# pip install "flights==0.9.0" "mcp[cli]<2" fastmcp pydantic-settings smolagents[mcp] smolagents[openai] ddgs
# MCP - https://github.com/flightclaw/agents
#     - https://github.com/punitarani/fli
import os

from mcp import StdioServerParameters
from smolagents import FinalAnswerStep, MCPClient, PlanningStep, ToolCallingAgent, CodeAgent, OpenAIServerModel

airbnb_params = StdioServerParameters(command="npx", args=[ "-y", "@openbnb/mcp-server-airbnb"])
# Path to where your fli-mcp is installed
fli_params = StdioServerParameters(command="./.venv/bin/fli-mcp")

airbnb_client = MCPClient(airbnb_params, structured_output=True)
fli_client = MCPClient(fli_params, structured_output=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = "gpt-4.1-mini"
model = "gpt-5.6-luna"

for t in airbnb_client.get_tools():
    print(t.name)
for t in fli_client.get_tools():
    print(t.name)

# Handle review
def review_plan(step: PlanningStep, agent: ToolCallingAgent) -> None:
    print(f"*** Proposed plan: {step.plan}")
    stop = False
    while not stop:
        choice = input("[Aa]pprove, [Ee]dit, [Cc]ancel: ").strip().lower()
        stop = choice in {"a", "e", "c"}
        if "e" == choice:
            print("Write your new plan. Enter a new empty line to end")
            lines = iter(input, "")
            step.plan += "## Update\n\n" + "\n".join(lines)
        elif "c" == choice:
            raise Exception("User rejected this plan")

# reasoning_effort = none, low, medium, high, xhigh
llm_model = OpenAIServerModel(model_id=model, api_key=OPENAI_API_KEY, reasoning_effort='none')

# True either ToolCallingAgent or CodeAgent
agent = ToolCallingAgent(
        tools=[ *airbnb_client.get_tools(), *fli_client.get_tools() ],
        model=llm_model, 
        max_tool_threads=2, max_steps=15, 
        add_base_tools=True, 
        planning_interval=5, 
        step_callbacks={PlanningStep: review_plan})
# agent = CodeAgent(
#         tools=[ *airbnb_client.get_tools(), *flightclaw_client.get_tools() ],
#         model=llm_model, 
#         add_base_tools=True, 
#         planning_interval=2, 
#         step_callbacks={PlanningStep: review_plan})

prompt = """
I would like to plan a 5 day trip to Tokyo. I would like to depart from Singapore next Sunday and return the following Saturday.
Include an overnight stay in Hakone. 
Give me all possible flights from Singapore to Tokyo and return. Do not include any 'budget' airline.
Look for 2 possible accomodations. I do not want to pay more than SGD130 per night for the accomodation.
The trip is for 2 adults.
"""
result = agent.run(prompt)

print("\n=============================================\n")
i = 0
for step in agent.memory.steps:
    if isinstance(step, PlanningStep):
        i += 1
        print(f'**** STEP: {i}')
        print("Step: {step.plan}")
        print(f"Token: {step.token_usage}")
    elif isinstance(step, FinalAnswerStep):
        print(f'**** STEP: {i}')
        print(f"Output: {step.output}")

airbnb_client.disconnect()
fli_client.disconnect()


