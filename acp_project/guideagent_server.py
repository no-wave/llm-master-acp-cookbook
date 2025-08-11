
from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool, ToolCallingAgent, ToolCollection
from mcp import StdioServerParameters

server = Server()

model = LiteLLMModel(
    model_id="openai/gpt-4o-mini",  
    max_tokens=2048
)

server_parameters = StdioServerParameters(
    command="uv",
    args=["run", "mcpserver.py"],
    env=None,
)

@server.agent()
async def guide_search_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """응급환자 발생시 관련 질문을 검색하여 지원하는 Agent다. 교내의 응급 발생 시 교직원과 학생이 응급 처치 방법을 검색하여 찾는 데 사용할 수 있다."""
    agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()], model=model)

    prompt = input[0].parts[0].content
    response = agent.run(prompt)

    yield Message(parts=[MessagePart(content=str(response))])

@server.agent()
async def hospital_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """이것은 사용자가 응급환자 발생 시 근처 병원을 찾는 데 도움을 주는 에이전트다."""
    with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        agent = ToolCallingAgent(tools=[*tool_collection.tools], model=model)
        prompt = input[0].parts[0].content
        response = agent.run(prompt)

        yield Message(parts=[MessagePart(content=str(response))])

if __name__ == "__main__":
    server.run(port=8001)
