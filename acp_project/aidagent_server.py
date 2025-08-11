
from collections.abc import AsyncGenerator
from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server

from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool

import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv

load_dotenv()

server = Server()
llm = LLM(model="openai/gpt-4o-mini", max_tokens=1024)

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4",
        }
    },
    "embedding_model": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-ada-002"
        }
    }
}
rag_tool = RagTool(config=config,
                   chunk_size=1200,
                   chunk_overlap=200,
                  )

rag_tool.add("../dataset/first_aid_manual.pdf", data_type="pdf_file")


@server.agent()
async def policy_agent(input: list[Message]) -> AsyncGenerator[RunYield, RunYieldResume]:
    """이 에이전트는 학교 내에서 응급 환자 발생 시 응급 처치에 대한 질문을 처리하며, 응급처치 가이드 문서를 기반으로 답변을 찾기 위해 RAG 패턴을 사용한다."""


    aid_agent = Agent(
        role="응급처치 가이드",
        goal="학교에서 응급 환자 발생 시 응급처치하는 방법을 가이드한다.",
        backstory="당신은 학교 내에서 응급 환자 발생 시 응급처치를 안내하는 에이전트다",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[rag_tool],
        max_retry_limit=5
    )

    task1 = Task(
         description=input[0].parts[0].content,
         expected_output = "사용자 질문에 대한 구체적인 답변",
         agent=aid_agent
    )
    crew = Crew(agents=[aid_agent], tasks=[task1], verbose=True)

    task_output = await crew.kickoff_async()
    yield Message(parts=[MessagePart(content=str(task_output))])

if __name__ == "__main__":
    server.run(port=8000)
