
from colorama import Fore
from mcp.server.fastmcp import FastMCP
import json 
import requests

mcp = FastMCP("hospitalserver")

# 서버 함수 빌드
@mcp.tool()
def list_hospital(town:str) -> str:
    """이 도구는 특정 동(town)에 있는 병원 목록을 반환합니다.
    Args:
        town: 검색할 동 이름 (예: "역삼동").

    Returns:
        str: 해당 동에 위치한 병원 목록
        """

    url = 'https://raw.githubusercontent.com/no-wave/llm-master-acp-cookbook/main/hospital_info.json'

    resp = requests.get(url)
    hospitals = json.loads(resp.text)

    # BUG FIX: docstring의 매개변수명(state->town)을 실제 함수와 일치시켰습니다.
    matches = [hosp for hosp in hospitals.values() if hosp['address']['town'] == town]    
    return str(matches) 

# 파일이 실행될 경우 서버 시작
if __name__ == "__main__":
    mcp.run(transport="stdio")
