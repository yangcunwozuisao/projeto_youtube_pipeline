import autogen
from config import config_list
from tools.run_collect import run_collect

collector_agent = autogen.AssistantAgent(
    name="CollectorAgent",
    llm_config={"config_list": config_list},
    system_message="""
Você coleta vídeos do YouTube usando a API.

Sua função:
- buscar vídeos
- salvar metadados
- criar o dataset inicial
"""
)

collector_agent.register_function(
    function_map={
        "collect_videos": run_collect
    }
)
