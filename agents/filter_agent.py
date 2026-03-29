import autogen
from config import config_list
from tools.run_filter import run_filter

filter_agent = autogen.AssistantAgent(
    name="FilterAgent",
    llm_config={"config_list": config_list},
    system_message="""
Você limpa e filtra o dataset de vídeos.

Suas tarefas incluem:
- remover duplicatas
- converter duração dos vídeos
- filtrar vídeos curtos
"""
)

filter_agent.register_function(
    function_map={
        "filter_dataset": run_filter
    }
)
