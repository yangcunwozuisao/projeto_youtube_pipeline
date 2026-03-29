import autogen
from config import config_list
from tools.run_brand import run_brand

brand_agent = autogen.AssistantAgent(
    name="BrandAgent",
    llm_config={"config_list": config_list},
    system_message="""
Você extrai entidades de marca dos vídeos.

Suas tarefas:
- identificar marcas de smartphones
- detectar possíveis modelos
"""
)

brand_agent.register_function(
    function_map={
        "run_brand": run_brand
    }
)
