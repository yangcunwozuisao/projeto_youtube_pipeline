import autogen
from config import config_list
from tools.run_analytics import run_analytics

analytics_agent = autogen.AssistantAgent(
    name="AnalyticsAgent",
    llm_config={"config_list": config_list},
    system_message="""
Você gera análises agregadas e datasets para BI.

Suas tarefas:
- agregações por canal
- agregações por marca
- exportação para dashboards
"""
)

analytics_agent.register_function(
    function_map={
        "run_analytics": run_analytics
    }
)
