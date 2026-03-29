import autogen
from config import config_list
from tools.run_comments import run_comments

comment_agent = autogen.AssistantAgent(
    name="CommentAgent",
    llm_config={"config_list": config_list},
    system_message="""
Você coleta e analisa comentários do YouTube.

Suas tarefas:
- coletar comentários
- executar análise de sentimento
"""
)

comment_agent.register_function(
    function_map={
        "run_comments": run_comments
    }
)
