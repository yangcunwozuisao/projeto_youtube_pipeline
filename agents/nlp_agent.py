import autogen
from config import config_list
from tools.run_nlp import run_nlp

nlp_agent = autogen.AssistantAgent(
    name="NLPAgent",
    llm_config={"config_list": config_list},
    system_message="""
Você executa análise NLP nas transcrições.

Suas tarefas:
- extrair palavras-chave
- análise de sentimento
"""
)

nlp_agent.register_function(
    function_map={
        "run_nlp": run_nlp
    }
)
