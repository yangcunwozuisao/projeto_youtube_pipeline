import autogen
from config import config_list
from tools.run_spam import run_spam

spam_agent = autogen.AssistantAgent(
    name="SpamAgent",
    llm_config={"config_list": config_list},
    system_message="""
Você executa detecção de spam e bots nos comentários do YouTube.

Suas tarefas:
- analisar comentários coletados
- detectar sinais de spam
- classificar comentários como legítimo, suspeito ou spam
"""
)

spam_agent.register_function(
    function_map={
        "run_spam": run_spam
    }
)