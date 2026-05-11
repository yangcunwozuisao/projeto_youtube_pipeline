import autogen

from agents.manager_agent import manager
from agents.collector_agent import collector_agent
from agents.filter_agent import filter_agent
from agents.asr_agent import asr_agent
from agents.nlp_agent import nlp_agent
from agents.topic_agent import topic_agent
from agents.comment_agent import comment_agent
from agents.brand_agent import brand_agent
from agents.analytics_agent import analytics_agent
from agents.spam_agent import spam_agent


user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    system_message="""
Você executa funções registradas pelos agentes quando solicitado.
Siga a ordem do pipeline e finalize quando todas as etapas forem concluídas.
"""
)


groupchat = autogen.GroupChat(
    agents=[
        user_proxy,
        manager,
        collector_agent,
        filter_agent,
        asr_agent,
        nlp_agent,
        topic_agent,
        comment_agent,
        brand_agent,
        analytics_agent,
        spam_agent,
    ],
    messages=[],
    max_round=30,
)


group_manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=manager.llm_config,
)


if __name__ == "__main__":
    user_proxy.initiate_chat(
        group_manager,
        message="""
Execute o pipeline completo na ordem correta:

1. Coletar vídeos do YouTube
2. Limpar e filtrar o dataset
3. Transcrever áudio dos vídeos
4. Executar análise NLP
5. Executar modelagem de tópicos
6. Coletar e analisar comentários
7. Extrair marcas de smartphones
8. Gerar agregações e exportação para BI
9. Executar detecção de spam / bots nos comentários
"""
    )