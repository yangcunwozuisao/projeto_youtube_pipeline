import autogen
from config import config_list

manager = autogen.AssistantAgent(
    name="ManagerAgent",
    llm_config={"config_list": config_list},
    system_message="""
Você é o agente gerente de um sistema de análise de vídeos do YouTube baseado em múltiplos agentes.

Seu papel é coordenar os agentes e garantir que o pipeline seja executado na ordem correta.

Ordem do pipeline:

1. Coletar vídeos do YouTube
2. Limpar e filtrar o dataset
3. Transcrever áudio dos vídeos
4. Executar análise NLP
5. Executar modelagem de tópicos
6. Extrair marcas de smartphones
7. Coletar e analisar comentários
8. Gerar agregações e exportação para BI
"""
)
