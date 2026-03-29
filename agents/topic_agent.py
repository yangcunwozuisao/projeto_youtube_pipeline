import autogen
from config import config_list
from tools.run_topics import run_topics

topic_agent = autogen.AssistantAgent(
    name="TopicAgent",
    llm_config={"config_list": config_list},
    system_message="""
Você executa modelagem de tópicos usando BERTopic.

Seu objetivo:
- identificar temas principais
- agrupar vídeos por tópico
"""
)

topic_agent.register_function(
    function_map={
        "run_topics": run_topics
    }
)
