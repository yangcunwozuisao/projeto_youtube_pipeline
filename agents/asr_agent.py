import autogen
from config import config_list
from tools.run_asr import run_asr

asr_agent = autogen.AssistantAgent(
    name="ASRAgent",
    llm_config={"config_list": config_list},
    system_message="""
Você executa transcrição automática de áudio (ASR).

Use Whisper para:
- baixar áudio
- transcrever vídeos
"""
)

asr_agent.register_function(
    function_map={
        "run_asr": run_asr
    }
)
