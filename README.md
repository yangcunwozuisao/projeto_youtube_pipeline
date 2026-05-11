# YouTube Analytics Pipeline

Sistema de coleta, transcrição e análise de vídeos do YouTube com NLP, modelagem de tópicos, detecção de spam e exportação para BI. Orquestrado via agentes AutoGen.

---

## Índice

- [Visão Geral](#visão-geral)
- [Arquitetura](#arquitetura)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Etapas do Pipeline](#etapas-do-pipeline)
- [Saídas Geradas](#saídas-geradas)
- [Agentes AutoGen](#agentes-autogen)
- [Bugs Conhecidos](#bugs-conhecidos)

---

## Visão Geral

O pipeline realiza 9 etapas sequenciais:

```
Coleta → Filtro → ASR → NLP → Tópicos → Comentários → Marcas → Analytics → Spam
```

Cada etapa é idempotente: se já foi executada com sucesso, é pulada automaticamente via `pipeline_state.json`. Os resultados são persistidos em CSV e sincronizados com um banco MongoDB.

---

## Arquitetura

```
main.py                          # Orquestrador principal (execução sequencial)
├── tools/                       # Wrappers de cada etapa
│   ├── run_collect.py
│   ├── run_filter.py
│   ├── run_asr.py
│   ├── run_nlp.py
│   ├── run_topics.py
│   ├── run_comments.py
│   ├── run_brand.py
│   ├── run_analytics.py
│   └── run_spam.py
├── pipelines/                   # Implementação de cada etapa
│   ├── yt_collect_videos.py
│   ├── step_eda.py / step_filter.py
│   ├── asr_whisper.py
│   ├── nlp_stage.py
│   ├── topics_bertopic.py
│   ├── yt_collect_comments.py / comments_sentiment.py
│   ├── brand_entity_extract.py
│   ├── make_aggregates.py / exports_bi.py
│   └── spam_detect.py
├── agents/                      # Agentes AutoGen (orquestração multi-agente)
│   ├── manager_agent.py
│   ├── collector_agent.py
│   ├── filter_agent.py
│   ├── asr_agent.py
│   ├── nlp_agent.py
│   ├── topic_agent.py
│   ├── comment_agent.py
│   ├── brand_agent.py
│   └── analytics_agent.py
├── shared_models.py             # Carregamento lazy dos modelos ML
├── state_manager.py             # Controle de estado do pipeline
├── db.py                        # Conexão singleton com MongoDB
├── mongo_writer.py              # Escrita bulk no MongoDB
├── config.py                    # Configuração do LLM (AutoGen)
└── outputs/                     # CSVs gerados pelo pipeline
```

---

## Pré-requisitos

- Python 3.9+
- MongoDB rodando localmente (ou via `MONGO_URI`)
- ffmpeg (ou `imageio-ffmpeg`)
- Chave de API do YouTube Data API v3
- Ollama com modelo `qwen2` (para os agentes AutoGen)

---

## Instalação

```bash
# Clonar o repositório
git clone <repo-url>
cd youtube-pipeline

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Instalar dependências
pip install -r requirements.txt
```

Dependências principais:

```
google-api-python-client
yt-dlp
openai-whisper
imageio-ffmpeg
pandas
pymongo
sentence-transformers
transformers
keybert
bertopic
umap-learn
hdbscan
gliner
pyautogen
filelock
tqdm
```

---

## Configuração

### Chave da API do YouTube

A chave é lida na seguinte ordem de prioridade:

1. Variável de ambiente: `export YT_API_KEY="sua_chave"`
2. Arquivo `.env` na raiz: `YT_API_KEY=sua_chave`
3. Arquivo `yt_api_key.txt` contendo apenas a chave

### MongoDB

Por padrão conecta em `mongodb://localhost:27017/youtube_db`. Para sobrescrever:

```bash
export MONGO_URI="mongodb://usuario:senha@host:27017/"
export MONGO_DB="nome_do_banco"
```

### Número de vídeos para ASR

Controla quantos vídeos são transcritos (padrão: 20):

```bash
export ASR_TOP_N=50
```

### LLM para os agentes (AutoGen)

Edite `config.py` com o modelo e endpoint desejados:

```python
config_list = [
    {
        "model": "qwen2",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama"
    }
]
```

---

## Uso

### Execução completa do pipeline

```bash
python main.py
```

### Resetar e re-executar uma etapa específica

```python
from state_manager import reset
reset("nlp")          # Re-executa apenas a etapa NLP
reset()               # Re-executa todo o pipeline
```

### Executar uma etapa isoladamente

```bash
python -m pipelines.asr_whisper
python -m pipelines.nlp_stage
python -m pipelines.spam_detect --no-embeddings   # Spam apenas com regras (mais rápido)
```

### Verificar transcrições

```bash
python check_transcripts.py
```

### Testar conexão com MongoDB

```bash
python test_mongo.py
python test_query.py
```

---

## Estrutura do Projeto

```
.
├── main.py
├── config.py
├── db.py
├── shared_models.py
├── state_manager.py
├── mongo_writer.py
├── utils_incremental.py
├── pipeline_state.json          # Criado automaticamente
├── yt_api_key.txt               # Opcional (ver Configuração)
├── .env                         # Opcional (ver Configuração)
├── outputs/                     # CSVs gerados
│   ├── videos.csv
│   ├── videos_clean.csv
│   ├── videos_filtered.csv
│   ├── videos_top50.csv
│   ├── transcripts.csv
│   ├── dataset_nlp.csv
│   ├── dataset_topics.csv
│   ├── topics_overview.csv
│   ├── comments.csv
│   ├── dataset_enriched.csv
│   ├── dataset_brands.csv
│   ├── comments_spam.csv
│   ├── agg_channel.csv
│   ├── bi_fato_videos.csv
│   ├── bi_agg_channel.csv
│   └── bi_agg_brand.csv
├── data/
│   └── audio/                   # Áudios baixados pelo ASR
├── tools/
├── pipelines/
└── agents/
```

---

## Etapas do Pipeline

### Step 1 — Coleta de vídeos (`yt_collect_videos.py`)

Busca vídeos via YouTube Data API v3 para os mercados BR, US e GLOBAL usando queries de review/unboxing de smartphones. Suporta coleta incremental (retoma a partir do último vídeo no banco). Salva em `outputs/videos.csv`.

### Step 2 — Filtro e EDA (`step_eda.py` + `step_filter.py`)

Converte duração ISO 8601 para segundos, remove vídeos com menos de 30 segundos. Gera `videos_clean.csv`, `videos_filtered.csv` e `videos_top50.csv` (top 50 por visualizações).

### Step 3 — Transcrição ASR (`asr_whisper.py`)

Baixa o áudio via `yt-dlp` e transcreve com OpenAI Whisper (modelo `base`). Reutiliza áudios já baixados e pula vídeos já transcritos. Processa os `ASR_TOP_N` vídeos mais vistos. Salva em `outputs/transcripts.csv`.

### Step 4 — NLP (`nlp_stage.py`)

Para cada transcrição: extrai keywords com KeyBERT e calcula sentimento com `cardiffnlp/twitter-xlm-roberta-base-sentiment`. Salva em `outputs/dataset_nlp.csv`.

### Step 5 — Modelagem de tópicos (`topics_bertopic.py`)

Gera embeddings com SentenceTransformer e clusteriza com BERTopic (UMAP + HDBSCAN). Fallback para KMeans em datasets pequenos. Salva em `outputs/dataset_topics.csv` e `outputs/topics_overview.csv`.

### Step 6 — Comentários (`yt_collect_comments.py` + `comments_sentiment.py`)

Coleta até 200 comentários por vídeo via API, seleciona o top comment por likes e calcula sentimento. Enriquece o dataset base gerando `outputs/dataset_enriched.csv`.

### Step 7 — Extração de marcas (`brand_entity_extract.py`)

Combina regras de regex com NER (GLiNER `gliner_multi-v2.1`) para identificar marca, série e modelo do smartphone. Salva em `outputs/dataset_brands.csv`.

### Step 8 — Analytics e BI (`make_aggregates.py` + `exports_bi.py`)

Gera agregações por canal e marca, e a tabela fato principal para dashboards. Salva em `outputs/bi_fato_videos.csv`, `bi_agg_channel.csv` e `bi_agg_brand.csv`.

### Step 9 — Detecção de spam (`spam_detect.py`)

Estratégia híbrida: sinais de regra (URLs, keywords promocionais, duplicatas exatas, ratio de maiúsculas, densidade de emoji) combinados com similaridade de embeddings para detectar quase-duplicatas. Classifica comentários como `legítimo`, `suspeito` ou `spam`. Salva em `outputs/comments_spam.csv`.

---

## Saídas Geradas

| Arquivo | Conteúdo |
|---|---|
| `videos.csv` | Metadados brutos dos vídeos |
| `videos_top50.csv` | Top 50 vídeos por visualizações |
| `transcripts.csv` | Transcrições ASR (videoId, language, text) |
| `dataset_nlp.csv` | Vídeos + keywords + sentimento |
| `dataset_topics.csv` | Vídeos + topic_id + topic_repr |
| `dataset_enriched.csv` | Dataset completo + top comment + sentimento do comentário |
| `dataset_brands.csv` | Dataset completo + brand_primary + brand_series + model_hint |
| `comments_spam.csv` | Comentários + spam_score + spam_label + spam_signals |
| `bi_fato_videos.csv` | Tabela fato consolidada para BI |
| `bi_agg_channel.csv` | Agregações por canal |
| `bi_agg_brand.csv` | Agregações por marca |

---

## Agentes AutoGen

Os agentes em `agents/` encapsulam cada etapa do pipeline para uso em modo multi-agente. Cada agente é um `AssistantAgent` com a função da sua etapa registrada.

> **Atenção:** a orquestração multi-agente (entrada `UserProxyAgent` + `GroupChat`) ainda não está implementada. Para execução completa, use `main.py` diretamente.

---
