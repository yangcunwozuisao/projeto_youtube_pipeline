Pipeline automatizado para mineração multimodal de vídeos de unboxing no YouTube

Projeto em Python para coleta, transcrição, processamento e análise de vídeos de unboxing/review no YouTube, com foco em inteligência de mercado aplicada ao varejo de importação.

O pipeline integra coleta de vídeos e comentários, transcrição automática de fala, extração de palavras-chave, análise de sentimentos, modelagem de tópicos, extração de marcas, persistência em MongoDB, visualização em dashboard e uma camada experimental multiagente.

Objetivo

Automatizar a mineração de dados de vídeos de unboxing e reviews no YouTube para gerar indicadores relevantes de:

sentimento
tópicos recorrentes
marcas, séries e modelos
engajamento
comentários do público
spam e comportamento suspeito em comentários

A proposta busca transformar dados públicos e não estruturados em informações úteis para análise de mercado.

Principais funcionalidades
Coleta de vídeos com YouTube Data API
Coleta incremental para evitar retrabalho e reduzir consumo de quota
Enriquecimento de canais com métricas adicionais
Filtragem e seleção de vídeos mais relevantes
Extração de áudio com yt-dlp e FFmpeg
Transcrição automática com Whisper
Extração de palavras-chave com KeyBERT
Análise de sentimentos com XLM-RoBERTa
Modelagem de tópicos com BERTopic
Extração híbrida de marcas, séries e modelos com regras + GLiNER
Coleta e análise de comentários
Detecção de spam e bots em comentários
Persistência em MongoDB com upsert
Dashboard analítico em Streamlit
Camada experimental multiagente com AutoGen
Arquitetura geral

O pipeline foi organizado em etapas modulares:

Coleta de vídeos
Limpeza e filtragem
Extração de áudio
Transcrição automática
Processamento NLP
Modelagem de tópicos
Extração de marcas
Coleta e análise de comentários
Detecção de spam/bots
Persistência em MongoDB
Exportações para BI
Dashboard em Streamlit
Camada experimental multiagente
Tecnologias utilizadas
Linguagem e ambiente
Python 3.11
Coleta e mídia
YouTube Data API v3
google-api-python-client
yt-dlp
FFmpeg
NLP / ML
Whisper
KeyBERT
sentence-transformers
XLM-RoBERTa
BERTopic
GLiNER
scikit-learn
pandas
numpy
Persistência e visualização
MongoDB
pymongo
Streamlit
Plotly
Automação experimental
AutoGen
Estrutura esperada do projeto
projeto_youtube_pipeline/
│
├── data/
│   └── audio/
│
├── outputs/
│   ├── videos.csv
│   ├── videos_clean.csv
│   ├── videos_top50.csv
│   ├── transcripts.csv
│   ├── dataset_nlp.csv
│   ├── dataset_topics.csv
│   ├── topics_overview.csv
│   ├── dataset_brands.csv
│   ├── comments.csv
│   ├── dataset_enriched.csv
│   ├── comments_spam.csv
│   ├── bi_fato_videos.csv
│   ├── bi_agg_channel.csv
│   └── bi_agg_brand.csv
│
├── tools/
│   ├── run_collect.py
│   ├── run_filter.py
│   ├── run_asr.py
│   ├── run_nlp.py
│   ├── run_topics.py
│   ├── run_brand.py
│   ├── run_comments.py
│   ├── run_analytics.py
│   └── utils_incremental.py
│
├── agents/
│   ├── collector_agent.py
│   ├── filter_agent.py
│   ├── asr_agent.py
│   ├── nlp_agent.py
│   ├── topic_agent.py
│   ├── brand_agent.py
│   ├── comment_agent.py
│   ├── analytics_agent.py
│   └── manager_agent.py
│
├── dashboard.py
├── main.py
├── db.py
├── mongo_writer.py
├── state_manager.py
├── spam_detect.py
└── README.md
Pré-requisitos

Antes de executar o projeto, é necessário ter instalado:

Python 3.11
FFmpeg
MongoDB local ou remoto
Git
ambiente virtual configurado

Também é necessário possuir uma chave da YouTube Data API.

Instalação
1. Clonar o projeto
git clone <url-do-repositorio>
cd projeto_youtube_pipeline
2. Criar e ativar ambiente virtual
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / Mac
python3 -m venv .venv
source .venv/bin/activate
3. Instalar dependências
pip install -r requirements.txt
4. Configurar variáveis de ambiente

Crie um arquivo .env com os dados necessários:

YT_API_KEY=YOUR_YOUTUBE_API_KEY
MONGO_URI=mongodb://localhost:27017/
ASR_TOP_N=20

Se houver uso de modelos ou serviços adicionais, inclua as demais variáveis necessárias no seu ambiente.

Como executar
Execução completa do pipeline
python main.py

O main.py orquestra as etapas principais do projeto.

Execução modular

Se quiser rodar etapas separadamente, use os wrappers run_*:

python -m tools.run_collect
python -m tools.run_filter
python -m tools.run_asr
python -m tools.run_nlp
python -m tools.run_topics
python -m tools.run_brand
python -m tools.run_comments
python -m tools.run_analytics
Lógica do pipeline
1. Coleta de vídeos

A coleta utiliza a YouTube Data API para buscar vídeos por múltiplas consultas relacionadas a smartphones, unboxing, reviews e comparativos.

Também pode enriquecer os resultados com dados dos canais, como:

subscriberCount
channelVideoCount
channelViewCount
country
2. Coleta incremental

O sistema consulta o MongoDB para identificar a data do vídeo mais recente já armazenado e tenta coletar apenas novos registros, reduzindo repetição e economizando quota.

3. Filtragem

Após a coleta:

remove duplicados
converte duração para segundos
remove vídeos muito curtos
seleciona os vídeos mais relevantes por visualizações

A base filtrada é salva em videos_clean.csv, e a amostra principal segue para videos_top50.csv.

4. Extração de áudio

Os áudios são baixados com yt-dlp e tratados com FFmpeg.

5. Transcrição

A transcrição automática é feita com Whisper.

Saída:

outputs/transcripts.csv

Campos principais:

videoId
language
text
6. NLP

O pipeline combina metadados e transcrições para:

extrair palavras-chave com KeyBERT
classificar sentimento com XLM-RoBERTa

Saída:

outputs/dataset_nlp.csv

Campos principais:

sent_label
sent_conf
sent_value
keywords
7. Modelagem de tópicos

A modelagem de tópicos é feita com BERTopic.

Quando necessário, o pipeline pode usar fallback com KMeans para bases pequenas.

Saídas:

outputs/dataset_topics.csv
outputs/topics_overview.csv
8. Extração de marcas

A identificação de marcas, séries e modelos usa uma abordagem híbrida:

regras textuais e regex
GLiNER / NER

Saída:

outputs/dataset_brands.csv

Campos esperados:

brand_primary
brand_series
model_hint
brand_source
brand_conf
9. Comentários

O pipeline coleta comentários dos vídeos selecionados e pode enriquecer a base com:

top_comment
topc_likes
topc_label
topc_conf
topc_value

Saídas:

outputs/comments.csv
outputs/dataset_enriched.csv
10. Spam e bots

O módulo spam_detect.py faz uma análise híbrida para detectar comentários suspeitos.

Ele combina:

sinais heurísticos
similaridade textual por embeddings

Saída:

outputs/comments_spam.csv

Campos principais:

spam_score
spam_label
spam_signals

Classificações:

legítimo
suspeito
spam
11. Persistência em MongoDB

Os dados são gravados em coleções específicas, como:

videos
transcripts
nlp
topics
brands
comments

A escrita usa upsert, evitando duplicações e permitindo atualização de registros existentes.

12. Dashboard

O dashboard.py permite visualizar os resultados em Streamlit.

Principais recursos:

filtros por marca, canal e sentimento
métricas gerais
correlação entre métricas
análise de marcas e sentimentos
ranking de canais
análise de spam/bots

Executar:

streamlit run dashboard.py
13. Multiagente

Existe uma camada experimental baseada em AutoGen, composta por agentes especializados, como:

CollectorAgent
FilterAgent
ASRAgent
NLPAgent
TopicAgent
BrandAgent
CommentAgent
AnalyticsAgent
ManagerAgent

Essa camada é experimental e funciona como apoio à orquestração futura do pipeline.

Arquivos de saída mais importantes
Arquivo	Descrição
videos.csv	base bruta coletada
videos_clean.csv	base limpa
videos_top50.csv	amostra principal para etapas analíticas
transcripts.csv	transcrições
dataset_nlp.csv	NLP e sentimentos
dataset_topics.csv	tópicos
dataset_brands.csv	marcas, séries e modelos
comments.csv	comentários coletados
dataset_enriched.csv	base com enriquecimento de comentários
comments_spam.csv	comentários classificados por suspeição
bi_fato_videos.csv	exportação principal para BI
bi_agg_channel.csv	agregação por canal
bi_agg_brand.csv	agregação por marca
Controle de estado

O projeto utiliza pipeline_state.json para registrar etapas concluídas e evitar reprocessamento desnecessário.

Exemplo de uso:

se a etapa de coleta já foi concluída, ela pode ser pulada em execuções futuras
se quiser rerodar uma etapa, pode ser necessário limpar o estado correspondente
Principais limitações atuais
Dependência da quota da YouTube Data API
Custo computacional da transcrição com Whisper
Tendência de textos técnicos serem classificados como neutros
Possíveis erros em reconhecimento de marcas
Qualidade do dashboard depende da qualidade dos dados anteriores
A camada multiagente ainda é experimental
Possíveis melhorias futuras
Aumentar a base de vídeos e comentários
Validar manualmente sentimentos e entidades
Melhorar a detecção de marcas em casos ambíguos
Refinar o módulo de spam/bots
Incluir análise visual por frames
Melhorar o dashboard
Tornar a camada multiagente mais autônoma
Expandir o pipeline para outras categorias além de smartphones
Exemplo de uso acadêmico

Este projeto foi desenvolvido como parte de um Trabalho de Conclusão de Curso voltado à aplicação de técnicas de mineração multimodal, NLP e inteligência de mercado em vídeos do YouTube.
