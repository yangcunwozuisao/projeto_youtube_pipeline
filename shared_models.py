"""
shared_models.py — Carregamento lazy dos modelos pesados.

Uso (sem mudança nos callers):
    from shared_models import embedding_model     # carrega na 1ª vez
    from shared_models import sentiment_pipeline  # carrega na 1ª vez
    from shared_models import ner_model           # carrega na 1ª vez

Graças ao __getattr__ de módulo (PEP 562 / Python 3.7+), cada modelo
só é instanciado quando efetivamente necessário, não ao fazer o import.
"""

from __future__ import annotations

_embedding_model = None
_sentiment_pipeline = None
_ner_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("[shared_models] Carregando embedding model...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("[shared_models] Embedding model pronto.")
    return _embedding_model


def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("[shared_models] Carregando sentiment model...")
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline,
        )
        SENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL)
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
            padding="max_length",
            device=-1,
        )
        print("[shared_models] Sentiment model pronto.")
    return _sentiment_pipeline


def get_ner_model():
    global _ner_model
    if _ner_model is None:
        print("[shared_models] Carregando NER model...")
        from gliner import GLiNER
        _ner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        print("[shared_models] NER model pronto.")
    return _ner_model


def __getattr__(name: str):
    """
    Intercepta acesso a atributos de módulo não definidos, permitindo
    backward-compatibility com `from shared_models import embedding_model`.
    """
    if name == "embedding_model":
        return get_embedding_model()
    if name == "sentiment_pipeline":
        return get_sentiment_pipeline()
    if name == "ner_model":
        return get_ner_model()
    raise AttributeError(f"module 'shared_models' has no attribute {name!r}")