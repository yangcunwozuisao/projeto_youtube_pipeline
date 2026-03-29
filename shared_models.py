# shared_models.py

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

print("🧠 Loading shared models... (only once)")

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

SENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512,
    padding="max_length",
    device=-1
)

print("✅ Models loaded")