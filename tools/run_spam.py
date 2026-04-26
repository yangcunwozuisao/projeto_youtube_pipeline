from pipelines.spam_detect import main as spam_main
import sys

def run_spam(use_embeddings: bool = True):
    spam_main(use_embeddings=use_embeddings)
    return "Detecção de spam finalizada"