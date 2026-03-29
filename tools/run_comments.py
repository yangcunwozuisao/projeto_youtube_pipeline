from pipelines import yt_collect_comments
from pipelines import comments_sentiment

def run_comments():
    yt_collect_comments.main()
    comments_sentiment.main()
    return "Analizacao dos comentarios finalizado"
