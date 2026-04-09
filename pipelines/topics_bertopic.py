from pathlib import Path
import pandas as pd

from shared_models import embedding_model

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans


def main():

    SRC = "outputs/dataset_nlp.csv"

    if not Path(SRC).exists():
        raise FileNotFoundError("dataset_nlp.csv não encontrado. Rode NLP primeiro.")

    df = pd.read_csv(SRC)

    texts = df["text_short"].fillna("").astype(str).tolist()

    if len(texts) == 0:
        raise SystemExit("Sem textos para tópicos.")

    embedder = embedding_model

    print("[info] gerando embeddings...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    N = len(texts)

    # BERTopic
    def bertopic_with_small_dataset(texts, embeddings):

        nn = max(2, min(10, N - 1)) if N > 2 else 2

        umap_model = UMAP(
            n_neighbors=nn,
            n_components=2,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
            low_memory=True,
        )

        mcs = max(2, min(5, N))

        hdb = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        topic_model = BERTopic(
            language="multilingual",
            umap_model=umap_model,
            hdbscan_model=hdb,
            min_topic_size=mcs,
            nr_topics="auto",
            calculate_probabilities=False,
            verbose=False,
        )

        topics, probs = topic_model.fit_transform(texts, embeddings)

        return topic_model, topics, probs

    # FALLBACK
    def kmeans_fallback(texts, embeddings, k=2):
        k = min(max(2, k), N)

        km = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )

        labels = km.fit_predict(embeddings)

        tmp = pd.DataFrame({
            "topic_id": labels,
            "title": df["title"].fillna("").astype(str)
            if "title" in df.columns
            else pd.Series([""] * N)
        })

        cluster_repr = {}

        for topic_id, group in tmp.groupby("topic_id"):
            titles = group["title"].tolist()

            tokens = []
            for title in titles:
                parts = [w for w in title.split() if w.strip()]
                tokens.extend(parts[:3])

            if tokens:
                rep = " ".join(tokens[:3])
            else:
                rep = f"Topic {topic_id}"

            cluster_repr[topic_id] = rep

        reps = [cluster_repr[label] for label in labels]

        return labels, reps

    # RUN
    topic_model = None

    try:

        print("[info] rodando BERTopic...")

        topic_model, topics, probs = bertopic_with_small_dataset(
            texts,
            embeddings
        )

        topic_repr = []

        for t in topics:

            if t == -1:
                topic_repr.append("Other")

            else:
                rep = topic_model.get_topic(t)
                topic_repr.append(rep[0][0] if rep else "Topic")

    except Exception as e:

        print("[warn] BERTopic falhou → fallback KMeans")
        print("[erro detalhado]:", e)

        topics, topic_repr = kmeans_fallback(texts, embeddings, k=2)

    # OUTPUT
    out = df.copy()

    out["topic_id"] = topics
    out["topic_repr"] = topic_repr

    out.to_csv(
        "outputs/dataset_topics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"dataset_topics.csv salvo com {len(out)} linhas")

    # OVERVIEW
    try:

        if topic_model is not None:

            topics_info = topic_model.get_topic_info()

            topics_info.to_csv(
                "outputs/topics_overview.csv",
                index=False,
                encoding="utf-8-sig"
            )

            print(f"topics_overview.csv salvo ({len(topics_info)} linhas)")

        else:
            raise ValueError("topic_model não disponível")

    except Exception as e:

        print("[warn] fallback overview:", e)

        ov = (
            pd.DataFrame({
                "topic_id": topics,
                "topic_repr": topic_repr
            })
            .value_counts(["topic_id", "topic_repr"])
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )

        ov.to_csv(
            "outputs/topics_overview.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("topics_overview.csv salvo (fallback)")


if __name__ == "__main__":
    main()