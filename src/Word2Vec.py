import pandas as pd
from gensim.models import Word2Vec
from src.utils import load_config
import os


def run_word2vec():
    config = load_config()
    df = pd.read_csv("data/mbti_cleaned.csv")
    tokenized = [s.split() for s in df['cleaned_posts']]

    model = Word2Vec(
        sentences=tokenized,
        vector_size=config['word2vec']['vector_size'],
        window=config['word2vec']['window'],
        min_count=config['word2vec']['min_count'],
        workers=config['word2vec']['workers']
    )
    model.save("models/w2v_model.model")
    print("✅ Word2Vec モデル保存完了")


if __name__ == "__main__":
    run_word2vec()
