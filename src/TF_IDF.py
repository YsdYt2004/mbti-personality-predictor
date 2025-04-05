import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import load_config
import pickle
import os


def run_tfidf():
    config = load_config()
    df = pd.read_csv("data/mbti_cleaned.csv")

    # NaN値を空文字列に置換
    df['cleaned_posts'] = df['cleaned_posts'].fillna('')

    # TF-IDFベクトル化（sparse形式のまま）
    vectorizer = TfidfVectorizer(max_features=config['tfidf']['max_features'])
    X = vectorizer.fit_transform(df['cleaned_posts'])

    # ベクトライザ保存
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # 特徴量（X）とラベル（y）をpkl保存
    with open("data/tfidf_features.pkl", "wb") as f:
        pickle.dump(X, f)

    df['type'].to_csv("data/labels.csv", index=False)
    print("✅ TF-IDF 特徴量保存完了（sparse形式）")


if __name__ == "__main__":
    run_tfidf()
