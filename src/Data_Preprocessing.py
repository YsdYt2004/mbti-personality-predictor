import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.utils import load_config
import os

# 必要なNLTKデータをダウンロード
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


def clean_text(text, stop_words):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)         # URL削除
    text = re.sub(r"[^a-z\s]", "", text)               # 英字・空白以外削除
    tokens = word_tokenize(text)                       # トークン化
    tokens = [t for t in tokens if t not in stop_words]  # ストップワード除去
    return " ".join(tokens)


def preprocess_data():
    config = load_config()
    input_path = "data/mbti_1.csv"
    output_path = "data/mbti_cleaned.csv"

    df = pd.read_csv(input_path)
    stop_words = set(stopwords.words(config['preprocessing']['stopwords']))

    df["cleaned_posts"] = df["posts"].apply(lambda x: clean_text(x, stop_words))
    df.to_csv(output_path, index=False)
    print(f"✅ 前処理完了: {output_path}")


if __name__ == "__main__":
    preprocess_data()