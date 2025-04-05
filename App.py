import streamlit as st
import pickle
import pandas as pd

# モデルとベクトライザの読み込み
with open("models/svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# タイトル
st.title("🧠 MBTI 性格タイプ予測アプリ")
st.write("あなたの投稿テキストから、MBTIタイプ（16タイプ）を予測します。")

# 入力欄
user_input = st.text_area("✏️ 投稿テキストを入力してください（英語）", height=200)

# 予測ボタン
if st.button("🚀 予測する"):
    if user_input.strip() == "":
        st.warning("⚠️ テキストを入力してください。")
    else:
        # 入力文をベクトル化して予測
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        st.success(f"🎯 予測されたMBTIタイプは **{prediction}** です！")
