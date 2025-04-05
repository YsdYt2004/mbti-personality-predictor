# MBTI Personality Analyzer 🧠

このプロジェクトは、MBTI（16パーソナリティーズ）の分類を目的とした自然言語処理プロジェクトです。

## 使用技術

- Python
- NLP (TF-IDF / Word2Vec)
- 機械学習 (SVM)
- Webアプリ (Streamlit)

## 実行方法

```bash
streamlit run App.py


--説明書--

# 1. 前処理
python src/Data_Preprocessing.py

# 2. 特徴量抽出
python src/TF_IDF.py     # または Word2Vec.py

# 3. モデル学習
python src/SVM.py


# 4. Webアプリ起動
仮想起動 source venv/bin/activate
streamlit run App.py
