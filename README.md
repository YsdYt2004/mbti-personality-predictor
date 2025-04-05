# MBTI Personality Predictor 🧠✨

このアプリは、ユーザーが入力した文章から16タイプのMBTI性格タイプを予測するアプリです。

## 💡 機能

- 自然言語処理（NLP）による前処理
- TF-IDF による特徴量抽出
- SVMモデルによる予測
- StreamlitによるWebアプリ化

## 🚀 使用技術

- Python
- scikit-learn
- NLTK
- Streamlit

## 🛠️ セットアップ方法

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# モデルの構築
PYTHONPATH=. python src/Data_Preprocessing.py
PYTHONPATH=. python src/TF_IDF.py
PYTHONPATH=. python src/SVM.py

# Webアプリ起動
streamlit run App.py
