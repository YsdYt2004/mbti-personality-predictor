#!/bin/bash

# 仮想環境を有効化
source venv/bin/activate

# 必要ライブラリをインストール（初回だけ）
pip install -r requirements.txt

# 特徴量抽出とモデル学習
PYTHONPATH=. python src/Data_Preprocessing.py
PYTHONPATH=. python src/TF_IDF.py
PYTHONPATH=. python src/SVM.py

# Streamlit 起動
streamlit run App.py

# 保存後の実行権限を付与
chmod +x run.sh
./run.sh
