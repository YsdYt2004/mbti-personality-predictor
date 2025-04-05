import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
from src.utils import load_config


def run_svm():
    config = load_config()

    # TF-IDF特徴量（sparse）をpklから読み込み
    with open("data/tfidf_features.pkl", "rb") as f:
        X = pickle.load(f)

    # ラベル読み込み
    y = pd.read_csv("data/labels.csv").squeeze()

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['svm']['test_size'],
        random_state=config['svm']['random_state']
    )

    # モデル学習
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # 精度出力
    acc = model.score(X_test, y_test)
    print(f"✅ SVM 精度: {acc:.3f}")

    # モデル保存
    with open("models/svm_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("logs/training.log", "a") as log:
        log.write(f"SVM accuracy: {acc:.3f}\n")


if __name__ == "__main__":
    run_svm()
