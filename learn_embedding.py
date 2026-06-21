import sys
import os
import json
import glob
import warnings
from typing import List

# gensim（NLPライブラリ）の安全なインポート
try:
    from gensim.models import Word2Vec
except ImportError:
    print("❌ gensim ライブラリがインストールされていません。")
    print("   学習を実行するには、ターミナルで 'pip install gensim' を実行してください。")
    sys.exit(1)


def extract_parties_from_logs(log_dir: str) -> List[List[str]]:
    """
    蓄積された自己対戦ログ (selfplay_gen_*.jsonl) から、
    Word2Vecの入力となる『6体構築のリスト』をすべて抽出する
    """
    parties = []
    log_files = glob.glob(os.path.join(log_dir, "selfplay_gen_*.jsonl"))

    if not log_files:
        print(f"⚠️ '{log_dir}' 配下に selfplay_gen_*.jsonl ファイルが見つかりません。")
        return []

    print(f"📊 ログファイル {len(log_files)} 個から構築データを読み込んでいます...")

    for file_path in log_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    match_data = json.loads(line)
                    # 対戦した両プレイヤーの6体構築を取得
                    for pl in [0, 1]:
                        team = match_data.get("teams", [])[pl]
                        # 6体の名前をリスト化して文（Sentence）として扱う
                        poke_names = [p["name"] for p in team if "name" in p]
                        if len(poke_names) >= 2:
                            parties.append(poke_names)
                except Exception:
                    continue

    return parties


def train_pokemon_word2vec():
    # 1. データの抽出
    log_directory = "log"
    sentences = extract_parties_from_logs(log_directory)

    if not sentences:
        print("❌ 学習に利用できる構築データが見つかりませんでした。中断します。")
        return

    print(f"✅ 合計 {len(sentences)} 件の構築文（センテンス）を抽出完了しました。")
    print("🚀 Word2Vec 構築共起モデルのトレーニングを開始します...")

    # 2. Word2Vecモデルの設定と訓練
    model = Word2Vec(
        sentences=sentences,
        vector_size=32,
        window=6,
        min_count=2,
        workers=4,
        epochs=30,
        sg=1
    )

    # 3. モデルの保存
    os.makedirs("data", exist_ok=True)
    model_save_path = "data/pokemon_word2vec.model"
    model.save(model_save_path)

    print("\n==================================================")
    print(f"  🎉 Word2Vec 学習完了 (総単語数(種類): {len(model.wv)})")
    print(f"  保存先: '{model_save_path}'")
    print("==================================================")

    # デバッグ用にいくつかのポケモンの「組まれやすいポケモン（類似度）」を表示
    test_pokes = ["サーフゴー", "ブリジュラス", "ガブリアス", "ラウドボーン"]
    print("\n【🔍 構築上の共起類似度（組まれやすさ）のテスト】")
    for poke in test_pokes:
        if poke in model.wv:
            print(f"👉 【{poke}】とセットで組まれやすいポケモン Top 3:")
            # 修正箇所: most_common ➔ most_similar に変更
            for sim_poke, score in model.wv.most_similar(positive=[poke], topn=3):
                print(f"   ┗ {sim_poke} (共起類似度: {score:.3f})")
        else:
            print(f"👉 【{poke}】は出現回数が不足しているため、モデルに未登録です。")


if __name__ == "__main__":
    train_pokemon_word2vec()