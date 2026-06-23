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


def normalize_poke_name(name: str) -> str:
    """
    🌟 [新設] フォルムチェンジ、表記揺れ、およびメガシンカによる
    Word2Vecの単語（ベクトル）空間の分裂を防止するための正規化
    """
    # 1. 全角括弧を半角に統一
    name = name.replace("（", "(").replace("）", ")")

    # 2. メガシンカやフォルム違いの表記をベース名に統合
    if "ギルガルド" in name:
        return "ギルガルド"

    # メガリザードンXなどの表記を「リザードン」に統合
    name = name.replace("メガ", "").rstrip("XYＸＹ ")

    return name


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

                        # 🌟 フォルム違いやメガシンカを正規化しながらリスト化
                        poke_names = [
                            normalize_poke_name(p["name"])
                            for p in team
                            if "name" in p
                        ]

                        # 構築の重複（ユニーク化）を排除して1つのSentenceにする
                        unique_pokes = list(dict.fromkeys(poke_names))
                        if len(unique_pokes) >= 2:
                            parties.append(unique_pokes)
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
    # window=6 (6体構築全体をカバー), sg=1 (Skip-gramを採用し、マイナーポケモンの表現力を高める)
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
        # 正規化した名称でテスト
        norm_poke = normalize_poke_name(poke)
        if norm_poke in model.wv:
            print(f"👉 【{norm_poke}】とセットで組まれやすいポケモン Top 3:")
            for sim_poke, score in model.wv.most_similar(positive=[norm_poke], topn=3):
                print(f"   ┗ {sim_poke} (共起類似度: {score:.3f})")
        else:
            print(f"👉 【{norm_poke}】は出現回数が不足しているため、モデルに未登録です。")


if __name__ == "__main__":
    train_pokemon_word2vec()