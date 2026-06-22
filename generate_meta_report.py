import os
import json


def generate_aegis_meta_report():
    weights_path = "log/meta_weights.json"
    report_path = "log/meta_evolution_report_gen192.md"

    if not os.path.exists(weights_path):
        print(f"❌ '{weights_path}' が見つかりません。ループを実行してデータを蓄積させてください。")
        return

    print("📊 192世代分の適応進化データをパース中...")
    with open(weights_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 出現重み (weight) に基づいて降順ソート
    sorted_pokes = sorted(data.items(), key=lambda x: -x[1].get("weight", 1.0))

    markdown_content = []
    markdown_content.append("# 👑 Project Aegis: 192世代適応進化・最終結論レポート")
    markdown_content.append(
        "本レポートは、AI同士の7,680対戦に及ぶ自然淘汰（遺伝アルゴリズム）と勝率連動学習を経て、自律的に抽出されたレギュレーションM-Bの結論メタデータである。\n")

    markdown_content.append("## 🏆 環境支配度（出現重み）ランキング Top 20")
    markdown_content.append("| 順位 | ポケモン名 | 出現重み (Max: 10.0) |")
    markdown_content.append("| :--- | :--- | :--- |")

    for rank, (name, details) in enumerate(sorted_pokes[:20], 1):
        weight = details.get("weight", 1.0)
        markdown_content.append(f"| {rank}位 | **{name}** | {weight:.2f} |")

    markdown_content.append("\n## ⚔️ 主要ポケモンの「最強の型」分析 (自律抽出された最適解)")
    markdown_content.append(
        "各ポケモンにおいて、対戦履歴の勝率実績から最も高い適応度（重み）を獲得した技・特性・性格の構成である。\n")

    for rank, (name, details) in enumerate(sorted_pokes[:20], 1):
        weight = details.get("weight", 1.0)

        # 技の勝率重みソート（上位4つ）
        moves_data = details.get("moves", {})
        sorted_moves = sorted(moves_data.items(), key=lambda x: -x[1])[:4]
        best_moves = [f"{m} ({w:.2f})" for m, w in sorted_moves] if sorted_moves else ["データなし"]

        # 特性の勝率重みソート（上位1つ）
        abilities_data = details.get("abilities", {})
        sorted_abilities = sorted(abilities_data.items(), key=lambda x: -x[1])[:1]
        best_ability = f"{sorted_abilities[0][0]} ({sorted_abilities[0][1]:.2f})" if sorted_abilities else "データなし"

        # 性格の勝率重みソート（上位1つ）
        natures_data = details.get("natures", {})
        sorted_natures = sorted(natures_data.items(), key=lambda x: -x[1])[:1]
        best_nature = f"{sorted_natures[0][0]} ({sorted_natures[0][1]:.2f})" if sorted_natures else "データなし"

        markdown_content.append(f"### {rank}位：【{name}】 (出現重み: {weight:.2f})")
        markdown_content.append(f"*   **推奨特性（最適特性）**: {best_ability}")
        markdown_content.append(f"*   **推奨性格（最適性格）**: {best_nature}")
        markdown_content.append(f"*   **採用率・勝率の高い技 Top 4**: {', '.join(best_moves)}")
        markdown_content.append(f"---")

    # ファイルに保存
    with open(report_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(markdown_content))

    print(f"\n==================================================")
    print(f"  🎉 レポート生成完了！")
    print(f"  保存先: '{report_path}'")
    print(f"==================================================")
    print(
        "このファイルをMarkdownエディタ、またはテキストエディタで開いて、AIが導き出した『結論』をその目で確認してください。")


if __name__ == "__main__":
    generate_aegis_meta_report()