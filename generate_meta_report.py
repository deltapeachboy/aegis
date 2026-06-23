import os
import json
import glob
import re


def generate_aegis_meta_report():
    weights_path = "log/meta_weights.json"

    # 🌟 1. 進行中のログファイルから「現在の最新世代数」を自動検出
    gen_files = glob.glob("log/selfplay_gen_*.jsonl")
    current_gen = 192  # フォールバック用デフォルト値
    if gen_files:
        try:
            # ファイル名 (例: selfplay_gen_152.jsonl) から数値のみを抽出して最大値を得る
            gen_numbers = [
                int(re.findall(r'\d+', os.path.basename(f))[0])
                for f in gen_files
                if re.findall(r'\d+', os.path.basename(f))
            ]
            if gen_numbers:
                current_gen = max(gen_numbers)
        except Exception:
            pass

    report_path = f"battle_data/meta_evolution_report_gen{current_gen}.md"

    if not os.path.exists(weights_path):
        print(f"❌ '{weights_path}' が見つかりません。ループを実行してデータを蓄積させてください。")
        return

    print(f"📊 {current_gen}世代分の適応進化データをパース中...")
    with open(weights_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 出現重み (weight) に基づいて降順ソート
    sorted_pokes = sorted(data.items(), key=lambda x: -x[1].get("weight", 1.0))

    markdown_content = []
    markdown_content.append(f"# 👑 Project Aegis: {current_gen}世代適応進化・最終結論レポート")
    markdown_content.append(
        f"本レポートは、AI同士の累積対戦と勝率連動学習を経て、自律的に抽出されたレギュレーションM-Bの結論メタデータ（最新第 {current_gen} 世代時点）である。\n"
    )

    markdown_content.append("## 🏆 環境支配度（出現重み）ランキング Top 20")
    markdown_content.append("| 順位 | ポケモン名 | 出現重み (Max: 10.0) |")
    markdown_content.append("| :--- | :--- | :--- |")

    for rank, (name, details) in enumerate(sorted_pokes[:20], 1):
        weight = details.get("weight", 1.0)
        markdown_content.append(f"| {rank}位 | **{name}** | {weight:.2f} |")

    markdown_content.append("\n## ⚔️ 主要ポケモンの「最強の型」分析 (自律抽出された最適解)")
    markdown_content.append(
        "各ポケモンにおいて、対戦履歴の勝率実績から最も高い適応度（重み）を獲得した技・特性・性格の構成である。\n"
    )

    for rank, (name, details) in enumerate(sorted_pokes[:20], 1):
        weight = details.get("weight", 1.0)

        # 技の勝率重みソート
        moves_data = details.get("moves", {})
        sorted_moves = sorted(moves_data.items(), key=lambda x: -x[1])[:4]
        # 🌟 IndexError 防止の安全ガードを適用
        best_moves = [f"{m} ({w:.2f})" for m, w in sorted_moves] if sorted_moves else ["データなし（未選出）"]

        # 特性の勝率重みソート
        abilities_data = details.get("abilities", {})
        sorted_abilities = sorted(abilities_data.items(), key=lambda x: -x[1])[:1]
        best_ability = f"{sorted_abilities[0][0]} ({sorted_abilities[0][1]:.2f})" if sorted_abilities else "データなし（未選出）"

        # 性格の勝率重みソート
        natures_data = details.get("natures", {})
        sorted_natures = sorted(natures_data.items(), key=lambda x: -x[1])[:1]
        best_nature = f"{sorted_natures[0][0]} ({sorted_natures[0][1]:.2f})" if sorted_natures else "データなし（未選出）"

        markdown_content.append(f"### {rank}位：【{name}】 (出現重み: {weight:.2f})")
        markdown_content.append(f"*   **推奨特性（最適特性）**: {best_ability}")
        markdown_content.append(f"*   **推奨性格（最適性格）**: {best_nature}")
        markdown_content.append(f"* adopt率・勝率の高い技 Top 4: {', '.join(best_moves)}")
        markdown_content.append(f"---")

    # 🌟 3. 出現重みの低いワースト10を抽出し、デフレ環境をあぶり出すセクション
    markdown_content.append(f"\n## 📉 環境デフレ（過小評価・要救済）ランキング")
    markdown_content.append(
        "現在の自己対戦環境において、出現率（Weight）が著しく低下し、淘汰のデッドロックに陥っているポケモンたちのリストである。\n")
    markdown_content.append("| 順位 | ポケモン名 | 出現重み (Min: 0.1) |")
    markdown_content.append("| :--- | :--- | :--- |")

    # 逆順ソートして下位15体
    sorted_pokes_asc = sorted(data.items(), key=lambda x: x[1].get("weight", 1.0))
    for rank, (name, details) in enumerate(sorted_pokes_asc[:15], 1):
        weight = details.get("weight", 1.0)
        markdown_content.append(f"| {rank}位 | {name} | {weight:.2f} |")

    # ファイルに保存
    with open(report_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(markdown_content))

    print(f"\n==================================================")
    print(f"  🎉 レポート生成完了！")
    print(f"  保存先: '{report_path}'")
    print(f"==================================================")
    print(f"最新の {current_gen} 世代までの自律進化データをまとめました。")
    print("AIがどのような『型』や『デフレ環境』を形成しているか、その目で直接確認してください。")


if __name__ == "__main__":
    generate_aegis_meta_report()