import os
import json
import glob
import re
from collections import defaultdict


def generate_aegis_meta_report():
    weights_path = "log/meta_weights.json"

    # 1. 進行中のログファイルから「現在の最新世代数」を安全に検出
    gen_files = glob.glob("log/selfplay_gen_*.jsonl")
    current_gen = 192  # フォールバック用デフォルト値
    gen_numbers = []

    if gen_files:
        for f in gen_files:
            # ファイル名パターンを厳密にマッチング
            match = re.search(r'selfplay_gen_(\d+)\.jsonl$', os.path.basename(f))
            if match:
                gen_numbers.append(int(match.group(1)))

        if gen_numbers:
            current_gen = max(gen_numbers)

    report_path = f"battle_data/meta_evolution_report_gen{current_gen}.md"

    if not os.path.exists(weights_path):
        print(f"❌ '{weights_path}' が見つかりません。ループを実行してデータを蓄積させてください。")
        return

    print(f"📊 {current_gen}世代分の適応進化データをパース中...")
    try:
        with open(weights_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 重みファイルの読み込みに失敗しました: {e}")
        return

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

    # =========================================================================
    # 2. 【改善：世代別使用率（Pick Rate）推移追跡セクション】
    # =========================================================================
    markdown_content.append(f"\n## 📈 環境最重要ポケモンの選出使用率（%）歴史的推移")
    markdown_content.append(
        "AIの世代交代に伴い、環境を支配していた強豪や、新たに適応してきた対策ポケモンたちの「実質選出率（1プレイヤーあたり %）」の推移データである。\n"
    )

    TARGET_TRACK_POKES = ["ガオガエン", "ブリガロン", "カイリュー", "サーフゴー", "ドドゲザン", "ミロカロス"]
    gen_pick_rates = defaultdict(dict)
    valid_gens = sorted(gen_numbers)

    if valid_gens:
        # パフォーマンス向上のため、あらかじめ表示対象の世代を決定
        step = max(1, len(valid_gens) // 15)
        displayed_gens = [valid_gens[i] for i in range(0, len(valid_gens), step)]
        if valid_gens[-1] not in displayed_gens:
            displayed_gens.append(valid_gens[-1])

        print(f"📈 選択された {len(displayed_gens)} 世代の対戦ログから、使用率の時系列変動を遡及抽出中...")

        # 必要なファイルのみを開いて解析
        for g in displayed_gens:
            g_log = f"log/selfplay_gen_{g}.jsonl"
            if not os.path.exists(g_log):
                continue

            total_battles = 0
            g_counts = defaultdict(int)

            try:
                with open(g_log, "r", encoding="utf-8") as f_gen:
                    for line in f_gen:
                        if not line.strip():
                            continue
                        match_m = json.loads(line)
                        winner_m = match_m.get("winner")
                        if winner_m is None or winner_m == -1:
                            continue

                        # 有効な対戦数カウント
                        total_battles += 1

                        selections = match_m.get("selections", [])
                        teams = match_m.get("teams", [])

                        # 構造データの安全な検証
                        if len(selections) < 2 or len(teams) < 2:
                            continue

                        for pl in [0, 1]:
                            pl_selections = selections[pl]
                            pl_team = teams[pl]
                            for idx_m in pl_selections:
                                if idx_m < len(pl_team):
                                    poke_name = pl_team[idx_m].get("name")
                                    if poke_name:
                                        g_counts[poke_name] += 1
            except Exception as e:
                # 特定ファイルのパースエラーでプロセス全体を落とさない
                print(f"⚠️ 世代 {g} の解析中にエラーが発生しました: {e}")
                continue

            # 使用率の計算：
            # 「その世代の総プレイヤー機会数（総バトル数 × 2）」を分母とし、
            # 各プレイヤーがそのポケモンを選出した割合（0%〜100%）に調整
            total_opportunities = total_battles * 2
            if total_opportunities > 0:
                for target_p in TARGET_TRACK_POKES:
                    pick_rate = (g_counts[target_p] / total_opportunities) * 100.0
                    gen_pick_rates[g][target_p] = round(pick_rate, 1)

        # Markdown テーブルヘッダーの作成
        headers = ["世代(Gen)"] + [f"【{tp}】" for tp in TARGET_TRACK_POKES]
        markdown_content.append("| " + " | ".join(headers) + " |")
        markdown_content.append("| :--- | " + " | ".join([":---:" for _ in TARGET_TRACK_POKES]) + " |")

        for g in displayed_gens:
            # ログファイルが欠損しているなどの理由で計算されなかった場合への配慮
            if g not in gen_pick_rates and any(gen_pick_rates.values()):
                continue
            row = [f"第 {g} 世代"]
            for target_p in TARGET_TRACK_POKES:
                rate = gen_pick_rates[g].get(target_p, 0.0)
                row.append(f"{rate:.1f}%")
            markdown_content.append("| " + " | ".join(row) + " |")

        markdown_content.append("\n*(※使用率(%) ＝ その世代における各プレイヤーの選出枠（3枠）での採用割合。最大100%。)*")

    # =========================================================================

    markdown_content.append("\n## ⚔️ 主要ポケモンの「最強の型」分析 (自律抽出された最適解)")
    markdown_content.append(
        "各ポケモンにおいて、対戦履歴の勝率実績から最も高い適応度（重み）を獲得した技・特性・性格・持ち物・努力値の構成である。\n"
    )

    for rank, (name, details) in enumerate(sorted_pokes[:20], 1):
        weight = details.get("weight", 1.0)

        # 技の勝率重みソート
        moves_data = details.get("moves", {})
        sorted_moves = sorted(moves_data.items(), key=lambda x: -x[1])[:4]
        best_moves = [f"{m} ({w:.2f})" for m, w in sorted_moves] if sorted_moves else ["データなし"]

        # 特性の勝率重みソート
        abilities_data = details.get("abilities", {})
        sorted_abilities = sorted(abilities_data.items(), key=lambda x: -x[1])[:1]
        best_ability = f"{sorted_abilities[0][0]} ({sorted_abilities[0][1]:.2f})" if sorted_abilities else "データなし"

        # 性格の勝率重みソート
        natures_data = details.get("natures", {})
        sorted_natures = sorted(natures_data.items(), key=lambda x: -x[1])[:1]
        best_nature = f"{sorted_natures[0][0]} ({sorted_natures[0][1]:.2f})" if sorted_natures else "データなし"

        # 持ち物の勝率重みソート
        items_data = details.get("items", {})
        sorted_items = sorted(items_data.items(), key=lambda x: -x[1])[:1]
        best_item = f"{sorted_items[0][0]} ({sorted_items[0][1]:.2f})" if sorted_items else "データなし"

        # 努力値配分カテゴリの勝率重みソート
        ev_data = details.get("ev_categories", {})
        sorted_evs = sorted(ev_data.items(), key=lambda x: -x[1])[:1]
        best_ev = f"{sorted_evs[0][0]} ({sorted_evs[0][1]:.2f})" if sorted_evs else "データなし"

        markdown_content.append(f"### {rank}位：【{name}】 (出現重み: {weight:.2f})")
        markdown_content.append(f"*   **推奨特性（最適特性）**: {best_ability}")
        markdown_content.append(f"*   **推奨性格（最適性格）**: {best_nature}")
        markdown_content.append(f"*   **推奨持物（最適持ち物）**: {best_item}")
        markdown_content.append(f"*   **推奨努力（最適合努力値）**: {best_ev}")
        markdown_content.append(f"* adopt率・勝率の高い技 Top 4: {', '.join(best_moves)}")
        markdown_content.append(f"---")

    # 3. 出現重みの低いワースト15
    markdown_content.append(f"\n## 📉 環境デフレ（過小評価・要救済）ランキング")
    markdown_content.append(
        "現在の自己対戦環境において、出現率（Weight）が著しく低下し、淘汰のデッドロックに陥っているポケモンたちのリストである。\n")
    markdown_content.append("| 順位 | ポケモン名 | 出現重み (Min: 0.1) |")
    markdown_content.append("| :--- | :--- | :--- |")

    sorted_pokes_asc = sorted(data.items(), key=lambda x: x[1].get("weight", 1.0))
    for rank, (name, details) in enumerate(sorted_pokes_asc[:15], 1):
        weight = details.get("weight", 1.0)
        markdown_content.append(f"| {rank}位 | {name} | {weight:.2f} |")

    # ファイルに保存
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(markdown_content))
        print(f"\n==================================================")
        print(f"  🎉 レポート生成完了！")
        print(f"  保存先: '{report_path}'")
        print(f"==================================================")
    except Exception as e:
        print(f"❌ レポートの保存に失敗しました: {e}")


if __name__ == "__main__":
    generate_aegis_meta_report()