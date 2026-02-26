"""
百家樂預測研究系統 — 主程式入口
================================
執行方式:
  python main.py              → 模擬模式 (Monte Carlo)
  python main.py --live       → 即時追蹤模式 (真實數據)
  python main.py --quick      → 快速模擬模式

模擬模式參數:
  --rounds    每次模擬局數 (預設 1000)
  --sims      Monte Carlo 模擬次數 (預設 100)
  --seed      隨機種子 (預設 42)
  --unit      基本注碼 (預設 100)
  --output    輸出目錄 (預設 output)
  --quick     快速模式 (10次×500局)

即時追蹤模式:
  --live      啟動即時預測系統，手動輸入真實牌局結果
"""

import argparse
import sys
import time
import os

# 確保當前目錄在 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baccarat_engine import BaccaratGame, calculate_base_probabilities, THEORETICAL
from strategies import get_all_strategies
from simulator import run_monte_carlo, run_single_simulation, aggregate_results
from analyzer import (
    ensure_output_dir, setup_chinese_font,
    plot_accuracy_comparison, plot_profit_comparison,
    plot_balance_curves, plot_accuracy_distribution,
    plot_base_probability, plot_streak_analysis,
    plot_heatmap_correlation,
    generate_report, save_results_csv,
)


def print_header():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          百家樂預測策略研究系統 v1.0                    ║")
    print("║     Baccarat Prediction Strategy Research System        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def progress_bar(current, total, prefix='進度', length=40):
    pct = current / total * 100
    filled = int(length * current // total)
    bar = '█' * filled + '░' * (length - filled)
    print(f'\r  {prefix}: |{bar}| {pct:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()


def main():
    parser = argparse.ArgumentParser(description='百家樂預測策略研究系統')
    parser.add_argument('--rounds', type=int, default=1000, help='每次模擬局數')
    parser.add_argument('--sims', type=int, default=100, help='Monte Carlo 模擬次數')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--unit', type=float, default=100, help='基本注碼')
    parser.add_argument('--output', type=str, default='output', help='輸出目錄')
    parser.add_argument('--quick', action='store_true', help='快速模式 (10次×500局)')
    parser.add_argument('--live', action='store_true', help='即時追蹤模式（真實數據）')
    parser.add_argument('--battle', action='store_true', help='實戰模式（娛樂城專用）')
    parser.add_argument('--web', action='store_true', help='Web 介面（手機可用）')
    parser.add_argument('--port', type=int, default=8888, help='Web 介面埠號')
    args = parser.parse_args()

    # === 無參數時預設啟動 Web 介面（適用雲端部署） ===
    if len(sys.argv) == 1:
        from web_app import start_server
        start_server()
        return

    # === Web 介面 ===
    if args.web:
        from web_app import start_server
        start_server(args.port)
        return

    # === 實戰模式 ===
    if args.battle:
        from battle import run_battle
        run_battle()
        return

    # === 即時追蹤模式 ===
    if args.live:
        from live_tracker import interactive_mode
        interactive_mode()
        return

    if args.quick:
        args.sims = 10
        args.rounds = 500

    print_header()
    print(f"  模式: {'快速' if args.quick else '標準'}")
    print(f"  模擬次數: {args.sims}")
    print(f"  每次局數: {args.rounds}")
    print(f"  總模擬局數: {args.sims * args.rounds:,}")
    print(f"  基本注碼: {args.unit}")
    print(f"  隨機種子: {args.seed}")
    print(f"  輸出目錄: {args.output}")
    print()

    output_dir = ensure_output_dir(args.output)

    # ====== 步驟 1: 單次詳細模擬 ======
    print("─" * 50)
    print("📊 步驟 1/4: 執行單次詳細模擬...")
    t0 = time.time()

    game_history, strategy_results, base_stats = run_single_simulation(
        n_rounds=args.rounds,
        base_unit=args.unit,
        seed=args.seed,
    )

    t1 = time.time()
    print(f"  ✅ 完成 ({t1 - t0:.2f}s)")
    print()

    # 顯示基礎統計
    print("  基礎機率:")
    for key in ['閒贏', '莊贏', '和局']:
        val = base_stats.get(key, 0)
        pct = base_stats.get(f'{key}%', 0)
        theo = THEORETICAL.get(key, 0)
        diff = pct - theo
        print(f"    {key}: {pct:>6.2f}% (理論 {theo:.2f}%, 差異 {diff:+.2f}%)")
    print()

    # 顯示單次結果
    print("  單次模擬策略排名:")
    sorted_results = sorted(strategy_results, key=lambda x: x.accuracy, reverse=True)
    for i, sr in enumerate(sorted_results):
        marker = "🏆" if i == 0 else "  "
        print(f"  {marker} #{i+1:>2} {sr.strategy_name:<12} "
              f"準確率: {sr.accuracy:>6.2f}%  損益: {sr.profit:>+10,.0f}")
    print()

    # ====== 步驟 2: Monte Carlo 模擬 ======
    print("─" * 50)
    print(f"📊 步驟 2/4: 執行 Monte Carlo 模擬 ({args.sims}×{args.rounds})...")
    t0 = time.time()

    all_results = run_monte_carlo(
        n_simulations=args.sims,
        n_rounds=args.rounds,
        base_unit=args.unit,
        seed_base=args.seed,
        progress_callback=progress_bar,
    )

    t1 = time.time()
    print(f"  ✅ 完成 ({t1 - t0:.2f}s)")
    print()

    # 彙整
    df_summary = aggregate_results(all_results)

    print("  Monte Carlo 策略排名:")
    for idx, row in df_summary.iterrows():
        marker = "🏆" if idx == 0 else "  "
        print(f"  {marker} #{idx+1:>2} {row['策略']:<12} "
              f"準確率: {row['平均準確率%']:>6.2f}% ± {row['準確率標準差']:.2f}%  "
              f"ROI: {row['平均ROI%']:>+7.2f}%")
    print()

    # ====== 步驟 3: 生成圖表 ======
    print("─" * 50)
    print("📊 步驟 3/4: 生成分析圖表...")
    t0 = time.time()

    charts = []
    charts.append(("準確率比較", plot_accuracy_comparison(df_summary, output_dir)))
    charts.append(("損益比較", plot_profit_comparison(df_summary, output_dir)))
    charts.append(("資金曲線", plot_balance_curves(strategy_results, output_dir)))
    charts.append(("準確率分佈", plot_accuracy_distribution(all_results, output_dir)))
    charts.append(("基礎機率", plot_base_probability(base_stats, output_dir)))
    charts.append(("連勝連敗", plot_streak_analysis(strategy_results, output_dir)))
    charts.append(("相關性矩陣", plot_heatmap_correlation(all_results, output_dir)))

    t1 = time.time()
    print(f"  ✅ 完成 ({t1 - t0:.2f}s)")
    for name, path in charts:
        print(f"    📈 {name}: {path}")
    print()

    # ====== 步驟 4: 生成報告 ======
    print("─" * 50)
    print("📊 步驟 4/4: 生成分析報告...")

    report = generate_report(
        base_stats=base_stats,
        strategy_results=strategy_results,
        all_results=all_results,
        df_summary=df_summary,
        n_simulations=args.sims,
        n_rounds=args.rounds,
        output_dir=output_dir,
    )

    save_results_csv(df_summary, strategy_results, output_dir)

    print(f"  ✅ 報告已儲存: {os.path.join(output_dir, 'report.txt')}")
    print(f"  ✅ CSV 已儲存: {os.path.join(output_dir, 'summary.csv')}")
    print()

    # 印出報告
    print(report)
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  所有結果已輸出至 output/ 目錄                          ║")
    print("║  包含: 7 張圖表 + 報告 + CSV 數據                      ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
