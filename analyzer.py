"""
分析報告與視覺化 — analyzer.py
================================
產生完整的統計報告、圖表、CSV 輸出
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 無 GUI 模式
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import List, Dict

from baccarat_engine import THEORETICAL, PAYOUTS
from simulator import StrategyResult, aggregate_results


# ====== 中文字型設定 ======
def setup_chinese_font():
    """設定 matplotlib 中文字型"""
    # Windows 常見中文字型
    chinese_fonts = [
        'Microsoft JhengHei',  # 微軟正黑體
        'Microsoft YaHei',     # 微軟雅黑
        'SimHei',              # 黑體
        'DFKai-SB',            # 標楷體
        'Arial Unicode MS',
    ]
    for font_name in chinese_fonts:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and 'LastResort' not in font_path:
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                return font_name
        except Exception:
            continue

    # 回退設定
    plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return 'fallback'


def ensure_output_dir(output_dir: str = "output"):
    """確保輸出目錄存在"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ====== 圖表生成 ======

def plot_accuracy_comparison(df: pd.DataFrame, output_dir: str = "output"):
    """策略準確率比較橫條圖"""
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = sns.color_palette("husl", len(df))
    bars = ax.barh(df['策略'], df['平均準確率%'], xerr=df['準確率標準差'],
                   color=colors, edgecolor='white', linewidth=0.5,
                   capsize=3, alpha=0.85)

    # 50% 基準線
    ax.axvline(x=50, color='red', linestyle='--', linewidth=1.5, label='50% 基準線')

    # 理論莊贏率（不計和局）
    banker_no_tie = 45.8597 / (45.8597 + 44.6247) * 100
    ax.axvline(x=banker_no_tie, color='orange', linestyle=':', linewidth=1.5,
               label=f'莊贏理論值 {banker_no_tie:.1f}%')

    # 數值標籤
    for bar, acc in zip(bars, df['平均準確率%']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{acc:.2f}%', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('準確率 (%)', fontsize=12)
    ax.set_title('各策略預測準確率比較（Monte Carlo 模擬）', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(40, 60)
    plt.tight_layout()
    path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_profit_comparison(df: pd.DataFrame, output_dir: str = "output"):
    """策略損益比較圖"""
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in df['平均損益']]
    bars = ax.barh(df['策略'], df['平均損益'], color=colors, edgecolor='white',
                   linewidth=0.5, alpha=0.85)

    ax.axvline(x=0, color='black', linewidth=1)

    for bar, val in zip(bars, df['平均損益']):
        offset = 50 if val >= 0 else -50
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:+,.0f}', va='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('平均損益', fontsize=12)
    ax.set_title('各策略平均損益比較（平注法）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'profit_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_balance_curves(strategy_results: List[StrategyResult], output_dir: str = "output"):
    """資金曲線圖（單次模擬詳細）"""
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(14, 8))

    for sr in strategy_results:
        if sr.balance_history:
            ax.plot(sr.balance_history, label=f'{sr.strategy_name} ({sr.accuracy:.1f}%)',
                    linewidth=1.2, alpha=0.8)

    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('局數', fontsize=12)
    ax.set_ylabel('累計損益', fontsize=12)
    ax.set_title('各策略資金曲線（單次模擬）', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'balance_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_accuracy_distribution(all_results: Dict[str, List[StrategyResult]],
                                output_dir: str = "output"):
    """各策略準確率分佈 (箱形圖)"""
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(14, 7))

    data = []
    labels = []
    for name, results in all_results.items():
        accs = [r.accuracy for r in results]
        data.append(accs)
        labels.append(name)

    bp = ax.boxplot(data, labels=labels, patch_artist=True, vert=True)

    colors = sns.color_palette("husl", len(data))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='50%')
    ax.set_ylabel('準確率 (%)', fontsize=12)
    ax.set_title('各策略準確率分佈（Monte Carlo）', fontsize=14, fontweight='bold')
    ax.legend()
    plt.xticks(rotation=30, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(output_dir, 'accuracy_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_base_probability(base_stats: dict, output_dir: str = "output"):
    """基礎機率 vs 理論值比較"""
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['閒贏', '莊贏', '和局']
    simulated = [base_stats.get(f'{c}%', 0) for c in categories]
    theoretical = [THEORETICAL[c] for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, simulated, width, label='模擬結果',
                   color='#3498db', alpha=0.85)
    bars2 = ax.bar(x + width/2, theoretical, width, label='理論值',
                   color='#e74c3c', alpha=0.85)

    # 數值標籤
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f'{h:.2f}%', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel('機率 (%)', fontsize=12)
    ax.set_title('模擬結果 vs 理論機率', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(simulated), max(theoretical)) * 1.15)
    plt.tight_layout()
    path = os.path.join(output_dir, 'base_probability.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_streak_analysis(strategy_results: List[StrategyResult], output_dir: str = "output"):
    """連勝連敗分析"""
    setup_chinese_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = [sr.strategy_name for sr in strategy_results]
    max_wins = [sr.max_consecutive_wins for sr in strategy_results]
    max_losses = [sr.max_consecutive_losses for sr in strategy_results]

    colors_win = sns.color_palette("Greens_r", len(names))
    colors_loss = sns.color_palette("Reds_r", len(names))

    ax1.barh(names, max_wins, color=colors_win, edgecolor='white')
    ax1.set_title('最大連贏', fontsize=13, fontweight='bold')
    ax1.set_xlabel('局數')
    for i, v in enumerate(max_wins):
        ax1.text(v + 0.2, i, str(v), va='center', fontweight='bold')

    ax2.barh(names, max_losses, color=colors_loss, edgecolor='white')
    ax2.set_title('最大連輸', fontsize=13, fontweight='bold')
    ax2.set_xlabel('局數')
    for i, v in enumerate(max_losses):
        ax2.text(v + 0.2, i, str(v), va='center', fontweight='bold')

    plt.suptitle('各策略連勝/連敗分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'streak_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_heatmap_correlation(all_results: Dict[str, List[StrategyResult]],
                              output_dir: str = "output"):
    """策略之間的準確率相關性熱力圖"""
    setup_chinese_font()

    # 取每次模擬中各策略的準確率建成矩陣
    names = list(all_results.keys())
    n_sims = len(next(iter(all_results.values())))
    matrix = np.zeros((n_sims, len(names)))

    for j, name in enumerate(names):
        for i, r in enumerate(all_results[name]):
            matrix[i, j] = r.accuracy

    df_corr = pd.DataFrame(matrix, columns=names).corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('策略準確率相關性矩陣', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ====== 報告生成 ======

def generate_report(
    base_stats: dict,
    strategy_results: List[StrategyResult],
    all_results: Dict[str, List[StrategyResult]],
    df_summary: pd.DataFrame,
    n_simulations: int,
    n_rounds: int,
    output_dir: str = "output",
) -> str:
    """生成完整的文字報告"""
    lines = []
    lines.append("=" * 70)
    lines.append("     百家樂預測策略研究報告")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"模擬參數：{n_simulations} 次模擬 × {n_rounds} 局/次")
    lines.append(f"總模擬局數：{n_simulations * n_rounds:,}")
    lines.append("")

    # 基礎機率
    lines.append("─" * 50)
    lines.append("▶ 基礎機率（單次模擬）")
    lines.append("─" * 50)
    for key in ['閒贏', '莊贏', '和局', '閒對', '莊對', '天牌']:
        val = base_stats.get(key, 0)
        pct = base_stats.get(f'{key}%', 0)
        theo = THEORETICAL.get(key, '-')
        if isinstance(theo, float):
            lines.append(f"  {key}: {val:>6} ({pct:>6.2f}%)  理論值: {theo:.2f}%")
        else:
            lines.append(f"  {key}: {val:>6} ({pct:>6.2f}%)")
    lines.append("")

    # 策略排名
    lines.append("─" * 50)
    lines.append("▶ 策略準確率排名（Monte Carlo 平均）")
    lines.append("─" * 50)
    for idx, row in df_summary.iterrows():
        lines.append(
            f"  #{idx+1:>2} {row['策略']:<12} "
            f"準確率: {row['平均準確率%']:>6.2f}% ± {row['準確率標準差']:>5.2f}%  "
            f"ROI: {row['平均ROI%']:>+7.2f}%  "
            f"勝率>50%: {row['勝率>50%比例']:>5.1f}%"
        )
    lines.append("")

    # 單次模擬詳細
    lines.append("─" * 50)
    lines.append("▶ 單次模擬詳細結果")
    lines.append("─" * 50)
    for sr in sorted(strategy_results, key=lambda x: x.accuracy, reverse=True):
        lines.append(
            f"  {sr.strategy_name:<12} "
            f"對{sr.correct} 錯{sr.wrong} 和{sr.ties_skipped}  "
            f"準確率: {sr.accuracy:>6.2f}%  "
            f"損益: {sr.profit:>+10,.0f}  "
            f"最大連贏: {sr.max_consecutive_wins}  "
            f"最大連輸: {sr.max_consecutive_losses}"
        )
    lines.append("")

    # 結論
    lines.append("─" * 50)
    lines.append("▶ 研究結論")
    lines.append("─" * 50)
    best = df_summary.iloc[0]
    lines.append(f"  最佳策略: {best['策略']} (平均準確率 {best['平均準確率%']:.2f}%)")
    lines.append(f"  所有策略平均準確率: {df_summary['平均準確率%'].mean():.2f}%")
    lines.append("")

    all_above_50 = df_summary[df_summary['平均準確率%'] > 50]
    if len(all_above_50) == 0:
        lines.append("  ⚠ 沒有任何策略的平均準確率穩定超過 50%")
        lines.append("  ⚠ 這符合百家樂的數學本質：莊家優勢無法被預測策略克服")
    else:
        lines.append(f"  {len(all_above_50)} 個策略平均準確率超過 50%:")
        for _, row in all_above_50.iterrows():
            lines.append(f"    - {row['策略']}: {row['平均準確率%']:.2f}%")
        lines.append("  注意: 壓莊的高準確率源於莊家本身的理論優勢 (45.86% vs 44.62%)")
        lines.append("  扣除 5% 佣金後，長期 ROI 仍為負值")

    lines.append("")
    lines.append("  💡 百家樂是負期望值遊戲，所有策略長期都無法獲利")
    lines.append("  💡 莊家優勢 (House Edge): 閒 1.24%, 莊 1.06%, 和 14.36%")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    # 儲存報告
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def save_results_csv(
    df_summary: pd.DataFrame,
    strategy_results: List[StrategyResult],
    output_dir: str = "output",
):
    """儲存結果為 CSV"""
    ensure_output_dir(output_dir)

    # 彙整表
    df_summary.to_csv(os.path.join(output_dir, "summary.csv"),
                      index=False, encoding="utf-8-sig")

    # 單次模擬詳細預測
    for sr in strategy_results:
        if sr.prediction_detail:
            df_detail = pd.DataFrame(sr.prediction_detail)
            safe_name = sr.strategy_name.replace(' ', '_')
            df_detail.to_csv(
                os.path.join(output_dir, f"detail_{safe_name}.csv"),
                index=False, encoding="utf-8-sig"
            )
