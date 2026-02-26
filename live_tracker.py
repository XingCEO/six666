"""
即時預測儀表板 — live_tracker.py
=================================
互動式追蹤真實牌局，即時顯示：
- 12 種策略的預測結果
- 算牌系統的真實條件機率（不是固定機率）
- 路紙（大路）
- 各策略歷史準確率
- 建議下注方向與信心度

使用方式：
  python live_tracker.py
  然後按指示輸入每局結果
"""

import os
import sys
import csv
import time
from typing import List, Dict, Optional
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from card_counter import CardCounter, ShoeTracker
from strategies import get_all_strategies, BaseStrategy
from baccarat_engine import PAYOUTS, THEORETICAL


class LivePredictor:
    """即時預測引擎"""

    def __init__(self):
        self.tracker = ShoeTracker()
        self.strategies = get_all_strategies(seed=None)
        self.strategy_stats: Dict[str, Dict] = {}
        for s in self.strategies:
            self.strategy_stats[s.name] = {
                'correct': 0, 'wrong': 0, 'total': 0,
                'predictions': [],  # (prediction, actual, correct)
            }
        self.round_count = 0

    def reset(self):
        """新靴重置"""
        self.tracker.reset()
        for s in self.strategies:
            s.reset()
        for name in self.strategy_stats:
            self.strategy_stats[name] = {
                'correct': 0, 'wrong': 0, 'total': 0,
                'predictions': [],
            }
        self.round_count = 0

    def get_predictions(self) -> List[Dict]:
        """取得所有策略的預測"""
        history = self.tracker.get_outcome_sequence()
        predictions = []

        for s in self.strategies:
            pred = s.predict(history)
            stats = self.strategy_stats[s.name]
            total = stats['correct'] + stats['wrong']
            accuracy = stats['correct'] / total * 100 if total > 0 else 0

            predictions.append({
                'strategy': s.name,
                'prediction': pred,
                'accuracy': accuracy,
                'correct': stats['correct'],
                'wrong': stats['wrong'],
                'total': total,
            })

        return predictions

    def get_consensus(self) -> Dict:
        """取得所有策略的投票共識"""
        predictions = self.get_predictions()
        votes = Counter()
        weighted_votes = Counter()

        for p in predictions:
            votes[p['prediction']] += 1
            # 加權投票：準確率越高權重越大
            weight = max(p['accuracy'], 50) / 100  # 至少 0.5 權重
            weighted_votes[p['prediction']] += weight

        total_strategies = len(predictions)
        banker_votes = votes.get('莊', 0)
        player_votes = votes.get('閒', 0)

        if banker_votes > player_votes:
            consensus = '莊'
        elif player_votes > banker_votes:
            consensus = '閒'
        else:
            consensus = '莊'  # 平手時偏莊（理論優勢）

        majority = max(banker_votes, player_votes)
        confidence = majority / total_strategies * 100

        return {
            'consensus': consensus,
            'confidence': confidence,
            'banker_votes': banker_votes,
            'player_votes': player_votes,
            'total': total_strategies,
            'weighted_banker': weighted_votes.get('莊', 0),
            'weighted_player': weighted_votes.get('閒', 0),
        }

    def record_result(self, outcome: str,
                      player_cards: Optional[List[str]] = None,
                      banker_cards: Optional[List[str]] = None):
        """記錄真實結果並更新所有策略的統計"""
        history = self.tracker.get_outcome_sequence()

        # 更新各策略的預測紀錄
        if outcome != '和':
            for s in self.strategies:
                pred = s.predict(history)
                stats = self.strategy_stats[s.name]
                stats['total'] += 1
                if pred == outcome:
                    stats['correct'] += 1
                else:
                    stats['wrong'] += 1
                stats['predictions'].append({
                    'round': self.round_count + 1,
                    'prediction': pred,
                    'actual': outcome,
                    'correct': pred == outcome,
                })

        # 記錄到追蹤器
        self.tracker.record_round(outcome, player_cards, banker_cards)
        self.round_count += 1

    def get_real_probabilities(self, sample_size: int = 30000) -> Dict[str, float]:
        """
        核心功能：取得基於剩餘牌組的真實條件機率
        這不是固定的 44.6/45.8/9.5 — 是根據已出的牌計算的真實機率！
        """
        return self.tracker.counter.calculate_exact_probabilities(sample_size)

    def get_display_data(self) -> Dict:
        """取得完整的顯示數據"""
        stats = self.tracker.get_statistics()
        predictions = self.get_predictions()
        consensus = self.get_consensus()
        edge = self.tracker.counter.get_edge_indicator()

        # 嘗試計算真實機率（僅在有出牌紀錄時）
        real_prob = None
        if self.tracker.counter.get_dealt_count() > 0:
            real_prob = self.get_real_probabilities()

        return {
            'round': self.round_count,
            'stats': stats,
            'predictions': predictions,
            'consensus': consensus,
            'edge': edge,
            'real_prob': real_prob,
            'road': self._build_road(),
        }

    def _build_road(self, width: int = 30) -> str:
        """建構大路路紙（文字版）"""
        sequence = self.tracker.get_outcome_sequence()
        if not sequence:
            return "  (尚無紀錄)"

        # 簡化路紙：莊=B(紅) 閒=P(藍) 和=T
        symbols = {'莊': '莊', '閒': '閒', '和': '和'}

        # 建構列
        columns = []
        current_col = []
        prev = None

        for outcome in sequence:
            if outcome == '和':
                # 和局標記在最後一個位置
                if current_col:
                    current_col[-1] = current_col[-1] + '*'
                continue

            if prev is None or outcome == prev:
                current_col.append(symbols[outcome])
            else:
                columns.append(current_col)
                current_col = [symbols[outcome]]
            prev = outcome

        if current_col:
            columns.append(current_col)

        # 取最近的欄位
        recent = columns[-width:] if len(columns) > width else columns

        # 找最大高度
        max_height = max((len(col) for col in recent), default=1)
        max_height = min(max_height, 6)  # 限制高度

        # 繪製
        lines = []
        for row in range(max_height):
            line = "  "
            for col in recent:
                if row < len(col):
                    cell = col[row]
                    if '莊' in cell:
                        line += f" 莊"
                    elif '閒' in cell:
                        line += f" 閒"
                    else:
                        line += f" ──"
                else:
                    line += "   "
            lines.append(line)

        return "\n".join(lines)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_dashboard(predictor: LivePredictor):
    """列印完整的即時預測儀表板"""
    data = predictor.get_display_data()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           百家樂即時預測系統 — 真實數據追蹤                ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # === 基本統計 ===
    stats = data.get('stats', {})
    if stats:
        print(f"\n  📋 已記錄 {stats.get('總局數', 0)} 局  "
              f"| 莊 {stats.get('莊贏', 0)}({stats.get('莊%', 0):.1f}%)  "
              f"| 閒 {stats.get('閒贏', 0)}({stats.get('閒%', 0):.1f}%)  "
              f"| 和 {stats.get('和局', 0)}({stats.get('和%', 0):.1f}%)")

        if stats.get('當前連勝數', 0) > 0:
            print(f"  🔥 當前連勝: {stats['當前連勝方']} 連 {stats['當前連勝數']}")
    else:
        print("\n  📋 尚無紀錄，請開始輸入牌局結果")

    # === 路紙 ===
    print(f"\n  ─── 大路 ───")
    print(data.get('road', '  (空)'))

    # === 算牌資訊 ===
    edge = data.get('edge', {})
    if edge and edge.get('已出牌數', 0) > 0:
        print(f"\n  ─── 算牌統計 ───")
        print(f"  剩餘 {edge.get('剩餘牌數', 416)} 張 "
              f"| 滲透率 {edge.get('滲透率', 0):.1f}% "
              f"| 高牌 {edge.get('高牌比例', 0):.1f}% "
              f"| 低牌 {edge.get('低牌比例', 0):.1f}% "
              f"| 中牌 {edge.get('中牌比例', 0):.1f}%")

    # === 真實條件機率 ===
    real_prob = data.get('real_prob')
    if real_prob:
        print(f"\n  ─── 真實條件機率（基於剩餘牌組計算） ───")
        p_diff = real_prob['閒'] - THEORETICAL['閒贏']
        b_diff = real_prob['莊'] - THEORETICAL['莊贏']
        t_diff = real_prob['和'] - THEORETICAL['和局']
        print(f"  閒: {real_prob['閒']:>6.2f}% (理論 {THEORETICAL['閒贏']:.2f}%, 差 {p_diff:+.2f}%)")
        print(f"  莊: {real_prob['莊']:>6.2f}% (理論 {THEORETICAL['莊贏']:.2f}%, 差 {b_diff:+.2f}%)")
        print(f"  和: {real_prob['和']:>6.2f}% (理論 {THEORETICAL['和局']:.2f}%, 差 {t_diff:+.2f}%)")
    else:
        print(f"\n  ─── 理論機率 ───")
        print(f"  閒: {THEORETICAL['閒贏']:.2f}%  莊: {THEORETICAL['莊贏']:.2f}%  和: {THEORETICAL['和局']:.2f}%")

    # === 預測結果 ===
    predictions = data.get('predictions', [])
    consensus = data.get('consensus', {})

    print(f"\n  ═══ 下一局預測 ═══")

    if consensus:
        result = consensus['consensus']
        conf = consensus['confidence']
        b_votes = consensus['banker_votes']
        p_votes = consensus['player_votes']

        # 信心度視覺化
        if conf >= 75:
            level = "★★★ 高信心"
        elif conf >= 60:
            level = "★★☆ 中信心"
        else:
            level = "★☆☆ 低信心"

        print(f"\n  ┌──────────────────────────────────────────┐")
        print(f"  │  推薦: 壓 【{result}】  {level} ({conf:.0f}%)      │")
        print(f"  │  投票: 莊 {b_votes} vs 閒 {p_votes}                │")
        print(f"  └──────────────────────────────────────────┘")

    # === 各策略詳細 ===
    print(f"\n  ─── 各策略預測 ───")
    print(f"  {'策略':<14} {'預測':>4} {'準確率':>8} {'對/錯':>8}")
    print(f"  {'─'*40}")

    sorted_preds = sorted(predictions, key=lambda x: x['accuracy'], reverse=True)
    for p in sorted_preds:
        acc_str = f"{p['accuracy']:.1f}%" if p['total'] > 0 else "─"
        record = f"{p['correct']}/{p['wrong']}" if p['total'] > 0 else "─"
        marker = "✓" if p['accuracy'] > 50 and p['total'] >= 5 else " "
        print(f"  {marker} {p['strategy']:<12} {p['prediction']:>4} {acc_str:>8} {record:>8}")

    print(f"\n  {'─'*60}")


def parse_input(user_input: str) -> tuple:
    """
    解析使用者輸入
    支援格式:
      莊 / 閒 / 和 / b / p / t / B / P / T / 1(莊) / 2(閒) / 3(和)
      帶牌面: 莊 K5 82  (莊贏，閒家K5，莊家82)
      帶牌面: p A3K 972 (閒贏，閒家A3K，莊家972)
    """
    parts = user_input.strip().split()
    if not parts:
        return None, None, None

    # 解析結果
    outcome_map = {
        '莊': '莊', 'b': '莊', 'B': '莊', '1': '莊', 'banker': '莊',
        '閒': '閒', 'p': '閒', 'P': '閒', '2': '閒', 'player': '閒',
        '和': '和', 't': '和', 'T': '和', '3': '和', 'tie': '和',
    }

    outcome = outcome_map.get(parts[0])
    if outcome is None:
        return None, None, None

    player_cards = None
    banker_cards = None

    # 解析牌面（可選）
    if len(parts) >= 3:
        player_cards = list(parts[1].upper())
        banker_cards = list(parts[2].upper())
        # 處理 10
        player_cards = _fix_ten(player_cards)
        banker_cards = _fix_ten(banker_cards)

    return outcome, player_cards, banker_cards


def _fix_ten(cards: list) -> list:
    """處理牌面中的 10（'1','0' → '10'）"""
    result = []
    i = 0
    while i < len(cards):
        if cards[i] == '1' and i + 1 < len(cards) and cards[i+1] == '0':
            result.append('10')
            i += 2
        elif cards[i] == 'T':
            result.append('10')
            i += 1
        else:
            result.append(cards[i])
            i += 1
    return result


def load_csv(filepath: str, predictor: LivePredictor) -> int:
    """從 CSV 載入歷史紀錄"""
    count = 0
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 支援欄位名: outcome/結果/result
            outcome = (row.get('outcome') or row.get('結果') or
                      row.get('result') or row.get('Result') or '').strip()

            outcome_map = {
                '莊': '莊', 'B': '莊', 'Banker': '莊', 'banker': '莊',
                '閒': '閒', 'P': '閒', 'Player': '閒', 'player': '閒',
                '和': '和', 'T': '和', 'Tie': '和', 'tie': '和',
            }
            outcome = outcome_map.get(outcome)
            if outcome is None:
                continue

            # 嘗試讀取牌面
            p_cards_str = (row.get('player_cards') or row.get('閒家牌') or '').strip()
            b_cards_str = (row.get('banker_cards') or row.get('莊家牌') or '').strip()

            p_cards = list(p_cards_str.upper()) if p_cards_str else None
            b_cards = list(b_cards_str.upper()) if b_cards_str else None

            if p_cards:
                p_cards = _fix_ten(p_cards)
            if b_cards:
                b_cards = _fix_ten(b_cards)

            predictor.record_result(outcome, p_cards, b_cards)
            count += 1

    return count


def interactive_mode():
    """互動模式主迴圈"""
    predictor = LivePredictor()

    clear_screen()
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        百家樂即時預測系統 — 真實數據追蹤 v2.0             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  輸入格式:                                                  ║")
    print("║    結果: 莊/閒/和  或  B/P/T  或  1/2/3                   ║")
    print("║    帶牌: 莊 K5 82  (莊贏, 閒K5, 莊82)                     ║")
    print("║    帶牌: P A3K 972 (閒贏, 閒A3K, 莊972)                   ║")
    print("║                                                            ║")
    print("║  指令:                                                      ║")
    print("║    cards  — 顯示剩餘牌面詳情                               ║")
    print("║    prob   — 計算真實條件機率                                ║")
    print("║    road   — 顯示完整路紙                                   ║")
    print("║    stats  — 顯示完整統計                                   ║")
    print("║    save   — 儲存紀錄到 CSV                                 ║")
    print("║    load   — 從 CSV 載入歷史紀錄                            ║")
    print("║    new    — 新靴（重置）                                   ║")
    print("║    q/quit — 離開                                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            user_input = input("  ▶ 輸入結果: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再見！")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ('q', 'quit', 'exit', '離開'):
            print("\n  再見！")
            break

        elif cmd == 'new' or cmd == '新靴':
            predictor.reset()
            clear_screen()
            print("\n  🔄 已重置 — 新靴開始")
            continue

        elif cmd == 'cards' or cmd == '牌':
            print(f"\n  ─── 剩餘牌面 ───")
            print(predictor.tracker.counter.get_status_display())
            continue

        elif cmd == 'prob' or cmd == '機率':
            print(f"\n  ⏳ 計算真實條件機率中...")
            prob = predictor.get_real_probabilities(sample_size=50000)
            print(f"  閒: {prob['閒']:.2f}%  莊: {prob['莊']:.2f}%  和: {prob['和']:.2f}%")
            continue

        elif cmd == 'road' or cmd == '路':
            print(f"\n  ─── 大路 ───")
            print(predictor._build_road())
            continue

        elif cmd == 'stats' or cmd == '統計':
            stats = predictor.tracker.get_statistics()
            if stats:
                print(f"\n  ─── 完整統計 ───")
                for k, v in stats.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.2f}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print("  尚無紀錄")
            continue

        elif cmd.startswith('save'):
            parts = cmd.split()
            filename = parts[1] if len(parts) > 1 else 'real_data.csv'
            _save_records(predictor, filename)
            print(f"  ✅ 已儲存到 {filename}")
            continue

        elif cmd.startswith('load'):
            parts = cmd.split()
            if len(parts) < 2:
                print("  用法: load filename.csv")
                continue
            filepath = parts[1]
            if not os.path.exists(filepath):
                print(f"  ❌ 檔案不存在: {filepath}")
                continue
            count = load_csv(filepath, predictor)
            print(f"  ✅ 已載入 {count} 局紀錄")
            print_dashboard(predictor)
            continue

        # 解析牌局結果
        outcome, p_cards, b_cards = parse_input(user_input)
        if outcome is None:
            print("  ❌ 無效輸入！格式: 莊/閒/和 或 B/P/T 或 1/2/3")
            continue

        # 先顯示預測（記錄前），再記錄結果
        predictor.record_result(outcome, p_cards, b_cards)

        # 更新儀表板
        clear_screen()
        print_dashboard(predictor)


def _save_records(predictor: LivePredictor, filename: str):
    """儲存紀錄到 CSV"""
    with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['局數', '結果', '閒家牌', '莊家牌'])
        for r in predictor.tracker.results:
            writer.writerow([
                r['round'],
                r['outcome'],
                ''.join(r['player_cards']),
                ''.join(r['banker_cards']),
            ])


if __name__ == "__main__":
    interactive_mode()
