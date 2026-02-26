"""
百家樂實戰系統 — battle.py
============================
專為真實娛樂城實戰設計：
- 單鍵快速輸入（1=莊 2=閒 3=和）
- 真實資金追蹤（本金、下注、淨損益）
- 即時預測 + 建議注碼
- 場次自動存檔
- 停損停利提醒
"""

import os
import sys
import json
import csv
import time
from datetime import datetime
from typing import List, Dict, Optional
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from card_counter import CardCounter, ShoeTracker
from strategies import get_all_strategies
from baccarat_engine import PAYOUTS, THEORETICAL


class BattleSession:
    """實戰場次"""

    def __init__(self, bankroll: float = 10000, base_bet: float = 100,
                 stop_loss: float = None, stop_win: float = None):
        self.bankroll = bankroll          # 本金
        self.current_balance = bankroll   # 當前餘額
        self.base_bet = base_bet          # 基本注碼
        self.stop_loss = stop_loss or bankroll * 0.3     # 停損線（預設虧30%）
        self.stop_win = stop_win or bankroll * 0.5       # 停利線（預設賺50%）

        self.tracker = ShoeTracker()
        self.strategies = get_all_strategies(seed=None)
        self.strategy_stats: Dict[str, Dict] = {}
        for s in self.strategies:
            self.strategy_stats[s.name] = {'correct': 0, 'wrong': 0}

        self.bets: List[Dict] = []        # 下注紀錄
        self.results: List[str] = []      # 結果序列
        self.round_count = 0
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")

        # 連勝/連敗追蹤
        self.current_streak = 0  # 正=連贏 負=連輸
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.total_wagered = 0
        self.total_won = 0
        self.total_lost = 0

    def get_prediction(self) -> Dict:
        """取得下一局預測"""
        history = list(self.results)
        predictions = []

        for s in self.strategies:
            pred = s.predict(history)
            stats = self.strategy_stats[s.name]
            total = stats['correct'] + stats['wrong']
            acc = stats['correct'] / total * 100 if total > 0 else 50.0
            predictions.append({
                'name': s.name,
                'pred': pred,
                'acc': acc,
                'correct': stats['correct'],
                'wrong': stats['wrong'],
            })

        # 投票（加權）
        banker_score = 0
        player_score = 0
        for p in predictions:
            weight = max(p['acc'], 45) / 100
            # 準確率高的策略權重更大
            if p['total_rounds'] >= 10 if hasattr(p, 'total_rounds') else True:
                w = weight * 1.5 if p['acc'] > 55 else weight
            else:
                w = weight
            if p['pred'] == '莊':
                banker_score += w
            else:
                player_score += w

        if banker_score >= player_score:
            consensus = '莊'
            confidence = banker_score / (banker_score + player_score) * 100
        else:
            consensus = '閒'
            confidence = player_score / (banker_score + player_score) * 100

        # 建議注碼
        suggested_bet = self._calc_suggested_bet(confidence)

        return {
            'consensus': consensus,
            'confidence': confidence,
            'banker_score': banker_score,
            'player_score': player_score,
            'suggested_bet': suggested_bet,
            'predictions': sorted(predictions, key=lambda x: x['acc'], reverse=True),
        }

    def _calc_suggested_bet(self, confidence: float) -> float:
        """根據信心度計算建議注碼"""
        # 信心度越高，注碼越大（但永遠有上限）
        if confidence >= 80:
            multiplier = 2.0
        elif confidence >= 70:
            multiplier = 1.5
        elif confidence >= 60:
            multiplier = 1.0
        else:
            multiplier = 0.5  # 低信心 → 小注

        bet = self.base_bet * multiplier

        # 不超過餘額的 5%
        max_bet = self.current_balance * 0.05
        bet = min(bet, max_bet)
        bet = max(bet, self.base_bet * 0.5)  # 最少半注

        return round(bet, 0)

    def record_result(self, outcome: str, my_bet_side: Optional[str] = None,
                      my_bet_amount: Optional[float] = None):
        """
        記錄結果
        outcome: '莊'/'閒'/'和'
        my_bet_side: 我實際壓的 ('莊'/'閒'/None=沒壓)
        my_bet_amount: 我實際下注金額
        """
        history = list(self.results)
        self.round_count += 1

        # 更新策略統計
        if outcome != '和':
            for s in self.strategies:
                pred = s.predict(history)
                stats = self.strategy_stats[s.name]
                if pred == outcome:
                    stats['correct'] += 1
                else:
                    stats['wrong'] += 1

        # 記錄結果
        self.results.append(outcome)
        self.tracker.record_round(outcome)

        # 計算我的損益
        my_profit = 0
        if my_bet_side and my_bet_amount:
            self.total_wagered += my_bet_amount

            if outcome == '和':
                if my_bet_side == '和':
                    my_profit = my_bet_amount * 8
                    self.total_won += my_profit
                else:
                    my_profit = 0  # 和局退注
            elif my_bet_side == outcome:
                # 贏了
                payout = PAYOUTS.get(my_bet_side, 1.0)
                my_profit = my_bet_amount * payout
                self.total_won += my_profit

                if self.current_streak > 0:
                    self.current_streak += 1
                else:
                    self.current_streak = 1
                self.max_win_streak = max(self.max_win_streak, self.current_streak)
            else:
                # 輸了
                my_profit = -my_bet_amount
                self.total_lost += my_bet_amount

                if self.current_streak < 0:
                    self.current_streak -= 1
                else:
                    self.current_streak = -1
                self.max_loss_streak = max(self.max_loss_streak, abs(self.current_streak))

            self.current_balance += my_profit

        bet_record = {
            'round': self.round_count,
            'time': datetime.now().strftime("%H:%M:%S"),
            'outcome': outcome,
            'my_side': my_bet_side or '-',
            'my_amount': my_bet_amount or 0,
            'profit': my_profit,
            'balance': self.current_balance,
        }
        self.bets.append(bet_record)

        return my_profit

    def check_limits(self) -> Optional[str]:
        """檢查停損停利"""
        loss = self.bankroll - self.current_balance
        gain = self.current_balance - self.bankroll

        if loss >= self.stop_loss:
            return f"⛔ 已達停損線！虧損 {loss:,.0f} (上限 {self.stop_loss:,.0f})"
        if gain >= self.stop_win:
            return f"🎉 已達停利線！獲利 {gain:,.0f} (目標 {self.stop_win:,.0f})"
        return None

    def get_summary(self) -> Dict:
        """場次摘要"""
        duration = datetime.now() - self.session_start
        minutes = duration.total_seconds() / 60

        bet_rounds = [b for b in self.bets if b['my_side'] != '-']
        win_rounds = [b for b in bet_rounds if b['profit'] > 0]
        loss_rounds = [b for b in bet_rounds if b['profit'] < 0]

        return {
            '場次': self.session_id,
            '時長': f"{minutes:.0f} 分鐘",
            '總局數': self.round_count,
            '下注局數': len(bet_rounds),
            '贏': len(win_rounds),
            '輸': len(loss_rounds),
            '勝率': len(win_rounds) / max(len(bet_rounds), 1) * 100,
            '本金': self.bankroll,
            '餘額': self.current_balance,
            '淨損益': self.current_balance - self.bankroll,
            'ROI': (self.current_balance - self.bankroll) / self.bankroll * 100,
            '總下注': self.total_wagered,
            '最大連贏': self.max_win_streak,
            '最大連輸': self.max_loss_streak,
            '莊次數': self.results.count('莊'),
            '閒次數': self.results.count('閒'),
            '和次數': self.results.count('和'),
        }

    def save_session(self, directory: str = "sessions"):
        """儲存場次"""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"session_{self.session_id}.json")

        data = {
            'session_id': self.session_id,
            'start_time': self.session_start.isoformat(),
            'bankroll': self.bankroll,
            'base_bet': self.base_bet,
            'summary': self.get_summary(),
            'bets': self.bets,
            'results': self.results,
            'strategy_stats': {
                name: stats for name, stats in self.strategy_stats.items()
            },
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 也存 CSV
        csv_path = os.path.join(directory, f"session_{self.session_id}.csv")
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['round', 'time', 'outcome',
                                                    'my_side', 'my_amount',
                                                    'profit', 'balance'])
            writer.writeheader()
            writer.writerows(self.bets)

        return filepath


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def display_battle_screen(session: BattleSession, prediction: Dict,
                          last_profit: Optional[float] = None,
                          alert: Optional[str] = None):
    """實戰畫面"""
    clear()
    summary = session.get_summary()
    net = summary['淨損益']
    net_color = '+' if net >= 0 else ''

    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    print("┃          百 家 樂 實 戰 系 統                        ┃")
    print("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

    # === 資金狀態 ===
    print(f"\n  💰 餘額: {session.current_balance:>10,.0f}  "
          f"│ 本金: {session.bankroll:,.0f}  "
          f"│ 淨損益: {net_color}{net:,.0f}  "
          f"│ ROI: {net_color}{summary['ROI']:.1f}%")

    # 餘額條
    pct = session.current_balance / session.bankroll * 100
    bar_len = 30
    filled = max(0, min(bar_len, int(bar_len * pct / 150)))
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f"  [{bar}] {pct:.0f}%")

    # === 上一局結果 ===
    if last_profit is not None:
        if last_profit > 0:
            print(f"\n  ✅ 上一局: 贏 +{last_profit:,.0f}")
        elif last_profit < 0:
            print(f"\n  ❌ 上一局: 輸 {last_profit:,.0f}")
        else:
            print(f"\n  ➖ 上一局: 和局/未下注")

    if alert:
        print(f"\n  {alert}")

    # === 牌局紀錄 ===
    print(f"\n  📋 第 {session.round_count + 1} 局  "
          f"│ 已玩 {session.round_count} 局  "
          f"│ 莊 {summary['莊次數']} 閒 {summary['閒次數']} 和 {summary['和次數']}  "
          f"│ 勝率 {summary['勝率']:.0f}%")

    # 路紙（最近結果）
    recent = session.results[-30:]
    if recent:
        road = " ".join(recent)
        print(f"\n  路: {road}")

        # 連勝提示
        filtered = [r for r in session.results if r != '和']
        if filtered:
            last = filtered[-1]
            streak = 0
            for r in reversed(filtered):
                if r == last:
                    streak += 1
                else:
                    break
            if streak >= 3:
                print(f"  🔥 {last} 連 {streak}！")

    # ════════════════════════════════════════
    #  核心：下一局預測
    # ════════════════════════════════════════
    pred = prediction
    rec = pred['consensus']
    conf = pred['confidence']
    bet = pred['suggested_bet']

    print()
    print("  ╔══════════════════════════════════════╗")
    if conf >= 70:
        stars = "★★★"
    elif conf >= 60:
        stars = "★★☆"
    else:
        stars = "★☆☆"

    print(f"  ║                                      ║")
    print(f"  ║   下一局建議:  壓 【{rec}】            ║")
    print(f"  ║   信心度: {conf:.0f}%  {stars}                ║")
    print(f"  ║   建議注碼: {bet:,.0f}                    ║")
    print(f"  ║                                      ║")
    print("  ╚══════════════════════════════════════╝")

    # === 前 5 策略 ===
    print(f"\n  ── 策略投票 ──")
    top5 = pred['predictions'][:5]
    for p in top5:
        acc_str = f"{p['acc']:.0f}%" if (p['correct'] + p['wrong']) > 0 else "—"
        print(f"    {p['name']:<12} → {p['pred']}  (準確率 {acc_str})")

    print(f"\n  ─────────────────────────────────────────")
    print(f"  輸入: 1=莊  2=閒  3=和  │  s=跳過  q=結算離開")
    print(f"        下注方式: 先輸入結果，系統問你有沒有跟")


def run_battle():
    """實戰主程式"""
    clear()
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║         百家樂實戰系統 — 娛樂城專用             ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║  開始前請設定:                                   ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    # 設定本金
    while True:
        try:
            bankroll_input = input("  💰 你的本金是多少？(預設 10000): ").strip()
            bankroll = float(bankroll_input) if bankroll_input else 10000
            break
        except ValueError:
            print("  請輸入數字")

    while True:
        try:
            bet_input = input(f"  🎲 基本注碼？(預設 {bankroll * 0.01:.0f}): ").strip()
            base_bet = float(bet_input) if bet_input else bankroll * 0.01
            break
        except ValueError:
            print("  請輸入數字")

    while True:
        try:
            sl_input = input(f"  ⛔ 停損線？(預設虧 {bankroll * 0.3:.0f}): ").strip()
            stop_loss = float(sl_input) if sl_input else bankroll * 0.3
            break
        except ValueError:
            print("  請輸入數字")

    while True:
        try:
            sw_input = input(f"  🎉 停利線？(預設賺 {bankroll * 0.5:.0f}): ").strip()
            stop_win = float(sw_input) if sw_input else bankroll * 0.5
            break
        except ValueError:
            print("  請輸入數字")

    session = BattleSession(
        bankroll=bankroll,
        base_bet=base_bet,
        stop_loss=stop_loss,
        stop_win=stop_win,
    )

    print(f"\n  ✅ 設定完成！")
    print(f"     本金: {bankroll:,.0f}")
    print(f"     基本注碼: {base_bet:,.0f}")
    print(f"     停損: -{stop_loss:,.0f}  停利: +{stop_win:,.0f}")
    print(f"\n  按 Enter 開始...")
    input()

    last_profit = None
    alert = None

    while True:
        # 取得預測
        prediction = session.get_prediction()

        # 顯示畫面
        display_battle_screen(session, prediction, last_profit, alert)
        alert = None

        # 取得輸入
        try:
            user_input = input("\n  ▶ 開牌結果 (1=莊 2=閒 3=和 / q=結算): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input in ('q', 'quit', 'exit'):
            break

        if user_input in ('s', 'skip', '跳過'):
            continue

        # 解析結果
        outcome_map = {
            '1': '莊', '莊': '莊', 'b': '莊',
            '2': '閒', '閒': '閒', 'p': '閒',
            '3': '和', '和': '和', 't': '和',
        }
        outcome = outcome_map.get(user_input)
        if not outcome:
            alert = "❌ 無效輸入！1=莊 2=閒 3=和"
            continue

        # 問是否有跟注
        rec = prediction['consensus']
        suggested = prediction['suggested_bet']

        print(f"\n  系統建議壓: {rec} {suggested:,.0f}")
        follow_input = input(f"  你有下注嗎？(Enter=跟系統建議 / 金額=自訂 / n=沒壓): ").strip().lower()

        my_side = None
        my_amount = None

        if follow_input in ('n', 'no', '沒', '沒有', ''):
            if follow_input == '':
                # Enter = 跟系統建議
                my_side = rec
                my_amount = suggested
            else:
                my_side = None
                my_amount = None
        else:
            # 自訂下注
            try:
                # 格式: 金額 或 莊500 或 1 500
                parts = follow_input.split()
                if len(parts) == 1:
                    # 只有金額 → 跟系統推薦方向
                    my_amount = float(parts[0])
                    my_side = rec
                elif len(parts) == 2:
                    side_map = {'1': '莊', '莊': '莊', 'b': '莊',
                                '2': '閒', '閒': '閒', 'p': '閒'}
                    my_side = side_map.get(parts[0], rec)
                    my_amount = float(parts[1])
                else:
                    my_side = rec
                    my_amount = suggested
            except ValueError:
                my_side = rec
                my_amount = suggested

        # 記錄結果
        last_profit = session.record_result(outcome, my_side, my_amount)

        # 檢查停損停利
        limit_alert = session.check_limits()
        if limit_alert:
            alert = limit_alert

        # 自動存檔（每 10 局）
        if session.round_count % 10 == 0:
            session.save_session()

    # === 結算 ===
    session.save_session()
    show_final_report(session)


def show_final_report(session: BattleSession):
    """結算報告"""
    clear()
    summary = session.get_summary()
    net = summary['淨損益']

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║              場 次 結 算 報 告                   ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  場次: {summary['場次']}                          ║")
    print(f"  ║  時長: {summary['時長']}                               ║")
    print("  ╠══════════════════════════════════════════════════╣")

    if net >= 0:
        print(f"  ║  🎉 本場獲利: +{net:,.0f}                          ║")
    else:
        print(f"  ║  💸 本場虧損: {net:,.0f}                          ║")

    print("  ╚══════════════════════════════════════════════════╝")

    print(f"\n  ── 詳細數據 ──")
    print(f"  本金:       {summary['本金']:>10,.0f}")
    print(f"  結算餘額:   {summary['餘額']:>10,.0f}")
    print(f"  淨損益:     {'+' if net >= 0 else ''}{net:>10,.0f}")
    print(f"  ROI:        {'+' if net >= 0 else ''}{summary['ROI']:.1f}%")
    print(f"  總下注額:   {summary['總下注']:>10,.0f}")
    print()
    print(f"  總局數:     {summary['總局數']}")
    print(f"  下注局數:   {summary['下注局數']}")
    print(f"  贏:         {summary['贏']}")
    print(f"  輸:         {summary['輸']}")
    print(f"  勝率:       {summary['勝率']:.1f}%")
    print(f"  最大連贏:   {summary['最大連贏']}")
    print(f"  最大連輸:   {summary['最大連輸']}")
    print()
    print(f"  莊: {summary['莊次數']}  閒: {summary['閒次數']}  和: {summary['和次數']}")

    # 策略排名
    print(f"\n  ── 策略準確率排名 ──")
    strategy_ranking = []
    for name, stats in session.strategy_stats.items():
        total = stats['correct'] + stats['wrong']
        acc = stats['correct'] / total * 100 if total > 0 else 0
        strategy_ranking.append((name, acc, stats['correct'], stats['wrong']))

    strategy_ranking.sort(key=lambda x: x[1], reverse=True)
    for i, (name, acc, c, w) in enumerate(strategy_ranking):
        marker = "🏆" if i == 0 else "  "
        print(f"  {marker} {name:<14} {acc:>5.1f}%  ({c}贏/{w}輸)")

    filepath = session.save_session()
    print(f"\n  📁 紀錄已儲存: {filepath}")
    print(f"  📁 CSV已儲存: sessions/session_{session.session_id}.csv")
    print()


if __name__ == "__main__":
    run_battle()
