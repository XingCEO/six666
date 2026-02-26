"""
百家樂實戰 Web 介面 v2.0 — web_app.py
=======================================
全新設計：五路路紙 + 22 種策略 + 10 種注碼 + 專業介面
"""

import os
import sys
import json
import http.server
import socketserver
import threading
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from card_counter import ShoeTracker
from strategies import get_all_strategies
from baccarat_engine import PAYOUTS, THEORETICAL
from roads import RoadManager

# ─── 設定常數 ────────────────────────────────────────────────
RECENT_WINDOW = 15       # 近期表現追蹤窗口
MIN_ROUNDS_SMART = 6     # 智能模式啟動最低局數
CONFIDENCE_SKIP = 55     # 低於此信心度 → 建議跳局
CONFIDENCE_HIGH = 72     # 高於此 → 加注
CONFIDENCE_ULTRA = 82    # 超高信心 → 重注
HOT_THRESHOLD = 60       # ≥60% 近期準確率→熱策略
COLD_THRESHOLD = 40      # ≤40% 近期準確率→冷策略

# ─── 全域 session ────────────────────────────────────────────
SESSION_LOCK = threading.Lock()
SESSION = {
    'bankroll': 10000,
    'balance': 10000,
    'base_bet': 100,
    'results': [],
    'bets': [],
    'strategies': None,
    'strategy_stats': {},
    'round_count': 0,
    'total_wagered': 0,
    'current_streak': 0,
    'max_win_streak': 0,
    'max_loss_streak': 0,
    'road_manager': None,
    'shoe_tracker': None,
    'skip_count': 0,       # 建議跳局次數
    'skip_correct': 0,     # 跳局如果有壓會對幾次
}


def init_session(bankroll=10000, base_bet=100):
    global SESSION
    SESSION['bankroll'] = bankroll
    SESSION['balance'] = bankroll
    SESSION['base_bet'] = base_bet
    SESSION['results'] = []
    SESSION['bets'] = []
    SESSION['strategies'] = get_all_strategies(seed=None)
    SESSION['strategy_stats'] = {}
    for s in SESSION['strategies']:
        SESSION['strategy_stats'][s.name] = {
            'correct': 0, 'wrong': 0,
            'recent': [],           # 最近 RECENT_WINDOW 筆 True/False
            'recent_acc': 50.0,     # 近期準確率
            'streak': 0,            # 當前連對/連錯 (正=對, 負=錯)
            'enabled': True,        # 是否參與投票
        }
    SESSION['round_count'] = 0
    SESSION['total_wagered'] = 0
    SESSION['current_streak'] = 0
    SESSION['max_win_streak'] = 0
    SESSION['max_loss_streak'] = 0
    SESSION['road_manager'] = RoadManager()
    SESSION['shoe_tracker'] = ShoeTracker()
    SESSION['skip_count'] = 0
    SESSION['skip_correct'] = 0


def _calc_recent_acc(recent_list):
    """計算近期準確率"""
    if not recent_list:
        return 50.0
    return sum(1 for x in recent_list if x) / len(recent_list) * 100


def _get_card_counting_signal():
    """
    算牌信號：根據剩餘牌組的真實條件機率，判斷偏向哪方。
    回傳 {'side': '莊'/'閒'/None, 'edge': float, 'prob': dict}
    """
    tracker = SESSION.get('shoe_tracker')
    if not tracker or tracker.counter.get_dealt_count() < 10:
        return {'side': None, 'edge': 0, 'prob': None}

    try:
        # 用較小的 sample 求速度（2000次夠用）
        prob = tracker.counter.calculate_exact_probabilities(sample_size=2000)
    except Exception:
        return {'side': None, 'edge': 0, 'prob': None}

    b_prob = prob.get('莊', 45.86)
    p_prob = prob.get('閒', 44.62)

    # 扣除佣金後的期望值
    b_ev = b_prob * 0.95 - p_prob  # 莊贏扣5%佣金
    p_ev = p_prob - b_prob * 0.95

    # 與理論值的偏差
    b_shift = b_prob - 45.86
    p_shift = p_prob - 44.62

    edge = abs(b_shift - p_shift)

    if b_shift > p_shift + 0.3:  # 莊有明顯偏高
        return {'side': '莊', 'edge': edge, 'prob': prob}
    elif p_shift > b_shift + 0.3:
        return {'side': '閒', 'edge': edge, 'prob': prob}
    return {'side': None, 'edge': edge, 'prob': prob}


def get_prediction():
    """
    核心預測引擎 v3.0
    三層信號疊加：
      Layer 1 — 加權策略投票（近期準確率加權,冷策略降權）
      Layer 2 — 五路路紙模式分析
      Layer 3 — 算牌真實條件機率
    最終信心度 = 綜合三層的一致性
    """
    history = list(SESSION['results'])
    round_num = SESSION['round_count']

    # ══════════════════════════════════════════════
    #  Layer 1: 加權策略投票
    # ══════════════════════════════════════════════
    preds = []
    weighted_scores = {'莊': 0.0, '閒': 0.0}
    raw_votes = {'莊': 0, '閒': 0}

    for s in SESSION['strategies']:
        pred = s.predict(history)
        stats = SESSION['strategy_stats'][s.name]
        total = stats['correct'] + stats['wrong']
        acc_all = stats['correct'] / total * 100 if total > 0 else 50
        acc_recent = stats['recent_acc']

        # 權重計算：近期準確率為主，總準確率為輔
        if total >= 5:
            # 近期權重 70% + 總體權重 30%
            weight = (acc_recent * 0.7 + acc_all * 0.3) / 100
        else:
            weight = 0.5  # 數據不足，中性權重

        # 冷策略（近期 < COLD_THRESHOLD）懲罰
        if acc_recent < COLD_THRESHOLD and len(stats['recent']) >= 5:
            weight *= 0.3  # 大幅降權

        # 熱策略（近期 > HOT_THRESHOLD）加成
        if acc_recent > HOT_THRESHOLD and len(stats['recent']) >= 5:
            weight *= 1.5

        # 連對加成 / 連錯懲罰
        if stats['streak'] >= 3:
            weight *= 1.3
        elif stats['streak'] <= -3:
            weight *= 0.4

        weighted_scores[pred] += weight
        raw_votes[pred] += 1

        # 判斷是否為熱門策略
        status = 'normal'
        if len(stats['recent']) >= 5:
            if acc_recent >= HOT_THRESHOLD:
                status = 'hot'
            elif acc_recent <= COLD_THRESHOLD:
                status = 'cold'

        preds.append({
            'name': s.name, 'desc': s.description,
            'pred': pred,
            'acc': acc_all, 'acc_recent': acc_recent,
            'correct': stats['correct'], 'wrong': stats['wrong'],
            'weight': round(weight, 2),
            'status': status,
            'streak': stats['streak'],
        })

    total_weight = weighted_scores['莊'] + weighted_scores['閒']
    if total_weight > 0:
        layer1_banker_pct = weighted_scores['莊'] / total_weight * 100
    else:
        layer1_banker_pct = 50

    if layer1_banker_pct >= 50:
        layer1_side = '莊'
        layer1_conf = layer1_banker_pct
    else:
        layer1_side = '閒'
        layer1_conf = 100 - layer1_banker_pct

    # ══════════════════════════════════════════════
    #  Layer 2: 路紙分析
    # ══════════════════════════════════════════════
    rm = SESSION['road_manager']
    road_pred = rm.predict_by_roads() if rm else {'side': '莊', 'confidence': 50, 'reasons': []}
    layer2_side = road_pred['side']
    layer2_conf = road_pred['confidence']
    layer2_reasons = road_pred.get('reasons', [])

    # ══════════════════════════════════════════════
    #  Layer 3: 算牌機率
    # ══════════════════════════════════════════════
    card_signal = _get_card_counting_signal()
    layer3_side = card_signal['side']  # 可能是 None
    layer3_edge = card_signal['edge']
    card_prob = card_signal['prob']

    # ══════════════════════════════════════════════
    #  綜合三層：一致性越高，信心越大
    # ══════════════════════════════════════════════
    signals = []
    signal_weights = []

    # Layer 1: 策略投票 (權重 40%)
    signals.append(layer1_side)
    signal_weights.append(40)

    # Layer 2: 路紙 (權重 35%）— 有模式時加權
    if layer2_reasons and layer2_reasons[0] not in ('數據不足', '無明顯模式'):
        signals.append(layer2_side)
        w2 = 35
        if layer2_conf > 70:
            w2 = 45  # 路紙信號很強時
        signal_weights.append(w2)

    # Layer 3: 算牌 (權重 25%) — 有信號時才加入
    if layer3_side:
        signals.append(layer3_side)
        w3 = 25 + min(layer3_edge * 5, 15)  # 偏差越大權重越高
        signal_weights.append(w3)

    # 計算綜合分數
    if not signals:
        consensus = '莊'
        confidence = 50.0
    else:
        banker_total = sum(w for s, w in zip(signals, signal_weights) if s == '莊')
        player_total = sum(w for s, w in zip(signals, signal_weights) if s == '閒')
        grand_total = banker_total + player_total

        if grand_total > 0:
            banker_pct = banker_total / grand_total * 100
        else:
            banker_pct = 50

        if banker_pct >= 50:
            consensus = '莊'
            confidence = banker_pct
        else:
            consensus = '閒'
            confidence = 100 - banker_pct

    # ══════════════════════════════════════════════
    #  下注建議
    # ══════════════════════════════════════════════
    base = SESSION['base_bet']
    skip = False

    if round_num < MIN_ROUNDS_SMART:
        # 前幾局觀望
        skip = True
        bet = 0
        skip_reason = f'觀察期（前{MIN_ROUNDS_SMART}局）'
    elif confidence < CONFIDENCE_SKIP:
        skip = True
        bet = 0
        skip_reason = f'信心不足 ({confidence:.0f}% < {CONFIDENCE_SKIP}%)'
    elif confidence >= CONFIDENCE_ULTRA:
        bet = base * 2.5
        skip_reason = ''
    elif confidence >= CONFIDENCE_HIGH:
        bet = base * 1.5
        skip_reason = ''
    else:
        bet = base
        skip_reason = ''

    # 當前連敗保護：連輸 3 局以上降注
    if SESSION['current_streak'] <= -3:
        bet = max(base * 0.5, bet * 0.5)
        skip_reason = f'連敗保護 (連輸{abs(SESSION["current_streak"])}局)'

    return {
        'consensus': consensus,
        'confidence': round(confidence, 1),
        'suggested_bet': round(bet, 0),
        'skip': skip,
        'skip_reason': skip_reason if skip else '',
        'banker_votes': raw_votes['莊'],
        'player_votes': raw_votes['閒'],
        'weighted_banker': round(weighted_scores['莊'], 1),
        'weighted_player': round(weighted_scores['閒'], 1),
        'strategies': sorted(preds, key=lambda x: x['acc_recent'], reverse=True),
        'road_prediction': road_pred,
        'card_signal': {
            'side': layer3_side,
            'edge': round(layer3_edge, 2),
            'prob': card_prob,
        },
        'layers': {
            'strategy': {'side': layer1_side, 'conf': round(layer1_conf, 1)},
            'road': {'side': layer2_side, 'conf': round(layer2_conf, 1), 'reasons': layer2_reasons},
            'card': {'side': layer3_side, 'edge': round(layer3_edge, 2)},
        },
    }


def record_result(outcome, my_side=None, my_amount=None):
    history = list(SESSION['results'])

    # 更新各策略的預測紀錄（含近期追蹤）
    if outcome != '和':
        for s in SESSION['strategies']:
            pred = s.predict(history)
            stats = SESSION['strategy_stats'][s.name]
            is_correct = (pred == outcome)

            if is_correct:
                stats['correct'] += 1
                stats['streak'] = max(1, stats['streak'] + 1) if stats['streak'] >= 0 else 1
            else:
                stats['wrong'] += 1
                stats['streak'] = min(-1, stats['streak'] - 1) if stats['streak'] <= 0 else -1

            # 近期窗口
            stats['recent'].append(is_correct)
            if len(stats['recent']) > RECENT_WINDOW:
                stats['recent'].pop(0)
            stats['recent_acc'] = _calc_recent_acc(stats['recent'])

    SESSION['results'].append(outcome)
    SESSION['round_count'] += 1

    # 更新路紙
    if SESSION['road_manager']:
        SESSION['road_manager'].add_result(outcome)

    # 更新 shoe tracker
    if SESSION['shoe_tracker']:
        SESSION['shoe_tracker'].record_round(outcome)

    profit = 0
    if my_side and my_amount:
        SESSION['total_wagered'] += my_amount
        if outcome == '和':
            profit = 0
        elif my_side == outcome:
            payout = PAYOUTS.get(my_side, 1.0)
            profit = my_amount * payout
            if SESSION['current_streak'] > 0:
                SESSION['current_streak'] += 1
            else:
                SESSION['current_streak'] = 1
            SESSION['max_win_streak'] = max(SESSION['max_win_streak'], SESSION['current_streak'])
        else:
            profit = -my_amount
            if SESSION['current_streak'] < 0:
                SESSION['current_streak'] -= 1
            else:
                SESSION['current_streak'] = -1
            SESSION['max_loss_streak'] = max(SESSION['max_loss_streak'], abs(SESSION['current_streak']))

        SESSION['balance'] += profit

    SESSION['bets'].append({
        'round': SESSION['round_count'],
        'outcome': outcome,
        'my_side': my_side or '-',
        'my_amount': my_amount or 0,
        'profit': profit,
        'balance': SESSION['balance'],
    })

    return profit


def get_state():
    pred = get_prediction() if SESSION['strategies'] else {}
    results = SESSION['results']
    b = results.count('莊')
    p = results.count('閒')
    t = results.count('和')
    total = len(results)

    filtered = [r for r in results if r != '和']
    streak_side = ''
    streak_count = 0
    if filtered:
        streak_side = filtered[-1]
        for r in reversed(filtered):
            if r == streak_side:
                streak_count += 1
            else:
                break

    bet_rounds = [b2 for b2 in SESSION['bets'] if b2['my_side'] != '-']
    wins = sum(1 for b2 in bet_rounds if b2['profit'] > 0)

    # 路紙數據
    rm = SESSION['road_manager']
    roads_data = rm.get_all_roads_data() if rm else {}

    # 計算歷史準確率（系統建議 vs 實際）
    system_record = []
    for b2 in SESSION['bets']:
        if b2['my_side'] != '-' and b2['outcome'] != '和':
            system_record.append(b2['profit'] > 0)
    sys_acc = sum(system_record) / len(system_record) * 100 if system_record else 0

    return {
        'round': SESSION['round_count'],
        'bankroll': SESSION['bankroll'],
        'balance': SESSION['balance'],
        'net': SESSION['balance'] - SESSION['bankroll'],
        'roi': (SESSION['balance'] - SESSION['bankroll']) / max(SESSION['bankroll'], 1) * 100,
        'banker_count': b,
        'player_count': p,
        'tie_count': t,
        'total': total,
        'results': results[-80:],
        'streak_side': streak_side,
        'streak_count': streak_count,
        'prediction': pred,
        'last_bets': SESSION['bets'][-15:],
        'bet_count': len(bet_rounds),
        'win_count': wins,
        'win_rate': wins / max(len(bet_rounds), 1) * 100,
        'system_accuracy': round(sys_acc, 1),
        'max_win_streak': SESSION['max_win_streak'],
        'max_loss_streak': SESSION['max_loss_streak'],
        'skip_count': SESSION['skip_count'],
        'roads': roads_data,
    }


# ─── HTML 頁面 ────────────────────────────────────────────────
HTML_PAGE = r'''<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>百家樂實戰系統 Pro</title>
<style>
:root {
  --bg: #0a0a14;
  --card: #12121f;
  --card2: #181830;
  --border: #252540;
  --banker: #e63946;
  --banker-dark: #991122;
  --player: #457b9d;
  --player-dark: #1d3557;
  --tie: #2a9d8f;
  --gold: #f4c430;
  --green: #06d6a0;
  --red: #ef476f;
  --text: #e8e8f0;
  --text2: #8888aa;
  --text3: #555570;
  --road-red: #e63946;
  --road-blue: #457b9d;
  --derived-red: #ef476f;
  --derived-blue: #118ab2;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
html { font-size: 14px; }
body {
  font-family: -apple-system, 'Microsoft JhengHei', 'PingFang TC', 'Noto Sans TC', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  max-width: 600px;
  margin: 0 auto;
  padding: 6px;
  -webkit-tap-highlight-color: transparent;
  user-select: none;
}

/* ── Header ── */
.header {
  text-align: center;
  padding: 10px 0 6px;
  background: linear-gradient(180deg, #1a1a38 0%, var(--bg) 100%);
  border-bottom: 1px solid var(--border);
  margin-bottom: 6px;
}
.header h1 { font-size: 1.15rem; color: var(--gold); letter-spacing: 2px; }
.header .sub { font-size: 0.7rem; color: var(--text3); margin-top: 2px; }

/* ── Balance Bar ── */
.bal-bar {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 4px;
  margin-bottom: 6px;
}
.bal-item {
  background: var(--card);
  border-radius: 8px;
  padding: 6px 4px;
  text-align: center;
  border: 1px solid var(--border);
}
.bal-item .l { font-size: 0.6rem; color: var(--text3); text-transform: uppercase; }
.bal-item .v { font-size: 1rem; font-weight: 700; margin-top: 2px; }
.pos { color: var(--green); }
.neg { color: var(--red); }

/* ── Prediction ── */
.pred-box {
  background: var(--card);
  border: 2px solid var(--gold);
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 6px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.pred-box::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at center, rgba(244,196,48,0.05) 0%, transparent 60%);
}
.pred-label { font-size: 0.75rem; color: var(--text2); }
.pred-main {
  font-size: 2.8rem;
  font-weight: 900;
  margin: 4px 0;
  text-shadow: 0 0 20px rgba(255,255,255,0.15);
}
.pred-main.bk { color: var(--banker); }
.pred-main.pl { color: var(--player); }
.pred-main.skip-mode { color: var(--text3); font-size: 2rem; }
.vote-bar {
  display: flex;
  height: 6px;
  border-radius: 3px;
  overflow: hidden;
  margin: 6px 0;
  background: #222;
}
.vote-bar .vb { background: var(--banker); transition: width 0.3s; }
.vote-bar .vp { background: var(--player); transition: width 0.3s; }
.pred-info { font-size: 0.8rem; color: var(--gold); }
.pred-bet { font-size: 0.85rem; color: var(--green); margin-top: 2px; }
.pred-bet.skip-bet { color: var(--red); }

/* ── Skip Banner ── */
.skip-banner {
  background: linear-gradient(90deg, rgba(239,71,111,0.15), rgba(239,71,111,0.05));
  border: 1px solid rgba(239,71,111,0.3);
  border-radius: 8px;
  padding: 8px 12px;
  margin-bottom: 6px;
  text-align: center;
  font-size: 0.85rem;
  color: var(--red);
  font-weight: 600;
  display: none;
}
.skip-banner .skip-reason { font-size: 0.7rem; color: var(--text2); font-weight: 400; margin-top: 2px; }

/* ── Three Layer Panel ── */
.layer-panel {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 4px;
  margin-bottom: 6px;
}
.layer-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 6px 4px;
  text-align: center;
}
.layer-card .lc-title { font-size: 0.55rem; color: var(--text3); margin-bottom: 3px; }
.layer-card .lc-side { font-size: 1.1rem; font-weight: 800; }
.layer-card .lc-side.bk { color: var(--banker); }
.layer-card .lc-side.pl { color: var(--player); }
.layer-card .lc-side.none { color: var(--text3); }
.layer-card .lc-conf { font-size: 0.65rem; color: var(--gold); margin-top: 2px; }
.layer-card .lc-detail { font-size: 0.55rem; color: var(--text3); margin-top: 1px; }
.layer-card.agree { border-color: var(--green); }
.layer-card.disagree { border-color: var(--red); }

/* ── Card Counting Display ── */
.card-prob {
  display: flex;
  gap: 6px;
  margin-bottom: 6px;
  font-size: 0.7rem;
}
.card-prob .cp-item {
  flex: 1;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 4px;
  text-align: center;
}
.card-prob .cp-label { color: var(--text3); font-size: 0.55rem; }
.card-prob .cp-val { font-weight: 700; font-size: 0.85rem; margin-top: 1px; }

/* ── System Accuracy ── */
.sys-acc-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 10px;
  font-size: 0.7rem;
}
.sys-acc-bar .sa-label { color: var(--text3); }
.sys-acc-bar .sa-val { font-weight: 700; font-size: 0.9rem; }
.sys-acc-bar .sa-skip { margin-left: auto; color: var(--text3); }

/* ── Pattern Alert ── */
.pattern-box {
  background: linear-gradient(90deg, rgba(244,196,48,0.1), transparent);
  border-left: 3px solid var(--gold);
  border-radius: 0 8px 8px 0;
  padding: 6px 10px;
  margin-bottom: 6px;
  font-size: 0.8rem;
  display: none;
}
.pattern-box .reason {
  color: var(--gold);
  font-weight: 600;
}

/* ── Road Tabs ── */
.road-tabs {
  display: flex;
  gap: 2px;
  margin-bottom: 4px;
  background: var(--card);
  border-radius: 8px 8px 0 0;
  overflow: hidden;
}
.road-tab {
  flex: 1;
  padding: 6px 0;
  text-align: center;
  font-size: 0.7rem;
  color: var(--text3);
  cursor: pointer;
  transition: all 0.2s;
  background: transparent;
  border: none;
}
.road-tab.active {
  color: var(--gold);
  background: var(--card2);
  font-weight: 700;
}

/* ── Road Container ── */
.road-container {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 0 0 8px 8px;
  padding: 6px;
  margin-bottom: 6px;
  overflow-x: auto;
  min-height: 90px;
  -webkit-overflow-scrolling: touch;
}
.road-grid {
  display: inline-grid;
  gap: 1px;
}
.road-grid .cell {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.45rem;
  font-weight: 700;
  color: white;
  position: relative;
}
.road-grid .cell.bk { background: var(--road-red); }
.road-grid .cell.pl { background: var(--road-blue); }
.road-grid .cell.ti { background: var(--tie); }
.road-grid .cell.empty { background: transparent; }

/* 對子標記 */
.cell .pair-dot {
  position: absolute;
  width: 5px; height: 5px;
  border-radius: 50%;
}
.cell .bp-dot { top: -1px; left: -1px; background: var(--banker); border: 1px solid #fff; }
.cell .pp-dot { bottom: -1px; right: -1px; background: var(--player); border: 1px solid #fff; }

/* 和局在大路上的標記 */
.cell .tie-line {
  position: absolute;
  width: 100%;
  height: 2px;
  background: var(--tie);
  top: 50%;
  transform: translateY(-50%);
}
.cell .tie-count {
  position: absolute;
  top: -8px;
  right: -4px;
  font-size: 0.4rem;
  color: var(--tie);
  font-weight: 700;
}

/* 衍生路的小圓 */
.road-grid.derived .cell {
  width: 12px;
  height: 12px;
  border-width: 2px;
  border-style: solid;
  background: transparent !important;
}
.road-grid.derived .cell.dr { border-color: var(--derived-red); }
.road-grid.derived .cell.db { border-color: var(--derived-blue); }
.road-grid.derived .cell.filled-r { background: var(--derived-red) !important; }
.road-grid.derived .cell.filled-b { background: var(--derived-blue) !important; }

/* ── Buttons ── */
.btn-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 6px;
  margin-bottom: 6px;
}
.btn {
  padding: 14px 0;
  border: none;
  border-radius: 10px;
  font-size: 1.4rem;
  font-weight: 900;
  cursor: pointer;
  color: white;
  transition: transform 0.08s, box-shadow 0.2s;
  position: relative;
  overflow: hidden;
}
.btn::after {
  content: '';
  position: absolute;
  top: 50%; left: 50%;
  width: 0; height: 0;
  background: rgba(255,255,255,0.2);
  border-radius: 50%;
  transform: translate(-50%, -50%);
  transition: width 0.3s, height 0.3s;
}
.btn:active::after { width: 200%; height: 200%; }
.btn:active { transform: scale(0.96); }
.btn-bk {
  background: linear-gradient(135deg, var(--banker) 0%, var(--banker-dark) 100%);
  box-shadow: 0 4px 12px rgba(230,57,70,0.3);
}
.btn-pl {
  background: linear-gradient(135deg, var(--player) 0%, var(--player-dark) 100%);
  box-shadow: 0 4px 12px rgba(69,123,157,0.3);
}
.btn-ti {
  background: linear-gradient(135deg, var(--tie) 0%, #1a7a6f 100%);
  box-shadow: 0 4px 12px rgba(42,157,143,0.3);
}
.btn .sub-text {
  display: block;
  font-size: 0.55rem;
  font-weight: 400;
  opacity: 0.7;
  margin-top: 2px;
}

/* ── Bet Input ── */
.bet-bar {
  display: flex;
  gap: 6px;
  margin-bottom: 6px;
}
.bet-input {
  flex: 1;
  padding: 8px 12px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-size: 1rem;
  text-align: center;
  outline: none;
}
.bet-input:focus { border-color: var(--gold); }
.bet-input::placeholder { color: var(--text3); }
.bet-toggle {
  padding: 8px 14px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text2);
  cursor: pointer;
  font-size: 0.8rem;
  white-space: nowrap;
}
.bet-toggle.on { background: var(--gold); color: #000; border-color: var(--gold); font-weight: 700; }

/* ── Section ── */
.section {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 6px;
  overflow: hidden;
}
.section-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 10px;
  font-size: 0.75rem;
  color: var(--text2);
  cursor: pointer;
  border-bottom: 1px solid var(--border);
}
.section-head .toggle { font-size: 0.6rem; color: var(--text3); }
.section-body { max-height: 300px; overflow-y: auto; }
.section-body.collapsed { display: none; }

/* ── Strategy Row ── */
.s-row {
  display: grid;
  grid-template-columns: 1fr 36px 36px 70px;
  align-items: center;
  padding: 5px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  font-size: 0.75rem;
}
.s-row:last-child { border-bottom: none; }
.s-row .nm { color: var(--text2); display: flex; align-items: center; gap: 4px; }
.s-row .pr { font-weight: 700; text-align: center; }
.s-row .pr.bk { color: var(--banker); }
.s-row .pr.pl { color: var(--player); }
.s-row .wt { text-align: center; color: var(--text3); font-size: 0.6rem; }
.s-row .ac { text-align: right; color: var(--gold); font-size: 0.65rem; }
.s-row .ac.hot { color: var(--green); }
.s-row .ac.cold { color: var(--red); }
.s-row.hot-row { background: rgba(6,214,160,0.06); }
.s-row.cold-row { background: rgba(239,71,111,0.04); opacity: 0.6; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.status-dot.hot { background: var(--green); }
.status-dot.cold { background: var(--red); }
.streak-badge { font-size: 0.55rem; padding: 1px 3px; border-radius: 3px; font-weight: 600; }
.streak-badge.pos { background: rgba(6,214,160,0.2); color: var(--green); }
.streak-badge.neg { background: rgba(239,71,111,0.2); color: var(--red); }

/* ── Stats Bar ── */
.stats-bar {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 4px;
  margin-bottom: 6px;
}
.stat-item {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 4px;
  text-align: center;
}
.stat-item .sl { font-size: 0.55rem; color: var(--text3); }
.stat-item .sv { font-size: 0.85rem; font-weight: 700; }

/* ── History ── */
.h-row {
  display: grid;
  grid-template-columns: 36px 32px 32px 50px 55px 60px;
  align-items: center;
  padding: 4px 8px;
  font-size: 0.7rem;
  border-bottom: 1px solid rgba(255,255,255,0.03);
}
.h-row:last-child { border-bottom: none; }
.h-header { color: var(--text3); font-weight: 600; font-size: 0.6rem; }

/* ── Controls ── */
.ctrl-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  margin-bottom: 20px;
}
.ctrl {
  padding: 10px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text2);
  cursor: pointer;
  font-size: 0.8rem;
  text-align: center;
}
.ctrl:active { background: var(--card2); }

/* ── Road Prediction Widget ── */
.road-pred {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background: var(--card2);
  border-radius: 6px;
  margin-bottom: 6px;
  font-size: 0.75rem;
}
.road-pred .rp-label { color: var(--text3); }
.road-pred .rp-side { font-weight: 700; font-size: 1rem; }
.road-pred .rp-side.bk { color: var(--banker); }
.road-pred .rp-side.pl { color: var(--player); }
.road-pred .rp-reasons { flex: 1; color: var(--text2); font-size: 0.65rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── Toast ── */
.toast {
  position: fixed;
  top: 20px;
  left: 50%;
  transform: translateX(-50%) translateY(-100px);
  background: var(--card2);
  border: 1px solid var(--gold);
  border-radius: 8px;
  padding: 8px 20px;
  font-size: 0.85rem;
  color: var(--gold);
  z-index: 999;
  transition: transform 0.3s;
  pointer-events: none;
}
.toast.show { transform: translateX(-50%) translateY(0); }
</style>
</head>
<body>

<div class="header">
  <h1>百家樂實戰系統 Pro</h1>
  <div class="sub">三層信號疊加 ・ 22策略加權投票 ・ 五路路紙 ・ 算牌機率</div>
</div>

<!-- Balance -->
<div class="bal-bar" id="balBar">
  <div class="bal-item"><div class="l">本金</div><div class="v" id="vBankroll">-</div></div>
  <div class="bal-item"><div class="l">餘額</div><div class="v" id="vBalance">-</div></div>
  <div class="bal-item"><div class="l">損益</div><div class="v" id="vNet">-</div></div>
  <div class="bal-item"><div class="l">ROI</div><div class="v" id="vRoi">-</div></div>
  <div class="bal-item"><div class="l">勝率</div><div class="v" id="vWinRate">-</div></div>
</div>

<!-- Prediction -->
<div class="pred-box" id="predBox">
  <div class="pred-label">第 <span id="nextRound">1</span> 局 ─ 預測</div>
  <div class="pred-main bk" id="predMain">莊</div>
  <div class="vote-bar"><div class="vb" id="voteB" style="width:50%"></div><div class="vp" id="voteP" style="width:50%"></div></div>
  <div class="pred-info" id="predInfo">信心度 50% (0莊 vs 0閒)</div>
  <div class="pred-bet" id="predBet">建議注碼: 100</div>
</div>

<!-- Skip Banner -->
<div class="skip-banner" id="skipBanner">
  ⚠ 建議此局觀望，不下注
  <div class="skip-reason" id="skipReason"></div>
</div>

<!-- Three Layer Breakdown -->
<div class="layer-panel" id="layerPanel">
  <div class="layer-card" id="layer1">
    <div class="lc-title">策略投票</div>
    <div class="lc-side none" id="l1Side">-</div>
    <div class="lc-conf" id="l1Conf">-</div>
  </div>
  <div class="layer-card" id="layer2">
    <div class="lc-title">路紙分析</div>
    <div class="lc-side none" id="l2Side">-</div>
    <div class="lc-conf" id="l2Conf">-</div>
    <div class="lc-detail" id="l2Detail"></div>
  </div>
  <div class="layer-card" id="layer3">
    <div class="lc-title">算牌機率</div>
    <div class="lc-side none" id="l3Side">-</div>
    <div class="lc-conf" id="l3Conf">-</div>
  </div>
</div>

<!-- Card Counting Probs -->
<div class="card-prob" id="cardProb" style="display:none">
  <div class="cp-item"><div class="cp-label">莊機率</div><div class="cp-val" id="cpBk" style="color:var(--banker)">-</div></div>
  <div class="cp-item"><div class="cp-label">閒機率</div><div class="cp-val" id="cpPl" style="color:var(--player)">-</div></div>
  <div class="cp-item"><div class="cp-label">和機率</div><div class="cp-val" id="cpTi" style="color:var(--tie)">-</div></div>
</div>

<!-- System Accuracy -->
<div class="sys-acc-bar" id="sysAccBar">
  <span class="sa-label">系統準確率:</span>
  <span class="sa-val" id="sysAcc">-</span>
  <span class="sa-skip" id="sysSkip"></span>
</div>

<!-- Road Prediction -->
<div class="road-pred" id="roadPred" style="display:none">
  <span class="rp-label">路紙:</span>
  <span class="rp-side" id="rpSide">-</span>
  <span class="rp-reasons" id="rpReasons">-</span>
</div>

<!-- Pattern Alert -->
<div class="pattern-box" id="patternBox"></div>

<!-- Road Tabs -->
<div class="road-tabs" id="roadTabs">
  <button class="road-tab active" data-road="big_road">大路</button>
  <button class="road-tab" data-road="bead">珠盤路</button>
  <button class="road-tab" data-road="big_eye">大眼仔</button>
  <button class="road-tab" data-road="small_road">小路</button>
  <button class="road-tab" data-road="cockroach">曱甴路</button>
</div>
<div class="road-container" id="roadContainer">
  <div style="color:var(--text3);text-align:center;padding:20px;font-size:0.8rem">等待開牌...</div>
</div>

<!-- Buttons -->
<div class="btn-row">
  <button class="btn btn-bk" onclick="rec('莊')">莊<span class="sub-text">按 1</span></button>
  <button class="btn btn-pl" onclick="rec('閒')">閒<span class="sub-text">按 2</span></button>
  <button class="btn btn-ti" onclick="rec('和')">和<span class="sub-text">按 3</span></button>
</div>

<!-- Bet Input -->
<div class="bet-bar">
  <input type="number" class="bet-input" id="betAmt" placeholder="下注金額 (空=跟建議)">
  <button class="bet-toggle on" id="followBtn" onclick="toggleFollow()">跟注 ON</button>
</div>

<!-- Stats -->
<div class="stats-bar" id="statsBar">
  <div class="stat-item"><div class="sl">莊</div><div class="sv" id="sBk" style="color:var(--banker)">0</div></div>
  <div class="stat-item"><div class="sl">閒</div><div class="sv" id="sPl" style="color:var(--player)">0</div></div>
  <div class="stat-item"><div class="sl">和</div><div class="sv" id="sTi" style="color:var(--tie)">0</div></div>
  <div class="stat-item"><div class="sl">局數</div><div class="sv" id="sTotal">0</div></div>
  <div class="stat-item"><div class="sl">當前連</div><div class="sv" id="sStreak">-</div></div>
  <div class="stat-item"><div class="sl">最長連</div><div class="sv" id="sMaxStreak">-</div></div>
</div>

<!-- Strategies -->
<div class="section" id="stratSection">
  <div class="section-head" onclick="toggleSection('stratBody')">
    <span>📊 策略投票 (22種) ─ 按近期準確率排序</span><span class="toggle" id="stratToggle">▼</span>
  </div>
  <div class="section-body" id="stratBody"></div>
</div>

<!-- History -->
<div class="section" id="histSection">
  <div class="section-head" onclick="toggleSection('histBody')">
    <span>📜 下注紀錄</span><span class="toggle" id="histToggle">▼</span>
  </div>
  <div class="section-body" id="histBody">
    <div class="h-row h-header"><span>#</span><span>開</span><span>壓</span><span>金額</span><span>損益</span><span>餘額</span></div>
  </div>
</div>

<!-- Controls -->
<div class="ctrl-row">
  <button class="ctrl" onclick="newShoe()">🔄 新靴</button>
  <button class="ctrl" onclick="undo()">↩ 撤回</button>
  <button class="ctrl" onclick="setup()">⚙ 設定</button>
</div>

<div class="toast" id="toast"></div>

<script>
// ─── State ───
let follow = true;
let currentRoad = 'big_road';
let STATE = {};

// ─── Helpers ───
const $ = id => document.getElementById(id);
const fmt = n => n.toLocaleString('zh-TW', {maximumFractionDigits: 0});

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 1500);
}

function toggleFollow() {
  follow = !follow;
  const b = $('followBtn');
  b.textContent = follow ? '跟注 ON' : '跟注 OFF';
  b.className = 'bet-toggle' + (follow ? ' on' : '');
}

function toggleSection(id) {
  const el = $(id);
  el.classList.toggle('collapsed');
  const togId = id.replace('Body', 'Toggle');
  const tog = $(togId);
  if (tog) tog.textContent = el.classList.contains('collapsed') ? '▶' : '▼';
}

// ─── Road Rendering ───
function renderBigRoad(data) {
  // data.big_road is array of columns, each column is array of cells
  const columns = data.big_road || [];
  if (!columns.length) return '<div style="color:var(--text3);text-align:center;padding:15px;font-size:0.8rem">等待數據...</div>';

  const maxRows = 6;
  // Build grid with dragon tail support
  const grid = {};
  let colOffset = 0;

  columns.forEach(colData => {
    let row = 0, col = colOffset;
    colData.forEach(cell => {
      if (row >= maxRows) {
        row = maxRows - 1;
        col++;
      }
      while (grid[`${row},${col}`]) col++;
      grid[`${row},${col}`] = cell;
      row++;
    });
    colOffset = col + 1;
  });

  const maxCol = Math.max(...Object.keys(grid).map(k => parseInt(k.split(',')[1]))) + 1;
  const totalCols = Math.max(maxCol, 20);

  let html = `<div class="road-grid" style="grid-template-columns:repeat(${totalCols},18px);grid-template-rows:repeat(${maxRows},18px)">`;
  for (let r = 0; r < maxRows; r++) {
    for (let c = 0; c < totalCols; c++) {
      const cell = grid[`${r},${c}`];
      if (cell) {
        const cls = cell.side === '莊' ? 'bk' : 'pl';
        let inner = cell.side === '莊' ? '莊' : '閒';
        let extras = '';
        if (cell.ties > 0) extras += `<span class="tie-count">${cell.ties}</span><span class="tie-line"></span>`;
        if (cell.bp) extras += '<span class="pair-dot bp-dot"></span>';
        if (cell.pp) extras += '<span class="pair-dot pp-dot"></span>';
        html += `<div class="cell ${cls}">${inner[0]}${extras}</div>`;
      } else {
        html += '<div class="cell empty"></div>';
      }
    }
  }
  html += '</div>';
  return html;
}

function renderBead(data) {
  const entries = data.bead || [];
  if (!entries.length) return '<div style="color:var(--text3);text-align:center;padding:15px;font-size:0.8rem">等待數據...</div>';

  const rows = 6;
  const cols = Math.max(Math.ceil(entries.length / rows), 10);
  let html = `<div class="road-grid" style="grid-template-columns:repeat(${cols},18px);grid-template-rows:repeat(${rows},18px)">`;

  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const idx = c * rows + r;
      if (idx < entries.length) {
        const e = entries[idx];
        const cls = e.outcome === '莊' ? 'bk' : e.outcome === '閒' ? 'pl' : 'ti';
        const ch = e.outcome[0];
        let extras = '';
        if (e.banker_pair) extras += '<span class="pair-dot bp-dot"></span>';
        if (e.player_pair) extras += '<span class="pair-dot pp-dot"></span>';
        html += `<div class="cell ${cls}">${ch}${extras}</div>`;
      } else {
        html += '<div class="cell empty"></div>';
      }
    }
  }
  html += '</div>';
  return html;
}

function renderDerived(entries, label) {
  if (!entries || !entries.length) return `<div style="color:var(--text3);text-align:center;padding:15px;font-size:0.8rem">${label}: 需更多數據</div>`;

  // Group into columns (same color = same column)
  const columns = [];
  let cur = [entries[0]];
  for (let i = 1; i < entries.length; i++) {
    if (entries[i] === cur[0]) { cur.push(entries[i]); }
    else { columns.push(cur); cur = [entries[i]]; }
  }
  columns.push(cur);

  const maxRows = 6;
  const totalCols = Math.max(columns.length, 15);
  let html = `<div class="road-grid derived" style="grid-template-columns:repeat(${totalCols},12px);grid-template-rows:repeat(${maxRows},12px)">`;

  for (let c = 0; c < totalCols; c++) {
    for (let r = 0; r < maxRows; r++) {
      if (c < columns.length && r < columns[c].length) {
        const v = columns[c][r];
        const cls = v === '紅' ? 'filled-r' : 'filled-b';
        html += `<div class="cell ${cls}"></div>`;
      } else {
        html += '<div class="cell empty" style="border-color:transparent"></div>';
      }
    }
  }
  html += '</div>';
  return html;
}

function updateRoad(data) {
  const container = $('roadContainer');
  if (!data || !data.roads) { container.innerHTML = '<div style="color:var(--text3);text-align:center;padding:15px">等待開牌...</div>'; return; }

  const roads = data.roads;
  switch (currentRoad) {
    case 'big_road': container.innerHTML = renderBigRoad(roads); break;
    case 'bead': container.innerHTML = renderBead(roads); break;
    case 'big_eye': container.innerHTML = renderDerived(roads.big_eye, '大眼仔'); break;
    case 'small_road': container.innerHTML = renderDerived(roads.small_road, '小路'); break;
    case 'cockroach': container.innerHTML = renderDerived(roads.cockroach, '曱甴路'); break;
  }

  // Auto scroll to right
  container.scrollLeft = container.scrollWidth;
}

// ─── Road Tab Switching ───
document.querySelectorAll('.road-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.road-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    currentRoad = tab.dataset.road;
    updateRoad(STATE);
  });
});

// ─── Update UI ───
function updateUI(data) {
  STATE = data;

  // Balance
  $('vBankroll').textContent = fmt(data.bankroll);
  $('vBalance').textContent = fmt(data.balance);
  const netEl = $('vNet');
  netEl.textContent = (data.net >= 0 ? '+' : '') + fmt(data.net);
  netEl.className = 'v ' + (data.net >= 0 ? 'pos' : 'neg');
  $('vRoi').textContent = data.bet_count > 0 ? data.roi.toFixed(1) + '%' : '-';
  $('vRoi').className = 'v ' + (data.roi >= 0 ? 'pos' : 'neg');
  $('vWinRate').textContent = data.bet_count > 0 ? data.win_rate.toFixed(0) + '%' : '-';

  // Prediction
  const pred = data.prediction;
  if (pred && pred.consensus) {
    $('nextRound').textContent = data.round + 1;
    const pm = $('predMain');
    const skipBanner = $('skipBanner');
    const predBet = $('predBet');

    if (pred.skip) {
      // 跳局模式
      pm.textContent = '⏸ 觀望';
      pm.className = 'pred-main skip-mode';
      skipBanner.style.display = 'block';
      $('skipReason').textContent = pred.skip_reason || '';
      predBet.textContent = '建議不下注';
      predBet.className = 'pred-bet skip-bet';
    } else {
      pm.textContent = pred.consensus;
      pm.className = 'pred-main ' + (pred.consensus === '莊' ? 'bk' : 'pl');
      skipBanner.style.display = 'none';
      predBet.textContent = `建議注碼: ${fmt(pred.suggested_bet)}`;
      predBet.className = 'pred-bet';
    }

    $('predInfo').textContent = `信心度 ${pred.confidence.toFixed(0)}% ─ 加權(莊${pred.weighted_banker} / 閒${pred.weighted_player})`;

    const total = pred.banker_votes + pred.player_votes;
    if (total > 0) {
      $('voteB').style.width = (pred.banker_votes / total * 100) + '%';
      $('voteP').style.width = (pred.player_votes / total * 100) + '%';
    }

    // ── Three Layer Panel ──
    const layers = pred.layers;
    if (layers) {
      const consensus = pred.consensus;
      // Layer 1: Strategy
      const l1 = layers.strategy;
      $('l1Side').textContent = l1.side;
      $('l1Side').className = 'lc-side ' + (l1.side === '莊' ? 'bk' : 'pl');
      $('l1Conf').textContent = l1.conf + '%';
      $('layer1').className = 'layer-card' + (l1.side === consensus ? ' agree' : ' disagree');

      // Layer 2: Road
      const l2 = layers.road;
      $('l2Side').textContent = l2.side;
      $('l2Side').className = 'lc-side ' + (l2.side === '莊' ? 'bk' : 'pl');
      $('l2Conf').textContent = l2.conf + '%';
      $('l2Detail').textContent = l2.reasons && l2.reasons.length ? l2.reasons[0] : '';
      $('layer2').className = 'layer-card' + (l2.side === consensus ? ' agree' : ' disagree');

      // Layer 3: Card counting
      const l3 = layers.card;
      if (l3.side) {
        $('l3Side').textContent = l3.side;
        $('l3Side').className = 'lc-side ' + (l3.side === '莊' ? 'bk' : 'pl');
        $('l3Conf').textContent = '偏差 ' + l3.edge;
        $('layer3').className = 'layer-card' + (l3.side === consensus ? ' agree' : ' disagree');
      } else {
        $('l3Side').textContent = '—';
        $('l3Side').className = 'lc-side none';
        $('l3Conf').textContent = '數據不足';
        $('layer3').className = 'layer-card';
      }
    }

    // ── Card Counting Probabilities ──
    const cs = pred.card_signal;
    if (cs && cs.prob) {
      $('cardProb').style.display = 'flex';
      $('cpBk').textContent = (cs.prob['莊'] || 0).toFixed(2) + '%';
      $('cpPl').textContent = (cs.prob['閒'] || 0).toFixed(2) + '%';
      $('cpTi').textContent = (cs.prob['和'] || 0).toFixed(2) + '%';
    } else {
      $('cardProb').style.display = 'none';
    }

    // ── System Accuracy ──
    $('sysAcc').textContent = data.system_accuracy > 0 ? data.system_accuracy.toFixed(1) + '%' : '-';
    $('sysAcc').style.color = data.system_accuracy >= 55 ? 'var(--green)' : data.system_accuracy >= 48 ? 'var(--gold)' : 'var(--red)';
    $('sysSkip').textContent = data.skip_count > 0 ? `跳過 ${data.skip_count} 局` : '';

    // ── Road Prediction ──
    const rp = pred.road_prediction;
    if (rp && rp.reasons && rp.reasons.length && rp.reasons[0] !== '數據不足' && rp.reasons[0] !== '無明顯模式') {
      $('roadPred').style.display = 'flex';
      const rpSide = $('rpSide');
      rpSide.textContent = rp.side;
      rpSide.className = 'rp-side ' + (rp.side === '莊' ? 'bk' : 'pl');
      $('rpReasons').textContent = rp.reasons.join(' | ');
    } else {
      $('roadPred').style.display = 'none';
    }

    // ── Strategies with hot/cold/weight ──
    let sHtml = '';
    pred.strategies.forEach(s => {
      const pCls = s.pred === '莊' ? 'bk' : 'pl';
      const total = s.correct + s.wrong;

      // Use status from backend (hot/cold/normal)
      let rowCls = '';
      let dotHtml = '';
      if (s.status === 'hot') {
        rowCls = ' hot-row';
        dotHtml = '<span class="status-dot hot"></span>';
      } else if (s.status === 'cold') {
        rowCls = ' cold-row';
        dotHtml = '<span class="status-dot cold"></span>';
      }

      // Streak badge
      let streakHtml = '';
      if (Math.abs(s.streak) >= 3) {
        const sc = s.streak > 0 ? 'pos' : 'neg';
        streakHtml = `<span class="streak-badge ${sc}">${s.streak > 0 ? '+' : ''}${s.streak}</span>`;
      }

      // Show recent accuracy (primary) and all-time
      const accRecent = s.acc_recent !== undefined ? s.acc_recent.toFixed(0) : '-';
      const accAll = total > 0 ? s.acc.toFixed(0) : '-';
      let accCls = '';
      if (s.status === 'hot') accCls = 'hot';
      else if (s.status === 'cold') accCls = 'cold';

      sHtml += `<div class="s-row${rowCls}">
        <span class="nm" title="${s.desc || ''}">${dotHtml}${s.name}${streakHtml}</span>
        <span class="pr ${pCls}">${s.pred}</span>
        <span class="wt">${s.weight || '-'}</span>
        <span class="ac ${accCls}">近${accRecent}% 全${accAll}%</span>
      </div>`;
    });
    $('stratBody').innerHTML = sHtml;
  }

  // Patterns
  const patterns = data.roads?.patterns;
  const pBox = $('patternBox');
  if (patterns && Object.keys(patterns).length > 0) {
    const pats = Object.values(patterns).filter(p => p.active);
    if (pats.length > 0) {
      pBox.style.display = 'block';
      pBox.innerHTML = pats.map(p =>
        `<span class="reason">${p.description}</span>`
      ).join(' ・ ');
    } else {
      pBox.style.display = 'none';
    }
  } else {
    pBox.style.display = 'none';
  }

  // Stats
  $('sBk').textContent = data.banker_count;
  $('sPl').textContent = data.player_count;
  $('sTi').textContent = data.tie_count;
  $('sTotal').textContent = data.total;

  if (data.streak_count > 0) {
    const sk = $('sStreak');
    sk.textContent = data.streak_side[0] + data.streak_count;
    sk.style.color = data.streak_side === '莊' ? 'var(--banker)' : 'var(--player)';
  } else {
    $('sStreak').textContent = '-';
    $('sStreak').style.color = '';
  }
  $('sMaxStreak').textContent = Math.max(data.max_win_streak, data.max_loss_streak) || '-';

  // Roads
  updateRoad(data);

  // History
  const bets = data.last_bets || [];
  let hHtml = '<div class="h-row h-header"><span>#</span><span>開</span><span>壓</span><span>金額</span><span>損益</span><span>餘額</span></div>';
  bets.slice().reverse().forEach(b => {
    const profStr = b.profit > 0 ? `<span class="pos">+${fmt(b.profit)}</span>` :
                    b.profit < 0 ? `<span class="neg">${fmt(b.profit)}</span>` : '0';
    hHtml += `<div class="h-row">
      <span>#${b.round}</span>
      <span style="color:${b.outcome==='莊'?'var(--banker)':b.outcome==='閒'?'var(--player)':'var(--tie)'}">${b.outcome}</span>
      <span>${b.my_side}</span>
      <span>${b.my_amount > 0 ? fmt(b.my_amount) : '-'}</span>
      <span>${profStr}</span>
      <span>${fmt(b.balance)}</span>
    </div>`;
  });
  $('histBody').innerHTML = hHtml;
}

// ─── API ───
async function fetchState() {
  try {
    const r = await fetch('/api/state');
    const d = await r.json();
    updateUI(d);
  } catch (e) { console.error(e); }
}

async function rec(outcome) {
  let mySide = null, myAmt = null;

  if (follow && STATE.prediction) {
    // 如果系統建議跳局，警告用戶
    if (STATE.prediction.skip) {
      if (!confirm('⚠ 系統建議此局觀望！\n' + (STATE.prediction.skip_reason || '信心不足') + '\n\n確定要下注嗎？')) {
        // 只記錄開牌結果，不下注
        try {
          const r = await fetch('/api/record', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({outcome, my_side: null, my_amount: null}),
          });
          await r.json();
          toast('已記錄（跳局）');
        } catch(e) { console.error(e); }
        await fetchState();
        return;
      }
    }

    const amt = $('betAmt').value;
    mySide = STATE.prediction.consensus;
    myAmt = amt ? parseFloat(amt) : STATE.prediction.suggested_bet;
  }

  try {
    const r = await fetch('/api/record', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({outcome, my_side: mySide, my_amount: myAmt}),
    });
    const d = await r.json();
    if (d.profit > 0) toast(`+${fmt(d.profit)} 💰`);
    else if (d.profit < 0) toast(`${fmt(d.profit)} 😤`);
    else if (outcome === '和') toast('和局');
  } catch(e) { console.error(e); }

  await fetchState();
  if (navigator.vibrate) navigator.vibrate(30);
}

async function newShoe() {
  if (!confirm('開新靴？當前記錄會清空。')) return;
  await fetch('/api/reset', {method:'POST'});
  await fetchState();
  toast('新靴已開');
}

async function undo() {
  await fetch('/api/undo', {method:'POST'});
  await fetchState();
  toast('已撤回');
}

function setup() {
  const bankroll = prompt('本金:', STATE.bankroll || 10000);
  const baseBet = prompt('基本注碼:', STATE.prediction?.suggested_bet || 100);
  if (bankroll && baseBet) {
    fetch('/api/setup', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({bankroll: parseFloat(bankroll), base_bet: parseFloat(baseBet)}),
    }).then(() => fetchState());
  }
}

// ─── Keyboard ───
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === '1') rec('莊');
  else if (e.key === '2') rec('閒');
  else if (e.key === '3') rec('和');
  else if (e.key === 'z' || e.key === 'Z') undo();
});

// ─── Init ───
fetchState();
</script>
</body>
</html>'''


# ─── HTTP Handler ─────────────────────────────────────────────

class BattleHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"  [HTTP] {args[0] if args else ''}", flush=True)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                with SESSION_LOCK:
                    state = get_state()
            except Exception as e:
                import traceback
                traceback.print_exc()
                state = {'error': str(e)}
            self.wfile.write(json.dumps(state, ensure_ascii=False).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8') if content_length else '{}'

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        if self.path == '/api/record':
            outcome = data.get('outcome')
            my_side = data.get('my_side')
            my_amount = data.get('my_amount')
            if outcome:
                try:
                    with SESSION_LOCK:
                        profit = record_result(outcome, my_side, my_amount)
                    self._json_response({'profit': profit, 'ok': True})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self._json_response({'error': str(e), 'ok': True, 'profit': 0})
            else:
                self._json_response({'error': 'missing outcome'}, 400)

        elif self.path == '/api/reset':
            init_session(SESSION['bankroll'], SESSION['base_bet'])
            self._json_response({'ok': True})

        elif self.path == '/api/undo':
            if SESSION['results']:
                SESSION['results'].pop()
                if SESSION['bets']:
                    last = SESSION['bets'].pop()
                    SESSION['balance'] -= last.get('profit', 0)
                SESSION['round_count'] = max(0, SESSION['round_count'] - 1)
                # Rebuild roads
                rm = RoadManager()
                for r in SESSION['results']:
                    rm.add_result(r)
                SESSION['road_manager'] = rm
            self._json_response({'ok': True})

        elif self.path == '/api/setup':
            bankroll = data.get('bankroll', 10000)
            base_bet = data.get('base_bet', 100)
            init_session(bankroll, base_bet)
            self._json_response({'ok': True})

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))


def start_server(port=8888):
    init_session()

    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║         百家樂實戰系統 Pro v2.0 — Web 介面          ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  🌐  http://localhost:{port}                         ║")
    print("  ║  📱  手機同一區域網路亦可連線使用                    ║")
    print("  ║  ⌨   快捷鍵: 1=莊  2=閒  3=和  Z=撤回              ║")
    print("  ║  🎯  22 種預測策略 + 10 種注碼管理                   ║")
    print("  ║  🛣   五路路紙: 大路/大眼仔/小路/曱甴路/珠盤路      ║")
    print("  ║  按 Ctrl+C 停止伺服器                                ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with ThreadedServer(("", port), BattleHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  🛑 伺服器已停止")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    start_server(port)
