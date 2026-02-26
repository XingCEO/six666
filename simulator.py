"""
Monte Carlo 批量模擬器 — simulator.py
======================================
對所有策略進行大量模擬，收集預測準確率、ROI、資金曲線等數據。
"""

import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from baccarat_engine import BaccaratGame, Outcome, PAYOUTS, calculate_base_probabilities
from strategies import (
    BaseStrategy, BettingSystem, FlatBetting,
    get_all_strategies, get_all_betting_systems,
)


@dataclass
class StrategyResult:
    """單一策略的模擬結果"""
    strategy_name: str
    total_rounds: int = 0
    predictions: int = 0        # 有效預測局數（排除預熱期）
    correct: int = 0
    wrong: int = 0
    ties_skipped: int = 0       # 和局不計
    accuracy: float = 0.0
    profit: float = 0.0         # 累計損益
    roi: float = 0.0            # 投資報酬率
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    balance_history: List[float] = field(default_factory=list)
    prediction_detail: List[dict] = field(default_factory=list)


def simulate_strategy(
    strategy: BaseStrategy,
    game: BaccaratGame,
    n_rounds: int,
    betting_system: Optional[BettingSystem] = None,
    base_unit: float = 100,
    record_detail: bool = False,
) -> StrategyResult:
    """
    模擬一個策略在 n_rounds 局中的表現
    """
    if betting_system is None:
        betting_system = FlatBetting(base_unit)

    result = StrategyResult(strategy_name=strategy.name)

    balance = 0.0
    current_streak = 0   # 正=連贏, 負=連輸
    max_win_streak = 0
    max_loss_streak = 0
    correct = 0
    wrong = 0
    ties = 0

    # 重播遊戲歷史
    history_seq = []

    for i in range(n_rounds):
        # 取得歷史
        prediction = strategy.predict(history_seq)

        # 玩一局
        hand = game.play_one_round()
        actual = hand.outcome.value
        history_seq.append(actual)

        result.total_rounds += 1

        # 和局處理
        if actual == '和':
            ties += 1
            # 和局不輸不贏（大部分策略排除和局）
            result.balance_history.append(balance)
            if record_detail:
                result.prediction_detail.append({
                    'round': i + 1,
                    'prediction': prediction,
                    'actual': actual,
                    'result': '和局',
                    'bet': 0,
                    'profit': 0,
                    'balance': balance,
                })
            continue

        result.predictions += 1
        bet_amount = betting_system.next_bet()

        if prediction == actual:
            correct += 1
            payout_ratio = PAYOUTS.get(prediction, 1.0)
            profit = bet_amount * payout_ratio
            balance += profit
            betting_system.update(won=True, payout_ratio=payout_ratio)

            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
            max_win_streak = max(max_win_streak, current_streak)

            if record_detail:
                result.prediction_detail.append({
                    'round': i + 1,
                    'prediction': prediction,
                    'actual': actual,
                    'result': '贏',
                    'bet': bet_amount,
                    'profit': profit,
                    'balance': balance,
                })
        else:
            wrong += 1
            balance -= bet_amount
            betting_system.update(won=False)

            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
            max_loss_streak = max(max_loss_streak, abs(current_streak))

            if record_detail:
                result.prediction_detail.append({
                    'round': i + 1,
                    'prediction': prediction,
                    'actual': actual,
                    'result': '輸',
                    'bet': bet_amount,
                    'profit': -bet_amount,
                    'balance': balance,
                })

        result.balance_history.append(balance)

    result.correct = correct
    result.wrong = wrong
    result.ties_skipped = ties
    result.accuracy = correct / max(correct + wrong, 1) * 100
    result.profit = balance
    total_wagered = result.predictions * base_unit  # 近似
    result.roi = balance / max(total_wagered, 1) * 100
    result.max_consecutive_wins = max_win_streak
    result.max_consecutive_losses = max_loss_streak

    return result


def run_monte_carlo(
    n_simulations: int = 100,
    n_rounds: int = 1000,
    base_unit: float = 100,
    seed_base: int = 42,
    record_detail: bool = False,
    progress_callback=None,
) -> Dict[str, List[StrategyResult]]:
    """
    執行 Monte Carlo 模擬
    - n_simulations: 模擬次數
    - n_rounds: 每次模擬的局數
    回傳 {策略名: [StrategyResult, ...]} 
    """
    all_results: Dict[str, List[StrategyResult]] = {}

    for sim_idx in range(n_simulations):
        seed = seed_base + sim_idx
        game = BaccaratGame(seed=seed)
        strategies = get_all_strategies(seed=seed)

        for strat in strategies:
            # 每個策略用相同的牌局
            game.reset(seed=seed)
            strat.reset()
            betting = FlatBetting(base_unit)

            result = simulate_strategy(
                strategy=strat,
                game=game,
                n_rounds=n_rounds,
                betting_system=betting,
                base_unit=base_unit,
                record_detail=record_detail and sim_idx == 0,  # 只記錄第一次
            )

            if strat.name not in all_results:
                all_results[strat.name] = []
            all_results[strat.name].append(result)

        if progress_callback:
            progress_callback(sim_idx + 1, n_simulations)

    return all_results


def run_single_simulation(
    n_rounds: int = 1000,
    base_unit: float = 100,
    seed: int = 42,
) -> tuple:
    """
    執行單次模擬，回傳 (game_results, strategy_results, base_stats)
    """
    game = BaccaratGame(seed=seed)
    strategies = get_all_strategies(seed=seed)

    strategy_results = []
    for strat in strategies:
        game.reset(seed=seed)
        strat.reset()
        betting = FlatBetting(base_unit)

        result = simulate_strategy(
            strategy=strat,
            game=game,
            n_rounds=n_rounds,
            betting_system=betting,
            base_unit=base_unit,
            record_detail=True,
        )
        strategy_results.append(result)

    # 重新跑一次取得基礎統計
    game.reset(seed=seed)
    game.simulate(n_rounds)
    base_stats = calculate_base_probabilities(game.history)

    return game.history, strategy_results, base_stats


def aggregate_results(all_results: Dict[str, List[StrategyResult]]) -> pd.DataFrame:
    """彙整 Monte Carlo 結果為 DataFrame"""
    rows = []
    for name, results in all_results.items():
        accuracies = [r.accuracy for r in results]
        profits = [r.profit for r in results]
        rois = [r.roi for r in results]
        max_wins = [r.max_consecutive_wins for r in results]
        max_losses = [r.max_consecutive_losses for r in results]

        rows.append({
            '策略': name,
            '模擬次數': len(results),
            '平均準確率%': np.mean(accuracies),
            '準確率標準差': np.std(accuracies),
            '最高準確率%': np.max(accuracies),
            '最低準確率%': np.min(accuracies),
            '平均損益': np.mean(profits),
            '損益標準差': np.std(profits),
            '平均ROI%': np.mean(rois),
            '平均最大連贏': np.mean(max_wins),
            '平均最大連輸': np.mean(max_losses),
            '勝率>50%比例': sum(1 for a in accuracies if a > 50) / len(accuracies) * 100,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('平均準確率%', ascending=False).reset_index(drop=True)
    return df
