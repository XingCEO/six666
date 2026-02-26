"""
算牌系統 — card_counter.py
============================
追蹤牌靴中已出現的牌，計算剩餘牌組的真實條件機率。
不是用固定機率 — 是根據「剩下哪些牌」去算真正的莊/閒勝率。
"""

import itertools
from typing import Dict, List, Tuple, Optional
from collections import Counter


# 8 副牌中每種牌的初始數量
CARDS_PER_DECK = {
    'A': 4, '2': 4, '3': 4, '4': 4, '5': 4,
    '6': 4, '7': 4, '8': 4, '9': 4, '10': 4,
    'J': 4, 'Q': 4, 'K': 4,
}

CARD_VALUES = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5,
    '6': 6, '7': 7, '8': 8, '9': 9,
    '10': 0, 'J': 0, 'Q': 0, 'K': 0,
}

NUM_DECKS = 8


class CardCounter:
    """
    真實算牌系統
    追蹤 8 副牌中每張牌的剩餘數量，
    計算基於「真正剩餘牌組」的莊/閒/和機率。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置（新靴）"""
        self.remaining: Dict[str, int] = {}
        for rank, count in CARDS_PER_DECK.items():
            self.remaining[rank] = count * NUM_DECKS
        self.total_remaining = 52 * NUM_DECKS  # 416
        self.dealt_cards: List[str] = []
        self.rounds_played = 0

    def remove_card(self, rank: str):
        """記錄一張已出的牌"""
        rank = rank.upper().strip()
        # 支援各種輸入格式
        aliases = {'1': 'A', '0': '10', 'T': '10'}
        rank = aliases.get(rank, rank)

        if rank not in self.remaining:
            raise ValueError(f"無效的牌面: {rank}")
        if self.remaining[rank] <= 0:
            raise ValueError(f"{rank} 已全部出完")

        self.remaining[rank] -= 1
        self.total_remaining -= 1
        self.dealt_cards.append(rank)

    def remove_cards(self, ranks: List[str]):
        """記錄多張已出的牌"""
        for r in ranks:
            self.remove_card(r)

    def get_remaining_count(self) -> Dict[str, int]:
        """取得剩餘牌面數量"""
        return dict(self.remaining)

    def get_remaining_total(self) -> int:
        return self.total_remaining

    def get_dealt_count(self) -> int:
        return len(self.dealt_cards)

    def get_value_distribution(self) -> Dict[int, int]:
        """
        取得剩餘牌按「百家樂點數值」分組的數量
        點數 0: 10,J,Q,K
        點數 1-9: A-9
        """
        dist = {}
        for rank, count in self.remaining.items():
            val = CARD_VALUES[rank]
            dist[val] = dist.get(val, 0) + count
        return dist

    def get_value_probability(self) -> Dict[int, float]:
        """取得下一張牌各點數值的機率"""
        dist = self.get_value_distribution()
        total = self.total_remaining
        if total == 0:
            return {v: 0.0 for v in range(10)}
        return {v: dist.get(v, 0) / total for v in range(10)}

    # ================================================================
    #  核心：計算真實條件機率
    # ================================================================

    def calculate_exact_probabilities(self, sample_size: int = 50000) -> Dict[str, float]:
        """
        用 Monte Carlo 抽樣法，從剩餘牌組計算莊/閒/和的真實機率。
        不是固定值 — 每次出牌後機率都會變化！

        sample_size: 抽樣次數（越大越準，但越慢）
        """
        import random

        if self.total_remaining < 6:
            return {'閒': 0.0, '莊': 0.0, '和': 0.0}

        # 建立剩餘牌池
        pool = []
        for rank, count in self.remaining.items():
            pool.extend([CARD_VALUES[rank]] * count)

        if len(pool) < 6:
            return {'閒': 0.0, '莊': 0.0, '和': 0.0}

        player_wins = 0
        banker_wins = 0
        ties = 0

        for _ in range(sample_size):
            # 從剩餘牌中隨機抽取（不放回）
            drawn = random.sample(pool, min(6, len(pool)))

            # 發牌: 閒1、莊1、閒2、莊2
            p1, b1, p2, b2 = drawn[0], drawn[1], drawn[2], drawn[3]

            p_score = (p1 + p2) % 10
            b_score = (b1 + b2) % 10

            # 天牌
            if p_score >= 8 or b_score >= 8:
                pass  # 不補牌
            else:
                card_idx = 4
                p_third_val = None

                # 閒家補牌
                if p_score <= 5 and card_idx < len(drawn):
                    p_third_val = drawn[card_idx]
                    p_score = (p_score + p_third_val) % 10
                    card_idx += 1

                # 莊家補牌
                if card_idx < len(drawn):
                    should_draw = self._banker_should_draw(b_score, p_third_val)
                    if should_draw:
                        b_score = (b_score + drawn[card_idx]) % 10

            if p_score > b_score:
                player_wins += 1
            elif b_score > p_score:
                banker_wins += 1
            else:
                ties += 1

        total = player_wins + banker_wins + ties
        return {
            '閒': player_wins / total * 100,
            '莊': banker_wins / total * 100,
            '和': ties / total * 100,
        }

    def _banker_should_draw(self, banker_score: int, player_third_val: Optional[int]) -> bool:
        """莊家補牌規則"""
        if player_third_val is None:
            return banker_score <= 5

        if banker_score <= 2:
            return True
        elif banker_score == 3:
            return player_third_val != 8
        elif banker_score == 4:
            return player_third_val in (2, 3, 4, 5, 6, 7)
        elif banker_score == 5:
            return player_third_val in (4, 5, 6, 7)
        elif banker_score == 6:
            return player_third_val in (6, 7)
        return False

    def get_edge_indicator(self) -> Dict[str, float]:
        """
        計算當前牌況的優勢指標
        正值 = 有利，負值 = 不利
        """
        val_dist = self.get_value_distribution()
        total = self.total_remaining
        if total == 0:
            return {'莊優勢': 0, '閒優勢': 0, '高牌比例': 0, '低牌比例': 0}

        # 高牌 (面值0的牌: 10,J,Q,K) 比例
        high_cards = val_dist.get(0, 0)
        high_ratio = high_cards / total

        # 低牌 (1-4) 比例
        low_cards = sum(val_dist.get(v, 0) for v in range(1, 5))
        low_ratio = low_cards / total

        # 中牌 (5-9)
        mid_cards = sum(val_dist.get(v, 0) for v in range(5, 10))
        mid_ratio = mid_cards / total

        # 理論基準比例
        base_high = (4 * 4) / 52  # 10,J,Q,K = 16/52
        base_low = (4 * 4) / 52   # A,2,3,4 = 16/52

        # 偏差指標
        high_deviation = (high_ratio - base_high) * 100
        low_deviation = (low_ratio - base_low) * 100

        return {
            '高牌比例': high_ratio * 100,
            '低牌比例': low_ratio * 100,
            '中牌比例': mid_ratio * 100,
            '高牌偏差': high_deviation,
            '低牌偏差': low_deviation,
            '剩餘牌數': total,
            '已出牌數': len(self.dealt_cards),
            '滲透率': len(self.dealt_cards) / (52 * NUM_DECKS) * 100,
        }

    def get_status_display(self) -> str:
        """取得當前牌況的格式化顯示"""
        lines = []
        lines.append(f"  剩餘牌數: {self.total_remaining}/416  "
                     f"(已出 {len(self.dealt_cards)} 張, "
                     f"滲透率 {len(self.dealt_cards)/416*100:.1f}%)")
        lines.append("")
        lines.append("  牌面   剩餘  原始  差異")
        lines.append("  " + "─" * 30)

        for rank in ['A','2','3','4','5','6','7','8','9','10','J','Q','K']:
            original = CARDS_PER_DECK[rank] * NUM_DECKS
            remain = self.remaining[rank]
            diff = remain - original
            bar = '█' * (remain * 20 // original) if original > 0 else ''
            lines.append(f"  {rank:>3}    {remain:>3}   {original:>3}  {diff:>+3}  {bar}")

        return "\n".join(lines)


class ShoeTracker:
    """
    整靴追蹤器 — 結合算牌 + 歷史結果
    """

    def __init__(self):
        self.counter = CardCounter()
        self.results: List[Dict] = []  # 每局結果
        self.outcome_sequence: List[str] = []  # 莊/閒/和 序列

    def reset(self):
        self.counter.reset()
        self.results.clear()
        self.outcome_sequence.clear()

    def record_round(self,
                     outcome: str,
                     player_cards: Optional[List[str]] = None,
                     banker_cards: Optional[List[str]] = None):
        """
        記錄一局結果
        outcome: '莊', '閒', '和'
        player_cards: 閒家的牌面 ['K','5','3'] (可選)
        banker_cards: 莊家的牌面 ['8','2'] (可選)
        """
        round_data = {
            'round': len(self.results) + 1,
            'outcome': outcome,
            'player_cards': player_cards or [],
            'banker_cards': banker_cards or [],
        }

        # 如果有提供具體牌面，記錄到算牌器
        if player_cards:
            self.counter.remove_cards(player_cards)
        if banker_cards:
            self.counter.remove_cards(banker_cards)

        self.results.append(round_data)
        self.outcome_sequence.append(outcome)

    def get_outcome_sequence(self) -> List[str]:
        return list(self.outcome_sequence)

    def get_statistics(self) -> Dict:
        """取得目前統計"""
        total = len(self.results)
        if total == 0:
            return {}

        p = sum(1 for r in self.results if r['outcome'] == '閒')
        b = sum(1 for r in self.results if r['outcome'] == '莊')
        t = sum(1 for r in self.results if r['outcome'] == '和')

        # 連勝/連敗分析
        streaks = self._analyze_streaks()

        return {
            '總局數': total,
            '閒贏': p, '閒%': p/total*100,
            '莊贏': b, '莊%': b/total*100,
            '和局': t, '和%': t/total*100,
            '當前連勝方': streaks['current_side'],
            '當前連勝數': streaks['current_count'],
            '最長莊連': streaks['max_banker'],
            '最長閒連': streaks['max_player'],
        }

    def _analyze_streaks(self) -> Dict:
        filtered = [o for o in self.outcome_sequence if o != '和']
        if not filtered:
            return {'current_side': '-', 'current_count': 0,
                    'max_banker': 0, 'max_player': 0}

        # 當前連勝
        current_side = filtered[-1]
        current_count = 0
        for o in reversed(filtered):
            if o == current_side:
                current_count += 1
            else:
                break

        # 歷史最長
        max_b = max_p = 0
        count = 0
        prev = None
        for o in filtered:
            if o == prev:
                count += 1
            else:
                count = 1
                prev = o
            if o == '莊':
                max_b = max(max_b, count)
            else:
                max_p = max(max_p, count)

        return {
            'current_side': current_side,
            'current_count': current_count,
            'max_banker': max_b,
            'max_player': max_p,
        }
