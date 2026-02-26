"""
百家樂核心模擬引擎 — baccarat_engine.py
========================================
完整實現真實百家樂規則：
- 8 副牌（416 張）洗牌、切牌、燒牌
- 閒家 / 莊家 第三張補牌規則
- 點數計算（A=1, 2-9面值, 10/J/Q/K=0）
- 結果判定：閒贏、莊贏、和局、莊對、閒對
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class Outcome(Enum):
    PLAYER = "閒"
    BANKER = "莊"
    TIE = "和"


@dataclass
class Card:
    rank: str   # 'A','2',...,'10','J','Q','K'
    suit: str   # '♠','♥','♦','♣'

    @property
    def value(self) -> int:
        if self.rank in ('10', 'J', 'Q', 'K'):
            return 0
        if self.rank == 'A':
            return 1
        return int(self.rank)

    def __repr__(self):
        return f"{self.suit}{self.rank}"


@dataclass
class HandResult:
    """單局結果"""
    round_no: int
    player_cards: List[Card]
    banker_cards: List[Card]
    player_score: int
    banker_score: int
    outcome: Outcome
    player_pair: bool
    banker_pair: bool
    natural: bool  # 天牌 (8 或 9)


class Shoe:
    """牌靴：8 副牌洗牌、切牌、燒牌"""

    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    SUITS = ['♠', '♥', '♦', '♣']
    NUM_DECKS = 8

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.cards: List[Card] = []
        self.cut_position: int = 0
        self.shuffle()

    def shuffle(self):
        """建立 8 副牌並洗牌"""
        self.cards = [
            Card(rank, suit)
            for _ in range(self.NUM_DECKS)
            for suit in self.SUITS
            for rank in self.RANKS
        ]
        self.rng.shuffle(self.cards)
        # 切牌：最後 14~26 張不使用（真實賭場慣例）
        cut = self.rng.randint(14, 26)
        self.cut_position = len(self.cards) - cut
        # 燒牌：翻開第一張，根據面值燒掉對應張數
        burn_card = self.cards.pop(0)
        burn_count = burn_card.value if burn_card.value > 0 else 10
        for _ in range(burn_count):
            if self.cards:
                self.cards.pop(0)

    def draw(self) -> Card:
        """抽一張牌"""
        if not self.cards:
            raise RuntimeError("牌靴已空")
        return self.cards.pop(0)

    def needs_reshuffle(self) -> bool:
        """是否需要重新洗牌"""
        return len(self.cards) < (416 - self.cut_position) or len(self.cards) < 6

    @property
    def remaining(self) -> int:
        return len(self.cards)


def hand_score(cards: List[Card]) -> int:
    """計算一手牌的點數（取個位數）"""
    return sum(c.value for c in cards) % 10


def should_player_draw(player_score: int) -> bool:
    """閒家補牌規則：0-5 補牌，6-7 不補"""
    return player_score <= 5


def should_banker_draw(banker_score: int, player_third: Optional[Card]) -> bool:
    """
    莊家補牌規則（完整真實規則）：
    - 閒家沒補牌 → 莊家 0-5 補，6-7 不補
    - 閒家有補牌 → 根據莊家點數與閒家第三張牌決定
    """
    if player_third is None:
        # 閒家沒有補牌
        return banker_score <= 5

    p3v = player_third.value  # 閒家第三張牌面值

    if banker_score <= 2:
        return True
    elif banker_score == 3:
        return p3v != 8
    elif banker_score == 4:
        return p3v in (2, 3, 4, 5, 6, 7)
    elif banker_score == 5:
        return p3v in (4, 5, 6, 7)
    elif banker_score == 6:
        return p3v in (6, 7)
    else:  # 7
        return False


class BaccaratGame:
    """百家樂遊戲模擬器"""

    def __init__(self, seed: Optional[int] = None):
        self.shoe = Shoe(seed)
        self.history: List[HandResult] = []
        self.round_no = 0

    def play_one_round(self) -> HandResult:
        """玩一局，回傳結果"""
        if self.shoe.needs_reshuffle():
            self.shoe.shuffle()

        self.round_no += 1

        # 初始發牌：閒1、莊1、閒2、莊2（交替發牌）
        player_cards = [self.shoe.draw()]
        banker_cards = [self.shoe.draw()]
        player_cards.append(self.shoe.draw())
        banker_cards.append(self.shoe.draw())

        p_score = hand_score(player_cards)
        b_score = hand_score(banker_cards)

        # 對子判定
        player_pair = player_cards[0].rank == player_cards[1].rank
        banker_pair = banker_cards[0].rank == banker_cards[1].rank

        natural = False
        player_third = None

        # 天牌判定：任一方 8 或 9 → 不補牌
        if p_score >= 8 or b_score >= 8:
            natural = True
        else:
            # 閒家補牌
            if should_player_draw(p_score):
                player_third = self.shoe.draw()
                player_cards.append(player_third)
                p_score = hand_score(player_cards)

            # 莊家補牌
            if should_banker_draw(b_score, player_third):
                banker_cards.append(self.shoe.draw())
                b_score = hand_score(banker_cards)

        # 判定勝負
        if p_score > b_score:
            outcome = Outcome.PLAYER
        elif b_score > p_score:
            outcome = Outcome.BANKER
        else:
            outcome = Outcome.TIE

        result = HandResult(
            round_no=self.round_no,
            player_cards=player_cards,
            banker_cards=banker_cards,
            player_score=p_score,
            banker_score=b_score,
            outcome=outcome,
            player_pair=player_pair,
            banker_pair=banker_pair,
            natural=natural,
        )
        self.history.append(result)
        return result

    def play_shoe(self) -> List[HandResult]:
        """打完一整靴牌"""
        results = []
        while not self.shoe.needs_reshuffle():
            results.append(self.play_one_round())
        return results

    def simulate(self, n_rounds: int) -> List[HandResult]:
        """模擬 n 局（自動重新洗牌）"""
        results = []
        for _ in range(n_rounds):
            if self.shoe.needs_reshuffle():
                self.shoe.shuffle()
            results.append(self.play_one_round())
        return results

    def get_outcome_sequence(self) -> List[str]:
        """回傳結果序列（'莊','閒','和'）"""
        return [h.outcome.value for h in self.history]

    def reset(self, seed: Optional[int] = None):
        """重置遊戲"""
        self.shoe = Shoe(seed)
        self.history.clear()
        self.round_no = 0


# ====== 統計工具函數 ======

def calculate_base_probabilities(results: List[HandResult]) -> dict:
    """計算基礎機率分佈"""
    total = len(results)
    if total == 0:
        return {}
    p_count = sum(1 for r in results if r.outcome == Outcome.PLAYER)
    b_count = sum(1 for r in results if r.outcome == Outcome.BANKER)
    t_count = sum(1 for r in results if r.outcome == Outcome.TIE)
    pp_count = sum(1 for r in results if r.player_pair)
    bp_count = sum(1 for r in results if r.banker_pair)
    nat_count = sum(1 for r in results if r.natural)

    return {
        "總局數": total,
        "閒贏": p_count, "閒贏%": p_count / total * 100,
        "莊贏": b_count, "莊贏%": b_count / total * 100,
        "和局": t_count, "和局%": t_count / total * 100,
        "閒對": pp_count, "閒對%": pp_count / total * 100,
        "莊對": bp_count, "莊對%": bp_count / total * 100,
        "天牌": nat_count, "天牌%": nat_count / total * 100,
    }


# 理論機率（不計和局時）
THEORETICAL = {
    "閒贏": 44.6247,
    "莊贏": 45.8597,
    "和局":  9.5156,
    "閒對":  7.47,
    "莊對":  7.47,
}

PAYOUTS = {
    "閒": 1.0,      # 1:1
    "莊": 0.95,     # 1:1 扣 5% 佣金
    "和": 8.0,      # 8:1
    "閒對": 11.0,   # 11:1
    "莊對": 11.0,   # 11:1
}
