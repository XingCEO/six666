"""
百家樂預測策略集合 — strategies.py
===================================
包含 20+ 種預測策略 + 10 種注碼管理系統
收錄亞洲、歐洲、美洲最有效打法

預測策略：
  1.  隨機預測（基準線）
  2.  跟前局 (Follow Last)
  3.  反前局 (Opposite Last)
  4.  跟連勝 (Streak Follower)
  5.  反連勝 (Streak Breaker)
  6.  頻率統計 (Frequency)
  7.  大路模式 (Big Road Pattern)
  8.  Markov Chain
  9.  貝氏推論 (Bayesian)
  10. 移動平均 (Moving Average)
  11. 機器學習 (ML)
  12. 組合投票 (Ensemble)
  --- 全球知名打法 ---
  13. Avant Dernier 壓前前局（法國）
  14. 三珠路打法（亞洲）
  15. 微笑心法（亞洲）
  16. 長莊長閒追擊
  17. 跳路打法（單跳/雙跳）
  18. 天地人打法（三珠變體）
  19. 五路路紙綜合（大路+衍生路）
  20. 雙龍打法
  21. 排排連打法
  22. 正反纜打法

注碼管理：
  1. 平注法
  2. Martingale 倍投
  3. Fibonacci
  4. Paroli 正注
  5. 1-3-2-6 系統
  6. 1-3-2-4 系統
  7. Oscar's Grind
  8. Labouchere（負進纜）
  9. D'Alembert
  10. Anti-Martingale（反倍投）
"""

import random
import numpy as np
from collections import Counter, defaultdict
from typing import List, Optional
from baccarat_engine import Outcome


# ============================================================
#  基底類別
# ============================================================

class BaseStrategy:
    """策略基底類別"""
    name: str = "基底策略"
    description: str = ""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def predict(self, history: List[str]) -> str:
        """
        根據歷史結果預測下一局
        history: ['莊','閒','和',...] 的列表
        return: '莊' 或 '閒'（預測下注目標）
        """
        raise NotImplementedError

    def reset(self):
        """重置策略內部狀態"""
        pass


# ============================================================
#  策略 1: 隨機預測（基準線）
# ============================================================

class RandomStrategy(BaseStrategy):
    name = "隨機預測"
    description = "隨機選擇莊或閒，作為所有策略的基準對照線"

    def predict(self, history: List[str]) -> str:
        return self.rng.choice(['莊', '閒'])


# ============================================================
#  策略 2: 跟前局 (Follow Last)
# ============================================================

class FollowLastStrategy(BaseStrategy):
    name = "跟前局"
    description = "壓上一局的贏家，和局時跟最近一個非和結果"

    def predict(self, history: List[str]) -> str:
        if not history:
            return '莊'  # 默認莊（機率稍高）
        # 找最近的非和結果
        for h in reversed(history):
            if h != '和':
                return h
        return '莊'


# ============================================================
#  策略 3: 反前局 (Opposite Last)
# ============================================================

class OppositeLastStrategy(BaseStrategy):
    name = "反前局"
    description = "壓上一局輸家的相反方，和局時反最近一個非和結果"

    def predict(self, history: List[str]) -> str:
        if not history:
            return '閒'
        for h in reversed(history):
            if h != '和':
                return '閒' if h == '莊' else '莊'
        return '閒'


# ============================================================
#  策略 4: 跟連勝 (Streak Follower)
# ============================================================

class StreakFollowerStrategy(BaseStrategy):
    name = "跟連勝"
    description = "若同一方連續贏 2 局以上就跟注，否則壓莊"

    def __init__(self, min_streak: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.min_streak = min_streak

    def predict(self, history: List[str]) -> str:
        if len(history) < self.min_streak:
            return '莊'

        # 過濾和局
        filtered = [h for h in history if h != '和']
        if len(filtered) < self.min_streak:
            return '莊'

        # 檢查最近是否連勝
        last = filtered[-1]
        streak = 0
        for h in reversed(filtered):
            if h == last:
                streak += 1
            else:
                break

        if streak >= self.min_streak:
            return last  # 跟注
        return '莊'


# ============================================================
#  策略 5: 反連勝 (Streak Breaker)
# ============================================================

class StreakBreakerStrategy(BaseStrategy):
    name = "反連勝"
    description = "若同一方連續贏 3 局以上則反壓，認為連勝即將結束"

    def __init__(self, min_streak: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.min_streak = min_streak

    def predict(self, history: List[str]) -> str:
        filtered = [h for h in history if h != '和']
        if len(filtered) < self.min_streak:
            return '莊'

        last = filtered[-1]
        streak = 0
        for h in reversed(filtered):
            if h == last:
                streak += 1
            else:
                break

        if streak >= self.min_streak:
            return '閒' if last == '莊' else '莊'
        return '莊'


# ============================================================
#  策略 6: 頻率統計 (Frequency)
# ============================================================

class FrequencyStrategy(BaseStrategy):
    name = "頻率統計"
    description = "壓目前出現次數較多的一方，利用局部偏差"

    def predict(self, history: List[str]) -> str:
        if not history:
            return '莊'

        counter = Counter(h for h in history if h != '和')
        if not counter:
            return '莊'

        p_count = counter.get('閒', 0)
        b_count = counter.get('莊', 0)

        if b_count >= p_count:
            return '莊'
        return '閒'


# ============================================================
#  策略 7: 大路模式識別 (Big Road Pattern)
# ============================================================

class BigRoadStrategy(BaseStrategy):
    name = "大路模式"
    description = "分析百家樂大路（路紙），識別龍、跳、拍拍等經典模式"

    def predict(self, history: List[str]) -> str:
        filtered = [h for h in history if h != '和']
        if len(filtered) < 4:
            return '莊'

        # 構建大路：將連續相同結果分組
        columns = []
        current_col = [filtered[0]]
        for h in filtered[1:]:
            if h == current_col[-1]:
                current_col.append(h)
            else:
                columns.append(current_col)
                current_col = [h]
        columns.append(current_col)

        if len(columns) < 2:
            return filtered[-1]

        # 模式判定
        last_col = columns[-1]
        prev_col = columns[-2]

        # 龍模式：上一列超過 3 個 → 趨勢繼續
        if len(last_col) >= 3:
            return last_col[-1]

        # 拍拍模式 (Ping-Pong)：最近幾列都是長度 1 → 交替
        if len(columns) >= 3:
            recent_lens = [len(c) for c in columns[-3:]]
            if all(l == 1 for l in recent_lens):
                # 拍拍 → 壓相反
                return '閒' if last_col[-1] == '莊' else '莊'

        # 對稱模式：當前列長度與前一列相同 → 可能切換
        if len(last_col) == len(prev_col):
            return '閒' if last_col[-1] == '莊' else '莊'

        # 默認跟當前列
        return last_col[-1]


# ============================================================
#  策略 8: Markov Chain
# ============================================================

class MarkovStrategy(BaseStrategy):
    name = "Markov Chain"
    description = "建立 1 階 Markov 鏈轉移矩陣，用條件機率預測下一局"

    def __init__(self, order: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.order = order

    def predict(self, history: List[str]) -> str:
        filtered = [h for h in history if h != '和']
        if len(filtered) <= self.order:
            return '莊'

        # 建立轉移計數
        transitions = defaultdict(Counter)
        for i in range(len(filtered) - self.order):
            state = tuple(filtered[i:i + self.order])
            next_val = filtered[i + self.order]
            transitions[state][next_val] += 1

        # 當前狀態
        current_state = tuple(filtered[-self.order:])

        if current_state in transitions:
            counts = transitions[current_state]
            if counts.get('莊', 0) >= counts.get('閒', 0):
                return '莊'
            return '閒'

        return '莊'


# ============================================================
#  策略 9: 貝氏推論 (Bayesian)
# ============================================================

class BayesianStrategy(BaseStrategy):
    name = "貝氏推論"
    description = "用 Beta 分佈動態更新莊閒後驗機率"

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = prior_alpha  # 莊贏的先驗
        self.beta = prior_beta   # 閒贏的先驗

    def predict(self, history: List[str]) -> str:
        alpha = self.alpha
        beta_param = self.beta

        for h in history:
            if h == '莊':
                alpha += 1
            elif h == '閒':
                beta_param += 1
            # 和局不更新

        # 後驗均值
        banker_prob = alpha / (alpha + beta_param)

        if banker_prob >= 0.5:
            return '莊'
        return '閒'

    def reset(self):
        self.alpha = 1.0
        self.beta = 1.0


# ============================================================
#  策略 10: 移動平均 (Moving Average)
# ============================================================

class MovingAverageStrategy(BaseStrategy):
    name = "移動平均"
    description = "最近 N 局的莊贏比率超過 50% 就壓莊，否則壓閒"

    def __init__(self, window: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.window = window

    def predict(self, history: List[str]) -> str:
        if len(history) < self.window:
            return '莊'

        recent = history[-self.window:]
        filtered = [h for h in recent if h != '和']
        if not filtered:
            return '莊'

        banker_ratio = sum(1 for h in filtered if h == '莊') / len(filtered)

        if banker_ratio >= 0.5:
            return '莊'
        return '閒'


# ============================================================
#  策略 11: 機器學習 (Logistic Regression + Random Forest)
# ============================================================

class MLStrategy(BaseStrategy):
    name = "機器學習"
    description = "用 Logistic Regression 從歷史特徵學習預測（需 50 局預熱）"

    def __init__(self, lookback: int = 5, retrain_every: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.retrain_every = retrain_every
        self.model = None
        self.last_train_size = 0

    def _encode(self, val: str) -> int:
        if val == '莊':
            return 1
        elif val == '閒':
            return 0
        return 2  # 和

    def _build_features(self, history: List[str], idx: int) -> np.ndarray:
        """構建滑動窗口特徵"""
        features = []
        for j in range(self.lookback):
            pos = idx - self.lookback + j
            if pos >= 0:
                features.append(self._encode(history[pos]))
            else:
                features.append(-1)

        # 追加統計特徵
        window = history[max(0, idx - 20):idx]
        if window:
            filtered = [h for h in window if h != '和']
            banker_ratio = sum(1 for h in filtered if h == '莊') / max(len(filtered), 1)
            streak = 1
            for k in range(len(filtered) - 2, -1, -1):
                if filtered[k] == filtered[-1]:
                    streak += 1
                else:
                    break
            features.append(banker_ratio)
            features.append(streak)
            features.append(len(filtered))
        else:
            features.extend([0.5, 0, 0])

        return np.array(features, dtype=float)

    def _train(self, history: List[str]):
        from sklearn.linear_model import LogisticRegression

        X, y = [], []
        for i in range(self.lookback, len(history)):
            feat = self._build_features(history, i)
            label = 1 if history[i] == '莊' else 0
            if history[i] == '和':
                continue  # 跳過和局
            X.append(feat)
            y.append(label)

        if len(X) < 20 or len(set(y)) < 2:
            self.model = None
            return

        X = np.array(X)
        y = np.array(y)

        self.model = LogisticRegression(max_iter=500, random_state=42)
        self.model.fit(X, y)
        self.last_train_size = len(history)

    def predict(self, history: List[str]) -> str:
        if len(history) < self.lookback + 20:
            return '莊'  # 預熱期

        # 定期重新訓練
        if self.model is None or (len(history) - self.last_train_size) >= self.retrain_every:
            self._train(history)

        if self.model is None:
            return '莊'

        feat = self._build_features(history, len(history)).reshape(1, -1)
        pred = self.model.predict(feat)[0]
        return '莊' if pred == 1 else '閒'

    def reset(self):
        self.model = None
        self.last_train_size = 0


# ============================================================
#  策略 12: 組合投票 (Ensemble Vote)
# ============================================================

class EnsembleVoteStrategy(BaseStrategy):
    name = "組合投票"
    description = "綜合多個策略的預測結果，以多數決投票決定"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sub_strategies = [
            FollowLastStrategy(),
            StreakFollowerStrategy(),
            StreakBreakerStrategy(),
            FrequencyStrategy(),
            BigRoadStrategy(),
            MarkovStrategy(),
            BayesianStrategy(),
            MovingAverageStrategy(),
        ]

    def predict(self, history: List[str]) -> str:
        votes = Counter()
        for s in self.sub_strategies:
            pred = s.predict(history)
            votes[pred] += 1

        # 多數決
        winner = votes.most_common(1)[0][0]
        return winner

    def reset(self):
        for s in self.sub_strategies:
            s.reset()


# ============================================================
#  策略 13: Avant Dernier 壓前前局（法國經典）
# ============================================================

class AvantDernierStrategy(BaseStrategy):
    name = "Avant Dernier"
    description = "法國經典打法：壓倒數第二局的結果，而非最後一局。在跳路和連路都有不錯表現"

    def predict(self, history: List[str]) -> str:
        filtered = [h for h in history if h != '和']
        if len(filtered) < 2:
            return '莊'
        return filtered[-2]  # 壓倒數第二局


# ============================================================
#  策略 14: 三珠路打法（亞洲流行）
# ============================================================

class ThreeBeadStrategy(BaseStrategy):
    name = "三珠路打法"
    description = "將結果每 3 局一組分析，根據組合模式預測。亞洲職業賭客常用"

    def predict(self, history: List[str]) -> str:
        filtered = [h for h in history if h != '和']
        if len(filtered) < 3:
            return '莊'

        # 取最後完整三珠組
        remainder = len(filtered) % 3
        if remainder == 0:
            # 剛好完成一組 → 看最後一組的模式預測下一組第一局
            last_group = filtered[-3:]
        else:
            # 看上一完整組
            last_group = filtered[-(remainder + 3):-remainder] if remainder + 3 <= len(filtered) else filtered[:3]

        b_count = sum(1 for x in last_group if x == '莊')
        p_count = 3 - b_count

        # 三珠規則:
        # BBB → 跟莊   PPP → 跟閒
        # BBP/BPB/PBB → 壓閒（莊多反壓）
        # PPB/PBP/BPP → 壓莊（閒多反壓）
        if b_count == 3:
            return '莊'
        elif p_count == 3:
            return '閒'
        elif b_count > p_count:
            return '閒'  # 莊多 → 反壓閒
        else:
            return '莊'  # 閒多 → 反壓莊


# ============================================================
#  策略 15: 微笑心法（亞洲經典）
# ============================================================

class SmileStrategy(BaseStrategy):
    name = "微笑心法"
    description = "贏→跟注，輸→反壓。核心：贏就繼續壓贏家，輸就換邊。簡單有效的跟輸反路法"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_bet: Optional[str] = None

    def predict(self, history: List[str]) -> str:
        if not history:
            self.last_bet = '莊'
            return '莊'

        filtered = [h for h in history if h != '和']
        if not filtered:
            self.last_bet = '莊'
            return '莊'

        last_result = filtered[-1]

        if self.last_bet is None:
            # 第一次 → 跟最後結果
            self.last_bet = last_result
            return last_result

        if self.last_bet == last_result:
            # 上一手贏了 → 繼續跟
            return self.last_bet
        else:
            # 上一手輸了 → 換邊
            self.last_bet = last_result
            return last_result

    def reset(self):
        self.last_bet = None


# ============================================================
#  策略 16: 長莊長閒追擊
# ============================================================

class LongRunChaseStrategy(BaseStrategy):
    name = "長莊長閒追擊"
    description = "專門追打長龍。連 3 局以上同方即跟注追擊，一直追到斷龍為止"

    def __init__(self, trigger: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.trigger = trigger

    def predict(self, history: List[str]) -> str:
        filtered = [h for h in history if h != '和']
        if len(filtered) < self.trigger:
            return '莊'

        # 計算當前連勝
        last = filtered[-1]
        streak = 0
        for h in reversed(filtered):
            if h == last:
                streak += 1
            else:
                break

        if streak >= self.trigger:
            return last  # 追龍
        else:
            return '莊'  # 無龍時默認壓莊

    def reset(self):
        pass


# ============================================================
#  策略 17: 跳路打法（單跳/雙跳識別）
# ============================================================

class ChopPatternStrategy(BaseStrategy):
    name = "跳路打法"
    description = "識別「單跳」(BPBP... 交替) 和「雙跳」(BBPPBBPP...) 模式並跟打"

    def predict(self, history: List[str]) -> str:
        filtered = [h for h in history if h != '和']
        if len(filtered) < 4:
            return '莊'

        # 建列
        columns = self._build_columns(filtered)

        if len(columns) < 4:
            return '莊'

        recent_lens = [len(c) for c in columns[-4:]]

        # 單跳: 1,1,1,1
        if all(l == 1 for l in recent_lens):
            last_side = columns[-1][0]
            return '閒' if last_side == '莊' else '莊'

        # 雙跳: 2,2,2,2
        if all(l == 2 for l in recent_lens):
            current_col_len = len(columns[-1])
            if current_col_len == 1:
                return columns[-1][0]  # 還差一個，跟
            else:
                return '閒' if columns[-1][0] == '莊' else '莊'

        # 節奏跳: 檢查 pattern 1,2,1,2 / 2,1,2,1
        if len(columns) >= 4:
            p4 = [len(c) for c in columns[-4:]]
            if p4 == [1, 2, 1, 2] or p4 == [2, 1, 2, 1]:
                expected_len = 1 if p4[-1] == 2 else 2
                if expected_len == 1:
                    return '閒' if columns[-1][0] == '莊' else '莊'
                else:
                    return columns[-1][0]

        # 無明確模式 → 壓莊
        return '莊'

    def _build_columns(self, filtered: List[str]) -> List[List[str]]:
        if not filtered:
            return []
        columns = [[filtered[0]]]
        for h in filtered[1:]:
            if h == columns[-1][0]:
                columns[-1].append(h)
            else:
                columns.append([h])
        return columns


# ============================================================
#  策略 18: 天地人打法（三柱分析法，亞洲進階）
# ============================================================

class TianDiRenStrategy(BaseStrategy):
    name = "天地人打法"
    description = "天地人三柱：將結果分天(1st)、地(2nd)、人(3rd)三個位置循環分析，找出各位置的趨勢"

    def predict(self, history: List[str]) -> str:
        filtered = [h for h in history if h != '和']
        if len(filtered) < 6:
            return '莊'

        # 分三柱
        tian = filtered[0::3]  # 天位
        di = filtered[1::3]    # 地位
        ren = filtered[2::3]   # 人位

        # 下一局在哪個位置？
        pos = len(filtered) % 3  # 0=天, 1=地, 2=人
        if pos == 0:
            pillar = tian
        elif pos == 1:
            pillar = di
        else:
            pillar = ren

        if not pillar:
            return '莊'

        # 分析該柱的最近趨勢
        recent = pillar[-3:] if len(pillar) >= 3 else pillar
        b_count = sum(1 for x in recent if x == '莊')
        p_count = len(recent) - b_count

        if b_count > p_count:
            return '莊'
        elif p_count > b_count:
            return '閒'
        else:
            # 平手→看該柱最後一局
            return pillar[-1]


# ============================================================
#  策略 19: 五路路紙綜合分析
# ============================================================

class FiveRoadStrategy(BaseStrategy):
    name = "五路路紙"
    description = "綜合大路、大眼仔、小路、曱甴路、珠盤路的模式分析進行預測"

    def predict(self, history: List[str]) -> str:
        try:
            from roads import RoadManager
        except ImportError:
            return '莊'

        rm = RoadManager()
        for h in history:
            rm.add_result(h)

        pred = rm.predict_by_roads()
        return pred['side']


# ============================================================
#  策略 20: 雙龍打法
# ============================================================

class DoubleDragonStrategy(BaseStrategy):
    name = "雙龍打法"
    description = "追蹤兩條路線（奇數局/偶數局），各自獨立分析趨勢再綜合判斷"

    def predict(self, history: List[str]) -> str:
        filtered = [h for h in history if h != '和']
        if len(filtered) < 6:
            return '莊'

        # 奇數局路線（第1,3,5,7...局）和偶數局路線（第2,4,6,8...局）
        dragon_odd = filtered[0::2]
        dragon_even = filtered[1::2]

        # 下一局屬於哪條龍？
        is_odd = len(filtered) % 2 == 0  # 下一局是奇數位

        if is_odd:
            primary = dragon_odd
            secondary = dragon_even
        else:
            primary = dragon_even
            secondary = dragon_odd

        # 分析主路線趨勢
        if len(primary) >= 2:
            recent = primary[-3:] if len(primary) >= 3 else primary
            b = sum(1 for x in recent if x == '莊')
            p = len(recent) - b
            if b > p:
                primary_pred = '莊'
            elif p > b:
                primary_pred = '閒'
            else:
                primary_pred = primary[-1]
        else:
            primary_pred = '莊'

        # 副路線作為參考
        if len(secondary) >= 2:
            recent_s = secondary[-2:]
            b_s = sum(1 for x in recent_s if x == '莊')
            secondary_pred = '莊' if b_s >= 1 else '閒'
        else:
            secondary_pred = '莊'

        # 兩條龍一致 → 高信心
        if primary_pred == secondary_pred:
            return primary_pred
        else:
            return primary_pred  # 以主路線為主


# ============================================================
#  策略 21: 排排連打法
# ============================================================

class SequentialStreakStrategy(BaseStrategy):
    name = "排排連打法"
    description = "分析大路列長變化趨勢（遞增/遞減/平穩），預測下一列長度並據此下注"

    def predict(self, history: List[str]) -> str:
        filtered = [h for h in history if h != '和']
        if len(filtered) < 4:
            return '莊'

        # 建列
        columns = []
        current = [filtered[0]]
        for h in filtered[1:]:
            if h == current[0]:
                current.append(h)
            else:
                columns.append(current)
                current = [h]
        columns.append(current)

        if len(columns) < 3:
            return '莊'

        lens = [len(c) for c in columns]
        last_3 = lens[-3:]

        # 判斷列長趨勢
        if last_3[-1] > last_3[-2] > last_3[-3]:
            # 遞增趨勢 → 當前連勝會繼續
            return columns[-1][0]
        elif last_3[-1] < last_3[-2] < last_3[-3]:
            # 遞減趨勢 → 快斷了
            other = '閒' if columns[-1][0] == '莊' else '莊'
            return other
        elif last_3[-1] == last_3[-2]:
            # 穩定 → 跟
            return columns[-1][0]
        else:
            # 不規則 → 看大眾
            return '莊'


# ============================================================
#  策略 22: 正反纜打法
# ============================================================

class CableStrategy(BaseStrategy):
    name = "正反纜打法"
    description = "正纜贏跟（追贏），反纜輸跟（追輸），根據近期勝率自動切換正反纜"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = '正纜'  # 正纜 or 反纜
        self.last_bet = None
        self.win_count = 0
        self.loss_count = 0

    def predict(self, history: List[str]) -> str:
        filtered = [h for h in history if h != '和']
        if not filtered:
            self.last_bet = '莊'
            return '莊'

        last_result = filtered[-1]

        # 計算近10局勝率
        recent = filtered[-10:]
        if len(recent) >= 5:
            b_count = sum(1 for x in recent if x == '莊')
            b_rate = b_count / len(recent)
            # 勝率高於55% → 正纜（追贏），低於45% → 反纜（追輸）
            if b_rate > 0.55:
                self.mode = '正纜'
            elif b_rate < 0.45:
                self.mode = '反纜'

        if self.last_bet is None:
            self.last_bet = last_result
            return last_result

        if self.last_bet == last_result:
            # 贏了
            if self.mode == '正纜':
                # 正纜：贏了繼續跟
                self.last_bet = last_result
                return last_result
            else:
                # 反纜：贏了換邊
                other = '閒' if last_result == '莊' else '莊'
                self.last_bet = other
                return other
        else:
            # 輸了
            if self.mode == '正纜':
                # 正纜：輸了換邊
                self.last_bet = last_result
                return last_result
            else:
                # 反纜：輸了繼續壓
                return self.last_bet

    def reset(self):
        self.mode = '正纜'
        self.last_bet = None
        self.win_count = 0
        self.loss_count = 0


# ============================================================
#  資金管理策略
# ============================================================

class BettingSystem:
    """注碼管理基底"""
    name: str = "基底"

    def __init__(self, base_unit: float = 100):
        self.base_unit = base_unit
        self.current_bet = base_unit
        self.history: List[float] = []

    def next_bet(self) -> float:
        return self.current_bet

    def update(self, won: bool, payout_ratio: float = 1.0):
        raise NotImplementedError

    def reset(self):
        self.current_bet = self.base_unit
        self.history.clear()


class FlatBetting(BettingSystem):
    name = "平注法"

    def update(self, won: bool, payout_ratio: float = 1.0):
        self.current_bet = self.base_unit


class MartingaleBetting(BettingSystem):
    name = "Martingale 倍投"

    def __init__(self, base_unit: float = 100, max_bet: float = 100000):
        super().__init__(base_unit)
        self.max_bet = max_bet

    def update(self, won: bool, payout_ratio: float = 1.0):
        if won:
            self.current_bet = self.base_unit
        else:
            self.current_bet = min(self.current_bet * 2, self.max_bet)


class FibonacciBetting(BettingSystem):
    name = "Fibonacci"

    def __init__(self, base_unit: float = 100):
        super().__init__(base_unit)
        self.seq = [1, 1]
        self.idx = 0

    def next_bet(self) -> float:
        return self.base_unit * self.seq[self.idx]

    def update(self, won: bool, payout_ratio: float = 1.0):
        if won:
            self.idx = max(0, self.idx - 2)
        else:
            self.idx += 1
            if self.idx >= len(self.seq):
                self.seq.append(self.seq[-1] + self.seq[-2])
        self.current_bet = self.base_unit * self.seq[self.idx]

    def reset(self):
        super().reset()
        self.seq = [1, 1]
        self.idx = 0


class ParoliBetting(BettingSystem):
    name = "Paroli 正注"

    def __init__(self, base_unit: float = 100, max_wins: int = 3):
        super().__init__(base_unit)
        self.max_wins = max_wins
        self.consecutive_wins = 0

    def update(self, won: bool, payout_ratio: float = 1.0):
        if won:
            self.consecutive_wins += 1
            if self.consecutive_wins >= self.max_wins:
                self.current_bet = self.base_unit
                self.consecutive_wins = 0
            else:
                self.current_bet *= 2
        else:
            self.current_bet = self.base_unit
            self.consecutive_wins = 0

    def reset(self):
        super().reset()
        self.consecutive_wins = 0


# ============================================================
#  注碼 5: 1-3-2-6 系統（經典正注法）
# ============================================================

class Betting1326(BettingSystem):
    name = "1-3-2-6"

    def __init__(self, base_unit: float = 100):
        super().__init__(base_unit)
        self.sequence = [1, 3, 2, 6]
        self.step = 0

    def next_bet(self) -> float:
        return self.base_unit * self.sequence[self.step]

    def update(self, won: bool, payout_ratio: float = 1.0):
        if won:
            self.step = min(self.step + 1, 3)
            if self.step > 3:
                self.step = 0
        else:
            self.step = 0
        self.current_bet = self.base_unit * self.sequence[self.step]

    def reset(self):
        super().reset()
        self.step = 0


# ============================================================
#  注碼 6: 1-3-2-4 系統（1326 改良版，風險更低）
# ============================================================

class Betting1324(BettingSystem):
    name = "1-3-2-4"

    def __init__(self, base_unit: float = 100):
        super().__init__(base_unit)
        self.sequence = [1, 3, 2, 4]
        self.step = 0

    def next_bet(self) -> float:
        return self.base_unit * self.sequence[self.step]

    def update(self, won: bool, payout_ratio: float = 1.0):
        if won:
            self.step += 1
            if self.step >= 4:
                self.step = 0
        else:
            self.step = 0
        self.current_bet = self.base_unit * self.sequence[self.step]

    def reset(self):
        super().reset()
        self.step = 0


# ============================================================
#  注碼 7: Oscar's Grind（穩健正注）
# ============================================================

class OscarsGrind(BettingSystem):
    name = "Oscar's Grind"

    def __init__(self, base_unit: float = 100):
        super().__init__(base_unit)
        self.session_profit = 0

    def next_bet(self) -> float:
        return self.current_bet

    def update(self, won: bool, payout_ratio: float = 1.0):
        if won:
            self.session_profit += self.current_bet
            if self.session_profit >= self.base_unit:
                # 目標達成，重設
                self.current_bet = self.base_unit
                self.session_profit = 0
            else:
                self.current_bet += self.base_unit
        else:
            self.session_profit -= self.current_bet
            # 輸了注碼不變

    def reset(self):
        super().reset()
        self.session_profit = 0


# ============================================================
#  注碼 8: Labouchere（負進纜 / 消數法）
# ============================================================

class LabouchereBetting(BettingSystem):
    name = "Labouchere"

    def __init__(self, base_unit: float = 100, sequence: list = None):
        super().__init__(base_unit)
        self.original_seq = sequence or [1, 2, 3, 4, 5]
        self.seq = list(self.original_seq)

    def next_bet(self) -> float:
        if not self.seq:
            return self.base_unit
        if len(self.seq) == 1:
            return self.base_unit * self.seq[0]
        return self.base_unit * (self.seq[0] + self.seq[-1])

    def update(self, won: bool, payout_ratio: float = 1.0):
        if not self.seq:
            self.seq = list(self.original_seq)

        if won:
            # 贏：消頭尾
            if len(self.seq) >= 2:
                self.seq.pop(0)
                self.seq.pop(-1)
            elif len(self.seq) == 1:
                self.seq.pop(0)
            if not self.seq:
                self.seq = list(self.original_seq)
        else:
            # 輸：把注碼數加到尾部
            bet_units = (self.seq[0] + self.seq[-1]) if len(self.seq) >= 2 else (self.seq[0] if self.seq else 1)
            self.seq.append(bet_units)

        self.current_bet = self.next_bet()

    def reset(self):
        super().reset()
        self.seq = list(self.original_seq)


# ============================================================
#  注碼 9: D'Alembert（等差遞增法）
# ============================================================

class DAlembertBetting(BettingSystem):
    name = "D'Alembert"

    def __init__(self, base_unit: float = 100):
        super().__init__(base_unit)
        self.level = 1

    def next_bet(self) -> float:
        return self.base_unit * self.level

    def update(self, won: bool, payout_ratio: float = 1.0):
        if won:
            self.level = max(1, self.level - 1)
        else:
            self.level += 1
        self.current_bet = self.base_unit * self.level

    def reset(self):
        super().reset()
        self.level = 1


# ============================================================
#  注碼 10: Anti-Martingale（反倍投）
# ============================================================

class AntiMartingaleBetting(BettingSystem):
    name = "Anti-Martingale"

    def __init__(self, base_unit: float = 100, max_doubles: int = 4):
        super().__init__(base_unit)
        self.max_doubles = max_doubles
        self.consecutive_wins = 0

    def update(self, won: bool, payout_ratio: float = 1.0):
        if won:
            self.consecutive_wins += 1
            if self.consecutive_wins >= self.max_doubles:
                self.current_bet = self.base_unit
                self.consecutive_wins = 0
            else:
                self.current_bet *= 2
        else:
            self.current_bet = self.base_unit
            self.consecutive_wins = 0

    def reset(self):
        super().reset()
        self.consecutive_wins = 0


# ============================================================
#  策略工廠
# ============================================================

def get_all_strategies(seed: Optional[int] = None) -> List[BaseStrategy]:
    """回傳所有策略的實例"""
    return [
        RandomStrategy(seed=seed),
        FollowLastStrategy(seed=seed),
        OppositeLastStrategy(seed=seed),
        StreakFollowerStrategy(seed=seed),
        StreakBreakerStrategy(seed=seed),
        FrequencyStrategy(seed=seed),
        BigRoadStrategy(seed=seed),
        MarkovStrategy(seed=seed),
        BayesianStrategy(seed=seed),
        MovingAverageStrategy(seed=seed),
        MLStrategy(seed=seed),
        EnsembleVoteStrategy(seed=seed),
        # --- 全球知名打法 ---
        AvantDernierStrategy(seed=seed),
        ThreeBeadStrategy(seed=seed),
        SmileStrategy(seed=seed),
        LongRunChaseStrategy(seed=seed),
        ChopPatternStrategy(seed=seed),
        TianDiRenStrategy(seed=seed),
        FiveRoadStrategy(seed=seed),
        DoubleDragonStrategy(seed=seed),
        SequentialStreakStrategy(seed=seed),
        CableStrategy(seed=seed),
    ]


def get_all_betting_systems(base_unit: float = 100) -> List[BettingSystem]:
    """回傳所有注碼管理系統"""
    return [
        FlatBetting(base_unit),
        MartingaleBetting(base_unit),
        FibonacciBetting(base_unit),
        ParoliBetting(base_unit),
        Betting1326(base_unit),
        Betting1324(base_unit),
        OscarsGrind(base_unit),
        LabouchereBetting(base_unit),
        DAlembertBetting(base_unit),
        AntiMartingaleBetting(base_unit),
    ]
