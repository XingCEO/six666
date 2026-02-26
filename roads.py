"""
百家樂完整路紙分析系統 — roads.py
===================================
五路完整實現：
  1. 珠盤路 (Bead Plate)     — 逐局記錄
  2. 大路   (Big Road)       — 主路，變色換列
  3. 大眼仔 (Big Eye Boy)    — 衍生路 1
  4. 小路   (Small Road)     — 衍生路 2
  5. 曱甴路 (Cockroach Pig)  — 衍生路 3

規則來源：澳門賭場標準 + 國際通用規則
"""

from typing import List, Tuple, Optional, Dict


# ============================================================
#  1. 珠盤路 (Bead Plate / Bead Road)
# ============================================================
class BeadPlate:
    """
    最簡單的路紙：逐局按順序排列，每格一局。
    莊=紅, 閒=藍, 和=綠, 莊對=紅點, 閒對=藍點
    通常排成 6 行 N 列
    """
    def __init__(self, rows: int = 6):
        self.rows = rows
        self.entries: List[dict] = []

    def add(self, outcome: str, banker_pair: bool = False, player_pair: bool = False):
        self.entries.append({
            'outcome': outcome,
            'banker_pair': banker_pair,
            'player_pair': player_pair,
        })

    def get_grid(self) -> List[List[Optional[dict]]]:
        """回傳 rows × cols 的格子"""
        total = len(self.entries)
        cols = (total + self.rows - 1) // self.rows if total > 0 else 1
        grid = [[None] * cols for _ in range(self.rows)]
        for i, entry in enumerate(self.entries):
            col = i // self.rows
            row = i % self.rows
            grid[row][col] = entry
        return grid

    def reset(self):
        self.entries.clear()


# ============================================================
#  2. 大路 (Big Road)
# ============================================================
class BigRoad:
    """
    百家樂最核心的路紙。
    規則：
    - 同一方連贏 → 同一列往下排
    - 換方 → 另起新列
    - 和局 → 在上一格畫綠線，不獨佔格子
    - 對子 → 在格子上加點
    - 列滿 6 格 → 往右拐（龍尾）
    """
    def __init__(self, max_rows: int = 6):
        self.max_rows = max_rows
        # columns: List of columns, each column is List of cells
        # cell: {'side': '莊'/'閒', 'ties': int, 'bp': bool, 'pp': bool}
        self.columns: List[List[dict]] = []
        self.last_side: Optional[str] = None

    def add(self, outcome: str, banker_pair: bool = False, player_pair: bool = False):
        if outcome == '和':
            # 和局：加到最後一個有效格子上
            if self.columns:
                last_col = self.columns[-1]
                if last_col:
                    last_col[-1]['ties'] += 1
            return

        side = outcome  # '莊' or '閒'

        if side == self.last_side and self.columns:
            # 同一方 → 同一列繼續往下
            self.columns[-1].append({
                'side': side, 'ties': 0,
                'bp': banker_pair, 'pp': player_pair,
            })
        else:
            # 換方 → 新一列
            self.columns.append([{
                'side': side, 'ties': 0,
                'bp': banker_pair, 'pp': player_pair,
            }])
            self.last_side = side

    def get_grid(self, max_cols: int = 40) -> List[List[Optional[dict]]]:
        """
        轉換成 rows × cols 的格子（處理龍尾）
        龍尾規則：列超過 max_rows 時往右拐
        """
        grid: Dict[Tuple[int, int], dict] = {}
        col_offset = 0

        for col_data in self.columns:
            row = 0
            col = col_offset
            for i, cell in enumerate(col_data):
                if row >= self.max_rows:
                    # 龍尾：往右走
                    row = self.max_rows - 1
                    col += 1
                # 檢查是否已被佔用（處理多列龍尾）
                while (row, col) in grid:
                    col += 1
                grid[(row, col)] = cell
                row += 1
            col_offset = col + 1

        if not grid:
            return [[None]]

        max_c = max(c for _, c in grid.keys()) + 1
        max_r = self.max_rows
        result = [[None] * min(max_c, max_cols) for _ in range(max_r)]
        for (r, c), cell in grid.items():
            if r < max_r and c < max_cols:
                result[r][c] = cell
        return result

    def get_column_lengths(self) -> List[int]:
        """取得每列的長度（用於衍生路計算）"""
        return [len(col) for col in self.columns]

    def reset(self):
        self.columns.clear()
        self.last_side = None


# ============================================================
#  衍生路通用邏輯
# ============================================================
def _derive_road(big_road: BigRoad, shift: int) -> List[str]:
    """
    計算衍生路（大眼仔 shift=1, 小路 shift=2, 曱甴路 shift=3）

    衍生路規則（核心）：
    每一個大路的新格子都會產生一個衍生路結果（紅/藍）

    大路新格子位於某列的第 N 行時：
    - 比較「當前列」和「前 shift 列」

    case 1: N=1（新列的第一格）
      → 比較 column[current - shift] 的長度和 column[current - shift - 1] 的長度
      → 若相同 → 紅 (齊整)
      → 若不同 → 藍 (不齊)

    case 2: N>1（列中的後續格）
      → 檢查 column[current - shift] 在第 N 行是否有格子
      → 若有 → 紅 (齊整)
      → 若沒有 → 藍 (不齊)
    """
    columns = big_road.columns
    result = []

    for col_idx, col_data in enumerate(columns):
        for row_idx in range(len(col_data)):
            # 衍生路起始位置
            if col_idx < shift:
                continue
            if col_idx == shift and row_idx == 0:
                continue  # 首格跳過

            if row_idx == 0:
                # Case 1: 新列第一格
                compare_col = col_idx - shift
                prev_col = compare_col - 1
                if compare_col >= 0 and prev_col >= 0:
                    len_compare = len(columns[compare_col])
                    len_prev = len(columns[prev_col])
                    result.append('紅' if len_compare == len_prev else '藍')
                elif compare_col >= 0:
                    result.append('藍')
            else:
                # Case 2: 列中的後續格
                compare_col = col_idx - shift
                if compare_col >= 0:
                    has_cell = row_idx < len(columns[compare_col])
                    result.append('紅' if has_cell else '藍')

    return result


# ============================================================
#  3. 大眼仔 (Big Eye Boy)
# ============================================================
class BigEyeBoy:
    """
    衍生路 1：從大路第 2 列第 2 行開始
    紅 = 齊整（趨勢延續） → 跟路
    藍 = 不齊（趨勢改變） → 反路
    """
    def __init__(self):
        self.entries: List[str] = []

    def calculate(self, big_road: BigRoad):
        self.entries = _derive_road(big_road, shift=1)

    def get_grid(self, rows: int = 6, max_cols: int = 40) -> List[List[Optional[str]]]:
        return self._to_grid(self.entries, rows, max_cols)

    @staticmethod
    def _to_grid(entries: List[str], rows: int, max_cols: int) -> List[List[Optional[str]]]:
        """轉成衍生路的格子（同色往下，變色換列）"""
        if not entries:
            return [[None] * max_cols for _ in range(rows)]

        columns: List[List[str]] = []
        current_col = [entries[0]]
        for e in entries[1:]:
            if e == current_col[-1]:
                current_col.append(e)
            else:
                columns.append(current_col)
                current_col = [e]
        columns.append(current_col)

        grid = [[None] * max_cols for _ in range(rows)]
        col_idx = 0
        for col_data in columns:
            for row, val in enumerate(col_data):
                if row < rows and col_idx < max_cols:
                    grid[row][col_idx] = val
            col_idx += 1
            if col_idx >= max_cols:
                break
        return grid

    def reset(self):
        self.entries.clear()


# ============================================================
#  4. 小路 (Small Road)
# ============================================================
class SmallRoad:
    """衍生路 2：shift=2, 從大路第 3 列第 2 行開始"""
    def __init__(self):
        self.entries: List[str] = []

    def calculate(self, big_road: BigRoad):
        self.entries = _derive_road(big_road, shift=2)

    def get_grid(self, rows: int = 6, max_cols: int = 40) -> List[List[Optional[str]]]:
        return BigEyeBoy._to_grid(self.entries, rows, max_cols)

    def reset(self):
        self.entries.clear()


# ============================================================
#  5. 曱甴路 (Cockroach Pig / Cockroach Road)
# ============================================================
class CockroachPig:
    """衍生路 3：shift=3, 從大路第 4 列第 2 行開始"""
    def __init__(self):
        self.entries: List[str] = []

    def calculate(self, big_road: BigRoad):
        self.entries = _derive_road(big_road, shift=3)

    def get_grid(self, rows: int = 6, max_cols: int = 40) -> List[List[Optional[str]]]:
        return BigEyeBoy._to_grid(self.entries, rows, max_cols)

    def reset(self):
        self.entries.clear()


# ============================================================
#  路紙管理器
# ============================================================
class RoadManager:
    """統一管理五路"""

    def __init__(self):
        self.bead = BeadPlate()
        self.big_road = BigRoad()
        self.big_eye = BigEyeBoy()
        self.small_road = SmallRoad()
        self.cockroach = CockroachPig()
        self.history: List[str] = []

    def add_result(self, outcome: str, banker_pair: bool = False, player_pair: bool = False):
        """記錄一局結果，更新五路"""
        self.history.append(outcome)
        self.bead.add(outcome, banker_pair, player_pair)
        self.big_road.add(outcome, banker_pair, player_pair)
        # 衍生路需要從大路重新計算
        self.big_eye.calculate(self.big_road)
        self.small_road.calculate(self.big_road)
        self.cockroach.calculate(self.big_road)

    def reset(self):
        self.bead.reset()
        self.big_road.reset()
        self.big_eye.reset()
        self.small_road.reset()
        self.cockroach.reset()
        self.history.clear()

    def get_all_roads_data(self) -> Dict:
        """取得全部路紙數據"""
        return {
            'bead': self._bead_to_json(),
            'big_road': self._big_road_to_json(),
            'big_eye': self._derived_to_json(self.big_eye.entries),
            'small_road': self._derived_to_json(self.small_road.entries),
            'cockroach': self._derived_to_json(self.cockroach.entries),
            'patterns': self.analyze_patterns(),
        }

    def _bead_to_json(self) -> List[dict]:
        return self.bead.entries

    def _big_road_to_json(self) -> List[List[dict]]:
        return [[cell for cell in col] for col in self.big_road.columns]

    def _derived_to_json(self, entries: List[str]) -> List[str]:
        return list(entries)

    # ============================================================
    #  路紙模式分析
    # ============================================================
    def analyze_patterns(self) -> Dict:
        """分析當前路紙模式"""
        patterns = {}

        columns = self.big_road.columns
        if not columns:
            return patterns

        # --- 龍 (Dragon) ---
        # 定義：連續 6 格以上同一方
        last_col = columns[-1]
        if len(last_col) >= 6:
            patterns['龍'] = {
                'active': True,
                'side': last_col[0]['side'],
                'length': len(last_col),
                'suggestion': last_col[0]['side'],
                'description': f"龍！{last_col[0]['side']}連 {len(last_col)}",
            }

        # --- 單跳 (Ping-Pong / Chop) ---
        # 定義：最近 4 列以上長度都是 1（交替出現）
        if len(columns) >= 4:
            recent_4 = [len(c) for c in columns[-4:]]
            if all(l == 1 for l in recent_4):
                next_side = '閒' if columns[-1][0]['side'] == '莊' else '莊'
                patterns['單跳'] = {
                    'active': True,
                    'length': self._count_chop_length(columns),
                    'suggestion': next_side,
                    'description': f"單跳模式，下一局壓 {next_side}",
                }

        # --- 雙跳 (Double Chop) ---
        # 定義：最近列長度是 2,2,2,...
        if len(columns) >= 4:
            recent_4 = [len(c) for c in columns[-4:]]
            if all(l == 2 for l in recent_4):
                current_col_len = len(columns[-1])
                if current_col_len < 2:
                    patterns['雙跳'] = {
                        'active': True,
                        'suggestion': columns[-1][0]['side'],
                        'description': '雙跳模式，跟當前方',
                    }

        # --- 排排連 (Streaks) ---
        # 定義：最近幾列長度遞增/遞減
        if len(columns) >= 3:
            recent_3 = [len(c) for c in columns[-3:]]
            if recent_3 == sorted(recent_3):
                patterns['遞增'] = {
                    'active': True,
                    'description': f"列長遞增 {recent_3}",
                }
            elif recent_3 == sorted(recent_3, reverse=True):
                patterns['遞減'] = {
                    'active': True,
                    'description': f"列長遞減 {recent_3}",
                }

        # --- 長莊/長閒 ---
        if columns and len(columns[-1]) >= 4:
            side = columns[-1][0]['side']
            patterns['長路'] = {
                'active': True,
                'side': side,
                'length': len(columns[-1]),
                'suggestion': side,
                'description': f"長{side} {len(columns[-1])} 局",
            }

        # --- 衍生路趨勢 ---
        if self.big_eye.entries:
            recent_eye = self.big_eye.entries[-6:]
            red_count = recent_eye.count('紅')
            blue_count = recent_eye.count('藍')
            if red_count >= 5:
                patterns['大眼仔紅'] = {
                    'active': True,
                    'description': '大眼仔連紅（齊整），跟路',
                }
            elif blue_count >= 5:
                patterns['大眼仔藍'] = {
                    'active': True,
                    'description': '大眼仔連藍（凌亂），反路',
                }

        return patterns

    def _count_chop_length(self, columns) -> int:
        count = 0
        for col in reversed(columns):
            if len(col) == 1:
                count += 1
            else:
                break
        return count

    def predict_by_roads(self) -> Dict:
        """
        路紙預測：根據當前模式給出建議
        回傳 {'side': '莊'/'閒', 'confidence': float, 'reasons': List[str]}
        """
        columns = self.big_road.columns
        if len(columns) < 2:
            return {'side': '莊', 'confidence': 50, 'reasons': ['數據不足']}

        scores = {'莊': 0, '閒': 0}
        reasons = []

        # 1. 大路趨勢
        last_col = columns[-1]
        last_side = last_col[0]['side']

        if len(last_col) >= 4:
            # 長路 → 跟
            scores[last_side] += 3
            reasons.append(f"長{last_side}({len(last_col)})→跟")
        elif len(last_col) == 1 and len(columns) >= 3:
            # 可能是跳路
            prev_lens = [len(c) for c in columns[-3:]]
            if all(l == 1 for l in prev_lens):
                other = '閒' if last_side == '莊' else '莊'
                scores[other] += 2
                reasons.append(f"單跳→壓{other}")

        # 2. 大眼仔趨勢
        if len(self.big_eye.entries) >= 3:
            recent = self.big_eye.entries[-3:]
            if all(e == '紅' for e in recent):
                scores[last_side] += 2
                reasons.append("大眼仔連紅→齊整跟路")
            elif all(e == '藍' for e in recent):
                other = '閒' if last_side == '莊' else '莊'
                scores[other] += 2
                reasons.append("大眼仔連藍→凌亂反路")

        # 3. 小路趨勢
        if len(self.small_road.entries) >= 3:
            recent = self.small_road.entries[-3:]
            if all(e == '紅' for e in recent):
                scores[last_side] += 1
                reasons.append("小路連紅→跟")
            elif all(e == '藍' for e in recent):
                other = '閒' if last_side == '莊' else '莊'
                scores[other] += 1
                reasons.append("小路連藍→反")

        # 4. 曱甴路趨勢
        if len(self.cockroach.entries) >= 3:
            recent = self.cockroach.entries[-3:]
            if all(e == '紅' for e in recent):
                scores[last_side] += 1
                reasons.append("曱甴路連紅→跟")
            elif all(e == '藍' for e in recent):
                other = '閒' if last_side == '莊' else '莊'
                scores[other] += 1
                reasons.append("曱甴路連藍→反")

        total = scores['莊'] + scores['閒']
        if total == 0:
            return {'side': '莊', 'confidence': 50, 'reasons': ['無明顯模式']}

        if scores['莊'] >= scores['閒']:
            side = '莊'
            conf = scores['莊'] / total * 100
        else:
            side = '閒'
            conf = scores['閒'] / total * 100

        return {'side': side, 'confidence': min(conf, 95), 'reasons': reasons}


# ============================================================
#  三珠路分析 (Three-Bead Road)
# ============================================================
class ThreeBeadRoad:
    """
    三珠路打法（亞洲流行）：
    將結果每 3 局分為一組，分析組合模式
    BBB=全莊, PPP=全閒, BBP/BPB/PBB=莊多, PPB/PBP/BPP=閒多
    然後看下一組趨勢
    """
    def __init__(self):
        self.groups: List[str] = []  # 每組的類型

    def analyze(self, results: List[str]) -> Dict:
        filtered = [r for r in results if r != '和']
        if len(filtered) < 6:
            return {'suggestion': None, 'pattern': None}

        # 分組
        self.groups = []
        for i in range(0, len(filtered) - 2, 3):
            group = filtered[i:i+3]
            # 轉代碼
            code = ''.join('B' if x == '莊' else 'P' for x in group)
            self.groups.append(code)

        if len(self.groups) < 2:
            return {'suggestion': None, 'pattern': None}

        # 分析最近的組
        last = self.groups[-1]
        prev = self.groups[-2] if len(self.groups) >= 2 else None

        # 規則：
        # 全同組(BBB/PPP) → 下一組跟同方
        # 混合組 → 看多數方
        b_in_last = last.count('B')
        p_in_last = last.count('P')

        if b_in_last == 3:
            return {'suggestion': '莊', 'pattern': '全莊組→跟莊', 'groups': self.groups}
        elif p_in_last == 3:
            return {'suggestion': '閒', 'pattern': '全閒組→跟閒', 'groups': self.groups}
        elif b_in_last > p_in_last:
            return {'suggestion': '莊', 'pattern': '莊多組→偏莊', 'groups': self.groups}
        else:
            return {'suggestion': '閒', 'pattern': '閒多組→偏閒', 'groups': self.groups}
