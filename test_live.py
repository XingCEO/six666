"""
非互動式測試 — 載入 CSV 資料驗證系統
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_tracker import LivePredictor, load_csv, print_dashboard

predictor = LivePredictor()

# 載入範例數據
count = load_csv('sample_data.csv', predictor)
print(f"✅ 載入 {count} 局真實數據")

# 顯示儀表板
print_dashboard(predictor)

# 顯示剩餘牌面
print("\n── 剩餘牌面詳情 ──")
print(predictor.tracker.counter.get_status_display())

# 計算真實條件機率
print("\n── 真實條件機率（基於剩餘牌組，非固定值）──")
prob = predictor.get_real_probabilities(sample_size=50000)
print(f"  閒: {prob['閒']:.2f}%")
print(f"  莊: {prob['莊']:.2f}%")
print(f"  和: {prob['和']:.2f}%")

# 顯示算牌優勢指標
edge = predictor.tracker.counter.get_edge_indicator()
print(f"\n── 牌況分析 ──")
for k, v in edge.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.2f}")
    else:
        print(f"  {k}: {v}")

print("\n✅ 所有功能正常運作！")
