"""測試 v3 預測引擎 — 驗證新欄位和三層信號"""
import json
import urllib.request

BASE = "http://localhost:8888"

def api_get(path):
    r = urllib.request.urlopen(f"{BASE}{path}")
    return json.loads(r.read())

def api_post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=body,
                                 headers={"Content-Type":"application/json"})
    r = urllib.request.urlopen(req)
    return json.loads(r.read())

# Reset
api_post("/api/reset", {})
print("✅ Reset OK")

# 輸入 25 局測試資料
results = ['莊','莊','莊','閒','閒','莊','閒','莊','莊','莊',
           '閒','閒','閒','莊','和','閒','莊','莊','閒','閒',
           '莊','莊','莊','閒','莊']

for i, r in enumerate(results):
    resp = api_post("/api/record", {"outcome": r})
    assert resp["ok"], f"Round {i+1} failed"

print(f"✅ {len(results)} 局記錄完成")

# 取得完整狀態
state = api_get("/api/state")

# 驗證基本欄位
assert state["round"] == 25, f"round={state['round']}"
assert "prediction" in state
print(f"✅ 總局數: {state['round']}")

pred = state["prediction"]

# 驗證新欄位存在
assert "skip" in pred, "Missing 'skip' field"
assert "skip_reason" in pred, "Missing 'skip_reason' field"
assert "weighted_banker" in pred, "Missing 'weighted_banker' field"
assert "weighted_player" in pred, "Missing 'weighted_player' field"
assert "card_signal" in pred, "Missing 'card_signal' field"
assert "layers" in pred, "Missing 'layers' field"
print(f"✅ 新欄位完整: skip={pred['skip']}, confidence={pred['confidence']}%")

# 驗證三層結構
layers = pred["layers"]
assert "strategy" in layers, "Missing layer: strategy"
assert "road" in layers, "Missing layer: road"
assert "card" in layers, "Missing layer: card"

l1 = layers["strategy"]
l2 = layers["road"]
l3 = layers["card"]
print(f"  Layer1 策略: {l1['side']} ({l1['conf']}%)")
print(f"  Layer2 路紙: {l2['side']} ({l2['conf']}%) - {l2.get('reasons',[''])[0]}")
print(f"  Layer3 算牌: {l3['side'] or '無信號'} (偏差 {l3['edge']})")

# 驗證策略有新欄位
strats = pred["strategies"]
assert len(strats) == 22, f"Expected 22 strategies, got {len(strats)}"
s0 = strats[0]
assert "acc_recent" in s0, "Missing 'acc_recent' in strategy"
assert "weight" in s0, "Missing 'weight' in strategy"
assert "status" in s0, "Missing 'status' in strategy"
assert "streak" in s0, "Missing 'streak' in strategy"

hot_count = sum(1 for s in strats if s["status"] == "hot")
cold_count = sum(1 for s in strats if s["status"] == "cold")
print(f"✅ 22策略: {hot_count} 熱 / {cold_count} 冷")

# 驗證算牌
cs = pred["card_signal"]
assert "side" in cs and "edge" in cs and "prob" in cs
print(f"✅ 算牌信號: side={cs['side']}, edge={cs['edge']}")
if cs["prob"]:
    print(f"  莊={cs['prob'].get('莊',0):.2f}% 閒={cs['prob'].get('閒',0):.2f}% 和={cs['prob'].get('和',0):.2f}%")

# 驗證 system_accuracy 和 skip_count
assert "system_accuracy" in state, "Missing system_accuracy"
assert "skip_count" in state, "Missing skip_count"
print(f"✅ 系統準確率: {state['system_accuracy']}%, 跳過: {state['skip_count']}局")

# 驗證路紙數據
roads = state.get("roads", {})
assert "big_road" in roads, "Missing big_road"
assert "bead" in roads, "Missing bead"
print(f"✅ 路紙: {len(roads.get('big_road',[]))} 大路列, {len(roads.get('bead',[]))} 珠盤")

# 驗證建議注碼
print(f"✅ 建議: {'跳局' if pred['skip'] else pred['consensus']} 注碼={pred['suggested_bet']}")

# 驗證預測一致性
print(f"\n✅ 最終預測: {pred['consensus']} (信心 {pred['confidence']}%)")
print(f"  投票: {pred['banker_votes']}莊 vs {pred['player_votes']}閒")

print("\n" + "="*50)
print("🎉 所有測試通過！v3 預測引擎運作正常")
print("="*50)
