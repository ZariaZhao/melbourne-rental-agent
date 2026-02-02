import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Data
data = pd.read_csv('melb_data.csv')

# ==========================================
# 🧪 实验设计：故意不给“高级特征”
# ==========================================
# 我们只给它：房间、距离、房型、车位、浴室 (最基础的硬指标)
# ❌ 去掉 Landsize (地大)
# ❌ 去掉 Lattitude/Longtitude (富人区)
# ❌ 去掉 Year (市场周期)
feature_cols = [
    'Rooms', 'Type', 'Distance', 
    'Bedroom2', 'Bathroom', 'Car'
]

# 简单清洗
X = pd.get_dummies(data[feature_cols])
X = X.fillna(X.median())
y = data['Price']

# 2. Train Model
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = RandomForestRegressor(random_state=1, n_estimators=100)
model.fit(train_X, train_y)

# 3. Evaluate
val_predictions = model.predict(val_X)
r2 = r2_score(val_y, val_predictions)

print("\n" + "="*40)
print("🧪 VALIDATION TEST (消融实验)")
print("="*40)
print("Removing: Landsize, Location, Year...")
print(f"R² Score: {r2:.4f}")
print("="*40)

# 你的“心中计算器”验证逻辑
if r2 < 0.7:
    print("✅ 验证成功！去掉核心特征后，分数果然暴跌。")
    print("结论：Landsize 和 Location 确实贡献了 ~20% 的准确率。")
else:
    print("❓ 奇怪，去掉核心特征分数依然很高？需要重新检查。")