import pandas as pd
from sklearn.model_selection import train_test_split
# 变化点 1: 引入 Random Forest
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. Load Data
# ==========================================
file_path = 'melb_data.csv' 
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print("❌ Error: melb_data.csv not found.")
    exit()

# ==========================================
# 2. Select Features (Same as Baseline)
# ==========================================
# 保持完全一样的特征，看看更聪明的模型能挖掘出什么
feature_cols = ['Rooms', 'Type', 'Distance']
target_col = 'Price'

baseline_data = data[feature_cols + [target_col]].dropna()

# ==========================================
# 3. Preprocessing (Same as Baseline)
# ==========================================
X = pd.get_dummies(baseline_data[feature_cols])
y = baseline_data[target_col]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# ==========================================
# 4. Train Model (❌ Linear -> ✅ Random Forest)
# ==========================================
print("⏳ Training Random Forest model... (this may take a few seconds)")

# RandomForestRegressor: 
# - random_state=1: 保证每次运行结果一样
# - n_estimators=100: 用 100 棵决策树来投票
model = RandomForestRegressor(random_state=1, n_estimators=100)
model.fit(train_X, train_y)

# ==========================================
# 5. Evaluate
# ==========================================
val_predictions = model.predict(val_X)

mae = mean_absolute_error(val_y, val_predictions)
r2 = r2_score(val_y, val_predictions)

print("\n" + "="*40)
print("🌲 RANDOM FOREST MODEL RESULTS")
print("="*40)
print(f"Mean Absolute Error (MAE): ${mae:,.0f}")
print(f"R² Score: {r2:.4f}")
print("="*40)

# 对比逻辑 (Simple Check)
print("Quick Analysis:")
if r2 > 0.4167:
    print(f"✅ Improvement! The model beat the baseline by {r2 - 0.4167:.4f}")
else:
    print("⚠️ No improvement. We need more data.")