import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. Load Data
# ==========================================
data = pd.read_csv('melb_data.csv')

# ==========================================
# 2. Feature Engineering (特征工程 - 核心步骤)
# ==========================================
print("⚙️ Processing Features...")

# A. 时间特征 (Time)
# 房价随年份波动很大，提取年份很重要
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data['Year'] = data['Date'].dt.year

# B. 房屋状态 (Condition)
# 算出房子几岁了。注意：YearBuilt 有空值，用中位数填补
data['YearBuilt'] = data['YearBuilt'].fillna(data['YearBuilt'].median())
data['House_Age'] = data['Year'] - data['YearBuilt']

# C. 土地价值 (Land Value) - 最关键的特征！
# Landsize 和 BuildingArea 也有很多空值，用中位数填补
data['Landsize'] = data['Landsize'].fillna(data['Landsize'].median())
data['BuildingArea'] = data['BuildingArea'].fillna(data['BuildingArea'].median())

# ==========================================
# 3. Select Enhanced Features
# ==========================================
feature_cols = [
    'Rooms', 'Type', 'Distance',          # 老三样
    'Bedroom2', 'Bathroom', 'Car',        # 更多功能
    'Landsize', 'BuildingArea',           # 土地大小 (关键增量)
    'Year', 'House_Age',                  # 时间维度
    'Lattitude', 'Longtitude'             # 精确坐标 (比 Distance 更准)
]
target_col = 'Price'

X = data[feature_cols]
y = data[target_col]

# One-Hot Encoding (处理 Type)
X = pd.get_dummies(X)

# 再次兜底：防止运算后产生极少数空值
X = X.fillna(X.median())

# ==========================================
# 4. Train Random Forest (更丰富的数据)
# ==========================================
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

print("⏳ Training Advanced Model (with 12+ features)...")
model = RandomForestRegressor(random_state=1, n_estimators=100)
model.fit(train_X, train_y)

# ==========================================
# 5. Evaluate
# ==========================================
val_predictions = model.predict(val_X)
mae = mean_absolute_error(val_y, val_predictions)
r2 = r2_score(val_y, val_predictions)

print("\n" + "="*40)
print("🚀 FINAL MODEL RESULTS (特征工程后)")
print("="*40)
print(f"Features Used: {len(X.columns)} (Increased from 3)")
print(f"Mean Absolute Error (MAE): ${mae:,.0f}")
print(f"R² Score: {r2:.4f}")
print("="*40)
print(f"Comparison: Baseline (0.42) -> RF Simple (0.59) -> This Model ({r2:.2f})")