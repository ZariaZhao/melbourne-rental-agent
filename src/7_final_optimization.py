import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # 用于保存模型
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# 1. 准备数据
df = pd.read_csv('melb_data.csv')

# 特征工程 (保持不变)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year
df['YearBuilt'] = pd.to_numeric(df['YearBuilt'], errors='coerce').fillna(df['YearBuilt'].median())
df['House_Age'] = df['Year'] - df['YearBuilt']
df['Landsize'] = df['Landsize'].fillna(df['Landsize'].median())
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].median())

# ==========================================
# ✂️ 瘦身时刻 (Feature Selection)
# ==========================================
# 之前的全量特征：
# features_full = ['Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Year', 'House_Age', 'Lattitude', 'Longtitude']

# ✅ 精简后的“特种部队” (只留 Top 8)：
features_slim = [
    'Lattitude', 'Longtitude',  # 核心地段
    'Rooms',                    # 核心大小
    'Distance',                 # 核心位置
    'Landsize', 'BuildingArea', # 核心资产价值
    'Type',                     # 核心房型 (Pipeline 会自动转 One-Hot，生成 Type_u 等)
    'Bathroom',                 # 核心配置 (富人区通常厕所多)
    'House_Age'                 # 核心折旧
]

# ❌ 删掉了：Bedroom2, Car, Year (注意：Type 会自动处理，不用手动删 Type_h)

X = df[features_slim]
y = df['Price']

# 2. Pipeline (保持不变)
numeric_cols = [c for c in X.columns if c != 'Type']
categorical_cols = ['Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=1))
])

# 3. 验证瘦身结果
cv = KFold(n_splits=5, shuffle=True, random_state=1)
print(f"⏳ Testing Slim Model with {len(features_slim)} features...")

scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')

print("\n" + "="*40)
print("🚀 FINAL SLIM MODEL RESULTS")
# ... (上面的代码保持不变)

print("\n" + "="*40)
print("🚀 FINAL SLIM MODEL RESULTS")
print("="*40)
print(f"Previous Full Model R²: 0.8044")
print(f"Current Slim Model R² : {scores.mean():.4f} (± {scores.std():.4f})")
print("="*40)

# ==========================================
# 🎨 1. 画图：预测值 vs 真实值 (Fixing NameError)
# ==========================================
print("🎨 Generating Prediction vs Actual plot...")

# 关键修复点：这里计算了 y_pred，你的报错就是因为缺了这一行！
y_pred = cross_val_predict(pipeline, X, y, cv=cv)

plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred, alpha=0.3, color='blue')
# 画一条红色的完美对角线
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Price (真实价格)')
plt.ylabel('Predicted Price (预测价格)')
plt.title('Truth vs. Prediction')
# 保存图片
plt.savefig('prediction_scatter.png')
print("✅ Plot saved as 'prediction_scatter.png'")
# plt.show() # 如果不想弹窗，就保持注释状态

# ==========================================
# 📦 2. 保存模型 (Saving Model)
# ==========================================
if scores.mean() >= 0.795:
    print("\n📦 Performance is good. Retraining on 100% data...")
    
    # 用全部数据重新训练
    pipeline.fit(X, y)
    
    # 保存文件
    model_filename = 'melbourne_housing_model.pkl'
    joblib.dump(pipeline, model_filename)
    
    print(f"✅ Model saved successfully as: {model_filename}")
else:
    print("❌ Performance not good enough. Model not saved.")