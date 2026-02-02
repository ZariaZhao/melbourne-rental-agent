import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# =========================
# 1. 准备数据 (Standard Setup)
# =========================
df = pd.read_csv('melb_data.csv')

# 特征工程复现
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year
df['YearBuilt'] = pd.to_numeric(df['YearBuilt'], errors='coerce').fillna(df['YearBuilt'].median())
df['House_Age'] = df['Year'] - df['YearBuilt']
df['Landsize'] = df['Landsize'].fillna(df['Landsize'].median())
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].median())

# 你的所有特征
features = [
    'Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 'Car',
    'Landsize', 'BuildingArea', 'Year', 'House_Age',
    'Lattitude', 'Longtitude'
]
X = df[features]
y = df['Price']

# =========================
# 2. 构建 Pipeline (必须用 Pipeline 防止数据泄露)
# =========================
# 只有在 Cross Validation 内部做填充(Impute)，才是真正严谨的
numeric_cols = [c for c in X.columns if c != 'Type']
categorical_cols = ['Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 模型管道
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=1))
])

# =========================
# 3. 核心：5-Fold Cross Validation
# =========================
# KFold: 把数据分成 5 份，不乱序 (shuffle=True 打乱顺序，但在时间序列中通常要小心)
# 这里我们要 shuffle，因为数据不是按时间严格排序的
cv = KFold(n_splits=5, shuffle=True, random_state=1)

print("⏳ Running 5-Fold Cross Validation (这可能需要一分钟)...")

# scoring='r2' 计算 R方
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')

# 同时计算 MAE (负数是因为 sklearn 的逻辑是 "得分越高越好"，MAE是越低越好，所以是负的)
mae_scores = -cross_val_score(pipeline, X, y, cv=cv, scoring='neg_mean_absolute_error')

# =========================
# 4. 最终审判 (Final Verdict)
# =========================
print("\n" + "="*40)
print("⚖️ 黄金标准验证结果 (Cross Validation)")
print("="*40)
print(f"5次运行的 R² 分数: {scores}")
print(f"平均 R² Score: {scores.mean():.4f} (± {scores.std():.4f})")
print(f"平均 MAE: ${mae_scores.mean():,.0f}")
print("="*40)

# 解读
if scores.mean() > 0.80 and scores.std() < 0.05:
    print("✅ 结论：0.83 是靠谱的！模型非常稳定。")
elif scores.mean() < 0.70:
    print("❌ 结论：之前的 0.83 是运气好（过拟合），真实水平只有不到 0.7。")
else:
    print("⚠️ 结论：还行，但有波动，需要小心。")