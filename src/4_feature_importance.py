import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. 准备数据
# ==========================================
# 读取数据
data = pd.read_csv('melb_data.csv')

# --- 特征工程 (保持一致) ---
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data['Year'] = data['Date'].dt.year
data['YearBuilt'] = data['YearBuilt'].fillna(data['YearBuilt'].median())
data['House_Age'] = data['Year'] - data['YearBuilt']
data['Landsize'] = data['Landsize'].fillna(data['Landsize'].median())
data['BuildingArea'] = data['BuildingArea'].fillna(data['BuildingArea'].median())

feature_cols = [
    'Rooms', 'Type', 'Distance', 'Bedroom2', 'Bathroom', 'Car',
    'Landsize', 'BuildingArea', 'Year', 'House_Age',
    'Lattitude', 'Longtitude'
]

# ==========================================
# 🛠️ 修复点在这里 (FIXED HERE)
# ==========================================
# 1. 先把分类变量 (Type) 变成数字 (One-Hot)
X_temp = pd.get_dummies(data[feature_cols])

# 2. 填充空值时，告诉它只计算数字列的中位数 (numeric_only=True)
# 这样它就会自动跳过 'Type' 这种文字列，不会报错了
X = X_temp.fillna(X_temp.median(numeric_only=True))

y = data['Price']

# ==========================================
# 2. 训练模型
# ==========================================
print("⏳ 正在重新训练模型并计算特征重要性...")
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = RandomForestRegressor(random_state=1, n_estimators=100)
model.fit(train_X, train_y)

# ==========================================
# 3. 核心：提取特征重要性
# ==========================================
importances = model.feature_importances_
feature_names = train_X.columns

# 整理表格
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 打印前 10 名
print("\n" + "="*40)
print("🏆 TOP 10 核心定价因素 (Feature Importance)")
print("="*40)
print(feature_df.head(10))

# ==========================================
# 4. 可视化
# ==========================================
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')

plt.title('What Drives Melbourne House Prices? (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()

# ==========================================
# 💾 保存图片 (Save Figure)
# ==========================================
# dpi=300 表示高清格式，bbox_inches='tight' 保证边缘不被切掉
filename = 'feature_importance.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')

print(f"\n✅ 图片已成功保存为: {filename}")
print("你可以去左侧文件列表里点开它看看！")

# 如果你还想弹窗看，这行可以留着，不想看就注释掉
# plt.show()