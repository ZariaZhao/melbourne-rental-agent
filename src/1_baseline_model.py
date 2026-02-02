import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. Load Data
# ==========================================
# Best Practice: Use relative path. 
# As long as the .csv is in the same folder as this script, this works everywhere.
file_path = 'melb_data.csv' 

try:
    data = pd.read_csv(file_path)
    print(f"✅ Dataset Loaded Successfully. Shape: {data.shape}")
except FileNotFoundError:
    print("❌ Error: 'melb_data.csv' not found. Make sure it's in the same folder as this script.")
    exit()

# ==========================================
# 2. Select Baseline Features
# ==========================================
# Theory: Price is driven by Location (Distance), Land (Type), and Utility (Rooms).
feature_cols = ['Rooms', 'Type', 'Distance']
target_col = 'Price'

# Filter data & Drop missing values
baseline_data = data[feature_cols + [target_col]].dropna()

print("\n--- Baseline Data Preview ---")
print(baseline_data.head())

# ==========================================
# 3. Preprocessing (One-Hot Encoding)
# ==========================================
# Convert 'Type' (h/u/t) into numerical columns
X = pd.get_dummies(baseline_data[feature_cols])
y = baseline_data[target_col]

print("\n--- Features after Encoding (X) ---")
print(X.head())

# ==========================================
# 4. Train Model
# ==========================================
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = LinearRegression()
model.fit(train_X, train_y)

# ==========================================
# 5. Evaluate
# ==========================================
val_predictions = model.predict(val_X)

mae = mean_absolute_error(val_y, val_predictions)
r2 = r2_score(val_y, val_predictions)

print("\n==========================================")
print("🎯 BASELINE MODEL RESULTS")
print("==========================================")
print(f"Mean Absolute Error (MAE): ${mae:,.0f}")
print(f"R² Score: {r2:.4f}")
print("==========================================")
print(f"Interpretation: With just 3 features, we explain {r2*100:.1f}% of the price variation.")