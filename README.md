Instead of blindly exploring features, I started with domain-driven hypotheses based on urban economics and housing market theory. These hypotheses were then tested using Melbourne housing data through exploratory analysis, modeling, and interpretability techniques. This approach ensures the model reflects real-world mechanisms rather than purely statistical correlations.

# 🏡 Melbourne Housing Market Intelligence Dashboard

![Dashboard Screenshot](output/images/image_b65a40.png)

---

## 🚀 Business Value
This project goes beyond price prediction. It provides **decision support for property investors and analysts** by identifying **undervalued listings** relative to a model-estimated fair value.

Using 13,000+ Melbourne property transactions, I demonstrate that **Land Value Density (price per sqm)** explains market premiums more effectively than raw distance to the CBD, enabling more accurate, location-sensitive valuation and investment screening.

---
## 📊 Interactive Dashboard

An interactive Streamlit dashboard that visualizes:
- Spatial price premium across Melbourne
- North–South divide independent of CBD distance
- Land value density (Price per sqm) as a key signal

![Dashboard](output/images/dashboard_overview.png)


## 📊 Key Insights
* **Market Structure Insight – The "North-South Divide":**  
  The interactive heatmap reveals a distinct price premium in Southern suburbs (e.g. Toorak, South Yarra) that cannot be explained by commute time alone.

* **Model Performance:**  
  Achieved **R² = 0.80 (±0.02)** using a Random Forest model, validated via 5-Fold Cross-Validation.

* **Actionable Intelligence:**  
  The dashboard flags properties where the listing price falls below the model’s predicted fair value, highlighting potential investment opportunities.

---

## 🛠️ Tech Stack
* **Analysis & Modeling:** Python, Pandas, Scikit-Learn (Random Forest)
* **Visualisation:** Streamlit (Interactive Dashboard), Matplotlib
* **Engineering:** Feature Engineering, ETL Pipeline, Cross-Validation

---

## 📂 Project Structure
```text
src/
├─ dashboard.py                # Streamlit interactive dashboard
├─ 1_baseline_model.py         # Baseline regression model
├─ 2_random_forest.py          # Random Forest modeling
├─ 3_feature_engineering.py    # Feature creation & transformation
├─ 4_feature_importance.py     # Model interpretability
├─ 5_validation_test.py        # Initial validation
├─ 6_rigorous_validation.py    # Cross-validation & robustness checks
└─ 7_final_optimization.py     # Final training & evaluation

output/
├─ models/                     # Trained model artifacts (.pkl)
└─ images/                     # Generated figures & visualisations

data/                          # Raw and processed datasets
THOUGHT_PROCESS.md             # Detailed reasoning and experiment log

## 🏃‍♂️ How to Run
# Python 3.9+
pip install -r requirements.txt
streamlit run src/dashboard.py

## 📌 Notes

This project follows an end-to-end analytics workflow, separating source code, data, and generated artifacts.

The step-by-step modeling pipeline (1_baseline → 7_final_optimization) is designed for transparency, reproducibility, and interview discussion.


---