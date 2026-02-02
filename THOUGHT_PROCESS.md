# 🧠 Project Thinking Log: Melbourne Housing Market Analysis

> **Role:** Data Scientist / Machine Learning Engineer
> **Objective:** Predict housing prices with high interpretability and stability.
> **Final Result:** R² Score **0.8028** (± 0.0217) on unseen data.

---

## 1. Problem Definition & Hypothesis
**Goal:** The Melbourne housing market is complex. Renters and investors need a data-driven way to identify "undervalued" properties, moving beyond simple intuition.

**Initial Hypotheses:**
1.  **Location is non-linear:** Distance to CBD matters, but "North vs. South" (e.g., Toorak vs. Footscray) matters more.
2.  **Land Value:** In Australia, "Land appreciates, buildings depreciate." Landsize should be a key driver.
3.  **Seasonality:** Prices fluctuate by year (2016-2018 market cycle).

---

## 2. Iterative Experiments (The "MLE" Workflow)

### Phase 1: The Baseline (Establishing a Benchmark)
* **Model:** Linear Regression
* **Features:** `Rooms`, `Type`, `Distance` (The "Naive" set).
* **Result:** R² = **0.42**.
* **Insight:** The model is underfitting. Linear regression cannot capture the complex, non-linear pricing rules of Melbourne suburbs.

### Phase 2: Model Selection (The Algorithm Shift)
* **Action:** Switched to **Random Forest Regressor** (Tree-based).
* **Rationale:** Trees can capture non-linear relationships and interactions (e.g., "If Suburb=Toorak AND Landsize>500, Price doubles").
* **Result:** R² improved to **0.59** (with same data).

### Phase 3: Feature Engineering (The "Information Gain")
* **Action:**
    1.  **Time:** Extracted `Year` from `Date` to capture market cycles.
    2.  **Asset Value:** Added `Landsize` and `BuildingArea` (imputed missing values with Median).
    3.  **Micro-Location:** Added `Lattitude` & `Longtitude` to capture specific neighborhood premiums.
* **Result:** R² jumped to **0.83**.
* **Key Discovery:** `Lattitude` became the #1 most important feature, validating the "Yarra River Divide" hypothesis (South is more expensive).

### Phase 4: Optimization & Slimming (Engineering Efficiency)
* **Action (Feature Selection):**
    * Analyzed Feature Importance Plot.
    * **Removed:** `Bedroom2` (Redundant with `Rooms`), `Car` (Low impact), `Type_h` (Redundant), `Year` (Low impact).
* **Result:**
    * **Full Model:** R² 0.8044 (± 0.0229)
    * **Slim Model:** R² 0.8028 (± 0.0217)
* **Conclusion:** The Slim Model retained **99.8%** of accuracy while reducing complexity by 30%. It is more stable (lower variance) and production-ready.

---

## 3. Business Value & Interpretability
**What drives prices in Melbourne?**
Based on the final model, the top 3 drivers are:
1.  **Lattitude (Precise Location):** Specific coordinates matter more than raw distance.
2.  **Rooms (Capacity):** The functional utility of the house.
3.  **Distance (Commute):** Proximity to the city center.

**Commercial Application:**
This model can now be deployed as an API or Web App to provide real-time valuation estimates for users inputting just 8 key fields.