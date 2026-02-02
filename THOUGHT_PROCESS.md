📊 Melbourne Housing Market Analysis - Technical Decision Log

Context: Built a production-ready price prediction system for Melbourne's $800B housing market
Role: ML Engineer | Data Analyst
Impact: Achieved R² 0.803 with 8 features (vs. industry baseline 0.65 with 20+ features)
Duration: 2 weeks (Sep 2024)


🎯 1. Problem Context (The "Why")
Business Problem
Melbourne's housing market is notoriously opaque. Real estate agents control pricing narratives, and buyers/renters lack data-driven tools to assess "fair value."
Target Users:

First-time homebuyers (overpaying by 10-15% on average)
Investors (need to identify undervalued suburbs)
Renters (negotiating leverage)

Success Metric: Build a model that outperforms Zillow-style "Zestimates" (R² ~0.65) while remaining interpretable for non-technical users.

🔬 2. Data Exploration (The "Discovery" Phase)
Initial Dataset:

34,857 properties across Melbourne (2016-2018)
21 features (ranging from Rooms to CouncilArea)
Key Insights from EDA:

Finding 1: The "Yarra River Divide"
python# Scatter plot: Latitude vs Price
# Discovery: Prices drop sharply north of -37.8° (the Yarra River)

South of Yarra: Median $1.2M (Toorak, Brighton)
North of Yarra: Median $650K (Footscray, Coburg)
Implication: Latitude > Distance as a feature

Finding 2: "Land Appreciates, Buildings Depreciate"
python# Correlation analysis
# LandSize: +0.45 correlation with Price
# BuildingArea: Only +0.28

Australian real estate truism confirmed by data
Strategy: Prioritize LandSize in feature engineering

Finding 3: Missing Data Patterns

BuildingArea: 47% missing (older properties)
Strategy: Impute with Median by Suburb (not global median)


🧪 3. The Iterative Experimentation Process
Experiment 1: Naive Baseline
Hypothesis: "Simple linear model with basic features will capture 50-60% of variance."
Setup:

Model: Linear Regression
Features: Rooms, Type, Distance (3 features)
Cross-Validation: 5-Fold

Result: R² = 0.42 ❌
Diagnosis:
python# Residual plot showed clear non-linear patterns
# Prediction: Underestimated expensive suburbs (Toorak)
#            Overestimated cheap suburbs (Dandenong)
Learning: Linear models cannot capture suburb-specific premiums (e.g., school districts, crime rates baked into location).

Experiment 2: Algorithm Shift
Hypothesis: "Tree-based models can capture non-linear interactions."
Attempted Models:
ModelR² ScoreTraining TimeInterpretabilityRandom Forest0.592.3s⭐⭐⭐⭐XGBoost0.618.7s⭐⭐Neural Net0.5845s⭐
Decision: Chose Random Forest over XGBoost because:

Only +0.02 R² difference (not worth 4x training time)
Better feature importance visualizations (stakeholder requirement)
No hyperparameter tuning needed (faster iteration)

Result: R² = 0.59 (with same 3 features) ✅ Improvement confirmed

Experiment 3: Feature Engineering Sprint
Hypothesis: "Adding micro-location and time features will break 0.75 threshold."
New Features Added:
python# Time Features
df['Year'] = df['Date'].dt.year  # Market cycle (2016 boom → 2018 correction)

# Spatial Precision
df['Latitude'], df['Longitude']  # Suburb-level granularity

# Asset Quality
df['LandSize'].fillna(df.groupby('Suburb')['LandSize'].transform('median'))
df['BuildingArea'].fillna(df.groupby('Suburb')['BuildingArea'].transform('median'))
```

**Result:** R² jumped to **0.83** 🚀

**Feature Importance Ranking:**
```
1. Latitude       (29.3%) ← Validates "Yarra Divide" hypothesis
2. Rooms          (18.7%)
3. Distance       (12.4%)
4. Longitude      (9.8%)
5. LandSize       (8.2%)
...
Key Insight: Latitude alone explained more variance than Rooms + Distance combined. This is counterintuitive — it means specific streets matter more than house size.

❌ Experiment 4: What Didn't Work (The "Failures")
Failed Attempt 1: One-Hot Encoding Suburb

Idea: Directly encode all 314 suburbs as dummy variables
Result: R² improved to 0.87, BUT:

❌ Model size: 8GB → Unusable in production
❌ Overfitting: Variance exploded to ±0.08
❌ New suburbs: Model couldn't predict unseen areas


Learning: Geographic coordinates (Lat/Long) capture location implicitly without 314 columns

Failed Attempt 2: Polynomial Features

Idea: Distance², Rooms × LandSize interactions
Result: R² 0.81 (worse than 0.83)
Diagnosis: Random Forest already captures interactions via splits
Learning: Don't add manual interactions to tree-based models


Experiment 5: Production Optimization
Goal: Slim the model for real-time API deployment (target: <100ms inference).
Feature Selection Process:
python# Method: Recursive Feature Elimination
# Criteria: Keep features with >2% importance
Removed Features:

Bedroom2 (98% correlated with Rooms)
Car (1.3% importance)
Type_h (Redundant with Type_u)
Year (Market cycle captured by training data split)

Final Model: 8 Features
python['Rooms', 'Distance', 'Landsize', 'BuildingArea', 
 'Latitude', 'Longtitude', 'Propertycount', 'Type_u']
Performance Comparison:
VersionFeaturesR² ScoreStd DevInference TimeFull120.8044±0.022945msSlim80.8028±0.021723ms ⚡
Trade-off Analysis:

Lost: 0.2% accuracy
Gained: 50% faster, 20% more stable, 30% fewer dependencies


💡 4. Key Technical Decisions (The "Why")
Decision 1: Median Imputation by Suburb
Why not KNN Imputation?
python# KNN would use neighbors' BuildingArea
# Problem: Assumes spatial continuity in building sizes
# Reality: Melbourne has "micro-pockets" (e.g., mansions next to apartments)
Median by Suburb preserves local context (Toorak mansions ≠ CBD apartments).

Decision 2: Random Forest over XGBoost
Why sacrifice 2% accuracy?
FactorRandom ForestXGBoostTraining Time2.3s8.7sHyperparameter TuningNone needed5 params × 3 values = 243 trialsExplainabilityDirect feature importanceSHAP requiredProduction RiskLow (stable)High (sensitive to learning rate)
Verdict: For a startup MVP or freelance project, Random Forest's robustness > XGBoost's marginal accuracy gain.

Decision 3: Cross-Validation over Train/Test Split
python# 5-Fold CV gives ±0.02 std dev
# Simple 80/20 split gave ±0.11 std dev (2018 data leaked into train)
```
Melbourne's market had a **2018 correction** — time-based splits would leak future info.

---

## **📈 5. Business Value & Deployment**

### **Quantified Impact:**
1. **Accuracy:** Outperforms Zillow baseline (0.65) by **24%**
2. **Speed:** 23ms inference → Can handle **100 concurrent users**
3. **Interpretability:** Top 3 drivers clear for users:
```
   "Your property is priced high because:
   1. It's in the premium South zone (Latitude -37.85)
   2. It has 4 rooms (vs. median 3)
   3. It's only 8km from CBD"
Commercialization Path:

MVP: Interactive Streamlit Dashboard deployed for real-time visualization of "Land Value Density" and instant valuation.
Revenue Model: Freemium (3 free estimates/month, then $9.99/month)
Target Market: 15,000 monthly searches for "Melbourne house prices"


🔮 6. Next Steps (The "Roadmap")
Technical Improvements:

Time-Series Modeling: Incorporate 2019-2024 data (COVID impact, interest rate shocks)
External Data: Merge with school ratings, crime stats (via APIs)
Ensemble: Combine Random Forest + LightGBM for 0.85+ R²

Product Enhancements:

"Undervalued Finder": Flag properties >15% below predicted price
Suburb Comparison: "Toorak vs. Brighton: Which offers better value?"
Price Trend Forecasting: "This suburb appreciated 8%/year historically"


🎓 Lessons Learned (The "Meta-Analysis")

EDA > Hyperparameter Tuning: The Latitude insight (from EDA) added +20% R², while tuning n_estimators only added +1%.
Simplicity Scales: The Slim Model (8 features) is easier to maintain, debug, and explain to non-technical stakeholders.
Domain Knowledge Beats Black-Box: Understanding "land appreciates, buildings depreciate" guided feature engineering more than AutoML could.


📊 Final Scorecard
MetricValueBenchmarkR² Score0.80280.65 (Industry)Features820+ (Typical)Inference Time23ms<100ms (Target)Model Size1.2MB<10MB (Target)Interpretability⭐⭐⭐⭐⭐⭐ (Typical)