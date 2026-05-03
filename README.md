# 🏠 House Price Prediction

This project builds a machine learning model to predict house prices using the **California Housing dataset**. It follows a complete ML workflow from data analysis to deployment-ready inference.

---

## 📌 Overview

The goal is to predict the **median house value** based on demographic and geographic features.

The project demonstrates:

* End-to-end ML pipeline
* Proper preprocessing using pipelines
* Model comparison with cross-validation
* Hyperparameter tuning
* Final evaluation on unseen data

---

## 📂 Dataset

* **Name:** California Housing Prices
* **Source:** Kaggle
* **Target Variable:** `median_house_value`

### Features:

* Numerical: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income
* Categorical: ocean_proximity

---

## ⚙️ Workflow

### 1. Exploratory Data Analysis (EDA)

* Checked data types and missing values
* Identified missing values in `total_bedrooms`
* Visualized distributions and outliers
* Correlation analysis

### 2. Data Preprocessing

* Missing value imputation (median for numerical, most frequent for categorical)
* One-hot encoding for categorical features
* Feature scaling for numerical variables
* Implemented using `Pipeline` and `ColumnTransformer`

### 3. Baseline Model

* Model: Linear Regression
* Established initial performance benchmark

### 4. Model Selection (Cross-Validation)

Compared multiple models:

* Linear Regression
* Ridge
* Lasso
* Random Forest
* HistGradientBoostingRegressor

✅ Best model: **HistGradientBoostingRegressor**

### 5. Hyperparameter Tuning

* Used `GridSearchCV`
* Optimized parameters like:

  * learning_rate
  * max_depth
  * max_leaf_nodes
  * min_samples_leaf
  * l2_regularization

### 6. Final Evaluation

Evaluated on unseen test data:

| Metric | Score  |
| ------ | ------ |
| RMSE   | 46,530 |
| MAE    | 30,649 |
| R²     | 0.835  |

---

## 📊 Key Insights

* `median_income` is the strongest predictor
* Dataset contains skewed features and outliers
* Target variable is capped and right-skewed
* Tree-based models outperform linear models

---

## 🧠 Model Pipeline

Final pipeline includes:

* Preprocessing (imputation + encoding + scaling)
* HistGradientBoostingRegressor (tuned)

This ensures:

* No data leakage
* Clean and reproducible workflow

---

## 🔮 Inference (Prediction Function)

You can predict house prices using:

```python
predict_house_price(model, longitude, latitude, housing_median_age,
                    total_rooms, total_bedrooms, population,
                    households, median_income, ocean_proximity)
```

### Example:

```python
predict_house_price(
    model=hgb_best,
    longitude=-122.23,
    latitude=37.88,
    housing_median_age=41,
    total_rooms=880,
    total_bedrooms=129,
    population=322,
    households=126,
    median_income=8.3252,
    ocean_proximity="NEAR BAY"
)
```

---

## 🚀 How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the notebook or script

---

## 🛠️ Tech Stack

* Python
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn

---

## 📈 Future Improvements

* Feature engineering (room ratios, population density)
* Log transformation of target
* Try advanced models (XGBoost, LightGBM)
* Deploy as a web app (Streamlit/Flask)

---
