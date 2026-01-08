# 📊 Time Series Forecasting in Machine Learning

> A comprehensive guide to time series analysis and forecasting using machine learning techniques

![Time Series Banner](Images(Notes)/01.webp)

## 🎯 Overview

This repository contains practical notes and guidelines for working with time series data in machine learning. Whether you're forecasting stock prices, predicting sales, or analyzing sensor data, this guide will walk you through the complete workflow from problem definition to model deployment.

**Key Philosophy**: *Time series = Past → Patterns → Stable behavior → Careful forecasting*

---

## 📑 Table of Contents

1. [Problem Understanding](#1-problem-understanding)
2. [Data Loading & Time Index](#2-data-loading--time-index)
3. [Exploratory Visualization](#3-exploratory-visualization)
4. [Baseline Models](#4-baseline-models-mandatory)
5. [Stationarity Check](#5-stationarity-check)
6. [Data Transformation](#6-data-transformation)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Selection](#8-model-selection)
9. [Train-Test Split](#9-train-test-split-time-aware)
10. [Evaluation](#10-evaluation)
11. [Diagnostics & Iteration](#11-diagnostics--iteration)

---

## 📌 STEP-BY-STEP WORKFLOW

### ① Problem Understanding

**Goal**: Know what you are predicting and why

#### Key Decisions:
- **Forecast horizon?** (next day / week / month)
- **Granularity?** (hourly, daily, monthly)
- **Univariate or multivariate?**

#### When to Use:
Always first, before touching data

#### Keywords:
`forecast_horizon`, `frequency`, `target_variable`

#### ✅ Checklist:
- [ ] Define what you're predicting
- [ ] Determine forecast horizon
- [ ] Identify data granularity
- [ ] Decide univariate vs multivariate approach

---

### ② Data Loading & Time Index

**Goal**: Make time explicit and correct

![Data Structure](Images(Notes)/02.png)

#### Key Decisions:
- Is datetime parsed correctly?
- Any missing timestamps?
- Correct frequency?

#### When to Use:
Raw data is loaded

#### Keywords:
`pd.to_datetime`, `set_index`, `sort_index`, `asfreq`

#### Code Example:
```python
import pandas as pd

# Load data with datetime parsing
df = pd.read_csv('data.csv', parse_dates=['date'])

# Set datetime as index
df = df.set_index('date')

# Sort by time
df = df.sort_index()

# Set explicit frequency
df = df.asfreq('D')  # Daily frequency
```

---

### ③ Exploratory Visualization

**Goal**: See patterns with your eyes

#### Key Decisions:
- Is there trend?
- Seasonality present?
- Outliers or sudden breaks?

#### When to Use:
Before preprocessing or modeling

#### Keywords:
`plot`, `trend`, `seasonality`, `outliers`

#### Code Example:
```python
import matplotlib.pyplot as plt

# Basic plot
df['value'].plot(figsize=(15, 6))
plt.title('Time Series Overview')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['value'], model='additive', period=12)
result.plot()
```

---

### ④ Baseline Models (MANDATORY)

**Goal**: Set a minimum performance bar

![Baseline Models](Images(Notes)/03.tif)

#### Key Decisions:
- Last value enough?
- Moving average window size?

#### When to Use:
Before advanced models

#### Keywords:
`naive_forecast`, `rolling_mean`

#### Code Example:
```python
# Naive forecast (last value)
naive_forecast = df['value'].shift(1)

# Moving average
ma_forecast = df['value'].rolling(window=7).mean()

# Seasonal naive
seasonal_naive = df['value'].shift(12)  # for monthly data with yearly seasonality
```

#### ⚠️ Critical Rule:
**Never deploy a model that can't beat the baseline!**

---

### ⑤ Stationarity Check

**Goal**: Decide if transformation is needed

#### Key Decisions:
- Mean changing over time?
- Variance increasing?
- Seasonality present?

#### When to Use:
Using AR / ARIMA family models

#### Keywords:
`stationarity`, `ADF_test`, `rolling_mean`, `unit_root`

#### Code Example:
```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries):
    # Rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # Plot
    plt.plot(timeseries, label='Original')
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.plot(rolling_std, label='Rolling Std')
    plt.legend()
    plt.show()
    
    # ADF Test
    result = adfuller(timeseries.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    
    if result[1] <= 0.05:
        print("Data is stationary")
    else:
        print("Data is non-stationary")

check_stationarity(df['value'])
```

---

### ⑥ Data Transformation

**Goal**: Stabilize patterns

#### Key Decisions:
- Need differencing?
- Seasonal differencing?
- Log transform?

#### When to Use:
Series is non-stationary

#### Keywords:
`differencing`, `seasonal_diff`, `log_transform`

#### Code Example:
```python
import numpy as np

# Log transform (for variance stabilization)
df['log_value'] = np.log(df['value'])

# First differencing (for trend removal)
df['diff_value'] = df['value'].diff()

# Seasonal differencing
df['seasonal_diff'] = df['value'].diff(12)  # 12 for monthly data

# Combined
df['combined'] = df['value'].diff().diff(12)
```

---

### ⑦ Feature Engineering

**Goal**: Inject past information

![Feature Engineering](Images(Notes)/04.png)

#### Key Decisions:
- How many lags to include?
- Which rolling statistics?
- Calendar features needed?

#### When to Use:
ML models or regression-based approaches

#### Keywords:
`lag_features`, `rolling_window`, `time_features`

#### Code Example:
```python
# Lag features
for i in range(1, 8):
    df[f'lag_{i}'] = df['value'].shift(i)

# Rolling features
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
df['rolling_std_7'] = df['value'].rolling(window=7).std()
df['rolling_min_7'] = df['value'].rolling(window=7).min()
df['rolling_max_7'] = df['value'].rolling(window=7).max()

# Time-based features
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['day_of_month'] = df.index.day
df['week_of_year'] = df.index.isocalendar().week

# Cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
```

---

### ⑧ Model Selection

**Goal**: Match model to data nature

#### Key Decisions:
- Small data → ARIMA
- Complex patterns → ML (XGBoost, Random Forest)
- Long sequences → LSTM (Deep Learning)

#### When to Use:
After preprocessing

#### Keywords:
`ARIMA`, `SARIMA`, `XGBoost`, `LSTM`

#### Model Comparison:

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **ARIMA** | Linear trends, small data | Interpretable, fast | Assumes linearity |
| **SARIMA** | Seasonal patterns | Handles seasonality | Complex parameter tuning |
| **Prophet** | Multiple seasonality | Robust to missing data | Less customizable |
| **XGBoost** | Non-linear patterns | High accuracy | Requires feature engineering |
| **LSTM** | Long sequences | Captures long-term dependencies | Needs lots of data |

#### Code Examples:

```python
# ARIMA
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)

# SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit()

# XGBoost
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=1000, learning_rate=0.01)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

### ⑨ Train–Test Split (TIME-AWARE)

**Goal**: Avoid future leakage

#### Key Decisions:
- Where to cut timeline?
- Single split or walk-forward validation?

#### When to Use:
Training any model

#### Keywords:
`chronological_split`, `walk_forward`

#### ⚠️ Critical Rules:
- **NEVER** shuffle time series data
- **ALWAYS** split chronologically
- Test set must come after training set

#### Code Example:
```python
# Simple chronological split
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

# Time-based split
split_date = '2023-01-01'
train = df[df.index < split_date]
test = df[df.index >= split_date]

# Walk-forward validation
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(df):
    train, test = df.iloc[train_index], df.iloc[test_index]
    # Train and evaluate model
```

---

### ⑩ Evaluation

**Goal**: Measure usefulness, not just accuracy

#### Key Decisions:
- MAE vs RMSE?
- Visual fit good?
- Residuals random?

#### When to Use:
Comparing models

#### Keywords:
`MAE`, `RMSE`, `MAPE`, `residual_analysis`

#### Metrics:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# MAE (Mean Absolute Error) - easy to interpret
mae = mean_absolute_error(y_true, y_pred)

# RMSE (Root Mean Squared Error) - penalizes large errors
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAPE (Mean Absolute Percentage Error) - scale-independent
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# R² Score
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.2f}")
```

#### Visual Evaluation:
```python
plt.figure(figsize=(15, 6))
plt.plot(test.index, y_true, label='Actual', marker='o')
plt.plot(test.index, y_pred, label='Predicted', marker='x')
plt.legend()
plt.title('Actual vs Predicted')
plt.show()
```

---

### ⑪ Diagnostics & Iteration

**Goal**: Improve model understanding

#### Key Decisions:
- Residual autocorrelation present?
- Missed seasonality?
- Model overfitting?

#### When to Use:
Model underperforms or before final deployment

#### Keywords:
`ACF`, `PACF`, `white_noise`, `model_refinement`

#### Code Example:
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Calculate residuals
residuals = y_true - y_pred

# Plot residuals
plt.figure(figsize=(15, 4))
plt.subplot(131)
plt.plot(residuals)
plt.title('Residuals Over Time')

plt.subplot(132)
plt.hist(residuals, bins=30)
plt.title('Residual Distribution')

plt.subplot(133)
plot_acf(residuals, lags=40)
plt.title('ACF of Residuals')
plt.tight_layout()
plt.show()

# Statistical tests
from scipy import stats

# Normality test
statistic, p_value = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test: p-value = {p_value:.4f}")

# Ljung-Box test (for autocorrelation)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_test)
```

#### Good Residuals Should:
- ✅ Have zero mean
- ✅ Be normally distributed
- ✅ Show no autocorrelation
- ✅ Have constant variance

---

## 🧠 ONE-LINE MEMORY RULE

> **Time series = Past → Patterns → Stable behavior → Careful forecasting**

### Critical Principles:

1. **If behavior isn't stable** → Transform it
2. **If model can't beat baseline** → Reject it
3. **Never use future information** → Time-aware splits only
4. **Residuals should be white noise** → Otherwise, model is incomplete

---

## 🛠️ Essential Libraries

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Deep Learning (optional)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
```

---

## 📚 Common Pitfalls to Avoid

### ❌ Don't:
- Shuffle time series data for train-test split
- Ignore baseline models
- Use future information in features
- Apply cross-validation without time awareness
- Forget to check stationarity for ARIMA models
- Scale before splitting (leads to data leakage)

### ✅ Do:
- Always start with baseline models
- Use chronological splits
- Check for stationarity
- Visualize your data and predictions
- Analyze residuals
- Document your assumptions

---

## 📊 Quick Reference: Model Selection Flowchart

```
Start
  │
  ├─ Is data stationary?
  │   ├─ Yes → Try ARIMA/AR models
  │   └─ No → Apply transformations → Try SARIMA
  │
  ├─ Is seasonality present?
  │   ├─ Yes → SARIMA, Prophet, or seasonal features
  │   └─ No → ARIMA, Moving Average
  │
  ├─ Is relationship linear?
  │   ├─ Yes → Statistical models (ARIMA)
  │   └─ No → ML models (XGBoost, Random Forest)
  │
  ├─ Do you have lots of data?
  │   ├─ Yes → Try Deep Learning (LSTM, Transformer)
  │   └─ No → Stick with statistical/simple ML
  │
  └─ Compare with baseline → Deploy best model
```

---

## 🎓 Learning Resources

- **Books**: 
  - "Forecasting: Principles and Practice" by Rob Hyndman
  - "Introduction to Time Series and Forecasting" by Brockwell & Davis
  
- **Online Courses**: 
  - Coursera: Practical Time Series Analysis
  - Fast.ai: Practical Deep Learning
  
- **Documentation**:
  - [Statsmodels](https://www.statsmodels.org/)
  - [Prophet](https://facebook.github.io/prophet/)
  - [XGBoost](https://xgboost.readthedocs.io/)

---

## 🤝 Contributing

Feel free to contribute by:
- Adding more examples
- Improving explanations
- Fixing errors
- Suggesting new techniques

---

## 📝 License

This repository is for educational purposes. Feel free to use and modify as needed.

---

## 📧 Contact

For questions or suggestions, please open an issue in this repository.

---

**Happy Forecasting! 📈**

*Remember: The best model is the one that beats the baseline and makes sense for your specific problem.*
