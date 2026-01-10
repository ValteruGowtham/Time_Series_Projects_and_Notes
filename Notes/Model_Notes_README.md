# 📊 Time Series Forecasting Models - Complete Guide

> *A comprehensive guide to time series forecasting models, from basic baselines to advanced deep learning approaches*

---

## 📋 Table of Contents

1. [Baseline Models](#1️⃣-baseline-models)
2. [Moving Averages & Exponential Smoothing](#2️⃣-moving-averages--exponential-smoothing)
3. [ARIMA Family](#3️⃣-arima-family)
4. [Machine Learning Approaches](#4️⃣-machine-learning-for-time-series)
5. [Prophet](#5️⃣-prophet)
6. [Model Selection Framework](#6️⃣-model-selection-framework)
7. [Advanced Techniques](#7️⃣-advanced-techniques)

---

## 🎯 Core Philosophy

**The Golden Rule:** *Model complexity must match data complexity.*

> A clean SARIMA often beats a sloppy LSTM. Always start simple and increase complexity only when necessary.

![Time Series Model Overview](Images(Notes)/model-01.tif)

---

## 1️⃣ Baseline Models (ALWAYS START HERE)

### 🤔 Why Baselines Exist

Before building anything complex, you must answer:

**"Is my model actually useful?"**

- Baselines give you a minimum performance bar
- If a complex model can't beat a baseline → reject it
- **Critical for preventing wasted effort on over-engineered solutions**

### 📌 Naive Forecast (Last Value)

**Intuition:** Tomorrow will be the same as today.

**Mathematical Formula:**

$$\hat{y}(t+h) = y(t)$$

**When It Works Surprisingly Well:**
- Random walk-like data
- Stock prices
- Economic indicators
- Short-term forecasts

**Why You MUST Use It:**
- ✅ Shockingly strong baseline
- ✅ Many ML models fail to beat it
- ✅ Fast to compute
- ✅ Zero parameters to tune

**🔑 Keywords:** `random_walk`, `benchmark`, `last_value`

---

### 📌 Seasonal Naive Forecast

**Intuition:** Tomorrow will be the same as the same day last season.

**Mathematical Formula:**

$$\hat{y}(t+h) = y(t-m+h)$$

where $m$ = seasonal period (e.g., 7 for weekly, 12 for monthly, 365 for yearly)

**Use Cases:**
- ✅ Strong repeating seasonality
- ✅ Retail sales (weekly/monthly patterns)
- ✅ Electricity demand (daily/weekly cycles)
- ✅ Temperature forecasting

**🔑 Keywords:** `seasonality`, `seasonal_period`, `repeat_pattern`

---

## 2️⃣ Moving Averages & Exponential Smoothing

> **Philosophy:** "Smooth the past to predict the future"

### 📈 Simple Moving Average (SMA)

**Intuition:** Predict using the average of last $k$ values.

**Mathematical Formula:**

$$\hat{y}(t+1) = \frac{1}{k}\sum_{i=0}^{k-1}y(t-i)$$

**Problems:**
- ❌ All past values treated equally
- ❌ Abrupt changes when window shifts
- ❌ Lags behind trend changes

**Use When:**
- Noise reduction needed
- Very simple smoothing
- Quick and dirty analysis

**🔑 Keywords:** `window_size`, `equal_weights`, `smoothing`

---

### 📈 Simple Exponential Smoothing (SES)

**Big upgrade over SMA!**

**Intuition:** Recent observations matter more than old ones.

**Mathematical Formulas:**

$$\hat{y}(t+1) = \alpha y(t) + (1-\alpha)\hat{y}(t)$$

**Alternative form (error correction):**

$$\hat{y}(t+1) = \hat{y}(t) + \alpha e(t)$$

**What $\alpha$ Controls:**
- Small $\alpha$ (0.1-0.3) → smoother, slow response, less reactive
- Large $\alpha$ (0.7-0.9) → fast response to changes, more noise

**When to Use:**
- ✅ Stationary data
- ✅ No trend
- ✅ No seasonality
- ✅ Need automatic weight optimization

**🔑 Keywords:** `exponential_decay`, `alpha`, `stationary_series`

---

### 📈 Holt's Linear Trend (Double Exponential Smoothing)

**Problem SES Can't Solve:** Trends

**Intuition:** Track level and trend separately.

**What It Models:**
- $\ell(t)$: current level
- $b(t)$: trend (slope)

**Forecast Equation:**

$$\hat{y}(t+h) = \ell(t) + h \cdot b(t)$$

**Update Equations:**
- Level: $\ell(t) = \alpha y(t) + (1-\alpha)[\ell(t-1) + b(t-1)]$
- Trend: $b(t) = \beta[\ell(t) - \ell(t-1)] + (1-\beta)b(t-1)$

**Use When:**
- ✅ Clear upward or downward trend
- ✅ No seasonality
- ✅ Linear growth patterns

**🔑 Keywords:** `level`, `trend`, `linear_growth`

---

### 📈 Holt-Winters (Triple Exponential Smoothing)

**The Full Smoothing Model**

**Intuition:** Model level + trend + seasonality simultaneously.

**Components:**
1. **Level** ($\ell$)
2. **Trend** ($b$)
3. **Seasonality** ($s$)

**Two Variants:**

| Type | Formula | When to Use |
|------|---------|-------------|
| **Additive** | $\hat{y}(t+h) = \ell(t) + hb(t) + s(t+h-m)$ | Seasonal effect constant over time |
| **Multiplicative** | $\hat{y}(t+h) = [\ell(t) + hb(t)] \times s(t+h-m)$ | Seasonal effect grows with level |

**Use When:**
- ✅ Trend + seasonality (MOST business data)
- ✅ Monthly revenue
- ✅ Website traffic
- ✅ Demand forecasting

**🔑 Keywords:** `trend + seasonality`, `additive`, `multiplicative`

---

## 3️⃣ ARIMA Family — The Statistical Backbone

> **Critical Requirement:** These models require stationarity (achieved through differencing)

![ARIMA Model Structure](Images(Notes)/model-02.png)

### 📊 AR (AutoRegressive) Model

**Intuition:** Present depends on past values.

**Mathematical Formula:**

$$y(t) = c + \sum_{i=1}^{p}\phi_i y(t-i) + \varepsilon(t)$$

**Parameters:**
- $p$: number of lag observations (AR order)
- $\phi_i$: coefficients
- $c$: constant term
- $\varepsilon(t)$: white noise error

**Use When:**
- ✅ Momentum exists
- ✅ Past values strongly influence future
- ✅ PACF shows significant lags

**Identification:** Use **PACF** (Partial Autocorrelation Function)

**🔑 Keywords:** `lags`, `momentum`, `PACF`

---

### 📊 MA (Moving Average) Model

**Important:** NOT the same as Simple Moving Average for smoothing!

**Intuition:** Current value depends on past forecast errors (shocks).

**Mathematical Formula:**

$$y(t) = \mu + \varepsilon(t) + \sum_{i=1}^{q}\theta_i \varepsilon(t-i)$$

**Parameters:**
- $q$: number of lagged forecast errors (MA order)
- $\theta_i$: coefficients
- $\varepsilon(t)$: white noise error terms

**Use When:**
- ✅ Shocks linger (promotions, disruptions)
- ✅ One-time events affect multiple periods
- ✅ ACF shows significant lags

**Identification:** Use **ACF** (Autocorrelation Function)

**🔑 Keywords:** `error_terms`, `shocks`, `ACF`

---

### 📊 ARMA(p, q) Model

**Intuition:** Combine AR + MA for stationary series.

**Formula:**

$$y(t) = c + \sum_{i=1}^{p}\phi_i y(t-i) + \varepsilon(t) + \sum_{i=1}^{q}\theta_i \varepsilon(t-i)$$

**Use When:**
- ✅ Series is already stationary
- ✅ No trend or seasonality
- ✅ Need more flexible modeling

---

### 📊 ARIMA(p, d, q) — The BIG Upgrade

**The Game Changer:** Adding differencing ($d$) to handle non-stationarity.

**Process (IMPORTANT):**

1. **Check stationarity** (Augmented Dickey-Fuller test)
2. **Difference** until stationary ($d$ times)
3. **Identify parameters:**
   - PACF → $p$ (AR order)
   - ACF → $q$ (MA order)
4. **Select model** via AIC/BIC
5. **Validate** residuals (white noise)

**Differencing:**
- First difference: $\Delta y(t) = y(t) - y(t-1)$
- Second difference: $\Delta^2 y(t) = \Delta y(t) - \Delta y(t-1)$

**Use When:**
- ✅ Non-stationary but no seasonality
- ✅ Trending data
- ✅ Need interpretability

**🔑 Keywords:** `differencing`, `ADF`, `AIC`, `BIC`

---

### 📊 SARIMA(p,d,q)(P,D,Q)m — Seasonal ARIMA

**The Complete Package for Seasonal Data**

**Notation:**
- $(p,d,q)$: Non-seasonal components
- $(P,D,Q)_m$: Seasonal components
- $m$: Seasonal period (4=quarterly, 12=monthly, 52=weekly)

**Adds:**
- Seasonal lags
- Seasonal differencing
- Seasonal MA terms

**Example:** SARIMA(1,1,1)(1,1,1)₁₂ for monthly data with seasonality

**Use When:**
- ✅ Strong repeating seasonal patterns
- ✅ Monthly sales data
- ✅ Quarterly GDP
- ✅ Daily/weekly patterns

**🔑 Keywords:** `seasonal_lag`, `seasonal_diff`, `period_m`

---

### 🤖 Auto ARIMA

**Intuition:** Let the algorithm search for best parameters automatically.

**What It Does:**
- Tests multiple (p,d,q)(P,D,Q)m combinations
- Uses information criteria (AIC/BIC/AICc)
- Performs stationarity tests
- Validates seasonality

**Why Use:**
- ✅ Fast baseline
- ✅ Strong starting point
- ✅ Reduces manual tuning
- ✅ Good for multiple series

**⚠️ Warning:**
- Not always optimal
- Still needs residual diagnostics
- May overfit on small samples

---

## 4️⃣ Machine Learning for Time Series

> **Critical Understanding:** ML does NOT understand time by default. You must teach time via features.

### 🛠️ Feature Engineering (CRITICAL)

**Core Feature Types:**

| Feature Type | Examples | Purpose |
|-------------|----------|---------|
| **Lags** | $y_{t-1}, y_{t-7}, y_{t-30}$ | Capture autocorrelation |
| **Rolling Stats** | `rolling_mean_7`, `rolling_std_30` | Smooth trends |
| **Calendar Features** | `day_of_week`, `month`, `quarter`, `is_weekend` | Capture patterns |
| **Trend Index** | `days_since_start` | Linear/polynomial trends |
| **Seasonal Encoding** | `sin(2π × day/365)`, `cos(2π × day/365)` | Cyclic patterns |
| **External Variables** | weather, holidays, promotions | Context |

**🔑 Keywords:** `lags`, `rolling`, `calendar_features`

---

### 🌳 XGBoost / LightGBM

**Why Powerful:**
- ✅ Handles non-linearity
- ✅ Robust to outliers
- ✅ Works with many features
- ✅ Fast training
- ✅ Feature importance built-in

**Limitations:**
- ❌ No native sequence memory
- ❌ Needs extensive feature engineering
- ❌ Can't extrapolate beyond training range

**Use When:**
- ✅ Many external variables (weather, promotions, etc.)
- ✅ Structured tabular data
- ✅ Non-linear relationships
- ✅ Need feature importance

**Best Practices:**
- Use time-based cross-validation (NOT random)
- Include multiple lag features
- Add rolling statistics
- Tune carefully to avoid overfitting

---

### 🧠 LSTM / GRU (Deep Learning)

**Intuition:** Learn sequences directly without manual feature engineering.

**Architecture:**
- **LSTM:** Long Short-Term Memory (complex, more parameters)
- **GRU:** Gated Recurrent Unit (simpler, faster)

**Use When:**
- ✅ Very long dependencies (100+ steps)
- ✅ Huge datasets (10,000+ observations)
- ✅ Multiple related time series
- ✅ Complex non-linear patterns

**⚠️ Warning:**
- Overkill for most problems
- Hard to debug
- Requires massive data
- Computationally expensive
- Often beaten by simpler models

**🔑 Keywords:** `sequence_model`, `overfitting`, `data_hungry`

---

### 🤖 Transformers (Attention Mechanisms)

**Intuition:** Apply attention over entire history, learning what matters most.

**Popular Models:**
- Temporal Fusion Transformer (TFT)
- Informer
- Autoformer

**Use When:**
- ✅ Massive datasets (100k+ observations)
- ✅ Multiple related series
- ✅ Long-range dependencies
- ✅ Need interpretable attention weights

**Requirements:**
- Significant computational resources
- Large datasets
- Time for experimentation

---

## 5️⃣ Prophet (Business-Friendly Model)

**Developed by Facebook (Meta)**

### 📐 Model Decomposition

$$y(t) = g(t) + s(t) + h(t) + \varepsilon(t)$$

**Components:**
- $g(t)$: Trend (piecewise linear or logistic)
- $s(t)$: Seasonality (Fourier series)
- $h(t)$: Holidays and events
- $\varepsilon(t)$: Error term

### 💼 Why Businesses Love It

- ✅ Handles holidays automatically
- ✅ Robust to missing data
- ✅ Minimal tuning required
- ✅ Interpretable components
- ✅ Easy to add domain knowledge
- ✅ Works out-of-the-box

### 📊 Use Cases

**Ideal For:**
- Business KPIs (revenue, users, conversions)
- Event-driven patterns
- Data with irregularities
- Non-technical stakeholders

**Not Great For:**
- Sub-daily high-frequency data
- Data with complex non-linear patterns
- When you need cutting-edge accuracy

---

## 6️⃣ Model Selection Framework (MOST IMPORTANT)

### 🎯 The Complexity Ladder

**Always start simple and climb only when necessary:**

```
1. Naive / Seasonal Naive
        ↓ (doesn't beat baseline?)
2. Exponential Smoothing (SES, Holt, Holt-Winters)
        ↓ (need statistical rigor?)
3. ARIMA / SARIMA
        ↓ (need business-friendly approach?)
4. Prophet
        ↓ (have many features?)
5. ML (XGBoost, LightGBM)
        ↓ (have massive data + complex patterns?)
6. Deep Learning (LSTM, Transformers)
```

---

### 🔍 Decision Matrix

| Scenario | Recommended Model |
|----------|------------------|
| **Univariate + Small Data** | ARIMA/SARIMA |
| **Univariate + Seasonality** | Holt-Winters, SARIMA |
| **Business Metrics + Events** | Prophet |
| **Many Features + Non-linear** | XGBoost/LightGBM |
| **Long Sequences + Massive Data** | LSTM/Transformer |
| **Need Interpretability** | ARIMA, Prophet, Linear Models |
| **Quick Baseline** | Naive, Seasonal Naive |

---

### ✅ Classical Models When:

- Univariate data
- Small to medium datasets (<10k points)
- Interpretability required
- Need statistical inference
- Limited computational resources

### ✅ ML/DL When:

- Many exogenous features
- Non-linear patterns
- Large datasets (10k+ observations)
- Computational resources available
- Accuracy is paramount

---

## 7️⃣ Advanced Techniques (Only After Mastery)

### 🔬 Ensemble Methods

**Philosophy:** Combine strengths of multiple models.

**Common Approaches:**
- Simple averaging
- Weighted averaging (based on validation performance)
- Stacking (meta-model learns to combine)

**Benefits:**
- Reduces variance
- More robust predictions
- Often wins competitions

---

### 📈 VAR (Vector Autoregression)

**Use Case:** Multiple related time series that influence each other.

**Examples:**
- GDP, inflation, unemployment (macroeconomics)
- Stock prices of related companies
- Supply chain metrics

**Formula:**

$$\mathbf{y}_t = \mathbf{c} + \mathbf{A}_1\mathbf{y}_{t-1} + \cdots + \mathbf{A}_p\mathbf{y}_{t-p} + \mathbf{\varepsilon}_t$$

---

### 🔮 State Space Models

**Key Idea:** Hidden dynamics drive observed data.

**Popular Implementations:**
- Kalman Filter
- Dynamic Linear Models (DLM)
- Structural Time Series

**Use When:**
- Noisy observations
- Need real-time filtering
- Model hidden states

---

### ⚡ Neural Prophet

**Hybrid Approach:** Combines:
- Prophet's decomposition framework
- Neural network flexibility
- PyTorch backend

**Benefits:**
- Faster than Prophet
- More flexible
- Supports deep learning components

---

## 🧠 FINAL GOLDEN RULES

### Rule #1: Start Simple
> "A model that works is better than a model that might work better but doesn't exist yet."

### Rule #2: Validate Properly
> Use time-based cross-validation. NEVER shuffle time series data.

### Rule #3: Check Residuals
> If residuals aren't white noise, your model missed something.

### Rule #4: Match Complexity to Data
> **A clean SARIMA often beats a sloppy LSTM.**

### Rule #5: Domain Knowledge Wins
> Understanding your data beats fancy algorithms every time.

---

## 📚 Quick Reference Cheat Sheet

### Stationarity Tests
- **ADF Test:** p-value < 0.05 → stationary
- **KPSS Test:** p-value > 0.05 → stationary

### Model Selection Criteria
- **AIC:** Akaike Information Criterion (penalizes complexity)
- **BIC:** Bayesian Information Criterion (stronger penalty)
- **AICc:** Corrected AIC (better for small samples)

**Rule:** Lower is better. Prefer BIC for parsimony.

### Cross-Validation Strategies
- **Time Series Split:** Expanding or sliding window
- **NEVER use K-Fold** (breaks temporal order)

---

## 🎓 Learning Path

1. **Week 1-2:** Master baselines and ETS models
2. **Week 3-4:** Deep dive into ARIMA/SARIMA
3. **Week 5:** Learn Prophet
4. **Week 6-7:** Feature engineering + ML models
5. **Week 8+:** Advanced techniques and ensembles

---

## 📖 Additional Resources

- **Books:** "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos
- **Python Libraries:** `statsmodels`, `prophet`, `pmdarima`, `sktime`, `darts`
- **Practice:** Kaggle time series competitions

---

**Last Updated:** January 2026  
**Version:** 1.0

---

*Remember: The best model is the one that solves your problem reliably, not the one with the most impressive name.* 🎯
