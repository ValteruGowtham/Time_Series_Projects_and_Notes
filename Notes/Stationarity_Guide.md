# 📊 Mastering Stationarity in Time Series Analysis

> *The single most critical concept that separates successful forecasters from failing ones*

---

## 🎯 Table of Contents

1. [Why Stationarity is Critical](#-why-stationarity-is-critical)
2. [What Stationarity Really Means](#-what-stationarity-really-means)
3. [Visual Tests - Always First](#1️⃣-visual-tests-always-first)
4. [Statistical Tests - Confirmation](#2️⃣-statistical-tests-confirmation-not-blind-trust)
5. [Types of Stationarity](#3️⃣-types-of-stationarity)
6. [Detrending vs Differencing](#-detrending-vs-differencing)
7. [Practical Workflow](#4️⃣-practical-workflow-step-by-step)
8. [Practice Exercise](#5️⃣-practice-exercise)
9. [Common Mistakes](#-common-mistakes)

---

## 🚨 WHY STATIONARITY IS CRITICAL (READ THIS FIRST)

### The Hard Truth

**~80% of time-series modeling decisions depend on stationarity**

If you misunderstand or skip this step, you will experience:

| Problem | Impact |
|---------|--------|
| ❌ **Meaningless ARIMA parameters** | Your (p,d,q) values are just random numbers |
| ❌ **Exploding forecasts** | Predictions drift to infinity or crash |
| ❌ **Wrong confidence intervals** | Uncertainty estimates are useless |
| ❌ **Production failures** | Model looks "accurate" in testing but fails live |

---

### 🎯 Stationarity Decides Everything

Stationarity is not just a "nice to check" step. It determines:

```
✅ Whether to difference
✅ How many times to difference  
✅ Whether ARIMA is even valid
✅ How to interpret ACF / PACF plots
✅ Whether trend/seasonality must be removed
```

### ⚡ The Golden Rule

> **No stationarity = No trustworthy model**

Without stationarity, every forecast is built on quicksand.

---

## 🧠 WHAT STATIONARITY REALLY MEANS (ONE-LINER)

### Core Definition

**A time series is stationary if its statistical behavior does not change over time.**

This means three things must be constant:

| Property | Mathematical | Intuitive Meaning |
|----------|-------------|-------------------|
| **Mean** | $E[y_t] = \mu$ | Average level stays flat |
| **Variance** | $Var[y_t] = \sigma^2$ | Spread stays consistent |
| **Autocovariance** | $Cov[y_t, y_{t-k}] = \gamma_k$ | Correlation pattern stable |

### Simple Test

**If any of these change over time → Non-stationary** ❌

![Stationarity Concepts](Images(Notes)/stationarity-01.tif)

---

## 1️⃣ VISUAL TESTS (ALWAYS FIRST)

### ⚠️ Critical Rule

> **Never start with statistical tests. Your eyes come first.**

Visual inspection is 10x faster and often more reliable than blind test application.

---

### 🔹 A. Raw Time Series Plot

#### Code

```python
import pandas as pd
import matplotlib.pyplot as plt

df['value'].plot(figsize=(12, 4), linewidth=2)
plt.title('Raw Time Series', fontsize=14)
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()
```

#### What to Ask Yourself

| Question | Interpretation |
|----------|----------------|
| Mean drifting up/down? | ❌ **Non-stationary** (trend present) |
| Variance increasing? | ❌ **Non-stationary** (heteroscedasticity) |
| Clear trend visible? | ❌ **Non-stationary** (requires differencing) |
| Fluctuates around flat line? | ✅ **Stationary** (good to proceed) |

#### 📌 Visual Red Flags

🚩 **Strong upward/downward slope** → Trend exists  
🚩 **Seasonal waves** → Seasonal component  
🚩 **Increasing volatility** → Variance instability  
🚩 **Sudden level shifts** → Structural breaks

---

### 🔹 B. Rolling Statistics Plot (VERY IMPORTANT)

#### Why This Works

Rolling statistics reveal whether mean and variance stay stable over time. This is one of the **most powerful visual diagnostics** available.

#### Code

```python
# Calculate rolling statistics
window = 12  # Adjust based on your data frequency
rolling_mean = df['value'].rolling(window=window).mean()
rolling_std = df['value'].rolling(window=window).std()

# Create comprehensive plot
fig, ax = plt.subplots(figsize=(14, 6))

# Original series
ax.plot(df.index, df['value'], label='Original Series', 
        color='blue', alpha=0.6, linewidth=1.5)

# Rolling mean
ax.plot(df.index, rolling_mean, label=f'Rolling Mean (window={window})', 
        color='red', linewidth=2.5)

# Rolling standard deviation
ax.plot(df.index, rolling_std, label=f'Rolling Std (window={window})', 
        color='green', linewidth=2.5)

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Rolling Statistics - Stationarity Check', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

#### Interpretation Matrix

| Observation | Meaning | Action |
|-------------|---------|--------|
| Rolling mean **flat** | Mean stationary ✅ | Good sign |
| Rolling std **flat** | Variance stationary ✅ | Good sign |
| **Both flat** | Likely stationary ✅ | Proceed to tests |
| Any **drifting** | Non-stationary ❌ | Transform needed |
| Rolling mean **trending** | Trend exists ❌ | Difference or detrend |
| Rolling std **increasing** | Heteroscedasticity ❌ | Log transform |

#### 📌 Critical Rule

> **If rolling mean or std moves → transformation needed immediately**

---

## 2️⃣ STATISTICAL TESTS (CONFIRMATION, NOT BLIND TRUST)

### The Right Mindset

Statistical tests **quantify** what you visually suspect. They don't replace visual inspection—they confirm it.

![Statistical Tests Overview](Images(Notes)/stationarity-02.png)

---

### 🔹 ADF TEST (Augmented Dickey-Fuller)

#### What ADF Actually Tests

**Core Question:** *"Does this series have a unit root (persistent trend)?"*

#### Hypotheses

- **H₀ (Null Hypothesis):** Series is **non-stationary** (has unit root)
- **H₁ (Alternative):** Series is **stationary**

#### Decision Rule

| p-value | Decision | Interpretation |
|---------|----------|----------------|
| **p < 0.05** | Reject H₀ | ✅ **Stationary** (safe to use) |
| **p ≥ 0.05** | Fail to reject H₀ | ❌ **Non-stationary** (transform needed) |

#### Python Implementation

```python
from statsmodels.tsa.stattools import adfuller
import numpy as np

def adf_test(series, name=''):
    """
    Perform Augmented Dickey-Fuller test
    """
    result = adfuller(series.dropna())
    
    print(f'--- ADF Test Results for {name} ---')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.3f}')
    
    # Interpretation
    if result[1] <= 0.05:
        print(f"\n✅ Result: Stationary (p-value = {result[1]:.4f})")
    else:
        print(f"\n❌ Result: Non-stationary (p-value = {result[1]:.4f})")
    
    return result

# Usage
adf_test(df['value'], name='Original Series')
```

#### Important Interpretation Notes

⚠️ **Low p-value ≠ perfect stationarity**  
⚠️ **ADF struggles with trend-stationary series**  
⚠️ **ADF may falsely say "stationary" if variance changes**

> **👉 ADF alone is NOT enough**

---

### 🔹 KPSS TEST (The Complement)

#### Why KPSS Matters

ADF and KPSS test **opposite assumptions**. Using both eliminates blind spots.

#### Hypotheses

- **H₀ (Null):** Series is **stationary**
- **H₁ (Alternative):** Series is **non-stationary**

**Notice:** This is the opposite of ADF!

#### Decision Rule

| p-value | Decision | Interpretation |
|---------|----------|----------------|
| **p < 0.05** | Reject H₀ | ❌ **Non-stationary** |
| **p ≥ 0.05** | Fail to reject H₀ | ✅ **Stationary** |

#### Python Implementation

```python
from statsmodels.tsa.stattools import kpss

def kpss_test(series, name=''):
    """
    Perform KPSS test
    """
    result = kpss(series.dropna(), regression='c', nlags='auto')
    
    print(f'--- KPSS Test Results for {name} ---')
    print(f'KPSS Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[3].items():
        print(f'  {key}: {value:.3f}')
    
    # Interpretation
    if result[1] >= 0.05:
        print(f"\n✅ Result: Stationary (p-value = {result[1]:.4f})")
    else:
        print(f"\n❌ Result: Non-stationary (p-value = {result[1]:.4f})")
    
    return result

# Usage
kpss_test(df['value'], name='Original Series')
```

#### Why KPSS is Critical

✅ Detects **trend-stationarity** (ADF may miss)  
✅ Catches cases where **ADF gives false positives**  
✅ More conservative (better for model safety)

---

### 🔹 WHY YOU MUST USE BOTH (ADF + KPSS)

#### The Truth Table

| ADF Test | KPSS Test | Conclusion | Action |
|----------|-----------|------------|--------|
| Reject H₀<br>(Stationary) | Fail to reject H₀<br>(Stationary) | ✅ **Stationary** | Proceed with modeling |
| Fail to reject H₀<br>(Non-stationary) | Reject H₀<br>(Non-stationary) | ❌ **Non-stationary** | Difference or transform |
| Reject H₀<br>(Stationary) | Reject H₀<br>(Non-stationary) | ⚠️ **Trend-stationary** | Detrend (don't difference) |
| Fail to reject H₀<br>(Non-stationary) | Fail to reject H₀<br>(Stationary) | ❓ **Inconclusive** | Use visual inspection |

#### Combined Testing Function

```python
def stationarity_check(series, name='Series'):
    """
    Comprehensive stationarity check using both ADF and KPSS
    """
    print("="*60)
    print(f"STATIONARITY CHECK: {name}")
    print("="*60)
    
    # ADF Test
    adf_result = adf_test(series, name)
    print("\n")
    
    # KPSS Test
    kpss_result = kpss_test(series, name)
    print("\n")
    
    # Combined interpretation
    print("="*60)
    print("COMBINED INTERPRETATION")
    print("="*60)
    
    adf_stationary = adf_result[1] <= 0.05
    kpss_stationary = kpss_result[1] >= 0.05
    
    if adf_stationary and kpss_stationary:
        print("✅ CONCLUSION: Series is STATIONARY")
        print("   → Safe to proceed with modeling")
    elif not adf_stationary and not kpss_stationary:
        print("❌ CONCLUSION: Series is NON-STATIONARY")
        print("   → Apply differencing or transformation")
    elif adf_stationary and not kpss_stationary:
        print("⚠️ CONCLUSION: Series is TREND-STATIONARY")
        print("   → Consider detrending instead of differencing")
    else:
        print("❓ CONCLUSION: INCONCLUSIVE")
        print("   → Rely on visual inspection and domain knowledge")
    
    return adf_result, kpss_result
```

### 📌 Golden Rule

> **Never trust a single test. Use visuals + ADF + KPSS together.**

---

## 3️⃣ TYPES OF STATIONARITY (EXAM + INTERVIEW GOLD)

Understanding these distinctions will place you in the top 10% of practitioners.

---

### 🔹 A. Strictly Stationary (Theoretical)

#### Definition

The **entire joint probability distribution** is invariant to time shifts.

$$P(y_{t_1}, y_{t_2}, ..., y_{t_n}) = P(y_{t_1+k}, y_{t_2+k}, ..., y_{t_n+k})$$

#### Reality Check

- ✅ Mathematically elegant
- ❌ Mostly theoretical
- ❌ Almost impossible to verify in practice
- ❌ Rarely used in real applications

---

### 🔹 B. Weakly (Covariance) Stationary (What We Actually Use)

#### Definition

Only first two moments are constant:

1. **Constant Mean:** $E[y_t] = \mu$ for all $t$
2. **Constant Variance:** $Var[y_t] = \sigma^2$ for all $t$
3. **Autocovariance depends only on lag:** $Cov[y_t, y_{t-k}] = \gamma_k$

#### Reality Check

- ✅ **Most models assume this**
- ✅ Practical and verifiable
- ✅ Sufficient for ARIMA, SARIMA
- ✅ What we mean by "stationary" in practice

**👉 When someone says "stationary" in time series, they mean this.**

![Stationarity Concepts](Images(Notes)/stationarity-03.jpg)

---

### 🔹 C. Trend-Stationary

#### Definition

Series has a **deterministic trend** that can be modeled as:

$$y_t = \alpha + \beta t + \varepsilon_t$$

where $\varepsilon_t$ is stationary noise.

#### Characteristics

| Feature | Description |
|---------|-------------|
| **Mean** | Changes predictably with time |
| **After detrending** | Becomes stationary |
| **Trend type** | Deterministic (fixed function) |

#### Example

```python
# Generate trend-stationary series
t = np.arange(100)
trend = 0.5 * t
noise = np.random.normal(0, 1, 100)
y_trend_stationary = trend + noise
```

#### What To Do

✅ **Remove trend** (regression or decomposition)  
❌ **Differencing NOT ideal** (creates artificial patterns)

#### Python Implementation

```python
from scipy import signal

# Method 1: Linear detrending
detrended = signal.detrend(df['value'])

# Method 2: Regression-based
from sklearn.linear_model import LinearRegression

X = np.arange(len(df)).reshape(-1, 1)
y = df['value'].values
model = LinearRegression().fit(X, y)
trend = model.predict(X)
detrended = y - trend
```

---

### 🔹 D. Difference-Stationary (MOST COMMON)

#### Definition

Series has a **stochastic trend** (random walk component).

$$y_t = y_{t-1} + \varepsilon_t$$

This is called having a **unit root**.

#### Characteristics

| Feature | Description |
|---------|-------------|
| **ADF test** | Fails initially (p > 0.05) |
| **After differencing** | Becomes stationary |
| **Trend type** | Stochastic (random) |
| **Prevalence** | **Most real-world series** |

#### Example

```python
# Generate difference-stationary series (random walk)
shocks = np.random.normal(0, 1, 100)
y_diff_stationary = np.cumsum(shocks)  # Random walk
```

#### What To Do

✅ **First difference:** $\Delta y_t = y_t - y_{t-1}$  
✅ **Sometimes seasonal difference** for seasonal data  
❌ **Detrending NOT appropriate** (trend is stochastic)

#### Python Implementation

```python
# First differencing
df_diff = df['value'].diff().dropna()

# Seasonal differencing (e.g., monthly data)
df_seasonal_diff = df['value'].diff(12).dropna()

# Both (if needed)
df_both = df['value'].diff(12).diff().dropna()
```

---

## 🔁 DETRENDING vs DIFFERENCING

### The Critical Decision

| Situation | Correct Method | Reason |
|-----------|----------------|--------|
| **Deterministic trend** | ✅ Detrending | Trend is predictable function |
| **Stochastic trend** | ✅ Differencing | Trend is random walk |
| **Unsure** | ✅ Try differencing first | More common in practice |

### Comparison Table

| Aspect | Detrending | Differencing |
|--------|-----------|--------------|
| **Removes** | Fixed trend component | Level shifts and stochastic trends |
| **Best for** | Trend-stationary series | Difference-stationary series |
| **Reversible** | Yes (add trend back) | Yes (cumsum) |
| **ARIMA parameter** | No impact on $d$ | Sets $d$ value |
| **Use when** | Regression-like trend | Random walk behavior |

### 📌 Practical Rule

> **Most real-world financial and economic series → difference-stationary**

Examples:
- Stock prices → Difference-stationary
- GDP → Often difference-stationary
- Sales data → Often difference-stationary

---

## 4️⃣ PRACTICAL WORKFLOW (STEP-BY-STEP)

### ✅ The Correct Order (NEVER SKIP STEPS)

```
1. Plot raw series
   ↓
2. Rolling mean & std
   ↓
3. ADF test
   ↓
4. KPSS test
   ↓
5. Decide transformation
   ↓
6. Apply transformation
   ↓
7. Re-check stationarity
   ↓
8. Proceed to modeling
```

---

### 🔹 Complete Python Workflow

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
import seaborn as sns

sns.set_style('whitegrid')

def complete_stationarity_workflow(df, column, seasonal_period=None):
    """
    Complete workflow for checking and achieving stationarity
    
    Parameters:
    -----------
    df : DataFrame
    column : str - name of the column to analyze
    seasonal_period : int - seasonal period (e.g., 12 for monthly, 7 for daily)
    """
    series = df[column].dropna()
    
    # STEP 1: Plot raw series
    print("="*80)
    print("STEP 1: RAW TIME SERIES PLOT")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Raw plot
    axes[0, 0].plot(series, linewidth=1.5)
    axes[0, 0].set_title('Raw Time Series', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # STEP 2: Rolling statistics
    print("\nSTEP 2: ROLLING STATISTICS")
    print("="*80)
    
    window = min(12, len(series) // 4)  # Adaptive window
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    axes[0, 1].plot(series, label='Original', alpha=0.6, linewidth=1.5)
    axes[0, 1].plot(rolling_mean, label=f'Rolling Mean ({window})', 
                    linewidth=2.5, color='red')
    axes[0, 1].plot(rolling_std, label=f'Rolling Std ({window})', 
                    linewidth=2.5, color='green')
    axes[0, 1].set_title('Rolling Statistics', fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    # STEP 3 & 4: Statistical tests
    print("\nSTEP 3 & 4: STATISTICAL TESTS")
    print("="*80)
    
    adf_result, kpss_result = stationarity_check(series, name='Original Series')
    
    # STEP 5: Decide transformation
    print("\n" + "="*80)
    print("STEP 5: TRANSFORMATION DECISION")
    print("="*80)
    
    adf_stationary = adf_result[1] <= 0.05
    kpss_stationary = kpss_result[1] >= 0.05
    
    if adf_stationary and kpss_stationary:
        print("✅ Series is already stationary. No transformation needed.")
        transformed = series
        transformation_type = "None"
    else:
        print("❌ Series is non-stationary. Applying transformation...")
        
        # Try first differencing
        transformed = series.diff().dropna()
        transformation_type = "First Difference"
        
        print(f"\n📊 Applied: {transformation_type}")
    
    # Plot transformed series
    axes[1, 0].plot(transformed, linewidth=1.5, color='purple')
    axes[1, 0].set_title(f'Transformed Series ({transformation_type})', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # STEP 6 & 7: Re-check stationarity
    if transformation_type != "None":
        print("\n" + "="*80)
        print("STEP 6 & 7: RE-CHECK STATIONARITY AFTER TRANSFORMATION")
        print("="*80)
        
        stationarity_check(transformed, name='Transformed Series')
    
    # Distribution comparison
    axes[1, 1].hist(series, bins=30, alpha=0.6, label='Original', density=True)
    axes[1, 1].hist(transformed, bins=30, alpha=0.6, label='Transformed', density=True)
    axes[1, 1].set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("✅ WORKFLOW COMPLETE")
    print("="*80)
    
    return transformed

# Usage example
# transformed_series = complete_stationarity_workflow(df, 'value', seasonal_period=12)
```

---

### 🔹 Differencing Examples

#### First Differencing

```python
# First difference
df_diff = df['value'].diff().dropna()

# Verify
stationarity_check(df_diff, name='First Difference')
```

#### Seasonal Differencing

```python
# For monthly data (period = 12)
df_seasonal_diff = df['value'].diff(12).dropna()

# For weekly data (period = 7)
df_weekly_diff = df['value'].diff(7).dropna()

# Verify
stationarity_check(df_seasonal_diff, name='Seasonal Difference')
```

#### Combined Differencing

```python
# First apply seasonal differencing, then regular differencing
df_combined = df['value'].diff(12).diff().dropna()

# Verify
stationarity_check(df_combined, name='Combined Difference')
```

**⚠️ Warning:** Over-differencing creates problems. Use maximum $d=2$ in ARIMA.

---

## 5️⃣ PRACTICE EXERCISE (YOU SHOULD ACTUALLY DO THIS)

### 🎯 The Exercise That Separates Experts from Beginners

> **This single exercise will place you in the top 10% of practitioners.**

---

### 📋 Exercise Instructions

**Take 10 different time series from various domains:**

1. **Stock prices** (e.g., S&P 500, Apple)
2. **Sales data** (retail, e-commerce)
3. **Temperature** (daily, monthly)
4. **Website traffic** (daily visitors)
5. **Exchange rates** (USD/EUR, USD/JPY)
6. **Electricity consumption**
7. **Air passenger numbers**
8. **GDP** (quarterly or annual)
9. **Unemployment rate**
10. **COVID-19 cases** (or any epidemic data)

---

### ✅ For EACH Series, Complete These Steps:

#### Step 1: Plot Raw Series
```python
df['value'].plot(figsize=(12, 4))
plt.title('Raw Series')
plt.show()
```

#### Step 2: Identify Type

Ask these questions:

| Question | Type | Evidence |
|----------|------|----------|
| Flat with constant variance? | Stationary | Rolling stats flat |
| Linear trend? | Trend-stationary | Straight line pattern |
| Random walk appearance? | Difference-stationary | Wandering behavior |
| Seasonal waves? | Seasonal non-stationary | Repeating patterns |

#### Step 3: Apply Correct Transformation

```python
# Based on your identification:

# If stationary → No transformation
stationary = df['value']

# If trend-stationary → Detrend
from scipy import signal
detrended = signal.detrend(df['value'])

# If difference-stationary → Difference
differenced = df['value'].diff().dropna()

# If seasonal → Seasonal difference
seasonal_diff = df['value'].diff(12).dropna()
```

#### Step 4: Re-test Stationarity

```python
stationarity_check(transformed_series, name='Transformed')
```

---

### 📊 Create a Summary Table

| Series | Initial Status | Transformation Applied | Final Status | Notes |
|--------|---------------|------------------------|--------------|-------|
| Stock Price | Non-stationary | First difference | Stationary ✅ | Classic random walk |
| Temperature | Seasonal non-stationary | Seasonal diff (12) | Stationary ✅ | Strong annual cycle |
| GDP | Trend non-stationary | First difference | Stationary ✅ | Growth trend |
| ... | ... | ... | ... | ... |

---

### 🎓 Learning Outcomes

After completing this exercise, you will:

✅ Recognize stationary vs. non-stationary patterns instantly  
✅ Choose correct transformation without trial-and-error  
✅ Understand why certain transformations work  
✅ Build intuition for real-world data behavior  
✅ Avoid 90% of common modeling mistakes

---

## 🚨 COMMON MISTAKES (STRICT WARNING)

### Mistake #1: Blindly Trusting ADF

❌ **The Error:**
```python
# BAD: Only using ADF
if adf_test(series)[1] < 0.05:
    print("Stationary!")  # WRONG!
```

✅ **The Fix:**
```python
# GOOD: Using both tests + visuals
plot_series(series)
adf_result = adf_test(series)
kpss_result = kpss_test(series)
# Make decision based on ALL evidence
```

**Why it matters:** ADF can give false positives with variance changes.

---

### Mistake #2: Over-Differencing

❌ **The Error:**
```python
# BAD: Differencing multiple times without checking
df_diff = df['value'].diff().diff().diff()  # OVERKILL!
```

✅ **The Fix:**
```python
# GOOD: Check after each difference
df_diff1 = df['value'].diff()
if not is_stationary(df_diff1):
    df_diff2 = df_diff1.diff()
    # Stop at d=2 maximum for ARIMA
```

**Why it matters:** Over-differencing introduces artificial patterns and reduces forecast accuracy.

---

### Mistake #3: Ignoring Variance Instability

❌ **The Error:**
```python
# BAD: Only checking mean stationarity
if rolling_mean.std() < threshold:
    print("Stationary!")  # INCOMPLETE!
```

✅ **The Fix:**
```python
# GOOD: Check variance too
if rolling_mean.std() < threshold and rolling_std.std() < threshold:
    print("Stationary!")
else:
    # Apply log transformation for variance
    df['log_value'] = np.log(df['value'])
```

**Why it matters:** ARIMA assumes constant variance. Heteroscedasticity violates this.

---

### Mistake #4: Applying ARIMA to Raw Trending Data

❌ **The Error:**
```python
# BAD: Fitting ARIMA without checking stationarity
model = ARIMA(raw_data, order=(1,0,1))  # d=0 on trending data!
```

✅ **The Fix:**
```python
# GOOD: Check stationarity first
if not is_stationary(raw_data):
    # Use d=1 or transform first
    model = ARIMA(raw_data, order=(1,1,1))
```

**Why it matters:** ARIMA(p,0,q) on non-stationary data produces garbage forecasts.

---

### Mistake #5: Forgetting Seasonal Stationarity

❌ **The Error:**
```python
# BAD: Only checking overall stationarity
adf_test(df['value'])  # Misses seasonal patterns
```

✅ **The Fix:**
```python
# GOOD: Check seasonal stationarity
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose first
decomposition = seasonal_decompose(df['value'], model='additive', period=12)

# Check residuals for stationarity
adf_test(decomposition.resid.dropna())

# Or use seasonal differencing
df_seasonal = df['value'].diff(12)
```

**Why it matters:** Regular differencing doesn't remove seasonal patterns.

---

## 🧠 FINAL MEMORY SUMMARY

### The One-Page Cheat Sheet

#### Core Definition
**Stationarity** = Stable statistical behavior over time
- Constant mean
- Constant variance  
- Constant autocovariance

---

#### The Workflow
1. **Visual** → Plot raw series + rolling stats
2. **Statistical** → ADF + KPSS tests (BOTH!)
3. **Transform** → Difference or detrend based on type
4. **Verify** → Re-check stationarity
5. **Model** → Only after achieving stationarity

---

#### The Decision Rules

| Evidence | Conclusion | Action |
|----------|-----------|--------|
| Visual: Flat + Tests: Both pass | Stationary ✅ | Model directly |
| Visual: Trend + ADF fails | Difference-stationary | Apply differencing |
| Visual: Trend + ADF passes | Trend-stationary | Detrend |
| Visual: Waves | Seasonal | Seasonal differencing |

---

#### The Warnings

```
⚠️ Never trust ADF alone
⚠️ Never over-difference (max d=2)
⚠️ Never ignore variance changes
⚠️ Never skip visual inspection
⚠️ Never apply ARIMA to raw trending data
```

---

### 🎯 The Ultimate Truth

> **If stationarity is wrong, everything downstream is garbage.**

No exceptions. No shortcuts.

---

## 📚 Additional Resources

### Python Libraries
- `statsmodels` - Statistical tests and ARIMA
- `pmdarima` - Auto ARIMA with stationarity checks
- `arch` - Advanced econometric tests

### Further Reading
- "Time Series Analysis" by James Hamilton
- "Forecasting: Principles and Practice" by Hyndman & Athanasopoulos

### Practice Datasets
- [Kaggle Time Series](https://www.kaggle.com/datasets?tags=13303-Time+Series)
- [FRED Economic Data](https://fred.stlouisfed.org/)
- [Yahoo Finance](https://finance.yahoo.com/)

---

**Last Updated:** January 2026  
**Version:** 1.0  
**Mastery Level:** Intermediate to Advanced

---

*Remember: Master stationarity, master time series. There are no shortcuts.* 🎯
