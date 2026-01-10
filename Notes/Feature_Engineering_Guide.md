# 🔧 Feature Engineering for Time Series

> *ML models don't understand time. Your job is to translate time into numbers they can learn from.*

---

## 📋 Table of Contents

1. [Why Feature Engineering is Critical](#-why-feature-engineering-is-critical)
2. [Time-Based (Calendar) Features](#1️⃣-time-based-calendar-features)
3. [Lag Features - Most Important](#2️⃣-lag-features-most-important)
4. [Rolling Window Features](#3️⃣-rolling-window-features)
5. [Cyclical Encoding](#4️⃣-cyclical-encoding-very-important)
6. [Feature Engineering Pipeline](#5️⃣-complete-feature-engineering-pipeline)
7. [Critical Rules](#-critical-feature-engineering-rules)
8. [Feature Selection](#6️⃣-feature-selection)
9. [Final Memory Summary](#-final-memory-summary)

---

## 🚨 WHY FEATURE ENGINEERING IS CRITICAL

### The Fundamental Problem

```
┌─────────────────────────────────────────┐
│  ML Models ≠ Time Series Models         │
│                                          │
│  ML thinks: rows are independent        │
│  Reality: rows depend on past           │
│                                          │
│  Solution: FEATURE ENGINEERING          │
└─────────────────────────────────────────┘
```

### The Core Truth

> **ML models do NOT understand time.**

They see:
```
Row 1: [value=100, ???, ???, ???]
Row 2: [value=105, ???, ???, ???]
Row 3: [value=102, ???, ???, ???]
```

They don't see:
- Yesterday's value influenced today
- Last week's pattern repeats
- January follows December
- Monday differs from Saturday

**Your job:** Translate temporal relationships into numeric features.

![Feature Engineering Concept](Images(Notes)/feature-01.tif)

---

### The Translation Problem

| Time Concept | Human Understanding | ML Understanding |
|--------------|---------------------|------------------|
| **"Last week"** | Clear reference | ??? Unknown |
| **"January"** | Month after December | Just number 1 |
| **"Monday"** | Start of workweek | Just number 0 |
| **"Recent trend"** | Last few days pattern | ??? Unknown |
| **"Volatility"** | How much variation | ??? Unknown |

**Feature Engineering = Building this bridge.**

---

### Impact on Model Performance

```
Without Features:           With Features:
──────────────────         ──────────────────
Accuracy: 60%              Accuracy: 85%
RMSE: High                 RMSE: Low
Training: Fast             Training: Fast
Understanding: Zero        Understanding: High
```

**Reality:**
- 70% of ML performance comes from features
- 20% from model choice
- 10% from hyperparameters

---

## 1️⃣ TIME-BASED (CALENDAR) FEATURES

### 🔹 Purpose

**Capture recurring calendar patterns that drive behavior.**

Business doesn't run on continuous time—it runs on:
- Weekdays vs weekends
- Holidays vs regular days
- Quarters (fiscal patterns)
- Months (seasonal shopping)

---

### Basic Calendar Features

```python
import pandas as pd
import numpy as np

def create_calendar_features(df):
    """
    Create time-based calendar features
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with datetime index
    
    Returns:
    --------
    df : DataFrame with added calendar features
    """
    # Basic time components
    df['day_of_week'] = df.index.dayofweek        # 0=Monday, 6=Sunday
    df['day_of_month'] = df.index.day             # 1-31
    df['day_of_year'] = df.index.dayofyear        # 1-365
    
    df['week_of_year'] = df.index.isocalendar().week  # 1-52
    df['month'] = df.index.month                  # 1-12
    df['quarter'] = df.index.quarter              # 1-4
    df['year'] = df.index.year
    
    # Hour features (if datetime has time component)
    if hasattr(df.index, 'hour'):
        df['hour'] = df.index.hour                # 0-23
        df['minute'] = df.index.minute            # 0-59
    
    return df

# Usage
# df = create_calendar_features(df)
```

---

### Business Logic Features

```python
def create_business_features(df):
    """
    Create business-logic calendar features
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with datetime index
    
    Returns:
    --------
    df : DataFrame with business features
    """
    # Weekend indicator
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    # Business day indicator
    df['is_business_day'] = df.index.dayofweek.isin([0, 1, 2, 3, 4]).astype(int)
    
    # Month start/end
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    
    # Quarter start/end
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    
    # Year start/end
    df['is_year_start'] = df.index.is_year_start.astype(int)
    df['is_year_end'] = df.index.is_year_end.astype(int)
    
    # Season (Northern Hemisphere)
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    
    df['season'] = df.index.month.map(get_season)
    
    return df

# Usage
# df = create_business_features(df)
```

---

### Holiday Features

```python
def create_holiday_features(df, country='US'):
    """
    Create holiday features
    
    Requires: pip install holidays
    """
    import holidays
    
    # Get holidays for country
    country_holidays = holidays.country_holidays(country)
    
    # Is holiday
    df['is_holiday'] = df.index.to_series().apply(
        lambda x: 1 if x in country_holidays else 0
    )
    
    # Days to/from nearest holiday
    holiday_dates = pd.DatetimeIndex([d for d in country_holidays.keys()])
    
    def days_to_nearest_holiday(date):
        if len(holiday_dates) == 0:
            return 0
        days_diff = (holiday_dates - date).days
        return days_diff[np.abs(days_diff).argmin()]
    
    df['days_to_holiday'] = df.index.to_series().apply(days_to_nearest_holiday)
    
    return df

# Usage
# df = create_holiday_features(df, country='US')
```

---

### 🔹 When Calendar Features Are Useful

| Domain | Key Features | Why |
|--------|--------------|-----|
| **Retail** | day_of_week, is_weekend, month, holidays | Weekend shopping, seasonal sales |
| **Traffic** | hour, is_weekend, is_holiday | Rush hours, weekend patterns |
| **Business Metrics** | quarter, is_month_end, is_business_day | Fiscal periods, reporting cycles |
| **Energy** | hour, day_of_week, season | Daily cycles, heating/cooling |
| **Finance** | is_business_day, quarter, is_month_end | Trading days, reporting |

---

### 📌 Why ML Models Love These

```python
# Example: Sales prediction

# Without calendar features
Model sees: [100, 105, 98, 110, 102, 95, 88, 112]
Pattern: ??? Confusing

# With calendar features
Model sees:
[100, Mon, NotWeekend]
[105, Tue, NotWeekend]
[98,  Wed, NotWeekend]
[110, Thu, NotWeekend]
[102, Fri, NotWeekend]
[95,  Sat, Weekend]     ← Lower sales
[88,  Sun, Weekend]     ← Lower sales
[112, Mon, NotWeekend]

Pattern: Weekends → Lower sales ✅ Clear!
```

**Models can learn:** "When is_weekend=1 → reduce prediction"

---

## 2️⃣ LAG FEATURES (MOST IMPORTANT)

### 🔹 Purpose

> **Give the model memory of the past.**

Without lags, model is **memoryless**:
```
Predict tomorrow using only today's features
↓
No knowledge of yesterday, last week, last month
↓
Can't learn temporal dependencies
```

With lags, model has **memory**:
```
Predict tomorrow using:
- Today's value
- Yesterday's value
- Last week's value
- Last month's value
↓
Can learn: "If last 3 days increased, tomorrow likely increases"
```

![Lag Features Concept](Images(Notes)/feature-02.png)

---

### Basic Lag Implementation

```python
def create_lag_features(df, target_col, lags):
    """
    Create lag features
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with time series
    target_col : str
        Column to create lags from
    lags : list of int
        List of lag values to create
    
    Returns:
    --------
    df : DataFrame with lag features
    """
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    return df

# Usage
# df = create_lag_features(df, 'sales', lags=[1, 2, 3, 7, 14, 30])
```

---

### 🔹 How to Choose Lags

**Domain knowledge + ACF/PACF analysis**

#### Standard Lag Choices

| Lag Value | Meaning | Captures | When to Use |
|-----------|---------|----------|-------------|
| **1** | Yesterday | Short-term momentum | Always include |
| **2-3** | 2-3 days ago | Very recent trend | High-frequency data |
| **7** | Last week same day | Weekly patterns | Daily data |
| **14, 21** | 2-3 weeks ago | Multi-week patterns | Daily data |
| **30** | ~Last month | Monthly trends | Daily/weekly data |
| **365** | Last year same day | Yearly seasonality | Daily data |

#### By Data Frequency

**Hourly Data:**
```python
lags = [1, 24, 168]  # 1 hour ago, 1 day ago, 1 week ago
```

**Daily Data:**
```python
lags = [1, 7, 14, 30, 365]  # Yesterday, last week, 2 weeks, month, year
```

**Weekly Data:**
```python
lags = [1, 4, 8, 52]  # Last week, last month, 2 months, last year
```

**Monthly Data:**
```python
lags = [1, 3, 6, 12]  # Last month, quarter, half-year, year
```

---

### Smart Lag Selection

```python
def select_lags_from_acf(series, threshold=0.3, max_lags=50):
    """
    Automatically select significant lags based on ACF
    
    Parameters:
    -----------
    series : array-like
        Time series
    threshold : float
        Significance threshold for ACF
    max_lags : int
        Maximum lags to check
    
    Returns:
    --------
    significant_lags : list
        List of significant lag values
    """
    from statsmodels.tsa.stattools import acf
    
    # Calculate ACF
    acf_values = acf(series, nlags=max_lags)
    
    # Find significant lags
    significant_lags = []
    for lag in range(1, len(acf_values)):
        if abs(acf_values[lag]) > threshold:
            significant_lags.append(lag)
    
    return significant_lags

# Usage
# lags = select_lags_from_acf(df['sales'], threshold=0.3, max_lags=50)
# df = create_lag_features(df, 'sales', lags)
```

---

### Multi-Step Lag Features

```python
def create_lag_differences(df, target_col, lags):
    """
    Create lag differences (change from lag period)
    
    Example: lag_diff_7 = value - value_7_days_ago
    """
    for lag in lags:
        df[f'lag_diff_{lag}'] = df[target_col] - df[target_col].shift(lag)
        df[f'lag_pct_change_{lag}'] = df[target_col].pct_change(lag)
    
    return df

# Usage
# df = create_lag_differences(df, 'sales', lags=[1, 7, 30])
```

---

### 📌 Lag Choice Reflects Domain Knowledge

**Example: Retail Sales**

```python
# Good lag choices
lags = [
    1,    # Yesterday (short-term momentum)
    7,    # Last week same day (weekly pattern)
    14,   # Two weeks ago (bi-weekly pattern)
    365   # Last year same day (seasonality)
]

# Why these work:
# - People shop weekly (groceries)
# - Bi-weekly paycheck patterns
# - Yearly seasons/holidays
```

**Example: Stock Prices**

```python
# Good lag choices
lags = [
    1,     # Previous close
    5,     # Last week (5 trading days)
    20,    # ~1 month
    60,    # ~3 months
    250    # ~1 year
]

# Why these work:
# - High autocorrelation at lag 1
# - Weekly patterns
# - Quarterly earnings
# - Annual cycles
```

---

### 🚨 Critical Lag Rules

#### Rule 1: NEVER Use Future Values

```python
# ❌ WRONG - Creates data leakage
df['lag_1'] = df['value'].shift(-1)  # Tomorrow's value!

# ✅ CORRECT - Uses past values
df['lag_1'] = df['value'].shift(1)   # Yesterday's value
```

#### Rule 2: Handle NaNs Properly

```python
# Lags create NaNs at the beginning
df = create_lag_features(df, 'sales', lags=[1, 7, 30])

# Option 1: Drop NaNs (safest)
df = df.dropna()

# Option 2: Fill with specific strategy
df = df.fillna(method='bfill', limit=30)  # Only if justified

# Option 3: Use partial data (only if you understand implications)
# Keep all rows but model handles NaNs
```

---

## 3️⃣ ROLLING WINDOW FEATURES

### 🔹 Purpose

**Capture local statistics and smooth noise.**

While lags give specific past values, rolling features give:
- **Local trend** (rolling mean)
- **Volatility** (rolling std)
- **Smoothing** (reduce noise)
- **Change detection** (rolling vs current)

---

### Basic Rolling Features

```python
def create_rolling_features(df, target_col, windows):
    """
    Create rolling window features
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with time series
    target_col : str
        Column to create rolling features from
    windows : list of int
        List of window sizes
    
    Returns:
    --------
    df : DataFrame with rolling features
    """
    for window in windows:
        # Rolling mean (trend)
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window).mean()
        
        # Rolling std (volatility)
        df[f'rolling_std_{window}'] = df[target_col].rolling(window).std()
        
        # Rolling min/max (range)
        df[f'rolling_min_{window}'] = df[target_col].rolling(window).min()
        df[f'rolling_max_{window}'] = df[target_col].rolling(window).max()
        
        # Rolling median (robust to outliers)
        df[f'rolling_median_{window}'] = df[target_col].rolling(window).median()
    
    return df

# Usage
# df = create_rolling_features(df, 'sales', windows=[7, 14, 30])
```

---

### Advanced Rolling Features

```python
def create_advanced_rolling_features(df, target_col, windows):
    """
    Create advanced rolling statistics
    """
    for window in windows:
        # Rolling sum
        df[f'rolling_sum_{window}'] = df[target_col].rolling(window).sum()
        
        # Rolling quantiles
        df[f'rolling_q25_{window}'] = df[target_col].rolling(window).quantile(0.25)
        df[f'rolling_q75_{window}'] = df[target_col].rolling(window).quantile(0.75)
        
        # Rolling skewness
        df[f'rolling_skew_{window}'] = df[target_col].rolling(window).skew()
        
        # Rolling kurtosis
        df[f'rolling_kurt_{window}'] = df[target_col].rolling(window).kurt()
        
        # Range (max - min)
        df[f'rolling_range_{window}'] = (
            df[target_col].rolling(window).max() - 
            df[target_col].rolling(window).min()
        )
    
    return df

# Usage
# df = create_advanced_rolling_features(df, 'sales', windows=[7, 30])
```

---

### Comparison Features

```python
def create_rolling_comparison_features(df, target_col, windows):
    """
    Compare current value to rolling statistics
    
    These are powerful features!
    """
    for window in windows:
        rolling_mean = df[target_col].rolling(window).mean()
        rolling_std = df[target_col].rolling(window).std()
        
        # Distance from mean
        df[f'distance_from_mean_{window}'] = df[target_col] - rolling_mean
        
        # Z-score (standardized distance)
        df[f'zscore_{window}'] = (df[target_col] - rolling_mean) / (rolling_std + 1e-8)
        
        # Ratio to mean
        df[f'ratio_to_mean_{window}'] = df[target_col] / (rolling_mean + 1e-8)
        
        # Above/below mean indicator
        df[f'above_mean_{window}'] = (df[target_col] > rolling_mean).astype(int)
    
    return df

# Usage
# df = create_rolling_comparison_features(df, 'sales', windows=[7, 30])
```

---

### Exponentially Weighted Features

```python
def create_ewm_features(df, target_col, spans):
    """
    Create exponentially weighted moving features
    
    EWM gives more weight to recent values
    Better than simple rolling for trending data
    """
    for span in spans:
        # EWM mean
        df[f'ewm_mean_{span}'] = df[target_col].ewm(span=span).mean()
        
        # EWM std
        df[f'ewm_std_{span}'] = df[target_col].ewm(span=span).std()
    
    return df

# Usage
# df = create_ewm_features(df, 'sales', spans=[7, 30])
```

---

### 🔹 What Rolling Features Capture

| Feature | What It Captures | Use Case |
|---------|------------------|----------|
| **rolling_mean** | Local trend | Smooth short-term direction |
| **rolling_std** | Local volatility | Risk, uncertainty periods |
| **rolling_min/max** | Recent extremes | Support/resistance levels |
| **distance_from_mean** | Current position | Overbought/oversold |
| **zscore** | Standardized anomaly | Outlier detection |
| **ewm_mean** | Weighted recent trend | Momentum indicators |

---

### 📌 Great for Tree-Based Models

Tree models (XGBoost, Random Forest) **love** these features because:

1. **Non-linear relationships** automatically captured
2. **Multiple scales** (windows) provide different perspectives
3. **Interaction effects** naturally learned

```python
# Tree model can learn rules like:
# IF rolling_mean_7 > rolling_mean_30 AND zscore_7 > 1.5
#    THEN predict high value
```

---

### 🚨 Critical Rolling Rules

#### Rule 1: Always Shift BEFORE Rolling

```python
# ❌ WRONG - Uses future data
df['rolling_mean_7'] = df['value'].rolling(7).mean()
# This includes today's value → data leakage!

# ✅ CORRECT - Only uses past
df['rolling_mean_7'] = df['value'].shift(1).rolling(7).mean()
# OR explicitly use past values only
```

#### Rule 2: Choose Appropriate Windows

```python
# Consider your forecast horizon
forecast_horizon = 7  # Predicting 7 days ahead

# Good windows
windows = [7, 14, 30]  # All < or = your data frequency

# Bad windows
windows = [3, 5]  # Too small, might not have enough history
windows = [365]  # Too large, might not have enough data
```

---

## 4️⃣ CYCLICAL ENCODING (VERY IMPORTANT)

### 🔹 The Problem

**Months as numbers are problematic:**

```python
month = [1, 2, 3, ..., 11, 12, 1, 2, ...]
```

**Issues:**
1. December (12) and January (1) are **11 units apart** numerically
2. But they're **1 month apart** in reality!
3. Model sees: 12 → 1 as huge jump
4. Reality: December → January is continuous

![Cyclical Encoding](Images(Notes)/feature-03.jpg)

---

### 🔹 The Solution: Sine & Cosine

**Mathematical representation:**

$$\text{month\_sin} = \sin\left(\frac{2\pi \times \text{month}}{12}\right)$$

$$\text{month\_cos} = \cos\left(\frac{2\pi \times \text{month}}{12}\right)$$

**Why this works:**

```
Circular Representation:
        Jan (1)
         / \
   Dec  /   \  Feb
  (12) |     | (2)
       |     |
   Nov |     | Mar
        \ | /
        Oct...

On unit circle:
Jan → (sin, cos) = (0.5, 0.87)
Dec → (sin, cos) = (-0.5, 0.87)
Distance = small ✅

With just numbers:
Jan = 1
Dec = 12
Distance = 11 ❌ Wrong!
```

---

### Basic Cyclical Encoding

```python
def create_cyclical_features(df):
    """
    Create cyclical encodings for time features
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with datetime index
    
    Returns:
    --------
    df : DataFrame with cyclical features
    """
    # Month (1-12)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Day of week (0-6)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # Day of year (1-365)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    
    # Hour (0-23) - if time component exists
    if hasattr(df.index, 'hour'):
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    return df

# Usage
# df = create_cyclical_features(df)
```

---

### Generalized Cyclical Encoding

```python
def encode_cyclical(data, col, max_val):
    """
    Generic cyclical encoding function
    
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame
    col : str
        Column name to encode
    max_val : int
        Maximum value in the cycle
    
    Returns:
    --------
    data : DataFrame with sin/cos columns added
    """
    data[f'{col}_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[f'{col}_cos'] = np.cos(2 * np.pi * data[col] / max_val)
    
    return data

# Usage
# df = encode_cyclical(df, 'month', 12)
# df = encode_cyclical(df, 'day_of_week', 7)
# df = encode_cyclical(df, 'hour', 24)
```

---

### 🔹 Why It Works

#### Preserves Circular Nature

```python
# Example: Month encoding

months = np.arange(1, 13)
month_sin = np.sin(2 * np.pi * months / 12)
month_cos = np.cos(2 * np.pi * months / 12)

# Plot
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Linear encoding (wrong)
axes[0].scatter(months, months)
axes[0].set_title('Linear Encoding (Wrong)')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Value')

# Sine only (incomplete)
axes[1].scatter(months, month_sin)
axes[1].set_title('Sine Only (Incomplete)')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Sin(Month)')

# Sine + Cosine (correct)
axes[2].scatter(month_cos, month_sin)
axes[2].set_title('Sine + Cosine (Correct - Circular)')
axes[2].set_xlabel('Cos(Month)')
axes[2].set_ylabel('Sin(Month)')
axes[2].set_aspect('equal')

plt.tight_layout()
plt.show()
```

---

#### Smooth Transitions

```python
# Distance between consecutive months

# With linear encoding
linear_dist = abs(1 - 12)  # = 11 (huge!)

# With cyclical encoding
jan_point = (np.sin(2*np.pi*1/12), np.cos(2*np.pi*1/12))
dec_point = (np.sin(2*np.pi*12/12), np.cos(2*np.pi*12/12))

cyclical_dist = np.sqrt(
    (jan_point[0] - dec_point[0])**2 + 
    (jan_point[1] - dec_point[1])**2
)
# = 0.52 (small!)

print(f"Linear distance: {linear_dist}")
print(f"Cyclical distance: {cyclical_dist:.2f}")
```

---

### 📌 When Cyclical Encoding is Mandatory

| Model Type | Need Cyclical? | Why |
|------------|----------------|-----|
| **Linear Models** | ✅ Yes | Assume linear relationships |
| **Neural Networks** | ✅ Yes | Need smooth representations |
| **SVMs** | ✅ Yes | Distance-based |
| **Tree Models** | ⚠️ Optional | Can learn splits, but helps |
| **Deep Learning** | ✅ Yes | Especially LSTMs, Transformers |

**Rule of thumb:** Always use cyclical encoding for cyclic features (month, day, hour).

---

### Common Cyclical Features

```python
def create_all_cyclical_features(df):
    """
    Create all common cyclical features
    """
    # Month (12 months)
    df = encode_cyclical(df.assign(month=df.index.month), 'month', 12)
    
    # Day of week (7 days)
    df = encode_cyclical(df.assign(day_of_week=df.index.dayofweek), 'day_of_week', 7)
    
    # Day of month (varies, use 31 as max)
    df = encode_cyclical(df.assign(day_of_month=df.index.day), 'day_of_month', 31)
    
    # Day of year (365 days)
    df = encode_cyclical(df.assign(day_of_year=df.index.dayofyear), 'day_of_year', 365)
    
    # Week of year (52 weeks)
    df = encode_cyclical(df.assign(week_of_year=df.index.isocalendar().week), 'week_of_year', 52)
    
    # Hour (if applicable)
    if hasattr(df.index, 'hour'):
        df = encode_cyclical(df.assign(hour=df.index.hour), 'hour', 24)
        
        # Minute (if applicable)
        if hasattr(df.index, 'minute'):
            df = encode_cyclical(df.assign(minute=df.index.minute), 'minute', 60)
    
    # Quarter (4 quarters)
    df = encode_cyclical(df.assign(quarter=df.index.quarter), 'quarter', 4)
    
    return df

# Usage
# df = create_all_cyclical_features(df)
```

---

## 5️⃣ COMPLETE FEATURE ENGINEERING PIPELINE

### All-in-One Feature Generator

```python
class TimeSeriesFeatureEngineer:
    """
    Complete feature engineering pipeline for time series
    """
    
    def __init__(self, 
                 target_col='value',
                 lag_values=[1, 7, 14, 30],
                 rolling_windows=[7, 14, 30],
                 use_calendar=True,
                 use_cyclical=True):
        """
        Initialize feature engineer
        
        Parameters:
        -----------
        target_col : str
            Target column name
        lag_values : list
            Lag values to create
        rolling_windows : list
            Rolling window sizes
        use_calendar : bool
            Create calendar features
        use_cyclical : bool
            Create cyclical encodings
        """
        self.target_col = target_col
        self.lag_values = lag_values
        self.rolling_windows = rolling_windows
        self.use_calendar = use_calendar
        self.use_cyclical = use_cyclical
    
    def fit(self, df):
        """Fit (store column names)"""
        self.original_columns = df.columns.tolist()
        return self
    
    def transform(self, df):
        """Transform data to create features"""
        df = df.copy()
        
        print("Starting feature engineering...")
        
        # 1. Calendar Features
        if self.use_calendar:
            print("  Creating calendar features...")
            df = self._create_calendar_features(df)
            df = self._create_business_features(df)
        
        # 2. Cyclical Features
        if self.use_cyclical:
            print("  Creating cyclical features...")
            df = self._create_cyclical_features(df)
        
        # 3. Lag Features
        print(f"  Creating {len(self.lag_values)} lag features...")
        df = self._create_lag_features(df)
        
        # 4. Rolling Features
        print(f"  Creating rolling features for {len(self.rolling_windows)} windows...")
        df = self._create_rolling_features(df)
        
        # 5. Interaction Features
        print("  Creating interaction features...")
        df = self._create_interaction_features(df)
        
        print(f"✅ Feature engineering complete! Created {len(df.columns) - len(self.original_columns)} new features.")
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform"""
        return self.fit(df).transform(df)
    
    def _create_calendar_features(self, df):
        """Create calendar features"""
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        if hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour
        
        return df
    
    def _create_business_features(self, df):
        """Create business logic features"""
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    def _create_cyclical_features(self, df):
        """Create cyclical encodings"""
        # Month
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # Day of week
        df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Day of year
        df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        
        if hasattr(df.index, 'hour'):
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        return df
    
    def _create_lag_features(self, df):
        """Create lag features"""
        for lag in self.lag_values:
            df[f'lag_{lag}'] = df[self.target_col].shift(lag)
            df[f'lag_diff_{lag}'] = df[self.target_col] - df[self.target_col].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df):
        """Create rolling window features"""
        for window in self.rolling_windows:
            # Shift first to avoid data leakage
            shifted = df[self.target_col].shift(1)
            
            df[f'rolling_mean_{window}'] = shifted.rolling(window).mean()
            df[f'rolling_std_{window}'] = shifted.rolling(window).std()
            df[f'rolling_min_{window}'] = shifted.rolling(window).min()
            df[f'rolling_max_{window}'] = shifted.rolling(window).max()
            
            # Distance from rolling mean
            df[f'distance_from_mean_{window}'] = df[self.target_col] - df[f'rolling_mean_{window}']
        
        return df
    
    def _create_interaction_features(self, df):
        """Create interaction features"""
        # Lag interactions
        if 'lag_1' in df.columns and 'lag_7' in df.columns:
            df['lag_1_7_interaction'] = df['lag_1'] * df['lag_7']
        
        # Calendar interactions
        if 'is_weekend' in df.columns and 'month' in df.columns:
            df['weekend_month'] = df['is_weekend'] * df['month']
        
        return df

# Usage
# engineer = TimeSeriesFeatureEngineer(
#     target_col='sales',
#     lag_values=[1, 7, 14, 30],
#     rolling_windows=[7, 14, 30],
#     use_calendar=True,
#     use_cyclical=True
# )
# 
# df_features = engineer.fit_transform(df)
```

---

### Quick Feature Creation

```python
def create_all_features(df, target_col='value', 
                       lags=[1, 7, 14, 30],
                       windows=[7, 14, 30]):
    """
    Quick all-in-one feature creation
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with datetime index and target column
    target_col : str
        Target column name
    lags : list
        Lag values to create
    windows : list
        Rolling window sizes
    
    Returns:
    --------
    df : DataFrame with all features
    """
    df = df.copy()
    
    print("="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # 1. Calendar
    print("\n1. Calendar Features")
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    print(f"   ✅ Created 4 calendar features")
    
    # 2. Cyclical
    print("\n2. Cyclical Features")
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    print(f"   ✅ Created 4 cyclical features")
    
    # 3. Lags
    print(f"\n3. Lag Features (lags: {lags})")
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    print(f"   ✅ Created {len(lags)} lag features")
    
    # 4. Rolling
    print(f"\n4. Rolling Features (windows: {windows})")
    for window in windows:
        shifted = df[target_col].shift(1)
        df[f'rolling_mean_{window}'] = shifted.rolling(window).mean()
        df[f'rolling_std_{window}'] = shifted.rolling(window).std()
    print(f"   ✅ Created {len(windows)*2} rolling features")
    
    # 5. Drop NaNs
    original_len = len(df)
    df = df.dropna()
    print(f"\n5. Removed {original_len - len(df)} rows with NaNs")
    
    print("\n" + "="*60)
    print(f"COMPLETE! Total features: {len(df.columns)}")
    print("="*60)
    
    return df

# Usage
# df_with_features = create_all_features(df, target_col='sales')
```

---

## 🚨 CRITICAL FEATURE ENGINEERING RULES

### Rule 1: ❌ NEVER Use Future Values

```python
# ❌ WRONG - Data leakage
df['next_day'] = df['value'].shift(-1)  # Tomorrow's value!
df['future_mean'] = df['value'].shift(-7).rolling(7).mean()  # Future data!

# ✅ CORRECT - Only past values
df['prev_day'] = df['value'].shift(1)   # Yesterday's value
df['past_mean'] = df['value'].shift(1).rolling(7).mean()  # Past 7 days
```

**Why this matters:**
```
Training:   Past data available → features use past → learns correctly
Production: Only past available → features use past → works correctly ✅

If you use future in training:
Training:   Uses future → learns incorrectly
Production: No future → features different → model fails ❌
```

---

### Rule 2: ❌ Always Shift BEFORE Rolling

```python
# ❌ WRONG - Includes current value
df['rolling_mean_7'] = df['value'].rolling(7).mean()
# Uses: [t-6, t-5, t-4, t-3, t-2, t-1, t] → includes today!

# ✅ CORRECT - Only past values
df['rolling_mean_7'] = df['value'].shift(1).rolling(7).mean()
# Uses: [t-7, t-6, t-5, t-4, t-3, t-2, t-1] → excludes today!
```

---

### Rule 3: ❌ Drop NaNs Created by Lags

```python
# Lags create NaNs at the beginning
df = create_lag_features(df, 'sales', lags=[1, 7, 30])

# Before dropping NaNs
print(df.head())
#         sales  lag_1  lag_7  lag_30
# Day 1    100    NaN    NaN     NaN    ← Can't predict
# Day 2    105   100.0   NaN     NaN    ← Can't predict
# ...

# ✅ CORRECT - Drop NaNs
df = df.dropna()

# OR specify minimum periods
max_lag = max([1, 7, 30])  # 30
df = df.iloc[max_lag:]  # Skip first 30 rows
```

---

### Rule 4: ❌ Use SAME Features for Train & Test

```python
# ✅ CORRECT - Same feature engineering for both

# Define features
def engineer_features(df):
    df = create_calendar_features(df)
    df = create_lag_features(df, 'sales', [1, 7, 30])
    df = create_rolling_features(df, 'sales', [7, 30])
    return df

# Apply to both train and test
train_features = engineer_features(train_data)
test_features = engineer_features(test_data)

# ❌ WRONG - Different features
train_features = create_lag_features(train_data, 'sales', [1, 7])
test_features = create_lag_features(test_data, 'sales', [1, 7, 30])  # Different!
```

---

### Rule 5: ❌ Handle Time Zones Properly

```python
# If dealing with time zones
df.index = df.index.tz_localize('UTC')  # Or appropriate timezone

# Features will then respect timezone
df['hour'] = df.index.hour  # Correct hour in that timezone
```

---

### Rule 6: ❌ Document Your Features

```python
# Keep a feature dictionary
feature_dict = {
    'lag_1': 'Previous day value',
    'lag_7': 'Same day last week',
    'rolling_mean_7': 'Average of past 7 days (excluding today)',
    'month_sin': 'Cyclical encoding of month (sine)',
    'is_weekend': 'Binary indicator for Saturday/Sunday',
}

# Save for production
import json
with open('features.json', 'w') as f:
    json.dump(feature_dict, f, indent=2)
```

---

## 6️⃣ FEATURE SELECTION

### Why Feature Selection Matters

```
Too many features:
❌ Overfitting
❌ Slow training
❌ Hard to interpret
❌ Multicollinearity

Too few features:
❌ Underfitting
❌ Missed patterns
❌ Poor performance

Goal: Optimal feature set
```

---

### Correlation-Based Selection

```python
def select_by_correlation(df, target_col, threshold=0.1):
    """
    Select features by correlation with target
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with features
    target_col : str
        Target column name
    threshold : float
        Minimum absolute correlation
    
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
    # Calculate correlations
    correlations = df.corr()[target_col].drop(target_col).abs()
    
    # Select features above threshold
    selected_features = correlations[correlations > threshold].sort_values(ascending=False)
    
    print(f"Selected {len(selected_features)} features (threshold={threshold})")
    print("\nTop 10 features:")
    print(selected_features.head(10))
    
    return selected_features.index.tolist()

# Usage
# selected = select_by_correlation(df, 'sales', threshold=0.1)
# df_selected = df[selected + ['sales']]
```

---

### Tree-Based Feature Importance

```python
def select_by_importance(X, y, n_features=20):
    """
    Select features using tree-based importance
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : pandas Series
        Target
    n_features : int
        Number of top features to select
    
    Returns:
    --------
    selected_features : list
    """
    from sklearn.ensemble import RandomForestRegressor
    
    # Train random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importances
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    
    # Select top features
    selected_features = importances.head(n_features)
    
    # Plot
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    selected_features.plot(kind='barh')
    plt.title(f'Top {n_features} Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    return selected_features.index.tolist()

# Usage
# X = df.drop('sales', axis=1)
# y = df['sales']
# selected = select_by_importance(X, y, n_features=20)
```

---

### Remove Multicollinear Features

```python
def remove_multicollinear_features(df, threshold=0.9):
    """
    Remove highly correlated features
    
    Parameters:
    -----------
    df : pandas DataFrame
        Feature DataFrame
    threshold : float
        Correlation threshold (remove if > threshold)
    
    Returns:
    --------
    df : DataFrame with multicollinear features removed
    """
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Get upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Removing {len(to_drop)} multicollinear features:")
    print(to_drop)
    
    return df.drop(columns=to_drop)

# Usage
# df_reduced = remove_multicollinear_features(df, threshold=0.9)
```

---

## 🧠 FINAL MEMORY SUMMARY

### The Complete Picture

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Visualization → Tells you WHAT the data is        │
│  Features      → Tell model HOW to learn           │
│                                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ACF/PACF         → ARIMA decisions                │
│  Decomposition    → Structure understanding        │
│                                                     │
│  Lags             → Memory                         │
│  Rolling stats    → Local behavior                 │
│  Cyclical         → True time representation       │
│  Calendar         → Business patterns              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

### The Feature Engineering Hierarchy

```
Priority 1: LAG FEATURES
└─ Give model memory of past
   Most important for time series ML

Priority 2: CALENDAR FEATURES  
└─ Capture business patterns
   Essential for retail, traffic, etc.

Priority 3: CYCLICAL ENCODING
└─ Represent time correctly
   Mandatory for linear/neural models

Priority 4: ROLLING FEATURES
└─ Capture local statistics
   Excellent for tree models

Priority 5: INTERACTION FEATURES
└─ Complex relationships
   Model can often learn these
```

---

### Feature Engineering Checklist

Before training any ML model on time series:

✅ **Lags created** (based on ACF/PACF)  
✅ **Rolling features** (for local patterns)  
✅ **Calendar features** (for business patterns)  
✅ **Cyclical encoding** (for periodic features)  
✅ **No future data used** (shift before rolling)  
✅ **NaNs handled** (dropped or filled properly)  
✅ **Same features train/test** (consistency)  
✅ **Features documented** (for production)  
✅ **Feature selection done** (if needed)  
✅ **Multicollinearity checked** (if many features)

---

### Quick Reference: Feature Types

| Feature Type | Purpose | Example | Best For |
|--------------|---------|---------|----------|
| **Lag** | Past values | `lag_7` | All models |
| **Rolling** | Local stats | `rolling_mean_30` | Trees |
| **Calendar** | Time patterns | `day_of_week` | All models |
| **Cyclical** | Periodic | `month_sin` | Linear/Neural |
| **Difference** | Changes | `lag_diff_1` | Capturing momentum |
| **Interaction** | Combined | `lag_1 * month` | Complex patterns |

---

### The Golden Rules

```
1. Visualization FIRST
   └─ Understand before engineering

2. Lag features ALWAYS
   └─ Memory is critical

3. NEVER use future data
   └─ Production will fail

4. Shift BEFORE rolling
   └─ Avoid data leakage

5. Cyclical for cyclic features
   └─ Preserve continuity

6. Document everything
   └─ Production needs clarity
```

---

### Success Metrics

**Good feature engineering means:**

✅ Model performance significantly improves  
✅ Features have business interpretation  
✅ Same pipeline works in production  
✅ No data leakage detected  
✅ Features stable across time  

---

## 📚 Additional Resources

### Python Libraries
- `pandas` - Time-based indexing and operations
- `numpy` - Mathematical operations
- `sklearn` - Feature selection tools
- `featuretools` - Automated feature engineering

### Further Reading
- "Feature Engineering for Machine Learning" by Alice Zheng
- "Forecasting: Principles and Practice" (Chapter 5)

### Production Tips
- Save feature engineering pipeline with model
- Version control feature definitions
- Monitor feature distributions in production
- Test features on out-of-sample data first

---

**Last Updated:** January 2026  
**Version:** 1.0  
**Mastery Level:** Critical (Must-Master)

---

*Remember: Features are the bridge between data and model. Build strong bridges.* 🔧
