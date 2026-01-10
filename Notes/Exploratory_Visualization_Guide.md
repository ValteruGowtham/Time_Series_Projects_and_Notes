# 🔍 Exploratory Visualization for Time Series

> *Before you model, you must see. Before you see, you must know what to look for.*

---

## 📋 Table of Contents

1. [Why Exploratory Visualization Matters](#-why-exploratory-visualization-matters)
2. [ACF - Autocorrelation Function](#1️⃣-acf--autocorrelation-function)
3. [PACF - Partial Autocorrelation Function](#2️⃣-pacf--partial-autocorrelation-function)
4. [ACF vs PACF - Side-by-Side](#-acf-vs-pacf--side-by-side-thinking)
5. [Decomposition Plots](#3️⃣-decomposition-plots-structure-detector)
6. [Complete Visualization Framework](#4️⃣-complete-visualization-framework)
7. [Practice Exercises](#-practice-non-negotiable)
8. [Common Patterns Reference](#-common-patterns-quick-reference)

---

## 🚨 WHY EXPLORATORY VISUALIZATION MATTERS

### The Foundation of Good Modeling

> **Before modeling, you must understand the memory of the data.**

Models don't work in a vacuum. They work when they match data structure.

### Critical Questions Visualization Answers

| Question | Why It Matters | What Changes |
|----------|----------------|--------------|
| **How far back does the past matter?** | Determines lag length | Number of features/lags |
| **Is the series AR-like, MA-like, or both?** | Determines model structure | ARIMA (p,d,q) order |
| **Is there seasonality?** | Requires seasonal terms | SARIMA vs ARIMA |
| **Is trend deterministic or stochastic?** | Differencing vs detrending | Transformation method |

![Exploratory Visualization Overview](Images(Notes)/exploratory-01.tif)

---

### The Two Pillars

```
┌─────────────────────────────────────┐
│  ACF & PACF → Guide ARIMA Structure │
│  Decomposition → Explain Patterns   │
└─────────────────────────────────────┘
```

**Without these:**
- You're guessing model parameters randomly
- You miss critical patterns
- Your model assumptions are wrong
- Production fails mysteriously

**With these:**
- You know exactly which model to try
- You understand what drives your data
- Your parameters have theoretical justification
- You can explain model choices

---

### The Learning Path

```
Visualize → Understand → Model
    ↓          ↓           ↓
   ACF      Patterns    ARIMA(p,d,q)
   PACF     Structure   Parameters
   Decomp   Components  Transformations
```

---

## 1️⃣ ACF — Autocorrelation Function

### 🔹 What ACF Shows (INTUITION)

**ACF measures:**

> *"How correlated is the series with itself after k time steps?"*

#### The Questions ACF Answers

| Lag (k) | Question | Business Meaning |
|---------|----------|------------------|
| **k=1** | Does today depend on yesterday? | Short-term memory |
| **k=7** | Does today depend on last week? | Weekly patterns |
| **k=12** | Does today depend on last month? | Monthly patterns |
| **k=365** | Does today depend on last year? | Yearly patterns |

---

### 🔹 How to Read an ACF Plot

#### Visual Components

```
ACF Plot Anatomy:
│
│     ┌── Correlation Value (Y-axis)
│     │
│  1.0├─────────────────
│     │  ▓▓▓
│  0.5├──▓▓▓──── Confidence Band (Blue Shaded)
│     │  ▓▓▓
│  0.0├─────────────────
│     │
│ -0.5├─────────────────
│     │
│     └──────────────────→ Lag (X-axis)
       1  2  3  4  5...
```

#### Reading Rules

| Element | Meaning |
|---------|---------|
| **X-axis** | Lag (time delay in periods) |
| **Y-axis** | Correlation coefficient (-1 to +1) |
| **Blue shaded area** | 95% confidence interval |
| **Bars outside band** | Statistically significant correlation |
| **Bars inside band** | Not significantly different from zero |

---

### Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

def plot_acf_detailed(series, lags=40, title='ACF Plot'):
    """
    Plot ACF with detailed annotations
    
    Parameters:
    -----------
    series : array-like
        Time series data
    lags : int
        Number of lags to display
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot ACF
    plot_acf(series, lags=lags, ax=ax, alpha=0.05)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key lags
    acf_values = acf(series, nlags=lags)
    
    # Highlight significant lags
    for lag in range(1, min(lags+1, len(acf_values))):
        if abs(acf_values[lag]) > 1.96 / np.sqrt(len(series)):
            ax.text(lag, acf_values[lag], f'{acf_values[lag]:.2f}',
                   ha='center', va='bottom' if acf_values[lag] > 0 else 'top',
                   fontsize=8, color='red')
    
    plt.tight_layout()
    plt.show()

# Usage
# plot_acf_detailed(df['value'], lags=40)
```

---

### 🔹 Classic ACF Patterns (MUST MEMORIZE)

These patterns are your diagnostic fingerprints for model selection.

![ACF Patterns](Images(Notes)/exploratory-02.png)

---

#### 🔸 Pattern 1: AR (Autoregressive) Process

**Visual Signature:**
```
ACF: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓...
     ████████▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░░
     Gradual, exponential decay
     No sharp cutoff
```

**Characteristics:**
- ACF decays **gradually** (exponential or damped sine wave)
- Multiple significant lags
- Slow decline to zero
- No clear cutoff point

**What It Means:**
> **Past values influence future continuously**

**Example:**
```python
# Generate AR(2) process
from statsmodels.tsa.arima_process import arma_generate_sample

ar_params = np.array([1, -0.6, 0.2])  # AR coefficients
ar_sample = arma_generate_sample(ar_params, [1], nsample=500)

plot_acf_detailed(ar_sample, lags=30, title='ACF: AR(2) Process')
```

**Model Implication:** Use **AR** terms

---

#### 🔸 Pattern 2: MA (Moving Average) Process

**Visual Signature:**
```
ACF: ▓▓▓▓▓ (then cuts off)
     ████▓▒░
     Sharp cutoff after lag q
     After lag q → correlations ≈ 0
```

**Characteristics:**
- ACF cuts off **sharply** after lag q
- Only first q lags significant
- Remaining lags near zero
- Clear boundary

**What It Means:**
> **Short-lived shocks that don't persist**

**Example:**
```python
# Generate MA(3) process
ma_params = np.array([1, 0.6, 0.3, 0.1])  # MA coefficients
ma_sample = arma_generate_sample([1], ma_params, nsample=500)

plot_acf_detailed(ma_sample, lags=30, title='ACF: MA(3) Process')
# Expect significant spikes at lags 1, 2, 3 only
```

**Model Implication:** Use **MA(q)** where q = last significant lag

---

#### 🔸 Pattern 3: Seasonal Component

**Visual Signature:**
```
ACF: ▓░░░░░░▓░░░░░░▓░░░░░░▓
     |       |       |       |
    lag 7   14      21      28  (weekly)
     
     ▓░░░░░░░░░░▓░░░░░░░░░░▓
     |           |           |
    lag 12      24          36  (monthly)
```

**Characteristics:**
- Strong spikes at regular intervals
- Spikes at multiples of seasonal period
- Pattern repeats

**Common Seasonal Lags:**

| Data Frequency | Seasonal Period | Spike Locations |
|----------------|----------------|-----------------|
| **Hourly** | 24 (daily pattern) | 24, 48, 72, ... |
| **Daily** | 7 (weekly pattern) | 7, 14, 21, ... |
| **Monthly** | 12 (yearly pattern) | 12, 24, 36, ... |
| **Quarterly** | 4 (yearly pattern) | 4, 8, 12, ... |

**What It Means:**
> **Seasonality present - need SARIMA**

**Example:**
```python
# Generate seasonal data
t = np.arange(365)
seasonal = 10 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
noise = np.random.normal(0, 1, 365)
series = seasonal + noise

plot_acf_detailed(series, lags=50, title='ACF: Weekly Seasonality')
# Expect spikes at lags 7, 14, 21, ...
```

**Model Implication:** Use **SARIMA** with seasonal terms

---

### 🔹 What ACF Helps You Decide

```python
def interpret_acf(series, lags=40):
    """
    Automated ACF interpretation
    
    Parameters:
    -----------
    series : array-like
        Time series data
    lags : int
        Number of lags to analyze
    
    Returns:
    --------
    insights : dict
    """
    acf_values = acf(series, nlags=lags)
    
    # Confidence bound
    conf_bound = 1.96 / np.sqrt(len(series))
    
    # Count significant lags
    significant_lags = np.where(np.abs(acf_values[1:]) > conf_bound)[0] + 1
    
    insights = {
        'num_significant_lags': len(significant_lags),
        'significant_lags': significant_lags.tolist()[:10],  # First 10
    }
    
    # Check for seasonality (look for repeating spikes)
    if len(significant_lags) > 0:
        # Check for patterns at 7, 12, 24
        seasonal_candidates = []
        for period in [7, 12, 24, 365]:
            if period in significant_lags:
                seasonal_candidates.append(period)
        
        insights['potential_seasonality'] = seasonal_candidates
    
    # Check for MA cutoff
    if len(significant_lags) > 0:
        # If no significant lags after certain point
        last_sig = significant_lags[-1] if len(significant_lags) > 0 else 0
        if last_sig < lags / 2:
            insights['ma_cutoff_at'] = last_sig
    
    # Decay pattern
    first_10_acf = acf_values[1:11]
    decay_ratio = abs(first_10_acf[-1] / first_10_acf[0]) if first_10_acf[0] != 0 else 1
    
    if decay_ratio > 0.5:
        insights['decay_pattern'] = 'slow (AR-like)'
    elif decay_ratio < 0.2:
        insights['decay_pattern'] = 'fast (MA-like)'
    else:
        insights['decay_pattern'] = 'moderate (mixed)'
    
    # Print insights
    print("="*60)
    print("ACF INTERPRETATION")
    print("="*60)
    print(f"Significant lags: {insights['num_significant_lags']}")
    print(f"Key lags: {insights['significant_lags']}")
    
    if 'potential_seasonality' in insights and insights['potential_seasonality']:
        print(f"⚠️ Potential seasonality at: {insights['potential_seasonality']}")
    
    if 'ma_cutoff_at' in insights:
        print(f"📊 Possible MA cutoff at lag: {insights['ma_cutoff_at']}")
    
    print(f"📈 Decay pattern: {insights['decay_pattern']}")
    print("="*60)
    
    return insights

# Usage
# insights = interpret_acf(df['value'], lags=40)
```

---

### 📌 Key Decisions from ACF

1. **Presence of seasonality** → Seasonal lags significant
2. **MA order (q)** → Where ACF cuts off
3. **Whether differencing worked** → If ACF shows slow decay after differencing, need more

---

## 2️⃣ PACF — Partial Autocorrelation Function

### 🔹 What PACF Shows (INTUITION)

**PACF measures:**

> *"Direct correlation with lag k, AFTER removing effects of intermediate lags"*

#### The Key Distinction

```
ACF  = Total effect (direct + indirect)
PACF = Direct effect only
```

**Analogy:**

Imagine lag relationships like this:
```
Today → Yesterday → 2 Days Ago → 3 Days Ago
```

- **ACF:** Correlation between Today and 3 Days Ago (includes indirect path through Yesterday)
- **PACF:** Correlation between Today and 3 Days Ago (removing Yesterday's influence)

---

### 🔹 How to Read a PACF Plot

**Same structure as ACF:**

| Element | Meaning |
|---------|---------|
| **X-axis** | Lag |
| **Y-axis** | Partial correlation |
| **Confidence band** | 95% significance threshold |
| **Bars outside band** | Direct relationship exists |

---

### Python Implementation

```python
def plot_pacf_detailed(series, lags=40, title='PACF Plot'):
    """
    Plot PACF with detailed annotations
    
    Parameters:
    -----------
    series : array-like
        Time series data
    lags : int
        Number of lags to display
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot PACF
    plot_pacf(series, lags=lags, ax=ax, alpha=0.05, method='ywm')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Partial Autocorrelation', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Calculate PACF values
    pacf_values = pacf(series, nlags=lags, method='ywm')
    
    # Highlight significant lags
    for lag in range(1, min(lags+1, len(pacf_values))):
        if abs(pacf_values[lag]) > 1.96 / np.sqrt(len(series)):
            ax.text(lag, pacf_values[lag], f'{pacf_values[lag]:.2f}',
                   ha='center', va='bottom' if pacf_values[lag] > 0 else 'top',
                   fontsize=8, color='red')
    
    plt.tight_layout()
    plt.show()

# Usage
# plot_pacf_detailed(df['value'], lags=40)
```

---

### 🔹 Classic PACF Patterns (MUST MEMORIZE)

![PACF Patterns](Images(Notes)/exploratory-03.jpg)

---

#### 🔸 Pattern 1: AR(p) Process

**Visual Signature:**
```
PACF: ▓▓▓▓▓ (then cuts off)
      ████▓░
      Sharp cutoff after lag p
      After lag p → insignificant
```

**Characteristics:**
- PACF cuts off **sharply** after lag p
- Only first p lags significant
- Clear boundary
- Remaining lags ≈ 0

**What It Means:**
> **Model needs p autoregressive terms**

**Example:**
```python
# Generate AR(3) process
from statsmodels.tsa.arima_process import arma_generate_sample

ar_params = np.array([1, -0.5, 0.3, -0.2])
ar_sample = arma_generate_sample(ar_params, [1], nsample=500)

plot_pacf_detailed(ar_sample, lags=30, title='PACF: AR(3) Process')
# Expect significant spikes at lags 1, 2, 3 only
```

**Model Implication:** Use **AR(p)** where p = last significant lag

---

#### 🔸 Pattern 2: MA Process

**Visual Signature:**
```
PACF: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓...
      ████████▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░
      Gradual decay
      No sharp cutoff
```

**Characteristics:**
- PACF decays **gradually**
- Multiple significant lags that slowly diminish
- No clear cutoff
- Exponential or damped decay

**What It Means:**
> **Process has moving average structure**

**Model Implication:** Use **MA** terms (check ACF for order)

---

### 🔹 What PACF Helps You Decide

```python
def interpret_pacf(series, lags=40):
    """
    Automated PACF interpretation
    
    Parameters:
    -----------
    series : array-like
        Time series data
    lags : int
        Number of lags to analyze
    
    Returns:
    --------
    insights : dict
    """
    pacf_values = pacf(series, nlags=lags, method='ywm')
    
    # Confidence bound
    conf_bound = 1.96 / np.sqrt(len(series))
    
    # Count significant lags
    significant_lags = np.where(np.abs(pacf_values[1:]) > conf_bound)[0] + 1
    
    insights = {
        'num_significant_lags': len(significant_lags),
        'significant_lags': significant_lags.tolist()[:10],
    }
    
    # Check for AR cutoff
    if len(significant_lags) > 0:
        # Look for clear cutoff
        consecutive_insignificant = 0
        cutoff_point = None
        
        for lag in range(1, lags+1):
            if abs(pacf_values[lag]) < conf_bound:
                consecutive_insignificant += 1
                if consecutive_insignificant >= 3 and cutoff_point is None:
                    cutoff_point = lag - 3
            else:
                consecutive_insignificant = 0
        
        if cutoff_point is not None and cutoff_point > 0:
            insights['ar_cutoff_at'] = cutoff_point
    
    # Decay pattern
    first_10_pacf = pacf_values[1:11]
    decay_ratio = abs(first_10_pacf[-1] / first_10_pacf[0]) if first_10_pacf[0] != 0 else 1
    
    if decay_ratio > 0.5:
        insights['decay_pattern'] = 'slow (MA-like)'
    elif decay_ratio < 0.2:
        insights['decay_pattern'] = 'fast (AR-like)'
    else:
        insights['decay_pattern'] = 'moderate (mixed)'
    
    # Print insights
    print("="*60)
    print("PACF INTERPRETATION")
    print("="*60)
    print(f"Significant lags: {insights['num_significant_lags']}")
    print(f"Key lags: {insights['significant_lags']}")
    
    if 'ar_cutoff_at' in insights:
        print(f"📊 Possible AR cutoff at lag: {insights['ar_cutoff_at']}")
        print(f"   → Suggested AR order (p): {insights['ar_cutoff_at']}")
    
    print(f"📈 Decay pattern: {insights['decay_pattern']}")
    print("="*60)
    
    return insights

# Usage
# insights = interpret_pacf(df['value'], lags=40)
```

---

### 📌 Key Decisions from PACF

1. **AR order (p)** → Where PACF cuts off
2. **Whether over-differencing happened** → All lags insignificant after differencing

---

## 🔁 ACF vs PACF — SIDE-BY-SIDE THINKING

### The Decision Matrix

This table is **the most important reference** for ARIMA model selection.

| ACF Pattern | PACF Pattern | Likely Model | Parameters |
|-------------|--------------|--------------|------------|
| **Gradual decay** | **Sharp cutoff at lag p** | AR(p) | p = cutoff lag |
| **Sharp cutoff at lag q** | **Gradual decay** | MA(q) | q = cutoff lag |
| **Gradual decay** | **Gradual decay** | ARMA(p,q) | Both p and q needed |
| **Seasonal spikes** | **Seasonal spikes** | SARIMA | Add seasonal terms |
| **All near zero** | **All near zero** | White noise | No model needed |

![ACF vs PACF Comparison](Images(Notes)/exploratory-04.webp)

---

### Detailed Pattern Recognition

#### Pattern 1: Pure AR Process

```
ACF:  ▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░░    (gradual decay)
PACF: ▓▓▓░░░░░░░░░░░░░░░░    (cutoff at lag 3)

Diagnosis: AR(3)
Model: ARIMA(3,d,0)
```

---

#### Pattern 2: Pure MA Process

```
ACF:  ▓▓▓▓░░░░░░░░░░░░░░░    (cutoff at lag 4)
PACF: ▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░    (gradual decay)

Diagnosis: MA(4)
Model: ARIMA(0,d,4)
```

---

#### Pattern 3: Mixed ARMA Process

```
ACF:  ▓▓▓▓▓▓▓▓▓▓▒▒▒░░░░░    (gradual decay)
PACF: ▓▓▓▓▓▓▓▓▓▓▒▒▒░░░░░    (gradual decay)

Diagnosis: ARMA(p,q)
Model: ARIMA(p,d,q) - use AIC/BIC to find p,q
```

---

#### Pattern 4: Seasonal Component

```
ACF:  ▓░░░░░░▓░░░░░░▓░░░    (spikes at 7, 14, 21...)
PACF: ▓░░░░░░▓░░░░░░▓░░░    (spikes at 7, 14, 21...)

Diagnosis: Weekly seasonality
Model: SARIMA(p,d,q)(P,D,Q)₇
```

---

### Combined Analysis Function

```python
def analyze_acf_pacf(series, lags=40, title=''):
    """
    Combined ACF and PACF analysis with interpretation
    
    Parameters:
    -----------
    series : array-like
        Time series data
    lags : int
        Number of lags to analyze
    title : str
        Series name
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # ACF
    plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f'ACF: {title}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # PACF
    plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title(f'PACF: {title}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Interpret
    print("\n" + "="*70)
    print(f"COMBINED ANALYSIS: {title}")
    print("="*70)
    
    acf_insights = interpret_acf(series, lags)
    print()
    pacf_insights = interpret_pacf(series, lags)
    
    # Suggest model
    print("\n" + "="*70)
    print("MODEL SUGGESTION")
    print("="*70)
    
    has_ma_cutoff = 'ma_cutoff_at' in acf_insights
    has_ar_cutoff = 'ar_cutoff_at' in pacf_insights
    
    if has_ar_cutoff and not has_ma_cutoff:
        p = acf_insights.get('ar_cutoff_at', 1)
        print(f"✅ Suggested Model: AR({p})")
        print(f"   → Try ARIMA({p}, d, 0)")
    
    elif has_ma_cutoff and not has_ar_cutoff:
        q = acf_insights.get('ma_cutoff_at', 1)
        print(f"✅ Suggested Model: MA({q})")
        print(f"   → Try ARIMA(0, d, {q})")
    
    elif has_ar_cutoff and has_ma_cutoff:
        p = pacf_insights.get('ar_cutoff_at', 1)
        q = acf_insights.get('ma_cutoff_at', 1)
        print(f"✅ Suggested Model: ARMA({p},{q})")
        print(f"   → Try ARIMA({p}, d, {q})")
    
    else:
        print("⚠️ No clear cutoff detected")
        print("   → Try Auto ARIMA")
        print("   → Or start with ARIMA(1,d,1)")
    
    # Check seasonality
    if 'potential_seasonality' in acf_insights and acf_insights['potential_seasonality']:
        periods = acf_insights['potential_seasonality']
        print(f"\n⚠️ Seasonality detected at periods: {periods}")
        print(f"   → Consider SARIMA with seasonal period m={periods[0]}")
    
    print("="*70 + "\n")

# Usage
# analyze_acf_pacf(df['value'], lags=40, title='Sales Data')
```

---

### 📌 Memory Aids

```
ACF → MA
  "A" for "After" → MA comes after in alphabet
  
PACF → AR
  "P" for "Partial" → AR is Partial (direct) effects
```

---

## 3️⃣ DECOMPOSITION PLOTS (STRUCTURE DETECTOR)

### 🔹 What Decomposition Does

**Splits series into interpretable components:**

```
Original = Trend + Seasonality + Residual
```

| Component | Meaning | What to Look For |
|-----------|---------|------------------|
| **Trend** | Long-term direction | Smooth curve, overall movement |
| **Seasonality** | Repeating patterns | Regular cycles, consistent amplitude |
| **Residual** | Everything else | Random noise (ideally) |

---

### Python Implementation

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def decompose_series(series, period=12, model='additive', title=''):
    """
    Decompose time series into components
    
    Parameters:
    -----------
    series : array-like
        Time series with datetime index
    period : int
        Seasonal period
    model : str
        'additive' or 'multiplicative'
    title : str
        Series name
    """
    # Perform decomposition
    result = seasonal_decompose(series, model=model, period=period)
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # Original
    axes[0].plot(series, linewidth=1.5, color='blue')
    axes[0].set_title(f'Original Series: {title}', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(result.trend, linewidth=2, color='orange')
    axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Trend')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    axes[2].plot(result.seasonal, linewidth=1.5, color='green')
    axes[2].set_title(f'Seasonal Component (period={period})', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Seasonal')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    axes[3].plot(result.resid, linewidth=1, color='red', alpha=0.7)
    axes[3].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[3].set_title('Residual (Noise)', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print("="*60)
    print(f"DECOMPOSITION ANALYSIS: {title}")
    print("="*60)
    print(f"Model type: {model}")
    print(f"Seasonal period: {period}")
    print(f"\nComponent Statistics:")
    print(f"  Trend range: {result.trend.min():.2f} to {result.trend.max():.2f}")
    print(f"  Seasonal range: {result.seasonal.min():.2f} to {result.seasonal.max():.2f}")
    print(f"  Residual std: {result.resid.std():.4f}")
    print("="*60)
    
    return result

# Usage
# result = decompose_series(df['value'], period=12, model='additive', title='Monthly Sales')
```

---

### 🔹 Additive vs Multiplicative

#### When to Use Each

| Type | Formula | When to Use | Example |
|------|---------|-------------|---------|
| **Additive** | y = T + S + R | Seasonal effect is **constant** | Temperature variations |
| **Multiplicative** | y = T × S × R | Seasonal effect **grows with level** | Sales revenue |

#### Visual Comparison

```python
def compare_decomposition_types(series, period=12, title=''):
    """
    Compare additive vs multiplicative decomposition
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Additive
    add_result = seasonal_decompose(series, model='additive', period=period)
    axes[0, 0].plot(add_result.trend, color='orange')
    axes[0, 0].set_title('Additive: Trend', fontweight='bold')
    axes[0, 1].plot(add_result.seasonal, color='green')
    axes[0, 1].set_title('Additive: Seasonal', fontweight='bold')
    axes[0, 2].plot(add_result.resid, color='red', alpha=0.7)
    axes[0, 2].set_title('Additive: Residual', fontweight='bold')
    
    # Multiplicative (if all values positive)
    if (series > 0).all():
        mult_result = seasonal_decompose(series, model='multiplicative', period=period)
        axes[1, 0].plot(mult_result.trend, color='orange')
        axes[1, 0].set_title('Multiplicative: Trend', fontweight='bold')
        axes[1, 1].plot(mult_result.seasonal, color='green')
        axes[1, 1].set_title('Multiplicative: Seasonal', fontweight='bold')
        axes[1, 2].plot(mult_result.resid, color='red', alpha=0.7)
        axes[1, 2].set_title('Multiplicative: Residual', fontweight='bold')
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Decomposition Comparison: {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Usage
# compare_decomposition_types(df['value'], period=12, title='Sales')
```

---

### 📌 Rule of Thumb

**Visual test:**

```
If seasonal waves grow → Multiplicative
If seasonal waves constant → Additive
```

**Quick fix:**

> Log transform often converts multiplicative → additive

```python
# If multiplicative suspected
df['log_value'] = np.log(df['value'])
result = seasonal_decompose(df['log_value'], model='additive', period=12)
```

---

### 🔹 What to Look For in Decomposition

#### 1. Trend Analysis

```python
def analyze_trend(trend):
    """Analyze trend component"""
    trend_clean = trend.dropna()
    
    # Calculate trend direction
    first_half = trend_clean[:len(trend_clean)//2].mean()
    second_half = trend_clean[len(trend_clean)//2:].mean()
    
    print("TREND ANALYSIS")
    print("-"*40)
    
    if second_half > first_half * 1.1:
        print("📈 Upward trend detected")
    elif second_half < first_half * 0.9:
        print("📉 Downward trend detected")
    else:
        print("➡️ Relatively flat trend")
    
    # Trend smoothness
    trend_diff = trend_clean.diff().dropna()
    volatility = trend_diff.std()
    
    if volatility < trend_clean.std() * 0.1:
        print("✅ Smooth trend (deterministic)")
    else:
        print("⚠️ Jagged trend (stochastic)")
```

---

#### 2. Seasonality Analysis

```python
def analyze_seasonality(seasonal, period):
    """Analyze seasonal component"""
    seasonal_clean = seasonal.dropna()
    
    print("\nSEASONALITY ANALYSIS")
    print("-"*40)
    
    # Check stability
    seasonal_range = seasonal_clean.max() - seasonal_clean.min()
    seasonal_std = seasonal_clean.std()
    
    print(f"Seasonal period: {period}")
    print(f"Seasonal range: {seasonal_range:.2f}")
    print(f"Seasonal std: {seasonal_std:.2f}")
    
    if seasonal_std > 0.01:
        print("✅ Strong seasonality detected")
    else:
        print("⚠️ Weak or no seasonality")
```

---

#### 3. Residual Analysis (CRITICAL!)

```python
def analyze_residuals(residuals):
    """
    Analyze residual component
    
    Residuals should be:
    1. Random (no patterns)
    2. Zero mean
    3. Constant variance
    4. Normally distributed
    """
    resid_clean = residuals.dropna()
    
    print("\nRESIDUAL ANALYSIS")
    print("-"*40)
    
    # Mean check
    mean_resid = resid_clean.mean()
    if abs(mean_resid) < 0.01:
        print("✅ Mean ≈ 0 (good)")
    else:
        print(f"⚠️ Mean = {mean_resid:.4f} (should be near 0)")
    
    # Variance stability
    first_half_std = resid_clean[:len(resid_clean)//2].std()
    second_half_std = resid_clean[len(resid_clean)//2:].std()
    
    if abs(first_half_std - second_half_std) / first_half_std < 0.5:
        print("✅ Constant variance (good)")
    else:
        print("⚠️ Variance changes over time (heteroscedasticity)")
    
    # Check for patterns (ACF of residuals)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    lb_test = acorr_ljungbox(resid_clean, lags=10, return_df=True)
    
    if (lb_test['lb_pvalue'] > 0.05).all():
        print("✅ No significant autocorrelation (white noise)")
    else:
        print("⚠️ Residuals still have patterns (model missing something)")
    
    print("-"*40)

# Complete analysis
def full_decomposition_analysis(series, period=12, model='additive'):
    """
    Complete decomposition with all diagnostics
    """
    result = seasonal_decompose(series, model=model, period=period)
    
    analyze_trend(result.trend)
    analyze_seasonality(result.seasonal, period)
    analyze_residuals(result.resid)
    
    return result
```

---

### 👉 Critical Rule

> **If residual still has patterns → model is missing something**

Possible issues:
- Wrong seasonal period
- Multiple seasonalities
- Non-linear trend
- Outliers not handled

---

## 4️⃣ COMPLETE VISUALIZATION FRAMEWORK

### All-in-One Exploratory Analysis

```python
class TimeSeriesExplorer:
    """
    Complete exploratory visualization framework
    """
    
    def __init__(self, series, name='Series'):
        """
        Initialize explorer
        
        Parameters:
        -----------
        series : pandas Series
            Time series with datetime index
        name : str
            Series name
        """
        self.series = series
        self.name = name
    
    def plot_overview(self):
        """Plot basic overview"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # Time series
        axes[0, 0].plot(self.series, linewidth=1.5)
        axes[0, 0].set_title('Time Series', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution
        axes[0, 1].hist(self.series, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling stats
        rolling_mean = self.series.rolling(12).mean()
        rolling_std = self.series.rolling(12).std()
        axes[1, 0].plot(self.series, alpha=0.5, label='Original')
        axes[1, 0].plot(rolling_mean, label='Rolling Mean', linewidth=2)
        axes[1, 0].plot(rolling_std, label='Rolling Std', linewidth=2)
        axes[1, 0].set_title('Rolling Statistics', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Boxplot by period (if monthly data)
        try:
            if hasattr(self.series.index, 'month'):
                df = pd.DataFrame({'value': self.series, 'month': self.series.index.month})
                df.boxplot(column='value', by='month', ax=axes[1, 1])
                axes[1, 1].set_title('Seasonality Check', fontweight='bold')
        except:
            axes[1, 1].text(0.5, 0.5, 'Seasonality plot unavailable',
                           ha='center', va='center')
        
        plt.suptitle(f'Overview: {self.name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_acf_pacf(self, lags=40):
        """Plot ACF and PACF"""
        analyze_acf_pacf(self.series, lags=lags, title=self.name)
    
    def plot_decomposition(self, period=12, model='additive'):
        """Plot decomposition"""
        return decompose_series(self.series, period=period, model=model, title=self.name)
    
    def full_report(self, lags=40, period=12):
        """
        Generate complete exploratory report
        """
        print("\n" + "="*70)
        print(f"COMPLETE EXPLORATORY ANALYSIS: {self.name}")
        print("="*70 + "\n")
        
        # Overview
        print("1. OVERVIEW PLOTS")
        self.plot_overview()
        
        # ACF/PACF
        print("\n2. ACF/PACF ANALYSIS")
        self.plot_acf_pacf(lags=lags)
        
        # Decomposition
        print("\n3. DECOMPOSITION ANALYSIS")
        result = self.plot_decomposition(period=period)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        return result

# Usage
# explorer = TimeSeriesExplorer(df['value'], name='Monthly Sales')
# explorer.full_report(lags=40, period=12)
```

---

## 🧪 PRACTICE (NON-NEGOTIABLE)

### The Path to Mastery

> **This single practice will make you better than 90% of practitioners.**

---

### Exercise 1: Pattern Recognition Training

**Task:** Look at 20+ ACF/PACF plots

**For each plot, identify:**

1. ✅ Is it AR, MA, or ARMA?
2. ✅ What are the orders (p, q)?
3. ✅ Is there seasonality?
4. ✅ What seasonal period?

**Resources:**
```python
# Generate practice data
from statsmodels.tsa.arima_process import arma_generate_sample

# AR(2)
ar2 = arma_generate_sample([1, -0.6, 0.2], [1], 500)
analyze_acf_pacf(ar2, title='Mystery Series 1')

# MA(3)
ma3 = arma_generate_sample([1], [1, 0.5, 0.3, 0.2], 500)
analyze_acf_pacf(ma3, title='Mystery Series 2')

# ARMA(2,2)
arma22 = arma_generate_sample([1, -0.5, 0.2], [1, 0.4, 0.3], 500)
analyze_acf_pacf(arma22, title='Mystery Series 3')

# With seasonality
t = np.arange(500)
seasonal = 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, 500)
analyze_acf_pacf(seasonal, title='Mystery Series 4')
```

---

### Exercise 2: Real Data Analysis

**Download 5 real datasets:**
1. Stock prices
2. Weather data
3. Sales data
4. Website traffic
5. Energy consumption

**For each:**
```python
# 1. Load data
df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')

# 2. Create explorer
explorer = TimeSeriesExplorer(df['value'], name='Dataset Name')

# 3. Full analysis
explorer.full_report(lags=40, period=12)

# 4. Document findings
# - What patterns did you see?
# - What model would you try?
# - What transformations needed?
```

---

### Exercise 3: Model Validation

**After modeling, check residuals:**

```python
# After fitting model
residuals = model.resid

# Analyze residuals
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Residual plot
axes[0, 0].plot(residuals)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals Over Time')

# Histogram
axes[0, 1].hist(residuals, bins=30, edgecolor='black')
axes[0, 1].set_title('Residual Distribution')

# ACF of residuals
plot_acf(residuals, ax=axes[1, 0], lags=40)
axes[1, 0].set_title('ACF of Residuals')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()

# If residuals show patterns → model inadequate
```

---

### 🎯 Learning Outcomes

After completing these exercises:

✅ **Instant pattern recognition** - See AR/MA/ARMA immediately  
✅ **Seasonal detection** - Spot seasonality patterns instantly  
✅ **Model selection confidence** - Know which model to try  
✅ **Diagnostic skills** - Validate models properly  
✅ **Professional intuition** - Understand data deeply

---

### 📌 The Truth

> **This builds visual intuition no textbook can replace.**

Hours spent on these exercises = years saved on wrong model choices.

---

## 📊 COMMON PATTERNS QUICK REFERENCE

### Pattern Cheat Sheet

```
┌─────────────────────────────────────────────────────┐
│  AR Process                                         │
│  ACF:  ▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░ (gradual decay)        │
│  PACF: ▓▓▓░░░░░░░░░░░░░░░░ (cutoff at p)          │
│  Model: AR(p)                                      │
├─────────────────────────────────────────────────────┤
│  MA Process                                         │
│  ACF:  ▓▓▓▓░░░░░░░░░░░░░░░ (cutoff at q)          │
│  PACF: ▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░ (gradual decay)        │
│  Model: MA(q)                                      │
├─────────────────────────────────────────────────────┤
│  ARMA Process                                       │
│  ACF:  ▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░ (gradual)               │
│  PACF: ▓▓▓▓▓▓▓▓▓▓▒▒▒▒░░░ (gradual)               │
│  Model: ARMA(p,q)                                  │
├─────────────────────────────────────────────────────┤
│  Seasonal Component                                 │
│  ACF:  ▓░░░░░░▓░░░░░░▓░░ (spikes at m, 2m, 3m)   │
│  PACF: ▓░░░░░░▓░░░░░░▓░░ (spikes at m, 2m, 3m)   │
│  Model: SARIMA                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🧠 FINAL MEMORY SUMMARY

### The Essential Flow

```
1. Plot raw series
   ↓
2. Check stationarity
   ↓
3. Plot ACF/PACF
   ↓
4. Identify patterns
   ↓
5. Decompose
   ↓
6. Select model
   ↓
7. Validate residuals
```

---

### Decision Rules

| What You See | What It Means | What You Do |
|--------------|---------------|-------------|
| **ACF gradual, PACF cutoff** | AR process | Use AR(p) |
| **ACF cutoff, PACF gradual** | MA process | Use MA(q) |
| **Both gradual** | Mixed process | Use ARMA(p,q) |
| **Seasonal spikes** | Seasonality | Use SARIMA |
| **All near zero** | White noise | No model needed |

---

### 🎯 The Ultimate Truth

> **Visualization comes before modeling. Always.**

No exceptions. No "I'll just try Auto ARIMA." Understanding the data structure is non-negotiable for reliable forecasting.

---

## 📚 Additional Resources

### Python Libraries
- `statsmodels.graphics.tsaplots` - ACF/PACF plots
- `statsmodels.tsa.seasonal` - Decomposition
- `matplotlib` / `seaborn` - Visualization

### Further Reading
- "Time Series Analysis" by James Hamilton (Chapters 2-3)
- "Forecasting: Principles and Practice" (Chapter 2)

### Practice Datasets
- [Kaggle Time Series](https://www.kaggle.com/datasets?tags=13303-Time+Series)
- Built-in datasets: `statsmodels.datasets`

---

**Last Updated:** January 2026  
**Version:** 1.0  
**Mastery Level:** Fundamental (Must-Master)

---

*Remember: Good forecasters see patterns invisible to others. Build that vision through practice.* 🔍
