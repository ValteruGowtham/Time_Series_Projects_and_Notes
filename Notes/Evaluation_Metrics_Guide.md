# 📈 Evaluation Metrics for Time Series Forecasting

> *Numbers without context are just noise. Learn to measure what matters.*

---

## 📋 Table of Contents

1. [Why Evaluation is Different](#-why-evaluation-is-different-in-time-series)
2. [Core Error Metrics](#1️⃣-core-error-metrics-you-must-master-these)
3. [Metric Selection Guide](#2️⃣-which-metric-when-save-this-table)
4. [Evaluation by Horizon](#3️⃣-evaluation-by-forecast-horizon-very-important)
5. [Visual Evaluation](#4️⃣-visual-evaluation-non-negotiable)
6. [Complete Evaluation Framework](#5️⃣-complete-evaluation-framework)
7. [Common Mistakes](#-common--dangerous-mistakes)
8. [Best Practices](#-best-practices-checklist)

---

## 🚨 WHY EVALUATION IS DIFFERENT IN TIME SERIES

### The Fundamental Difference

| Traditional ML | Time Series |
|----------------|-------------|
| Accuracy is **static** | Errors **change over time** |
| Single test set | **Multiple horizons** |
| One accuracy number | **Errors grow with horizon** |
| Direction doesn't matter | **Direction and scale matter** |
| Single metric often enough | **Multiple metrics required** |

### The Critical Truth

> **A single metric number is NEVER enough in time series.**

Why? Because:

1. **Temporal dynamics** - Performance varies across time
2. **Horizon sensitivity** - 1-day vs 30-day forecasts behave differently
3. **Error asymmetry** - Over-forecasting ≠ Under-forecasting in business impact
4. **Trend vs noise** - Model might catch trend but miss volatility

![Evaluation Landscape](Images(Notes)/evaluation-01.tif)

---

### Real-World Example

```python
# Bad evaluation
accuracy = 95%  # Meaningless!

# Good evaluation
metrics = {
    '1-day': {'RMSE': 10, 'MAE': 7, 'MAPE': 5%},
    '7-day': {'RMSE': 25, 'MAE': 18, 'MAPE': 12%},
    '30-day': {'RMSE': 50, 'MAE': 35, 'MAPE': 25%}
}
```

The second tells a complete story.

---

## 1️⃣ CORE ERROR METRICS (YOU MUST MASTER THESE)

### Overview Table

| Metric | Outlier Sensitivity | Scale | Interpretability | Best For |
|--------|-------------------|-------|------------------|----------|
| **RMSE** | High | Same as data | Medium | Penalizing large errors |
| **MAE** | Low | Same as data | High | Robust evaluation |
| **MAPE** | Medium | Percentage | Very High | Cross-scale comparison |
| **SMAPE** | Medium | Percentage | High | Symmetric percentage |

---

## 🔹 RMSE — Root Mean Squared Error

### Mathematical Formula

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{true} - y_{pred})^2}$$

### Properties

| Feature | Description |
|---------|-------------|
| **Error penalty** | Quadratic (large errors penalized heavily) |
| **Units** | Same as original data |
| **Outlier sensitivity** | High (squares amplify large errors) |
| **Optimization** | Commonly used in model training |

### Python Implementation

```python
import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    rmse : float
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# Usage
y_true = np.array([100, 110, 105, 115, 120])
y_pred = np.array([98, 112, 103, 118, 119])

rmse = calculate_rmse(y_true, y_pred)
print(f"RMSE: {rmse:.2f}")
```

### When to Use RMSE

✅ **Large errors are very costly**
- Example: Stockouts in inventory management
- Over-forecasting production capacity

✅ **Comparing models on same scale**
- All models trained on same data
- Need consistent metric

✅ **Optimization objective**
- Many algorithms minimize squared error

### ⚠️ Cautions

❌ **Sensitive to outliers** - One extreme error dominates
❌ **Different scales** - Can't compare across different products
❌ **Not interpretable** - Squared units confuse stakeholders

### 📌 Key Insight

> **Use RMSE when big mistakes are disasters.**

Example: Predicting hospital bed demand during pandemic.

---

## 🔹 MAE — Mean Absolute Error

### Mathematical Formula

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_{true} - y_{pred}|$$

### Properties

| Feature | Description |
|---------|-------------|
| **Error penalty** | Linear (all errors weighted equally) |
| **Units** | Same as original data |
| **Outlier sensitivity** | Low (robust) |
| **Interpretability** | High (easy to explain) |

### Python Implementation

```python
from sklearn.metrics import mean_absolute_error

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    mae : float
    """
    mae = mean_absolute_error(y_true, y_pred)
    return mae

# Usage
mae = calculate_mae(y_true, y_pred)
print(f"MAE: {mae:.2f}")

# Interpretation
print(f"On average, predictions are off by {mae:.2f} units")
```

### When to Use MAE

✅ **Errors equally bad in all directions**
- Over and under-forecasting have same cost

✅ **Outliers exist in data**
- Robust to extreme values
- Won't be dominated by single bad prediction

✅ **Business reporting**
- Easy to explain to non-technical stakeholders
- Direct interpretation: "average error"

### Advantages Over RMSE

```python
# Example: Why MAE is robust
y_true = [100, 100, 100, 100, 100]
y_pred_normal = [98, 102, 99, 101, 100]  # Small errors
y_pred_outlier = [98, 102, 99, 101, 500] # One huge outlier

rmse_normal = calculate_rmse(y_true, y_pred_normal)
mae_normal = calculate_mae(y_true, y_pred_normal)

rmse_outlier = calculate_rmse(y_true, y_pred_outlier)
mae_outlier = calculate_mae(y_true, y_pred_outlier)

print(f"Normal - RMSE: {rmse_normal:.2f}, MAE: {mae_normal:.2f}")
print(f"Outlier - RMSE: {rmse_outlier:.2f}, MAE: {mae_outlier:.2f}")

# RMSE explodes, MAE increases moderately
```

### 📌 Key Insight

> **Often preferred in business contexts for its simplicity and robustness.**

---

## 🔹 MAPE — Mean Absolute Percentage Error

### Mathematical Formula

$$MAPE = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_{true} - y_{pred}}{y_{true}}\right|$$

### Properties

| Feature | Description |
|---------|-------------|
| **Units** | Percentage (%) |
| **Scale independence** | Can compare across different scales |
| **Interpretability** | Very high ("5% error") |
| **Symmetry** | No (biased) |

![MAPE Visualization](Images(Notes)/evaluation-02.png)

### Python Implementation

```python
def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    
    Parameters:
    -----------
    y_true : array-like
        Actual values (must be non-zero)
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    mape : float (percentage)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    
    if not mask.any():
        return np.inf
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

# Usage
mape = calculate_mape(y_true, y_pred)
print(f"MAPE: {mape:.2f}%")
```

### Pros

✅ **Scale-independent**
- Compare forecasts across products
- Compare $10 items vs $10,000 items

✅ **Easy for stakeholders**
- "We're off by 5%" is universally understood
- Natural business language

✅ **Intuitive threshold**
- < 10% = Excellent
- 10-20% = Good
- \> 20% = Needs improvement

### ⚠️ Critical Problems

#### Problem 1: Division by Zero

```python
# ❌ FAILS
y_true = [0, 10, 20]
y_pred = [1, 12, 18]
mape = calculate_mape(y_true, y_pred)  # Undefined!
```

#### Problem 2: Asymmetric Penalty

```python
# Over-forecasting penalized MORE than under-forecasting
y_true = 100

# Under-forecast by 50%
y_pred_under = 50
error_under = abs((100 - 50) / 100) = 50%

# Over-forecast by 50%
y_pred_over = 150
error_over = abs((100 - 150) / 100) = 50%

# Same percentage error, but different business impact!
```

#### Problem 3: Bias Toward Low Values

```python
# Small values dominate MAPE
y_true = [10, 1000]
y_pred = [15, 1050]

errors = [
    abs((10 - 15) / 10) = 50%,      # Small value, large %
    abs((1000 - 1050) / 1000) = 5%  # Large value, small %
]

# Average MAPE = 27.5%
# Dominated by the small value!
```

### 📌 Use With Caution

> **MAPE looks attractive but has dangerous pitfalls. Always validate results.**

---

## 🔹 SMAPE — Symmetric MAPE

### Mathematical Formula

$$SMAPE = \frac{100}{n}\sum_{i=1}^{n}\frac{2|y_{true} - y_{pred}|}{|y_{true| + |y_{pred}|}$$

### Why Better Than MAPE

| Issue | MAPE | SMAPE |
|-------|------|-------|
| **Zero handling** | Fails | Works better |
| **Symmetry** | No | Yes |
| **Bounded** | No (0-∞) | Yes (0-200%) |
| **Over/under bias** | Biased | Balanced |

### Python Implementation

```python
def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    smape : float (percentage)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    mask = denominator != 0
    
    if not mask.any():
        return 0.0
    
    smape = np.mean(numerator[mask] / denominator[mask]) * 100
    return smape

# Usage
smape = calculate_smape(y_true, y_pred)
print(f"SMAPE: {smape:.2f}%")
```

### Comparison: MAPE vs SMAPE

```python
# Example showing asymmetry
y_true = np.array([100, 100])
y_pred_under = np.array([50, 50])   # Under-forecast
y_pred_over = np.array([150, 150])  # Over-forecast

mape_under = calculate_mape(y_true, y_pred_under)
mape_over = calculate_mape(y_true, y_pred_over)

smape_under = calculate_smape(y_true, y_pred_under)
smape_over = calculate_smape(y_true, y_pred_over)

print(f"Under-forecast - MAPE: {mape_under:.2f}%, SMAPE: {smape_under:.2f}%")
print(f"Over-forecast - MAPE: {mape_over:.2f}%, SMAPE: {smape_over:.2f}%")

# MAPE: different values
# SMAPE: same values (symmetric!)
```

### 📌 Preferred Percentage Metric

> **Use SMAPE instead of MAPE for more balanced evaluation.**

---

## 2️⃣ WHICH METRIC WHEN (SAVE THIS TABLE)

### The Decision Matrix

| Scenario | Best Metric | Why |
|----------|-------------|-----|
| **Same scale comparison** | RMSE / MAE | Direct comparison, same units |
| **Outliers present** | MAE | Robust, not dominated by extremes |
| **Different scales** | MAPE / SMAPE | Scale-independent comparison |
| **Large errors costly** | RMSE | Penalizes big mistakes heavily |
| **Business reporting** | MAE or SMAPE | Easy to explain |
| **Balanced view** | MAE + RMSE | Shows central tendency + spread |
| **Zero values exist** | MAE or SMAPE | Avoid MAPE division issues |
| **Production monitoring** | MAE + SMAPE | Robust + comparable |

![Metric Selection Guide](Images(Notes)/evaluation-03.jpg)

---

### Multi-Metric Evaluation Framework

```python
def comprehensive_evaluation(y_true, y_pred, series_name=''):
    """
    Calculate all major metrics at once
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    series_name : str
        Name for the series (optional)
    
    Returns:
    --------
    metrics : dict
    """
    import pandas as pd
    
    metrics = {
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAE': calculate_mae(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'SMAPE': calculate_smape(y_true, y_pred)
    }
    
    # Additional metrics
    metrics['MSE'] = np.mean((y_true - y_pred) ** 2)
    metrics['Max_Error'] = np.max(np.abs(y_true - y_pred))
    metrics['Mean_Bias'] = np.mean(y_true - y_pred)
    
    # Display
    print("="*60)
    print(f"COMPREHENSIVE EVALUATION: {series_name}")
    print("="*60)
    print(f"RMSE:       {metrics['RMSE']:.4f}")
    print(f"MAE:        {metrics['MAE']:.4f}")
    print(f"MAPE:       {metrics['MAPE']:.2f}%")
    print(f"SMAPE:      {metrics['SMAPE']:.2f}%")
    print(f"Max Error:  {metrics['Max_Error']:.4f}")
    print(f"Mean Bias:  {metrics['Mean_Bias']:.4f}")
    print("="*60)
    
    return metrics

# Usage
metrics = comprehensive_evaluation(y_true, y_pred, series_name='Sales Forecast')
```

---

### 📌 Golden Rule

> **Always report MORE THAN ONE metric.**

Why?
- RMSE alone hides outlier issues
- MAE alone hides large error penalties
- MAPE alone has zero-value problems
- Together, they tell the complete story

---

## 3️⃣ EVALUATION BY FORECAST HORIZON (VERY IMPORTANT)

### The Horizon Problem

**Accuracy is NOT constant across time.**

```python
# This is WRONG
overall_accuracy = 95%  # For what horizon?!

# This is RIGHT
horizon_performance = {
    '1-day':  {'RMSE': 10, 'MAE': 7},
    '7-day':  {'RMSE': 25, 'MAE': 18},
    '30-day': {'RMSE': 50, 'MAE': 35}
}
```

---

### Typical Error Growth Pattern

```
Horizon (h)    Error      Pattern
─────────────────────────────────
1-step         Low        ✓ High confidence
7-step         Moderate   ~ Acceptable
30-step        High       ✗ Unreliable
```

### Why Errors Grow

1. **Uncertainty compounds** - Each step adds noise
2. **Model drift** - Assumptions break down over time
3. **External factors** - Unforeseen events accumulate
4. **Information decay** - Recent data matters more

---

### Multi-Horizon Evaluation Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_multiple_horizons(y_true, y_pred_dict, horizons=[1, 7, 14, 30]):
    """
    Evaluate forecasts at multiple horizons
    
    Parameters:
    -----------
    y_true : array-like
        True values (full series)
    y_pred_dict : dict
        {horizon: predictions} for each horizon
    horizons : list
        Forecast horizons to evaluate
    
    Returns:
    --------
    results : DataFrame
    """
    results = []
    
    print("="*80)
    print("MULTI-HORIZON EVALUATION")
    print("="*80)
    
    for h in horizons:
        if h not in y_pred_dict:
            continue
        
        y_pred = y_pred_dict[h]
        
        # Align lengths
        min_len = min(len(y_true), len(y_pred))
        y_t = y_true[:min_len]
        y_p = y_pred[:min_len]
        
        # Calculate metrics
        rmse = calculate_rmse(y_t, y_p)
        mae = calculate_mae(y_t, y_p)
        mape = calculate_mape(y_t, y_p)
        smape = calculate_smape(y_t, y_p)
        
        results.append({
            'Horizon': f'{h}-step',
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'SMAPE': smape
        })
        
        print(f"\n{h}-step ahead forecast:")
        print(f"  RMSE:  {rmse:.4f}")
        print(f"  MAE:   {mae:.4f}")
        print(f"  MAPE:  {mape:.2f}%")
        print(f"  SMAPE: {smape:.2f}%")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    return results_df

# Visualization
def plot_horizon_performance(results_df):
    """
    Visualize performance degradation across horizons
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    horizons = range(len(results_df))
    
    # RMSE
    axes[0, 0].plot(horizons, results_df['RMSE'], 
                    marker='o', linewidth=2, markersize=8, color='blue')
    axes[0, 0].set_title('RMSE vs Forecast Horizon', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Horizon')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_xticks(horizons)
    axes[0, 0].set_xticklabels(results_df['Horizon'])
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(horizons, results_df['MAE'], 
                    marker='o', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_title('MAE vs Forecast Horizon', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Horizon')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_xticks(horizons)
    axes[0, 1].set_xticklabels(results_df['Horizon'])
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAPE
    axes[1, 0].plot(horizons, results_df['MAPE'], 
                    marker='o', linewidth=2, markersize=8, color='red')
    axes[1, 0].set_title('MAPE vs Forecast Horizon', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Horizon')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].set_xticks(horizons)
    axes[1, 0].set_xticklabels(results_df['Horizon'])
    axes[1, 0].grid(True, alpha=0.3)
    
    # SMAPE
    axes[1, 1].plot(horizons, results_df['SMAPE'], 
                    marker='o', linewidth=2, markersize=8, color='green')
    axes[1, 1].set_title('SMAPE vs Forecast Horizon', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Horizon')
    axes[1, 1].set_ylabel('SMAPE (%)')
    axes[1, 1].set_xticks(horizons)
    axes[1, 1].set_xticklabels(results_df['Horizon'])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage example
# results = evaluate_multiple_horizons(y_true, predictions_by_horizon)
# plot_horizon_performance(results)
```

---

### 📌 Always Report Horizon

**Bad reporting:**
> "Our model has 5% MAPE"

**Good reporting:**
> "Our model has 5% MAPE at 1-day horizon, 12% at 7-day, and 25% at 30-day horizon"

The second gives actionable information.

---

## 4️⃣ VISUAL EVALUATION (NON-NEGOTIABLE)

### Why Metrics Alone Lie

```python
# Both have same MAE = 10
Model A: Consistently off by 10
Model B: Alternates between -20 and +20

# Metrics say "same"
# Reality: Model A is better (stable bias)
```

**You MUST visualize to understand behavior.**

![Visual Evaluation Methods](Images(Notes)/evaluation-04.webp)

---

### Essential Visualizations

#### 1. Predictions vs Actuals

```python
import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast_vs_actual(y_true, y_pred, dates=None, title='Forecast vs Actual'):
    """
    Plot predicted vs actual values
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    dates : array-like (optional)
        Date indices
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if dates is None:
        dates = range(len(y_true))
    
    # Plot actual
    ax.plot(dates, y_true, label='Actual', 
            color='blue', linewidth=2, marker='o', markersize=4)
    
    # Plot predicted
    ax.plot(dates, y_pred, label='Predicted', 
            color='red', linewidth=2, marker='x', markersize=4, alpha=0.7)
    
    # Add metrics annotation
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    
    textstr = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage
# plot_forecast_vs_actual(y_true, y_pred, dates=df.index)
```

---

#### 2. Residual Plot (Critical!)

```python
def plot_residuals(y_true, y_pred, dates=None):
    """
    Plot residuals to check for patterns
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    dates : array-like (optional)
        Date indices
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    if dates is None:
        dates = range(len(residuals))
    
    # 1. Residuals over time
    axes[0, 0].plot(dates, residuals, color='purple', linewidth=1.5, alpha=0.7)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].fill_between(dates, residuals, 0, alpha=0.3, color='purple')
    axes[0, 0].set_title('Residuals Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Residual (Actual - Predicted)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual histogram
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add normality annotation
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)
    axes[0, 1].text(0.05, 0.95, f'Mean: {mean_resid:.2f}\nStd: {std_resid:.2f}',
                    transform=axes[0, 1].transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Predicted vs Residuals (check for patterns)
    axes[1, 0].scatter(y_pred, residuals, alpha=0.5, color='blue')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted Value')
    axes[1, 0].set_ylabel('Residual')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot (normality check)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    print("="*60)
    print("RESIDUAL DIAGNOSTICS")
    print("="*60)
    print(f"Mean Residual: {mean_resid:.4f}")
    print(f"Std Residual: {std_resid:.4f}")
    print(f"Min Residual: {np.min(residuals):.4f}")
    print(f"Max Residual: {np.max(residuals):.4f}")
    
    # Check for bias
    if abs(mean_resid) < 0.01 * np.mean(np.abs(y_true)):
        print("✅ No systematic bias detected")
    else:
        if mean_resid > 0:
            print("⚠️ Model tends to UNDER-forecast")
        else:
            print("⚠️ Model tends to OVER-forecast")
    
    print("="*60)

# Usage
# plot_residuals(y_true, y_pred, dates=df.index)
```

---

#### 3. Error Distribution by Time Period

```python
def plot_error_by_period(y_true, y_pred, dates, period='month'):
    """
    Analyze error patterns across time periods
    
    Parameters:
    -----------
    y_true, y_pred : array-like
        Actual and predicted values
    dates : DatetimeIndex
        Date indices
    period : str
        'month', 'quarter', 'weekday', etc.
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'error': y_true - y_pred,
        'abs_error': np.abs(y_true - y_pred),
        'pct_error': np.abs((y_true - y_pred) / y_true) * 100
    }, index=dates)
    
    # Group by period
    if period == 'month':
        df['period'] = df.index.month
    elif period == 'quarter':
        df['period'] = df.index.quarter
    elif period == 'weekday':
        df['period'] = df.index.dayofweek
    
    # Calculate metrics per period
    grouped = df.groupby('period').agg({
        'abs_error': 'mean',
        'error': 'mean',
        'pct_error': 'mean'
    })
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # MAE by period
    axes[0].bar(grouped.index, grouped['abs_error'], alpha=0.7, color='orange')
    axes[0].set_title(f'MAE by {period.capitalize()}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(period.capitalize())
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].grid(True, alpha=0.3)
    
    # Bias by period
    axes[1].bar(grouped.index, grouped['error'], alpha=0.7, color='blue')
    axes[1].axhline(y=0, color='red', linestyle='--')
    axes[1].set_title(f'Bias by {period.capitalize()}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(period.capitalize())
    axes[1].set_ylabel('Mean Error (Positive = Under-forecast)')
    axes[1].grid(True, alpha=0.3)
    
    # MAPE by period
    axes[2].bar(grouped.index, grouped['pct_error'], alpha=0.7, color='green')
    axes[2].set_title(f'MAPE by {period.capitalize()}', fontsize=14, fontweight='bold')
    axes[2].set_xlabel(period.capitalize())
    axes[2].set_ylabel('Mean Absolute Percentage Error (%)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage
# plot_error_by_period(y_true, y_pred, dates=df.index, period='month')
```

---

### 📌 Critical Insight

> **A model with slightly worse RMSE but stable residuals is often better than one with good RMSE but erratic residuals.**

Why? Stable errors are predictable and can be managed. Erratic errors are risks.

---

## 5️⃣ COMPLETE EVALUATION FRAMEWORK

### All-in-One Evaluation Class

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class TimeSeriesEvaluator:
    """
    Complete evaluation framework for time series forecasts
    """
    
    def __init__(self, y_true, y_pred, dates=None, model_name='Model'):
        """
        Initialize evaluator
        
        Parameters:
        -----------
        y_true : array-like
            Actual values
        y_pred : array-like
            Predicted values
        dates : array-like (optional)
            Date indices
        model_name : str
            Name of the model
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.dates = dates if dates is not None else range(len(y_true))
        self.model_name = model_name
        self.residuals = self.y_true - self.y_pred
        
    def calculate_all_metrics(self):
        """Calculate all standard metrics"""
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            'MAE': mean_absolute_error(self.y_true, self.y_pred),
            'MAPE': self._calculate_mape(),
            'SMAPE': self._calculate_smape(),
            'Mean_Bias': np.mean(self.residuals),
            'Std_Residuals': np.std(self.residuals),
            'Max_Error': np.max(np.abs(self.residuals)),
            'R2': 1 - (np.sum(self.residuals**2) / np.sum((self.y_true - np.mean(self.y_true))**2))
        }
        return metrics
    
    def _calculate_mape(self):
        """MAPE calculation"""
        mask = self.y_true != 0
        if not mask.any():
            return np.inf
        return np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
    
    def _calculate_smape(self):
        """SMAPE calculation"""
        numerator = np.abs(self.y_true - self.y_pred)
        denominator = (np.abs(self.y_true) + np.abs(self.y_pred)) / 2
        mask = denominator != 0
        if not mask.any():
            return 0.0
        return np.mean(numerator[mask] / denominator[mask]) * 100
    
    def print_report(self):
        """Print comprehensive evaluation report"""
        metrics = self.calculate_all_metrics()
        
        print("\n" + "="*70)
        print(f"TIME SERIES FORECAST EVALUATION: {self.model_name}")
        print("="*70)
        print("\n📊 ERROR METRICS")
        print("-"*70)
        print(f"  RMSE:                {metrics['RMSE']:.4f}")
        print(f"  MAE:                 {metrics['MAE']:.4f}")
        print(f"  MAPE:                {metrics['MAPE']:.2f}%")
        print(f"  SMAPE:               {metrics['SMAPE']:.2f}%")
        print(f"  Max Error:           {metrics['Max_Error']:.4f}")
        print(f"  R²:                  {metrics['R2']:.4f}")
        
        print("\n📈 BIAS ANALYSIS")
        print("-"*70)
        print(f"  Mean Bias:           {metrics['Mean_Bias']:.4f}")
        print(f"  Std of Residuals:    {metrics['Std_Residuals']:.4f}")
        
        if abs(metrics['Mean_Bias']) < 0.01 * np.mean(np.abs(self.y_true)):
            print("  Status:              ✅ No systematic bias")
        elif metrics['Mean_Bias'] > 0:
            print("  Status:              ⚠️ Tends to UNDER-forecast")
        else:
            print("  Status:              ⚠️ Tends to OVER-forecast")
        
        print("\n" + "="*70)
        
        return metrics
    
    def plot_comprehensive(self):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Forecast vs Actual
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.dates, self.y_true, label='Actual', 
                color='blue', linewidth=2, marker='o', markersize=3)
        ax1.plot(self.dates, self.y_pred, label='Predicted', 
                color='red', linewidth=2, marker='x', markersize=3, alpha=0.7)
        ax1.fill_between(self.dates, self.y_true, self.y_pred, alpha=0.2, color='gray')
        ax1.set_title(f'{self.model_name}: Forecast vs Actual', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals over time
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.dates, self.residuals, color='purple', linewidth=1.5)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.fill_between(self.dates, self.residuals, 0, alpha=0.3, color='purple')
        ax2.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Residual')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residual distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(self.residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Residual')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Predicted vs Actual scatter
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.scatter(self.y_true, self.y_pred, alpha=0.5, color='blue')
        
        # Perfect prediction line
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        ax4.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Actual')
        ax4.set_ylabel('Predicted')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Error metrics bar chart
        ax5 = fig.add_subplot(gs[2, 1])
        metrics = self.calculate_all_metrics()
        
        # Normalize metrics for display
        display_metrics = {
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE'],
            'SMAPE': metrics['SMAPE']
        }
        
        colors = ['blue', 'orange', 'red', 'green']
        bars = ax5.bar(display_metrics.keys(), display_metrics.values(), 
                      alpha=0.7, color=colors)
        ax5.set_title('Error Metrics Summary', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Value')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate complete evaluation report"""
        metrics = self.print_report()
        self.plot_comprehensive()
        return metrics

# Usage
# evaluator = TimeSeriesEvaluator(y_true, y_pred, dates=df.index, model_name='ARIMA')
# metrics = evaluator.generate_report()
```

---

## 🚨 COMMON & DANGEROUS MISTAKES

### Mistake #1: Evaluating on Training Data

❌ **The Error:**
```python
# Train model
model.fit(train)

# Evaluate on training data ❌
train_accuracy = model.evaluate(train)
print(f"Accuracy: {train_accuracy}")  # Meaningless!
```

✅ **The Fix:**
```python
# Train model
model.fit(train)

# Evaluate on TEST data ✅
test_accuracy = model.evaluate(test)
print(f"Test Accuracy: {test_accuracy}")
```

**Why it matters:** Training accuracy always looks good. Test accuracy reveals truth.

---

### Mistake #2: Using Only One Metric

❌ **The Error:**
```python
# Only looking at RMSE
rmse = calculate_rmse(y_true, y_pred)
print(f"Model performance: {rmse}")  # Incomplete!
```

✅ **The Fix:**
```python
# Multiple metrics
metrics = {
    'RMSE': calculate_rmse(y_true, y_pred),
    'MAE': calculate_mae(y_true, y_pred),
    'MAPE': calculate_mape(y_true, y_pred),
    'SMAPE': calculate_smape(y_true, y_pred)
}
print("Model performance:", metrics)
```

**Why it matters:** Each metric reveals different aspects of performance.

---

### Mistake #3: Ignoring Forecast Horizon

❌ **The Error:**
```python
# Overall accuracy without horizon context
overall_mape = 10%  # For what horizon?!
```

✅ **The Fix:**
```python
# Horizon-specific accuracy
horizon_performance = {
    '1-day': {'MAPE': 5%},
    '7-day': {'MAPE': 12%},
    '30-day': {'MAPE': 25%}
}
```

**Why it matters:** Same model performs differently at different horizons.

---

### Mistake #4: Trusting MAPE Blindly

❌ **The Error:**
```python
# Using MAPE with zeros or near-zeros
mape = calculate_mape([0, 1, 2], [1, 2, 3])  # Division by zero!
```

✅ **The Fix:**
```python
# Use SMAPE or check for zeros
if (y_true == 0).any():
    metric = calculate_smape(y_true, y_pred)
else:
    metric = calculate_mape(y_true, y_pred)
```

**Why it matters:** MAPE fails with zero values and has asymmetric bias.

---

### Mistake #5: No Visual Inspection

❌ **The Error:**
```python
# Just looking at numbers
print(f"RMSE: {rmse}")  # Missing the full picture
```

✅ **The Fix:**
```python
# Always visualize
plot_forecast_vs_actual(y_true, y_pred)
plot_residuals(y_true, y_pred)
```

**Why it matters:** Metrics can be identical but behavior completely different.

---

## ✅ BEST PRACTICES CHECKLIST

### Before Evaluation

- [ ] Split data chronologically (train → validate → test)
- [ ] Ensure no data leakage
- [ ] Define forecast horizon clearly
- [ ] Choose appropriate metrics for your business context

### During Evaluation

- [ ] Calculate multiple metrics (RMSE, MAE, MAPE/SMAPE)
- [ ] Evaluate at multiple horizons (1-step, 7-step, 30-step)
- [ ] Check for bias (over-forecasting vs under-forecasting)
- [ ] Analyze residuals for patterns

### Visualization

- [ ] Plot predictions vs actuals
- [ ] Plot residuals over time
- [ ] Check residual distribution (histogram)
- [ ] Create Q-Q plot for normality
- [ ] Analyze errors by time period (month, weekday, etc.)

### Reporting

- [ ] Report metrics with confidence intervals
- [ ] Specify forecast horizon clearly
- [ ] Include visualizations in reports
- [ ] Document any assumptions or limitations
- [ ] Compare against baseline (naive forecast)

---

## 🧠 FINAL MEMORY SUMMARY

### The Core Principles

```
┌─────────────────────────────────────────────────┐
│  Time Series Evaluation = Correctness Over Time │
│                                                 │
│  1. Split chronologically                       │
│  2. Test on the future                          │
│  3. Evaluate by horizon                         │
│  4. Use multiple metrics                        │
│  5. Always visualize                            │
└─────────────────────────────────────────────────┘
```

---

### Quick Reference Card

| Element | Best Practice |
|---------|--------------|
| **Metrics** | RMSE + MAE + SMAPE (minimum) |
| **Horizon** | Evaluate at 1-step, 7-step, 30-step |
| **Visualization** | Forecast plot + Residual plot (mandatory) |
| **Reporting** | Always include horizon and multiple metrics |
| **Comparison** | Use same test set for all models |

---

### The Ultimate Truth

> **If split or evaluation is wrong, your model is lying to you.**

No amount of hyperparameter tuning can fix bad evaluation methodology.

---

## 📚 Additional Resources

### Python Libraries
- `sklearn.metrics` - Standard metrics
- `statsmodels` - Statistical tests
- `matplotlib` / `seaborn` - Visualization

### Further Reading
- "Forecasting: Principles and Practice" (Chapter 3.3-3.4)
- "Time Series Analysis" - Hamilton (Chapter 4)

### Practice Datasets
- M5 Competition data (Walmart sales)
- Tourism forecasting dataset
- Electricity demand data

---

**Last Updated:** January 2026  
**Version:** 1.0  
**Mastery Level:** Essential (Core Competency)

---

*Remember: Good evaluation separates lucky guesses from reliable forecasts.* 📊
