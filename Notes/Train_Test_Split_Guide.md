# ⏰ Train-Test Split for Time Series

> *The single rule that breaks 90% of beginner models: Never shuffle time*

---

## 📋 Table of Contents

1. [Why Train-Test Split is Critical](#-why-traintestsplit-is-critical)
2. [What You Must Never Do](#-what-you-must-never-do)
3. [Core Principle](#-core-principle)
4. [Correct Splitting Methods](#-correct-time-series-splitting-methods)
5. [Forecast Horizon](#-forecast-horizon-matters)
6. [Critical Rules](#-critical-splitting-rules)
7. [Complete Implementation Guide](#-complete-implementation-guide)
8. [Practice Tasks](#-practice-task)
9. [Common Mistakes](#-common-mistakes-and-solutions)

---

## 🚨 WHY TRAIN-TEST SPLIT IS CRITICAL

### The Fundamental Problem

```
Random splitting in time series
        ↓
Data leakage
        ↓
Fake accuracy
        ↓
💥 Production disaster
```

### 🎯 Time Series is Different

Unlike standard ML, in time series:

| Requirement | Reason | Violation Impact |
|-------------|--------|------------------|
| **Order matters** | Time flows forward | Results become meaningless |
| **Future must NEVER influence past** | Causality violation | Invalid predictions |
| **Test set = unseen future** | Real-world simulation | Model can't generalize |

### The Deceptive Trap

**If you break this rule:**

✅ Your model looks **amazing** in testing  
❌ Your production performance **collapses**  
💰 Your business **loses money**  
😱 Your credibility **disappears**

---

### Real-World Example of Failure

```python
# Someone uses shuffle=True
model_accuracy = 0.95  # "Wow, amazing!"

# Deploy to production
production_accuracy = 0.62  # "What happened?!"
```

**What went wrong?**
- Training data contained information from the "future"
- Model learned patterns it would never see in real deployment
- Test accuracy was a **lie**

---

## 🚫 WHAT YOU MUST NEVER DO

### ❌ THE DEADLY SIN

```python
# ❌ WRONG — NEVER USE THIS FOR TIME SERIES
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(
    data, 
    test_size=0.2, 
    shuffle=True  # 💀 THE KILLER
)
```

### Why This is Cheating

| Problem | Explanation | Consequence |
|---------|-------------|-------------|
| **Past sees future** | 2025 data in training, 2024 in test | Temporal causality broken |
| **Data leakage** | Model learns information it shouldn't have | Overly optimistic metrics |
| **Invalid accuracy** | Test accuracy meaningless | Can't trust any results |
| **Production failure** | Real-world has no future data | Model fails completely |

### Visual Comparison

```
❌ WRONG (shuffled):
Train: [2024-03, 2023-01, 2024-12, 2023-08, ...]
Test:  [2024-06, 2023-05, 2024-02, 2023-11, ...]
        ↑ Past and future mixed! ↑

✅ CORRECT (ordered):
Train: [2023-01, 2023-02, 2023-03, ..., 2024-08]
Test:  [2024-09, 2024-10, 2024-11, 2024-12]
        ↑ Past → Future ↑
```

---

### 📌 Critical Warning

> **This is the #1 beginner mistake in time series forecasting.**

More common than:
- Wrong model selection
- Poor feature engineering
- Incorrect transformations

Why? Because `sklearn`'s default behavior seems so convenient.

---

## 🧠 CORE PRINCIPLE (MEMORIZE THIS)

### The Golden Rule

```
┌─────────────────────────────────────┐
│  Train on the PAST                  │
│          ↓                          │
│  Test on the FUTURE                 │
│                                     │
│  ALWAYS. NO EXCEPTIONS.             │
└─────────────────────────────────────┘
```

### Mathematical Representation

Given time series: $y_1, y_2, y_3, ..., y_T$

**Correct split:**
- Training: $y_1, y_2, ..., y_t$
- Testing: $y_{t+1}, y_{t+2}, ..., y_T$

where $t < T$ and $t$ is the split point.

**Constraint:** $\max(\text{train indices}) < \min(\text{test indices})$

![Train Test Split Concept](Images(Notes)/train test-01.tif)

---

### The Timeline Visualization

```
Time ────────────────────────────────────────→

Past                      Now            Future
│◄────── Train ──────►│◄──── Test ────►│
                      ↑
                  Split point
                  (Today)
```

---

## ✅ CORRECT TIME SERIES SPLITTING METHODS

### Overview Table

| Method | Best For | Complexity | Compute Cost | Robustness |
|--------|----------|------------|--------------|------------|
| **Hold-out Split** | Quick tests, baselines | Low | Low | Low |
| **Rolling Window** | Model comparison | Medium | Medium | Medium |
| **Expanding Window** | Production simulation | Medium | High | High |

---

## 1️⃣ SIMPLE HOLD-OUT SPLIT (MOST BASIC, MOST IMPORTANT)

### Core Idea

**Split once, chronologically.**

The simplest and most fundamental approach. Master this first.

---

### 📐 Mathematical Definition

Given series length $N$ and split ratio $r$:

$$t_{\text{split}} = \lfloor r \times N \rfloor$$

- Train: indices $[0, t_{\text{split}})$
- Test: indices $[t_{\text{split}}, N)$

---

### Python Implementation

#### Basic Version

```python
import pandas as pd
import numpy as np

# Simple hold-out split
train_size = int(0.8 * len(df))  # 80% train, 20% test

train = df[:train_size]
test = df[train_size:]

print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")
print(f"Train period: {train.index[0]} to {train.index[-1]}")
print(f"Test period: {test.index[0]} to {test.index[-1]}")
```

#### Complete Function

```python
def simple_train_test_split(df, train_ratio=0.8, target_col='value'):
    """
    Perform simple hold-out split for time series
    
    Parameters:
    -----------
    df : DataFrame with datetime index
    train_ratio : float, proportion for training (default 0.8)
    target_col : str, name of target column
    
    Returns:
    --------
    train, test : DataFrames
    """
    # Ensure data is sorted by time
    df = df.sort_index()
    
    # Calculate split point
    n = len(df)
    train_size = int(train_ratio * n)
    
    # Split
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    # Validation
    assert len(train) > 0, "Training set is empty"
    assert len(test) > 0, "Test set is empty"
    assert train.index[-1] < test.index[0], "Train and test overlap!"
    
    # Summary
    print("="*60)
    print("SIMPLE HOLD-OUT SPLIT")
    print("="*60)
    print(f"Total samples: {n}")
    print(f"Train samples: {len(train)} ({len(train)/n*100:.1f}%)")
    print(f"Test samples: {len(test)} ({len(test)/n*100:.1f}%)")
    print(f"Train period: {train.index[0]} to {train.index[-1]}")
    print(f"Test period: {test.index[0]} to {test.index[-1]}")
    print("="*60)
    
    return train, test

# Usage
train, test = simple_train_test_split(df, train_ratio=0.8)
```

---

### Visual Intuition

```
Data: [════════════════════════════════]
      
Split: |─────────── TRAIN ──────────|──── TEST ────|
       ↑                            ↑              ↑
     Start                       Split          End
     (Past)                    (Decision)     (Future)

Example with dates:
Train: [2020-01-01 ──────────────→ 2023-12-31]
Test:  [2024-01-01 ──────→ 2024-12-31]
```

---

### Visualization Code

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_split(train, test, figsize=(14, 6)):
    """
    Visualize train-test split
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot train
    ax.plot(train.index, train['value'], 
            label='Train', color='blue', linewidth=2, alpha=0.7)
    
    # Plot test
    ax.plot(test.index, test['value'], 
            label='Test', color='red', linewidth=2, alpha=0.7)
    
    # Add vertical line at split
    split_point = test.index[0]
    ax.axvline(x=split_point, color='green', 
               linestyle='--', linewidth=2, label='Split Point')
    
    # Shaded regions
    ax.axvspan(train.index[0], train.index[-1], 
               alpha=0.1, color='blue')
    ax.axvspan(test.index[0], test.index[-1], 
               alpha=0.1, color='red')
    
    # Labels
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Train-Test Split (Hold-out)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage
visualize_split(train, test)
```

---

### When to Use Hold-Out Split

✅ **Baseline models** - Quick sanity checks  
✅ **Quick experiments** - Fast iteration  
✅ **Small datasets** - Not enough data for CV  
✅ **Final evaluation** - After all tuning done  
✅ **Production deployment** - Final train before deployment

---

### Pros and Cons

#### ✅ Pros

- **Simple** - Easy to implement and understand
- **Fast** - No repeated training
- **No leakage** - Clear temporal boundary
- **Low compute** - Train once, test once

#### ❌ Cons

- **Variance** - Performance depends on split point
- **No confidence interval** - Single estimate only
- **Data waste** - Doesn't use all data efficiently
- **Sensitive to outliers** - One bad test period affects results

---

### 📌 Best Practice

> **Always start with hold-out split. It's your baseline.**

Before trying complex cross-validation, ensure your model works on simple hold-out.

![Hold-out Split Visualization](Images(Notes)/train test-02.png)

---

## 2️⃣ ROLLING WINDOW CROSS-VALIDATION (TimeSeriesSplit)

### Core Idea

**Train on a window → test → slide forward → repeat.**

This gives you multiple train-test splits while maintaining temporal order.

![Rolling Window Concept](Images(Notes)/train test-03.jpg)

---

### 📐 Visual Intuition

```
Fold 1: [Train────────] [Test]
Fold 2:     [Train────────] [Test]
Fold 3:         [Train────────] [Test]
Fold 4:             [Train────────] [Test]
Fold 5:                 [Train────────] [Test]

Each fold:
- Uses fixed-size training window
- Tests on next period
- Slides forward in time
```

---

### Python Implementation

#### Basic sklearn Version

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Create time series splitter
tscv = TimeSeriesSplit(n_splits=5)

# Iterate through splits
for fold, (train_idx, test_idx) in enumerate(tscv.split(data), 1):
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    
    print(f"Fold {fold}:")
    print(f"  Train: {train.index[0]} to {train.index[-1]} ({len(train)} samples)")
    print(f"  Test:  {test.index[0]} to {test.index[-1]} ({len(test)} samples)")
    print()
```

---

#### Complete Custom Implementation

```python
def rolling_window_split(df, n_splits=5, test_size=None):
    """
    Perform rolling window cross-validation
    
    Parameters:
    -----------
    df : DataFrame with datetime index
    n_splits : int, number of folds
    test_size : int, size of test set (if None, auto-calculated)
    
    Yields:
    -------
    train, test : DataFrames for each fold
    """
    n = len(df)
    
    if test_size is None:
        test_size = n // (n_splits + 1)
    
    indices = np.arange(n)
    
    for i in range(n_splits):
        # Calculate split points
        test_start = (i + 1) * test_size
        test_end = test_start + test_size
        
        if test_end > n:
            break
            
        # Create indices
        train_idx = indices[:test_start]
        test_idx = indices[test_start:test_end]
        
        # Get data
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        yield i + 1, train, test

# Usage
for fold, train, test in rolling_window_split(df, n_splits=5):
    print(f"Fold {fold}: Train={len(train)}, Test={len(test)}")
```

---

#### Complete Workflow with Evaluation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def rolling_cv_evaluation(df, model, n_splits=5, target_col='value'):
    """
    Complete rolling window cross-validation with evaluation
    
    Parameters:
    -----------
    df : DataFrame
    model : fitted model with fit() and predict() methods
    n_splits : int, number of CV folds
    target_col : str, name of target column
    
    Returns:
    --------
    results : DataFrame with metrics for each fold
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    
    print("="*80)
    print("ROLLING WINDOW CROSS-VALIDATION")
    print("="*80)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), 1):
        # Split data
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        # Prepare features and target
        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]
        X_test = test.drop(columns=[target_col])
        y_test = test[target_col]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Store results
        results.append({
            'Fold': fold,
            'Train_Start': train.index[0],
            'Train_End': train.index[-1],
            'Test_Start': test.index[0],
            'Test_End': test.index[-1],
            'Train_Size': len(train),
            'Test_Size': len(test),
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })
        
        print(f"\nFold {fold}:")
        print(f"  Train: {train.index[0]} to {train.index[-1]}")
        print(f"  Test:  {test.index[0]} to {test.index[-1]}")
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Average RMSE: {results_df['RMSE'].mean():.4f} ± {results_df['RMSE'].std():.4f}")
    print(f"Average MAE:  {results_df['MAE'].mean():.4f} ± {results_df['MAE'].std():.4f}")
    print(f"Average MAPE: {results_df['MAPE'].mean():.2f}% ± {results_df['MAPE'].std():.2f}%")
    print("="*80)
    
    return results_df

# Usage
# results = rolling_cv_evaluation(df, model=my_model, n_splits=5)
```

---

### Visualization

```python
def visualize_rolling_splits(df, n_splits=5, figsize=(14, 10)):
    """
    Visualize rolling window splits
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fig, axes = plt.subplots(n_splits, 1, figsize=figsize, sharex=True)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        ax = axes[fold]
        
        # Plot full data in gray
        ax.plot(df.index, df['value'], color='gray', alpha=0.3, linewidth=1)
        
        # Highlight train
        train = df.iloc[train_idx]
        ax.plot(train.index, train['value'], 
                color='blue', linewidth=2, label='Train')
        
        # Highlight test
        test = df.iloc[test_idx]
        ax.plot(test.index, test['value'], 
                color='red', linewidth=2, label='Test')
        
        # Add split line
        ax.axvline(x=test.index[0], color='green', 
                   linestyle='--', alpha=0.7)
        
        # Labels
        ax.set_ylabel(f'Fold {fold+1}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Time', fontsize=12)
    fig.suptitle('Rolling Window Cross-Validation', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Usage
visualize_rolling_splits(df, n_splits=5)
```

---

### When to Use Rolling Window

✅ **Model comparison** - Compare different algorithms  
✅ **Hyperparameter tuning** - Find best parameters  
✅ **Medium datasets** - Enough data for multiple folds  
✅ **Research validation** - Robust performance estimation  
✅ **Feature selection** - Test feature importance

---

### Pros and Cons

#### ✅ Pros

- **Robust evaluation** - Multiple test periods
- **Better variance estimate** - Standard deviation of metrics
- **Uses more data** - Efficient data utilization
- **Fair comparison** - Same CV for all models

#### ❌ Cons

- **Computationally heavier** - Multiple model trainings
- **Training window size** - Fixed window may not be optimal
- **Still has leakage risk** - If features look ahead

---

### 📌 Best Practice

> **Use rolling window CV for model selection and hyperparameter tuning.**

This gives you confidence that your chosen model generalizes well.

---

## 3️⃣ EXPANDING WINDOW (MOST REALISTIC)

### Core Idea

**Keep all past data, keep adding more.**

This mimics real production systems where you accumulate data over time.

![Expanding Window Concept](Images(Notes)/train test-04.webp)

---

### 📐 Visual Intuition

```
Fold 1: [Train────] [Test]
Fold 2: [Train────────] [Test]
Fold 3: [Train────────────] [Test]
Fold 4: [Train────────────────] [Test]
Fold 5: [Train────────────────────] [Test]

Key difference from rolling:
- Training window GROWS (doesn't slide)
- Uses all historical data
- More realistic for production
```

---

### What It Simulates

| Aspect | Simulation |
|--------|-----------|
| **Real production** | Model retrains with accumulated data |
| **Continuous learning** | Always uses full history |
| **Business reality** | You don't forget old data |
| **Forecasting pipelines** | How models actually operate |

---

### Python Implementation

#### Custom Expanding Window

```python
def expanding_window_split(df, n_splits=5, initial_train_size=None, test_size=None):
    """
    Perform expanding window cross-validation
    
    Parameters:
    -----------
    df : DataFrame with datetime index
    n_splits : int, number of folds
    initial_train_size : int, minimum training size
    test_size : int, size of each test set
    
    Yields:
    -------
    fold, train, test : fold number and DataFrames
    """
    n = len(df)
    
    # Auto-calculate sizes if not provided
    if test_size is None:
        test_size = n // (n_splits + 2)
    
    if initial_train_size is None:
        initial_train_size = n // 2
    
    for i in range(n_splits):
        # Calculate split points
        train_end = initial_train_size + (i * test_size)
        test_start = train_end
        test_end = test_start + test_size
        
        if test_end > n:
            break
        
        # Create splits
        train = df.iloc[:train_end]
        test = df.iloc[test_start:test_end]
        
        yield i + 1, train, test

# Usage
for fold, train, test in expanding_window_split(df, n_splits=5):
    print(f"Fold {fold}:")
    print(f"  Train: {len(train)} samples ({train.index[0]} to {train.index[-1]})")
    print(f"  Test:  {len(test)} samples ({test.index[0]} to {test.index[-1]})")
    print()
```

---

#### Complete Evaluation Function

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def expanding_window_evaluation(df, model, n_splits=5, target_col='value'):
    """
    Complete expanding window cross-validation
    
    Parameters:
    -----------
    df : DataFrame
    model : model object with fit() and predict()
    n_splits : int, number of folds
    target_col : str, target column name
    
    Returns:
    --------
    results : DataFrame with metrics
    """
    results = []
    
    print("="*80)
    print("EXPANDING WINDOW CROSS-VALIDATION")
    print("="*80)
    
    for fold, train, test in expanding_window_split(df, n_splits=n_splits):
        # Prepare data
        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]
        X_test = test.drop(columns=[target_col])
        y_test = test[target_col]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({
            'Fold': fold,
            'Train_Size': len(train),
            'Test_Size': len(test),
            'Train_End': train.index[-1],
            'Test_Period': f"{test.index[0]} to {test.index[-1]}",
            'RMSE': rmse,
            'MAE': mae
        })
        
        print(f"\nFold {fold}:")
        print(f"  Train: {len(train)} samples (growing)")
        print(f"  Test:  {len(test)} samples")
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Average RMSE: {results_df['RMSE'].mean():.4f}")
    print(f"Average MAE:  {results_df['MAE'].mean():.4f}")
    print("="*80)
    
    return results_df

# Usage
# results = expanding_window_evaluation(df, model=my_model, n_splits=5)
```

---

### Visualization

```python
def visualize_expanding_splits(df, n_splits=5, figsize=(14, 10)):
    """
    Visualize expanding window splits
    """
    fig, axes = plt.subplots(n_splits, 1, figsize=figsize, sharex=True)
    
    for fold, train, test in expanding_window_split(df, n_splits=n_splits):
        ax = axes[fold-1]
        
        # Full data in gray
        ax.plot(df.index, df['value'], color='gray', alpha=0.3, linewidth=1)
        
        # Train (expanding)
        ax.plot(train.index, train['value'], 
                color='blue', linewidth=2, label='Train (Expanding)')
        
        # Test
        ax.plot(test.index, test['value'], 
                color='red', linewidth=2, label='Test')
        
        # Split line
        ax.axvline(x=test.index[0], color='green', 
                   linestyle='--', alpha=0.7)
        
        # Labels
        ax.set_ylabel(f'Fold {fold}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Annotate train size
        ax.text(0.02, 0.95, f'Train: {len(train)} samples', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Time', fontsize=12)
    fig.suptitle('Expanding Window Cross-Validation (Growing Training Set)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Usage
visualize_expanding_splits(df, n_splits=5)
```

---

### When to Use Expanding Window

✅ **Forecasting pipelines** - Production-ready evaluation  
✅ **Production-like evaluation** - Simulate real deployment  
✅ **Financial forecasting** - Standard in finance  
✅ **Business forecasting** - Realistic business scenarios  
✅ **Final model validation** - Before deployment

---

### Pros and Cons

#### ✅ Pros

- **Most realistic** - Mirrors production behavior
- **Uses all history** - No data thrown away
- **Production simulation** - Exactly how models operate
- **Accumulates knowledge** - Model improves over time

#### ❌ Cons

- **Computationally expensive** - Training on growing data
- **Longer computation time** - Each fold trains on more data
- **Memory intensive** - Large datasets become problematic

---

### 📌 Critical Insight

> **This mirrors how models live in the real world.**

If your production system retrains monthly with all historical data, use expanding window for evaluation.

---

## 🔮 FORECAST HORIZON MATTERS (CRITICAL)

### Understanding Horizon Types

| Type | Horizon | Example | Use Case |
|------|---------|---------|----------|
| **1-step ahead** | h = 1 | Tomorrow | Short-term decisions |
| **Multi-step** | h = 7 | Next week | Weekly planning |
| **Long-term** | h = 30 | Next month | Strategic planning |

---

### Why Horizon Matters

#### 1. **Errors Grow with Horizon**

```python
# Typical error growth
h=1:  RMSE = 10   # Very accurate
h=7:  RMSE = 25   # Moderate accuracy
h=30: RMSE = 60   # Poor accuracy
```

**Why:** Uncertainty compounds over time.

---

#### 2. **Models Behave Differently**

| Model | 1-Step Performance | Multi-Step Performance |
|-------|-------------------|----------------------|
| **ARIMA** | Excellent | Degrades quickly |
| **ML (XGBoost)** | Good | Needs iterative forecasting |
| **Deep Learning** | Good | Better for longer horizons |

---

#### 3. **Business Impact Varies**

```
Tomorrow's forecast wrong by 10% → Minor issue
Next quarter wrong by 10% → Major business problem
```

---

### Python Implementation

```python
def evaluate_multiple_horizons(df, model, horizons=[1, 7, 14, 30], target_col='value'):
    """
    Evaluate model performance at different forecast horizons
    
    Parameters:
    -----------
    df : DataFrame
    model : forecasting model
    horizons : list of integers, forecast horizons to test
    target_col : str, target column
    
    Returns:
    --------
    results : DataFrame with metrics for each horizon
    """
    results = []
    
    print("="*80)
    print("MULTI-HORIZON EVALUATION")
    print("="*80)
    
    for h in horizons:
        print(f"\n📊 Evaluating horizon h={h}")
        
        # Create lagged features for horizon h
        df_lagged = df.copy()
        df_lagged[f'target_h{h}'] = df_lagged[target_col].shift(-h)
        df_lagged = df_lagged.dropna()
        
        # Split
        train_size = int(0.8 * len(df_lagged))
        train = df_lagged[:train_size]
        test = df_lagged[train_size:]
        
        # Prepare data
        feature_cols = [col for col in df_lagged.columns if col != f'target_h{h}']
        X_train = train[feature_cols]
        y_train = train[f'target_h{h}']
        X_test = test[feature_cols]
        y_test = test[f'target_h{h}']
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results.append({
            'Horizon': h,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })
        
        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    
    results_df = pd.DataFrame(results)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(results_df['Horizon'], results_df['RMSE'], 
                 marker='o', linewidth=2, markersize=8)
    axes[0].set_title('RMSE vs Forecast Horizon', fontweight='bold')
    axes[0].set_xlabel('Horizon (h)')
    axes[0].set_ylabel('RMSE')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(results_df['Horizon'], results_df['MAE'], 
                 marker='o', linewidth=2, markersize=8, color='orange')
    axes[1].set_title('MAE vs Forecast Horizon', fontweight='bold')
    axes[1].set_xlabel('Horizon (h)')
    axes[1].set_ylabel('MAE')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(results_df['Horizon'], results_df['MAPE'], 
                 marker='o', linewidth=2, markersize=8, color='red')
    axes[2].set_title('MAPE vs Forecast Horizon', fontweight='bold')
    axes[2].set_xlabel('Horizon (h)')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results_df

# Usage
# results = evaluate_multiple_horizons(df, model, horizons=[1, 7, 14, 30])
```

---

### 📌 Always Specify Horizon

When reporting model performance:

❌ **Bad:** "My model has 95% accuracy"  
✅ **Good:** "My model has 95% accuracy for 1-day ahead forecasts"

The horizon completely changes what the numbers mean.

---

## 🚨 CRITICAL SPLITTING RULES (NON-NEGOTIABLE)

### The Four Commandments

```
┌─────────────────────────────────────────────┐
│  1. ❌ Never shuffle                        │
│  2. ❌ Never train on future                │
│  3. ✅ Always preserve temporal order       │
│  4. ✅ Always match split to horizon        │
└─────────────────────────────────────────────┘
```

---

### Rule #1: Never Shuffle

```python
# ❌ WRONG
X_train, X_test = train_test_split(data, shuffle=True)

# ✅ CORRECT
X_train, X_test = train_test_split(data, shuffle=False)

# ✅ BETTER
train = data[:split_point]
test = data[split_point:]
```

---

### Rule #2: Never Train on Future

```python
# ❌ WRONG - Test period before train
train = data['2024-01-01':'2024-12-31']
test = data['2023-01-01':'2023-12-31']  # Earlier!

# ✅ CORRECT - Train before test
train = data['2023-01-01':'2023-12-31']
test = data['2024-01-01':'2024-12-31']  # Later!
```

---

### Rule #3: Preserve Temporal Order

```python
# ✅ Validation function
def validate_temporal_order(train, test):
    """
    Ensure train comes before test
    """
    assert train.index[-1] < test.index[0], \
        f"Temporal order violated! Train ends at {train.index[-1]}, Test starts at {test.index[0]}"
    print("✅ Temporal order preserved")

# Usage
validate_temporal_order(train, test)
```

---

### Rule #4: Match Split to Forecast Horizon

```python
# If forecasting 7 days ahead
test_size = 7  # Match your horizon

# If forecasting 30 days ahead
test_size = 30  # Match your horizon

# General rule
test_size = forecast_horizon
```

---

### Rule #5: Same Split for Fair Comparison

```python
# When comparing models
train, test = simple_train_test_split(df)

# Use SAME split for all models
model1_score = evaluate_model(model1, train, test)
model2_score = evaluate_model(model2, train, test)
model3_score = evaluate_model(model3, train, test)

# Now comparison is fair!
```

---

## 📊 COMPLETE IMPLEMENTATION GUIDE

### All-in-One Splitting Framework

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

class TimeSeriesSplitter:
    """
    Complete time series splitting framework
    """
    
    def __init__(self, df, target_col='value'):
        self.df = df.sort_index()
        self.target_col = target_col
        self.n = len(df)
    
    def holdout_split(self, train_ratio=0.8):
        """Simple hold-out split"""
        split_idx = int(train_ratio * self.n)
        train = self.df.iloc[:split_idx]
        test = self.df.iloc[split_idx:]
        return train, test
    
    def rolling_window_splits(self, n_splits=5):
        """Rolling window CV"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        for train_idx, test_idx in tscv.split(self.df):
            train = self.df.iloc[train_idx]
            test = self.df.iloc[test_idx]
            splits.append((train, test))
        return splits
    
    def expanding_window_splits(self, n_splits=5, test_size=None):
        """Expanding window CV"""
        if test_size is None:
            test_size = self.n // (n_splits + 2)
        
        initial_train = self.n // 2
        splits = []
        
        for i in range(n_splits):
            train_end = initial_train + (i * test_size)
            test_end = train_end + test_size
            
            if test_end > self.n:
                break
            
            train = self.df.iloc[:train_end]
            test = self.df.iloc[train_end:test_end]
            splits.append((train, test))
        
        return splits
    
    def visualize_splits(self, method='holdout', **kwargs):
        """Visualize splits"""
        if method == 'holdout':
            train, test = self.holdout_split(**kwargs)
            self._plot_single_split(train, test, 'Hold-out Split')
        
        elif method == 'rolling':
            splits = self.rolling_window_splits(**kwargs)
            self._plot_multiple_splits(splits, 'Rolling Window')
        
        elif method == 'expanding':
            splits = self.expanding_window_splits(**kwargs)
            self._plot_multiple_splits(splits, 'Expanding Window')
    
    def _plot_single_split(self, train, test, title):
        """Plot single train-test split"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(train.index, train[self.target_col], 
                label='Train', color='blue', linewidth=2)
        ax.plot(test.index, test[self.target_col], 
                label='Test', color='red', linewidth=2)
        ax.axvline(x=test.index[0], color='green', 
                   linestyle='--', linewidth=2, label='Split')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel(self.target_col)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_multiple_splits(self, splits, title):
        """Plot multiple splits"""
        n_splits = len(splits)
        fig, axes = plt.subplots(n_splits, 1, figsize=(14, 3*n_splits), sharex=True)
        
        if n_splits == 1:
            axes = [axes]
        
        for i, (train, test) in enumerate(splits):
            ax = axes[i]
            
            ax.plot(self.df.index, self.df[self.target_col], 
                    color='gray', alpha=0.3, linewidth=1)
            ax.plot(train.index, train[self.target_col], 
                    color='blue', linewidth=2, label='Train')
            ax.plot(test.index, test[self.target_col], 
                    color='red', linewidth=2, label='Test')
            ax.axvline(x=test.index[0], color='green', 
                       linestyle='--', alpha=0.7)
            
            ax.set_ylabel(f'Fold {i+1}', fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

# Usage
splitter = TimeSeriesSplitter(df, target_col='value')

# Hold-out
train, test = splitter.holdout_split(train_ratio=0.8)
splitter.visualize_splits(method='holdout')

# Rolling
splits = splitter.rolling_window_splits(n_splits=5)
splitter.visualize_splits(method='rolling', n_splits=5)

# Expanding
splits = splitter.expanding_window_splits(n_splits=5)
splitter.visualize_splits(method='expanding', n_splits=5)
```

---

## 🧪 PRACTICE TASK (DO NOT SKIP)

### The Complete Mastery Exercise

> **This exercise will teach you more than 10 tutorials.**

---

### Task 1: Implement All Three Methods

```python
# 1. Hold-out split
train_ho, test_ho = simple_train_test_split(df, train_ratio=0.8)

# 2. Rolling window
results_rolling = rolling_cv_evaluation(df, model, n_splits=5)

# 3. Expanding window
results_expanding = expanding_window_evaluation(df, model, n_splits=5)
```

---

### Task 2: Compare Model Accuracy Across Splits

```python
# Compare same model on different split methods
accuracies = {
    'Hold-out': evaluate(model, train_ho, test_ho),
    'Rolling': results_rolling['RMSE'].mean(),
    'Expanding': results_expanding['RMSE'].mean()
}

# Plot comparison
import matplotlib.pyplot as plt

plt.bar(accuracies.keys(), accuracies.values())
plt.title('Model Performance: Different Split Methods')
plt.ylabel('RMSE')
plt.show()
```

---

### Task 3: Observe Performance Changes

Answer these questions:

1. **Which method gave the best performance?**
2. **Which method has highest variance?**
3. **Which method is most realistic for production?**
4. **How does performance change with different horizons?**

---

### Task 4: Test Data Leakage

```python
# Intentionally create leakage
X_train_wrong, X_test_wrong = train_test_split(df, shuffle=True)

# Compare to correct split
X_train_correct, X_test_correct = train_test_split(df, shuffle=False)

# Observe accuracy difference
# Wrong split will have artificially high accuracy!
```

---

### 🎓 Expected Learning Outcomes

After completing these tasks:

✅ **Understand** why shuffling breaks time series  
✅ **Can implement** all three splitting methods  
✅ **Can choose** appropriate method for your problem  
✅ **Can detect** data leakage in time series  
✅ **Can explain** to stakeholders why temporal order matters

---

### 📌 Key Insight

> **This builds intuition, not just coding skill.**

You'll develop an instinct for when something "feels wrong" with a time series split.

---

## 🚨 COMMON MISTAKES AND SOLUTIONS

### Mistake #1: Using `shuffle=True`

**The Problem:**
```python
# ❌ WRONG
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(data, test_size=0.2, shuffle=True)
```

**The Fix:**
```python
# ✅ CORRECT
split_point = int(0.8 * len(data))
X_train = data[:split_point]
X_test = data[split_point:]
```

---

### Mistake #2: Test Period Before Train Period

**The Problem:**
```python
# ❌ WRONG - Dates out of order
train = df['2024-01-01':'2024-06-30']
test = df['2023-01-01':'2023-12-31']  # Earlier than train!
```

**The Fix:**
```python
# ✅ CORRECT
train = df['2023-01-01':'2023-12-31']
test = df['2024-01-01':'2024-06-30']  # After train
```

---

### Mistake #3: Using Same Test Set for Multiple Experiments

**The Problem:**
```python
# ❌ WRONG - Test set contamination
# Tune on test set
model1 = tune_model(train, test)  # Using test for tuning
model2 = tune_model(train, test)  # Using test again
final_accuracy = evaluate(best_model, test)  # Not valid!
```

**The Fix:**
```python
# ✅ CORRECT - Use validation set
train, val, test = split_three_way(df)

# Tune on validation
model1 = tune_model(train, val)
model2 = tune_model(train, val)

# Final evaluation on test (once!)
final_accuracy = evaluate(best_model, test)
```

---

### Mistake #4: Ignoring Forecast Horizon

**The Problem:**
```python
# ❌ WRONG - Test size doesn't match horizon
forecast_horizon = 30  # Want to forecast 30 days
test_size = 100  # But using 100 days for testing
```

**The Fix:**
```python
# ✅ CORRECT - Match test size to horizon
forecast_horizon = 30
test_size = forecast_horizon  # Same size
```

---

### Mistake #5: Not Validating Temporal Order

**The Problem:**
```python
# ❌ WRONG - No validation
train, test = some_split_function(df)
# Assume it's correct... but is it?
```

**The Fix:**
```python
# ✅ CORRECT - Always validate
train, test = some_split_function(df)

# Validate temporal order
assert train.index[-1] < test.index[0], "Temporal order violated!"
print(f"✅ Train ends: {train.index[-1]}")
print(f"✅ Test starts: {test.index[0]}")
```

---

## 🧠 FINAL MEMORY SUMMARY

### The One-Page Cheat Sheet

#### Core Principle
```
Train on PAST → Test on FUTURE
NEVER shuffle. ALWAYS preserve order.
```

---

#### Three Methods

| Method | When to Use | Key Feature |
|--------|-------------|-------------|
| **Hold-out** | Quick tests, baselines | Simple, fast |
| **Rolling** | Model comparison, tuning | Fixed window |
| **Expanding** | Production simulation | Growing window |

---

#### Critical Rules

```
✅ DO:
- Split chronologically
- Preserve temporal order
- Match test size to forecast horizon
- Validate splits before using

❌ DON'T:
- Use shuffle=True
- Train on future data
- Ignore forecast horizon
- Use test set for tuning
```

---

#### Decision Tree

```
What's your goal?
│
├─ Quick baseline → Hold-out split
│
├─ Compare models → Rolling window CV
│
└─ Production-ready → Expanding window CV
```

---

### 🎯 The Ultimate Truth

> **If you shuffle time series data, every result is a lie.**

No exceptions. No "but my data is different." No shortcuts.

---

## 📚 Additional Resources

### Python Libraries
- `sklearn.model_selection.TimeSeriesSplit` - Rolling CV
- `pandas` - Time series handling
- `numpy` - Array operations

### Further Reading
- "Forecasting: Principles and Practice" (Chapter 3.4)
- "Time Series Analysis" by James Hamilton

### Practice Datasets
- Stock prices (Yahoo Finance)
- Weather data (NOAA)
- Sales data (Kaggle)

---

**Last Updated:** January 2026  
**Version:** 1.0  
**Mastery Level:** Fundamental (Must-Know)

---

*Remember: Time only flows forward. Your splits should too.* ⏰
