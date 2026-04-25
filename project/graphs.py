"""
EARTHQUAKE HAZARD MODELING & RESEARCH DEMO
⚠️  DOES NOT PREDICT EARTHQUAKES. Demonstrates:
   1. Probabilistic forecasting (Gutenberg-Richter + Poisson)
   2. Educational ML pattern exploration on synthetic catalogs
For operational use, consult USGS, GEM, or national seismic agencies.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ──────────────────────────────────────────────────────────────
# 1. PROBABILISTIC FORECASTING (Standard PSHA Approach)
# ──────────────────────────────────────────────────────────────
def gutenberg_richter_probability(m_threshold, a_val=4.5, b_val=1.0, years=10):
    """
    Calculate probability of ≥1 earthquake >= M_threshold in 'years'.
    Based on Gutenberg-Richter law: log10(N) = a - b*M
    Uses Poisson process (standard in Probabilistic Seismic Hazard Analysis).
    """
    log_n = a_val - b_val * m_threshold
    annual_rate = 10**log_n
    expected_count = annual_rate * years
    prob_at_least_one = 1 - poisson.pmf(0, expected_count)
    return prob_at_least_one, annual_rate

# Example: Regional hazard assessment
m_thresh = 6.0
prob, rate = gutenberg_richter_probability(m_thresh, a_val=4.8, b_val=1.0, years=30)
print(f"📊 Prob of M≥{m_thresh} in next 30 years: {prob:.1%}")
print(f"   Annual occurrence rate: {rate:.3f} events/year")

# ──────────────────────────────────────────────────────────────
# 2. EDUCATIONAL ML: Pattern Recognition on Synthetic Catalog
# ──────────────────────────────────────────────────────────────
"""
⚠️  This is a RESEARCH DEMO ONLY. 
Real seismic ML (e.g., lab acoustic emission studies) does NOT 
translate to real-world prediction. Accuracy here reflects synthetic 
data properties, not Earth physics.
"""
np.random.seed(42)

# Generate synthetic seismic catalog (replace with real USGS data in practice)
n_events = 2000
magnitudes = np.random.exponential(scale=0.8, size=n_events) + 2.0  # M≥2
magnitudes = np.clip(magnitudes, 2.0, 8.5)
inter_event_days = np.random.exponential(scale=15, size=n_events)  # ~15 day avg
cumulative_days = np.cumsum(inter_event_days)

df = pd.DataFrame({'magnitude': magnitudes, 'day': cumulative_days})

# Rolling features (research-style temporal statistics)
df['ma_3'] = df['magnitude'].rolling(3, min_periods=1).mean()
df['std_3'] = df['magnitude'].rolling(3, min_periods=1).std().fillna(0)
df['rate_3'] = 3.0 / df['day'].diff().rolling(3, min_periods=1).mean().fillna(999)

# Target: Will the NEXT event be ≥ M5.0? (NOT prediction, just classification demo)
df['next_large'] = (df['magnitude'].shift(-1) >= 5.0).astype(int)
df = df.dropna()

X = df[['magnitude', 'ma_3', 'std_3', 'rate_3']]
y = df['next_large']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n🤖 ML Classification Report (Synthetic Data Only):")
print(classification_report(y_test, y_pred, zero_division=0))
print("⚠️  Real earthquakes do not follow these synthetic patterns. This code is for educational exploration only.")

# ──────────────────────────────────────────────────────────────
# 3. VISUALIZATION (Hazard Curve)
# ──────────────────────────────────────────────────────────────
mag_range = np.arange(2.0, 8.5, 0.1)
probs = [gutenberg_richter_probability(m, a_val=4.8, b_val=1.0, years=30)[0] for m in mag_range]

plt.figure(figsize=(8, 4))
plt.plot(mag_range, probs, marker='o', markersize=4, linestyle='-')
plt.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='50% probability')
plt.xlabel('Earthquake Magnitude')
plt.ylabel('Probability of ≥1 Event in 30 Years')
plt.title('Probabilistic Seismic Hazard Curve (Demonstration)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()