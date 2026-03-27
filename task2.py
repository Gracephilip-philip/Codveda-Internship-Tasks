import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Load the data
df = pd.read_csv('sentiment_data.csv', sep=None, engine='python')

# 2. Data Preparation
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')
df.set_index('Timestamp', inplace=True)

# 3. Moving Average
df['Smooth_Likes'] = df['Likes'].rolling(window=3).mean()

# 4. Decomposition
result = seasonal_decompose(df['Likes'].fillna(0), model='additive', period=7)

# --- THE CHART SECTION (This makes them appear) ---
plt.figure(figsize=(10, 8))

# Top Graph: Original vs Smoothed
plt.subplot(4,1,1)
plt.plot(df['Likes'], label='Original', alpha=0.3)
plt.plot(df['Smooth_Likes'], label='Smoothed', color='red')
plt.title('Codveda Level 2: Time Series Analysis')
plt.legend()

# Trend Graph
plt.subplot(4,1,2)
plt.plot(result.trend, color='orange')
plt.ylabel('Trend')

# Seasonality Graph
plt.subplot(4,1,3)
plt.plot(result.seasonal, color='green')
plt.ylabel('Seasonality')

# Noise (Residuals) Graph
plt.subplot(4,1,4)
plt.plot(result.resid, color='gray', marker='o', linestyle='None')
plt.ylabel('Noise')

plt.tight_layout()
plt.show() # <--- CRITICAL: This line opens the window!