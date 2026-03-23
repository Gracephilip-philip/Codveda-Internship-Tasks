import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('stock_price.csv')

# Convert date to datetime objects
df['date'] = pd.to_datetime(df['date'])

# --- OBJECTIVE 1: Summary Statistics ---
print("--- Summary Statistics ---")
print(df.describe())

print(f"\nMedian Price: {df['close'].median()}")
print(f"Most Frequent Stock: {df['symbol'].mode()[0]}")

# --- OBJECTIVE 2: Visualize Data ---
# This will open a window with your charts
plt.figure(figsize=(10, 6))
sns.histplot(df['close'], bins=30, kde=True)
plt.title('Stock Closing Price Distribution')
plt.show()