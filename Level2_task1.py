import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
# We define the column names because the raw data doesn't have them
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']
df = pd.read_csv('house_prediction.csv', header=None, names=columns, sep=r'\s+')

# 2. Select Features
# We are predicting 'PRICE' based on 'RM' (Average number of rooms)
X = df[['RM']] 
y = df['PRICE']

# 3. Split the data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred = model.predict(X_test)
print(f"Model Accuracy (R-squared): {r2_score(y_test, y_pred):.2f}")

# 6. Create the Visualization
plt.figure(figsize=(10,6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prediction Line')
plt.title('Level 2: House Price Regression (Rooms vs Price)')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Price')
plt.legend()
plt.savefig('level2_regression_plot.png')
plt.show()