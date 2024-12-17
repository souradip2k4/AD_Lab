import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path = 'housing.csv'
df = pd.read_csv(file_path)

data = df[['total_rooms', 'median_house_value']].copy()

data = data.dropna()

X = data[['total_rooms']]
y = data['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Slope (Coefficient): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='green', label='Actual Prices', alpha=0.5)
plt.plot(X_test, y_pred, color='orange', linewidth=2, label='Regression Line')


plt.title('Simple Linear Regression: Total Rooms vs Median House Value')
plt.xlabel('Total Rooms (Area)')
plt.ylabel('Median House Value (Price)')
plt.legend()
plt.savefig("housing.png")
plt.show()

print("\nSouradip Saha")
print("22052939")