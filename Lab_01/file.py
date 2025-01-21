import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('IRIS.csv')

print("Original Dataset:\n", df.head())

print("\nMissing values before imputation:\n", df.isnull().sum())

for col in df.select_dtypes(include=[np.number]):
  mean_value = df[col].mean()
fill_values = {col: mean_value for col in df.select_dtypes(include=[np.number])}
df.fillna(fill_values, inplace=True)

print("\nMissing values after imputation:\n", df.isnull().sum())

label_encoders = {}
for col in df.select_dtypes(include=[object]):
  label_encoders[col] = LabelEncoder()
  df[col] = label_encoders[col].fit_transform(df[col])

print("\nDataset after encoding categorical data:\n", df.head())

scaler = StandardScaler()
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("\nDataset after feature scaling:\n", df.head())

first_numeric_col = numeric_columns[0]
plt.figure(figsize=(8, 6))
plt.hist(df[first_numeric_col], bins=20, color='blue', edgecolor='black')
plt.title(f"Distribution of {first_numeric_col} (scaled)")
plt.xlabel(f"{first_numeric_col} (scaled)")
plt.ylabel("Frequency")
plt.savefig('sepal_length_distribution.png')
plt.show()

if len(numeric_columns) > 1:
  plt.figure(figsize=(8, 6))
  sns.scatterplot(x=df[numeric_columns[0]], y=df[numeric_columns[1]])
  plt.title(f"{numeric_columns[0]} vs {numeric_columns[1]}")
  plt.xlabel(numeric_columns[0])
  plt.ylabel(numeric_columns[1])
  plt.savefig('scatter_plot.png')
  plt.show()

plt.figure(figsize=(8, 6))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig('heatmap.png')
plt.show()

print("\nSouradip Saha")
print("22052939")


