# HOUSE PRICE PREDICTION 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA

data = pd.read_csv("kc_house_data[1].csv", encoding='latin1')

print("Dataset Loaded ")
print(data.head())

# 2. DATA CLEANING

print("\nColumns:\n", data.columns)
print("\nInfo:\n")
print(data.info())

# Remove missing values (if any)

data = data.dropna()

# 3. REMOVE OUTLIERS

# Remove top 1% expensive houses
data = data[data['price'] < data['price'].quantile(0.99)]

# 4. FEATURE SELECTION

features = [
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'floors',
    'sqft_lot',
    'grade',
    'condition'
]

X = data[features]
y = data['price']

# 5. TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 7. TRAIN MODEL


model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Trained ")

# 8. PREDICTIONS

y_pred = model.predict(X_test)


# 9. EVALUATION

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance ")
print("MSE:", mse)
print("R² Score:", r2)

# 10. COEFFICIENTS

print("\nModel Details ")
print("Intercept:", model.intercept_)

for name, coef in zip(features, model.coef_):
    print(f"{name}: {coef}")

# 11. VISUALIZATION

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.75)

# perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# 12. RESIDUAL PLOT (ADVANCED)

residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.756)
plt.axhline(y=0, linestyle='--')

plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# 13. SAVE MODEL

joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel Saved ")


# 14. USER INPUT PREDICTION

print("\n Predict House Price")

bed = int(input("Bedrooms (>1): "))
bath = float(input("Bathrooms (>1): "))
sqft = int(input("Sqft Living (>0): "))
floor = float(input("Floors (>0): "))
lot = int(input("Sqft Lot (>0): "))
grade = int(input("Grade (1-13): "))
condition = int(input("Condition (1-5): "))



# VALIDATION

if (bed < 1 or bath < 1 or sqft <= 0 or floor < 1 or 
    lot <= 0 or grade < 1 or grade > 13 or 
    condition < 1 or condition > 5):
    
    print("\n❌ Invalid input! Please enter realistic values.")
else:
    new_house = [[bed, bath, sqft, floor, lot, grade, condition]]
    
    new_house_scaled = scaler.transform(new_house)
    price = model.predict(new_house_scaled)[0]

    print(f"\n🏠 Predicted Price: ₹{price:,.2f}")
