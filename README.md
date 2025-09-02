# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
# Date:01.09.2025
# Name: Sarish Varshan V
# Reg No: 212223230196
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
A - LINEAR TREND ESTIMATION
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset (with space in file name)
df = pd.read_excel("Data_Train.xlsx")

# Convert journey date to datetime
df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")

# Group by date and calculate mean price
daily_price = df.groupby("Date_of_Journey")["Price"].mean().reset_index()

# Prepare X (time in ordinal numbers) and y
X = daily_price["Date_of_Journey"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
y = daily_price["Price"].values

# ------------------ A. Linear Trend ------------------
linear_model = LinearRegression()
linear_model.fit(X, y)
daily_price["Linear_Trend"] = linear_model.predict(X)

plt.figure(figsize=(12,6))
plt.plot(daily_price["Date_of_Journey"], y, marker="o", label="Original Data", alpha=0.6)
plt.plot(daily_price["Date_of_Journey"], daily_price["Linear_Trend"], color="orange", label="Linear Trend")
plt.title("Linear Trend Estimation - Flight Price")
plt.xlabel("Date of Journey")
plt.ylabel("Average Price")
plt.legend()
plt.grid(True)
plt.show()
```
B- POLYNOMIAL TREND ESTIMATION
```
# ------------------ B. Polynomial Trend ------------------
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
daily_price["Poly_Trend"] = poly_model.predict(X_poly)

plt.figure(figsize=(12,6))
plt.plot(daily_price["Date_of_Journey"], y, marker="o", alpha=0.6, label="Original Data")
plt.plot(daily_price["Date_of_Journey"], daily_price["Poly_Trend"], color="red", label="Polynomial Trend (Degree 2)")
plt.title("Polynomial Trend Estimation - Flight Price")
plt.xlabel("Date of Journey")
plt.ylabel("Average Price")
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT
A - LINEAR TREND ESTIMATION

<img width="886" height="431" alt="Screenshot 2025-09-01 151405" src="https://github.com/user-attachments/assets/0aa49f69-44b5-494d-8953-e05fbb8ca9dc" />



B- POLYNOMIAL TREND ESTIMATION

<img width="885" height="466" alt="Screenshot 2025-09-01 151415" src="https://github.com/user-attachments/assets/16f5be6f-8b41-4afe-b6f8-1c12d230a25e" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
