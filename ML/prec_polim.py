import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_csv('./assets/fuel_speed.csv')
X = data[['speed_kmh']]
Y = data[['fuel_consumption_l_per_100km']]

degree = 6
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

model.fit(X, Y)

X_test = np.array([10, 38, 40, 60, 75, 93, 100]).reshape(-1,1)
Y_test = np.array([10, 8.3, 7.2, 6, 7, 7.5, 7.9])

y_pred = model.predict(X_test)

plt.figure(figsize=(10,6))
plt.plot(X_test, Y_test, label='My function', color='blue')
plt.plot(X_test, y_pred, label='Predicted function (Poly degree 3)', color='red', linestyle='--')
plt.xlabel('kph')
plt.ylabel('fuel_consumption')
plt.title('GGG')
plt.legend()
plt.grid(True)
plt.show()