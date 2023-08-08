#Random Forest Regression
#Pre-Processing Template
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import DataSet
dataset = pd.read_csv("/content/drive/MyDrive/Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fit the Regression with the data set
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators = 300, random_state = 0)
regression.fit(x, y)

#Predict with the Modells
y_pred = regression.predict([[6.5]])


#Visualize the results of the LPR
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = "red")
plt.plot(x_grid, regression.predict(x_grid), color = "blue")
plt.title("Model for Regression")
plt.xlabel("Posicion del Empleado")
plt.ylabel("Income (en $)")
plt.show()

y_pred
