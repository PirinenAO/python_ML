import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# READ DATA FROM CSV FILE
data = pd.read_csv("../datasets/linreg_data.csv", skiprows=0, names=["x", "y"])
xpd = data["x"]
ypd = data["y"]

# CALCULATIONS

# x and y means
xbar = np.mean(xpd)
ybar = np.mean(ypd)

# number of values x values
n = xpd.size

# terms for b's formula
term1 = np.sum(xpd*ypd)
term2 = np.sum(xpd**2)

# calculating b and b
b = (term1-n*xbar*ybar)/(term2-n*xbar*xbar)
a = ybar - b*xbar

# for drawing the regression line
x = np.linspace(0,2,100)
y = a+b*x

# predicting multiple y values
xval = np.array([0.5,0.75,0.90])
yval = a+b*xval

# MODEL EVALUAION
yhat = a+b*xpd
RSS = np.sum((ypd-yhat)**2)
print("RSS =",RSS)
RMSE = np.sqrt(np.sum((ypd-yhat)**2)/n)
print("RMSE=",RMSE)
MAE = np.sum(np.abs(ypd-yhat))/n
print("MAE =",MAE)
MSE = np.sum((ypd-yhat)**2)/n
print("MSE =",MSE)
R2 = 1-np.sum((ypd-yhat)**2)/np.sum((ypd-ybar)**2)
print("R2  =",R2)

# DISPLAYING DATA
plt.plot(x,y,color="black")
plt.scatter(xpd,ypd)
plt.scatter(xbar,ybar,color="red")
plt.scatter(xval,yval,color="orange")
plt.show()
