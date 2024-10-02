import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("../datasets/quadreg_data.csv",skiprows=0,names=["x","y"])
xpd = np.array(data[["x"]])
ypd = np.array(data[["y"]])
xpd = xpd.reshape(-1,1)
ypd = ypd.reshape(-1,1)

poly_reg = PolynomialFeatures(degree=2) # generate polynomial and interaction features
X_poly = poly_reg.fit_transform(xpd) # transform xpd array to one  where each column contains different powers of the original features
pol_reg = LinearRegression() # will be used to fit the polynomial features generated earlier
pol_reg.fit(X_poly, ypd) # fits the linear regression model to the transformed data

plt.scatter(xpd, ypd)
x = np.linspace(-1,1,10).reshape(-1,1)
y = pol_reg.predict(poly_reg.fit_transform(x))
plt.plot(x, pol_reg.predict(poly_reg.fit_transform(x)), color='black')
plt.show()