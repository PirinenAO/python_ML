import numpy as np
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt

# READING DATA AND CONVERTING TO MATRICES
data = np.genfromtxt("./datasets/linreg_data.csv", delimiter=",")
xp = data[:,0]
yp = data[:,1]
xp = xp.reshape(-1,1)
yp = yp.reshape(-1,1)

# CREATE AND TRAIN MODEL
regr = linear_model.LinearRegression()
regr.fit(xp, yp)
# regr.coef_ == b , regr.intercept_ == a
print(regr.coef_,regr.intercept_)

# PREDICTING
xval = np.full((1,1),0.5)   # creates 1x1 matrix with single element 0,5
yval = regr.predict(xval)
print(yval)

# PLOTTING THE DATA
xval = np.linspace(-1,2,20).reshape(-1,1)
yval = regr.predict(xval)
plt.plot(xval,yval, color="black") # this plots the regression line
plt.scatter(xp,yp)
plt.show()

# MODEL EVALUATION
yhat = regr.predict(xp)
print('Mean Absolute Error:', metrics.mean_absolute_error(yp, yhat))  
print('Mean Squared Error:', metrics.mean_squared_error(yp, yhat))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yp, yhat)))
print('R2 value:', metrics.r2_score(yp, yhat))