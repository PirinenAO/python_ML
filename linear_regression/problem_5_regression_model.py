"""
Consider the data from the file weight-height.csv.

1) Inspect the dependence between height and weight using a scatter plot. You may use either of the variables as independent variable.

2) Choose appropriate model for the dependence

3) Perform regression on the data using your model of choice

4) Plot the results

5) Compute RMSE and R2 value

6) Assess the quality of the regression (visually and using numbers) in your own words.

You are not required to split the dataset into training and testing sets. Of course you are completely free to experiment it here already.

It is recommended that you use the module sklearn for all your computations.

"""
import numpy as np
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt

# retrieve data from file
data = np.genfromtxt("../datasets/weight-height.csv", delimiter=",", skip_header=1)
# parse the data to variables and reshape them into 1x1 matrices
x = data[:,1]
y = data[:,2]
x = x.reshape(-1,1)
y = y.reshape(-1,1)

# unit conversions
x = x * 2.54        # inches to cm
y = y * 0.453592    # pounds to kg

# initialize and train the linear regression model
regr = linear_model.LinearRegression()
regr.fit(x, y)

# generate x values and calculate the corresponding y values 
xval = np.linspace(130,210,100).reshape(-1,1)
yval = regr.predict(xval)

# model evaluation
yhat = regr.predict(x)
RMSE = np.sqrt(metrics.mean_squared_error(y, yhat))
R2 = metrics.r2_score(y, yhat)
print(("RMSE: " + str(RMSE) + "\nR2: "+ str(R2)))

# plot the data and the regression line
plt.plot(xval,yval, color="red")    # plots the regression line
plt.scatter(x,y)                    # plots the data
plt.title("Relationship Between Height and Weight")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.show()