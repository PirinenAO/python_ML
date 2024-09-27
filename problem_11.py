import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import linear_model


# retrieve the data from csv file
df = pd.read_csv("./datasets/Auto.csv", delimiter=",")
x = df[['cylinders','displacement','horsepower','weight','acceleration','year']]
y = df['mpg']

# split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

# ridge regression
alphas = np.linspace(80,100,1000) 
r2values_ridge = []

for alp in alphas:
    rr = Ridge(alpha=alp)
    rr.fit(x_train, y_train)
    y_train_predict = rr.predict(x_train)
    r2_test = r2_score(y_test, rr.predict(x_test))
    r2values_ridge.append(r2_test)

# print the highest r2 value and its alpha
max_index = np.argmax(r2values_ridge)
print(r2values_ridge[max_index], alphas[max_index])

plt.plot(alphas,r2values_ridge)
plt.show()

# lasso regression
alphas = np.linspace(0,1,1000)
r2values_lasso = []

for alp in alphas:
    lasso = linear_model.Lasso(alpha=alp)
    lasso.fit(x_train, y_train)
    sc = lasso.score(x_test, y_test)
    r2values_lasso.append(sc)

# print the highest r2 value and its alpha value
max_index = np.argmax(r2values_lasso)
print(r2values_lasso[max_index], alphas[max_index])

plt.plot(alphas,r2values_lasso)
plt.show()


