import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# retrieve the data from csv file
df = pd.read_csv("./datasets/50_Startups.csv", delimiter=",")
x = df[['R&D Spend', 'Marketing Spend']]
y = df['Profit']

# split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

# create and train the model
lm = LinearRegression()
lm.fit(x_train, y_train)

# MODEL EVALUATION
# training data evaluation
y_train_predict = lm.predict(x_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)
print("Training data evaluation\n" + "RMSE: " +str(rmse) + "\nR2: " + str(r2))

# testing data evaluation
y_test_predict = lm.predict(x_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)
print("\nTesting data evaluation \n" + "RMSE: " +str(rmse_test) + "\nR2: " + str(r2_test))

# MAKING PREDICTIONS ON NEW DATA
"""
new_data = pd.DataFrame({
    'R&D Spend': [200000],  # Example values
    'Marketing Spend': [50000]  # Example values
})

# Predicting for the new data
new_predictions = lm.predict(new_data)
print(new_predictions)
"""

