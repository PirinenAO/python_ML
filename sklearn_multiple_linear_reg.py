import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes(as_frame=True)
print(data.DESCR)

df = data.frame

x = pd.DataFrame(df[['bmi','s5']], columns = ['bmi','s5'])
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)
lm = LinearRegression()
lm.fit(x_train, y_train)


# train data evaluation
y_train_predict = lm.predict(x_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

# test data evaluation
y_test_predict = lm.predict(x_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)

print(rmse,r2)
print(rmse_test,r2_test)
