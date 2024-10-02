import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# retrieve data
data = load_diabetes(as_frame=True)
df = data.frame
x = pd.DataFrame(df[['bmi','s5', 'bp', 's3']], columns = ['bmi','s5', 'bp', 's3'])
#x = pd.DataFrame(df[['bmi','s5', 'bp']], columns = ['bmi','s5', 'bp'])
#x = pd.DataFrame(df[['bmi','s5']], columns = ['bmi','s5'])

# display correlation map
#sns.heatmap(data=df.corr().round(2), annot=True)
#plt.show()

y = df['target']

# split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

# create and train the model
lm = LinearRegression()
lm.fit(x_train, y_train)

# training data evaluation
y_train_predict = lm.predict(x_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

# testing data evaluation
y_test_predict = lm.predict(x_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)

print("Training data evaluation\n" + "RMSE: " +str(rmse) + "\nR2: " + str(r2))
print("\nTesting data evaluation \n" + "RMSE: " +str(rmse_test) + "\nR2: " + str(r2_test))

