import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('./datasets/Admission_Predict.csv',skiprows=0,delimiter=",")

# parse the CGPA and Chance of Admit columns
x = data[['CGPA']]
y = data[['Chance of Admit ']]

# split the data into training data and testing data
# testing data will be 20% of data and the rest 80% will be used for training
# splitting is random so it does not follow any
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# train (fit) the model using training data only
lm = linear_model.LinearRegression()
model = lm.fit(x_train, y_train)

print("R2=",lm.score(x_test,y_test)) # using linear model

# display the data
plt.legend(["train","test"])
plt.xlabel("CGPA")
plt.ylabel("Chance of Admit")
plt.title("Prediction")

plt.scatter(x,y)
plt.plot(x_test,lm.predict(x_test),color="red")
plt.show()