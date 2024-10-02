import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class KNN:
    k = 1
    x_train = 0
    y_train = 0

    def __init__(self, kval):
        self.k = kval

    def fit(self,xt,yt):
        self.x_train = xt
        self.y_train = yt
    
    def predict(self,xpt,ypt):
        x = np.array(self.x_train.iloc[:, 0])
        y = np.array(self.x_train.iloc[:, 1])
        label = np.array(self.y_train)
        d = np.sqrt((x - xpt) ** 2 + (y - ypt) ** 2)
        d2 = d.argsort()
        idx2 = d2[0:self.k]
        n0 = np.sum(label[idx2] == 0)
        n1 = np.sum(label[idx2] == 1)
        if n0 > n1:
            return 0
        else:
            return 1

df = pd.read_csv("../datasets/exams.csv",skiprows=0,delimiter=",")
X = df.iloc[:, 0:2]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

knn = KNN(3)
knn.fit(X_train,y_train)
ypred = knn.predict(50,60)

print(ypred)