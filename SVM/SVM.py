import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

df2 = pd.read_csv("../datasets/iris.csv")

x = df2.drop('species', axis=1)
y = df2['species']

x_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=20)

# linear kernel
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)
y_pred = svclassifier.predict(X_test)

"""
# second degree polynomial kernel function
svclassifier = SVC(kernel='poly', degree=2)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
"""

"""
# radial basis kernel function
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
"""


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
