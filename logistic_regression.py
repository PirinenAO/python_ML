import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns

data = pd.read_csv("./datasets/exams.csv", skiprows=0, delimiter=",")

x = data.iloc[:, 0:2]
y = data.iloc[:, -1]

admit_yes = data.loc[y == 1]
admit_no = data.loc[y == 0]

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=0)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))

metrics.ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
plt.show()