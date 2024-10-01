import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# retrieve data from csv
df = pd.read_csv("./datasets/bank.csv", delimiter=';')

# create dataframe with desired variables
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

# convert string values to numerical values
df3 = pd.get_dummies(df2,columns=['job','marital','default','housing','poutcome'])
df3['y'] = df3['y'].map({'yes': 1, 'no': 0})

# explanatory and target variables
x = df3.drop(columns=['y'])
y = df3['y']

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

# Logical regression
model = LogisticRegression()
model.fit(x_train, y_train)

# confusion matrix
y_pred_log = model.predict(x_test)
cnf_matrix_log = metrics.confusion_matrix(y_test, y_pred_log)
accuracy_log = accuracy_score(y_test, y_pred_log)
print("Logistic regression cnf matrix and accuracy: ")
print(cnf_matrix_log)
print(accuracy_log)


# KNN
kvalue = 3
knn = KNeighborsClassifier(n_neighbors=kvalue)
knn.fit(x_train, y_train)

# confusion matrix
y_pred_knn = knn.predict(x_test)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("\nKNN cnf matrix and accuracy: ")
print("K = " + str(kvalue))
print(conf_matrix_knn)
print(accuracy_knn)

