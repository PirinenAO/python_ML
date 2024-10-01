import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 0) Read data into a pandas dataframe.
df = pd.read_csv("./datasets/data_banknote_authentication.csv")

# 1) Pick the column named "class" as target variable y and all other columns as feature variables X.
x = df.drop('class', axis=1)
y = df['class']

# 2) Split the data into training and testing sets with 80/20 ratio and random_state=20.
x_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=20)

# 3) Use support vector classifier with linear kernel to fit to the training data.
svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)

# 4) Predict on the testing data and compute the confusion matrix and classification report.
y_pred = svclassifier.predict(X_test)
print("Linear kernel function evaluation")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# 5) Repeat steps 3 and 4 for the radial basis function kernel.
svclassifier_rbf = SVC(kernel='rbf')
svclassifier_rbf.fit(x_train, y_train)
y_pred_rbf = svclassifier_rbf.predict(X_test)
print("Radial basis function evaluation")
print(confusion_matrix(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))


