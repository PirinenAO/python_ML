from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# 0) Read the data into a pandas dataframe.
df = pd.read_csv('../datasets/suv.csv',skiprows=0,delimiter=",")

# 1) Pick Age and Estimated Salary as the features and Purchased as the target variable.
x = df[['Age','EstimatedSalary']]
y = df[["Purchased"]]

# 2) Split the data into training and testing sets with 80/20 ratio.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=30)

# 3) Scale the features using standard scaler.
x_train_std = StandardScaler().fit_transform(x_train)
x_test_std = StandardScaler().fit_transform(x_test)

# 4) Train a decision tree classifier with entropy criterion and predict on test set.
classifier_entropy = DecisionTreeClassifier(criterion="entropy",max_depth=4)
classifier_entropy.fit(x_train_std, y_train)
y_pred = classifier_entropy.predict(x_test_std)

# 5) Print the confusion matrix and the classification report.
print("Model evaluation with entropy criterion")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6) Repeat steps 4 and 5 with the gini criterion.
classifier_gini = DecisionTreeClassifier()
classifier_gini.fit(x_train_std, y_train)
y_pred_gini = classifier_gini.predict(x_test_std)

print("Model evaluation with gini index criterion")
print(confusion_matrix(y_test, y_pred_gini))
print(classification_report(y_test, y_pred_gini))