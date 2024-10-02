from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("../datasets/emails.csv")

X = df["text"]
y = df["spam"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=10)

vect = CountVectorizer(stop_words="english")
vect.fit(X_train)
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)

model = svm.SVC()
model.fit(X_train_df,y_train)
y_pred = model.predict(X_test_df)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred,target_names=["not spam","spam"]))


"""
FOR TESTING A CUSTOM EMAIL

new_data = pd.DataFrame({
    'text': ['Subject: <YOUR SUBJECT HERE ']
})

new_data_transformed = vect.transform(new_data['text'])
new_prediction = model.predict(new_data_transformed)
print(new_prediction) # 0 == no spam, 1 == spam

"""



