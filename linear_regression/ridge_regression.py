from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

df = pd.read_csv("../datasets/ridgereg_data.csv")
x = df[['x']]
y = df[['y']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

alphas = np.linspace(0,4,50) # past 4 the score decreases
r2values = []
for alp in alphas:
    rr = Ridge(alpha=alp)
    rr.fit(X_train, y_train)
    r2_test = r2_score(y_test, rr.predict(X_test))
    r2values.append(r2_test)

plt.plot(alphas,r2values)
plt.show()