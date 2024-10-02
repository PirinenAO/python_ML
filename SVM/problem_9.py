import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# 0) Read data into a pandas dataframe.
df = pd.read_csv("../datasets/weight-height.csv")

# 1) Pick the target variable y as weight in kilograms, and the feature variable X as height in centimeters.
y = df['Weight']
x = df['Height'].values.reshape(-1, 1) # convert array 1D into matrix

# unit conversions
x = x * 2.54        # inches to centimeters
y = y * 0.453592    # pounds to kilograms

# 2) Split the data into training and testing sets with 80/20 ratio.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# 3) Scale the training and testing data using normalization and standardization.
scaler_mm = MinMaxScaler()
scaler_std = StandardScaler()

x_mm_train = scaler_mm.fit_transform(x_train) 
x_mm_test = scaler_mm.transform(x_test) 

x_std_train = scaler_std.fit_transform(x_train)
x_std_test = scaler_std.transform(x_test)  

# 4) Fit a KNN regression model with k=5 to the training data without scaling, predict on unscaled testing data and compute the R2 value.
knn_unscaled = KNeighborsRegressor(n_neighbors=5)
knn_unscaled.fit(x_train, y_train)
y_pred_unscaled = knn_unscaled.predict(x_test)
r2_unscaled = r2_score(y_test, y_pred_unscaled)
print("R2 (unscaled) =", r2_unscaled)

# 5) Repeat step 4 for normalized data.
knn_normalized = KNeighborsRegressor(n_neighbors=5)
knn_normalized.fit(x_mm_train, y_train)
y_pred_normalized = knn_normalized.predict(x_mm_test)
r2_normalized = r2_score(y_test, y_pred_normalized)
print("R2 (normalized) =", r2_normalized)

# 6) Repeat step 4 for standardize data.
knn_standardized = KNeighborsRegressor(n_neighbors=5)
knn_standardized.fit(x_std_train, y_train)
y_pred_standardized = knn_standardized.predict(x_std_test)
r2_standardized = r2_score(y_test, y_pred_standardized)
print("R2 (standardized) =", r2_standardized)