import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Metrics
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)

# Adjusted R-squared
n = len(y)        # number of observations
p = X.shape[1]    # number of predictors
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)


print("R-squared:", r2)
print("Adjusted R-squared:", adj_r2)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)