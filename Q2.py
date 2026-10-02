import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

TRAIN_FILE = "Datasets/BT2024210_train_var2.csv"

train = pd.read_csv(TRAIN_FILE)

features = ["x1", "x2", "x3"]

X = train[features].values
y = train["y"].values


def fit_polynomial_regression(X, y, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)
    theta = np.linalg.pinv(X_poly) @ y

    return poly, theta


def predict_polynomial(X, poly, theta):
    X_poly = poly.transform(X)
    return X_poly @ theta


kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

print("Polynomial Degree Selection")
print("-" * 65)
print(f"{'Degree':>8}" f"{'Terms':>12}" f"{'CV MSE':>16}" f"{'CV R2':>16}")
print("-" * 65)

for degree in range(1, 21):
    fold_mse = []
    fold_r2 = []

    for train_idx, val_idx in kf.split(X):
        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        poly, theta = fit_polynomial_regression(X_train, y_train, degree)
        y_pred = predict_polynomial(X_val, poly, theta)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        fold_mse.append(mse)
        fold_r2.append(r2)

    mean_mse = np.mean(fold_mse)
    mean_r2 = np.mean(fold_r2)

    num_terms = (
        PolynomialFeatures(degree=degree, include_bias=True).fit(X).n_output_features_
    )

    results.append(
        {"degree": degree, "terms": num_terms, "mse": mean_mse, "r2": mean_r2}
    )

    print(f"{degree:>8}" f"{num_terms:>12}" f"{mean_mse:>16.6f}" f"{mean_r2:>16.6f}")


degrees = [r["degree"] for r in results]
mse_values = [r["mse"] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(degrees, mse_values, marker="o")

plt.xlabel("Polynomial Degree")
plt.ylabel("Cross-Validation MSE")
plt.title("Polynomial Degree vs Cross-Validation MSE")
plt.xticks(degrees)
plt.grid(True)

plt.savefig("Q2_Validation.png")

best_result = min(results, key=lambda r: (r["mse"], -r["r2"]))

best_degree = best_result["degree"]

print("-" * 65)
print(f"Selected degree : {best_degree}")
print(f"CV MSE : {best_result['mse']:.6f}")
print(f"CV R2 : {best_result['r2']:.6f}")
print(f"Polynomial terms: {best_result['terms']}")


poly, theta = fit_polynomial_regression(X, y, best_degree)

train_pred = predict_polynomial(X, poly, theta)

train_mse = mean_squared_error(y, train_pred)
train_r2 = r2_score(y, train_pred)

print("\nFinal Model Performance on Training Data")
print(f"MSE: {train_mse:.6f}")
print(f"R2 : {train_r2:.6f}")

TEST_FILE = "Datasets/BT2024210_test_var2.csv"

test = pd.read_csv(TEST_FILE)
X_test = test[features].values
test_pred = predict_polynomial(X_test, poly, theta)
predictions = pd.DataFrame({"y": test_pred})
predictions.to_csv("BT2024210_pred_var2.csv", index=False)
