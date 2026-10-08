import warnings
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import lasso_path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TRAIN_FILE = "Datasets/BT2024210_train_var2.csv"

train = pd.read_csv(TRAIN_FILE)

features = ["x1", "x2", "x3"]
X = train[features].values
y = train["y"].values

DEGREES = range(1, 21)
RIDGE_ALPHAS = np.logspace(-3, 2.5, 12)
LASSO_ALPHAS = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5])

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def expand(X_train, X_val, degree):
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train = poly.fit_transform(X_train)
    X_val = poly.transform(X_val)
    scaler = StandardScaler().fit(X_train)

    return scaler.transform(X_train), scaler.transform(X_val)


def cross_validate(degree):
    mse = {"OLS": [], "Ridge": [], "Lasso": []}
    r2 = {"OLS": [], "Ridge": [], "Lasso": []}

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = expand(X[train_idx], X[val_idx], degree)
        y_mean = y[train_idx].mean()
        y_centered = y[train_idx] - y_mean

        U, s, Vt = np.linalg.svd(X_train, full_matrices=False)
        Uy = U.T @ y_centered
        X_val_v = X_val @ Vt.T

        keep = s > 1e-10 * s[0]
        ols_pred = X_val_v[:, keep] @ (Uy[keep] / s[keep]) + y_mean
        ridge_pred = (
            X_val_v @ ((s * Uy)[:, None] / (s[:, None] ** 2 + RIDGE_ALPHAS)) + y_mean
        )

        _, lasso_coef, _ = lasso_path(
            X_train, y_centered, alphas=LASSO_ALPHAS, max_iter=3000, tol=1e-3
        )
        lasso_pred = X_val @ lasso_coef + y_mean

        predictions = {
            "OLS": ols_pred[:, None],
            "Ridge": ridge_pred,
            "Lasso": lasso_pred,
        }

        for name, pred in predictions.items():
            mse[name].append(
                [
                    mean_squared_error(y[val_idx], pred[:, i])
                    for i in range(pred.shape[1])
                ]
            )
            r2[name].append(
                [r2_score(y[val_idx], pred[:, i]) for i in range(pred.shape[1])]
            )

    return {
        name: (np.mean(mse[name], axis=0), np.mean(r2[name], axis=0)) for name in mse
    }


alpha_grid = {"OLS": [0.0], "Ridge": RIDGE_ALPHAS, "Lasso": LASSO_ALPHAS}

results = []

print(
    f"{'Method':>7}{'Degree':>8}{'Terms':>8}{'Best alpha':>12}{'CV MSE':>18}{'CV R2':>18}"
)
print("-" * 71)

for degree in DEGREES:
    terms = (
        PolynomialFeatures(degree=degree, include_bias=False).fit(X).n_output_features_
    )

    for method, (mse, r2) in cross_validate(degree).items():
        best = np.argmin(mse)

        results.append(
            {
                "method": method,
                "degree": degree,
                "terms": terms,
                "alpha": alpha_grid[method][best],
                "mse": mse[best],
                "r2": r2[best],
                "all_mse": mse,
            }
        )

        print(
            f"{method:>7}{degree:>8}{terms:>8}"
            f"{alpha_grid[method][best]:>12.4f}"
            f"{mse[best]:>18.6f}{r2[best]:>18.6f}"
        )

print()

plt.figure(figsize=(10, 6))

for method, style in [("OLS", "o--"), ("Ridge", "s-"), ("Lasso", "^-")]:
    rows = [r for r in results if r["method"] == method]
    plt.plot([r["degree"] for r in rows], [r["mse"] for r in rows], style, label=method)

plt.yscale("log")
plt.xlabel("Polynomial Degree")
plt.ylabel("Cross-Validation MSE (log scale)")
plt.title("OLS vs Ridge vs Lasso: CV MSE by Polynomial Degree")
plt.xticks(list(DEGREES))
plt.grid(True, which="both", alpha=0.4)
plt.legend()
plt.savefig("Q2_Validation.png", dpi=150, bbox_inches="tight")

best = min(results, key=lambda r: (r["mse"], -r["r2"]))
ols_best = min((r for r in results if r["method"] == "OLS"), key=lambda r: r["mse"])

print("-" * 57)
print(
    f"Selected model : {best['method']}  "
    f"degree={best['degree']}  alpha={best['alpha']:.5f}"
)
print(f"CV MSE : {best['mse']:.6f}")
print(f"CV R2  : {best['r2']:.6f}")

poly = PolynomialFeatures(best["degree"], include_bias=False)
scaler = StandardScaler().fit(poly.fit_transform(X))
A = scaler.transform(poly.transform(X))

y_mean = y.mean()

if best["method"] == "Lasso":
    coef = lasso_path(A, y - y_mean, alphas=[best["alpha"]], max_iter=20000, tol=1e-5)[
        1
    ][:, 0]
elif best["method"] == "Ridge":
    coef = np.linalg.solve(
        A.T @ A + best["alpha"] * np.eye(A.shape[1]), A.T @ (y - y_mean)
    )
else:
    coef = np.linalg.pinv(A) @ (y - y_mean)

train_pred = A @ coef + y_mean

print("\nFinal Model Performance on Training Data")
print(f"MSE: {mean_squared_error(y, train_pred):.6f}")
print(f"R2 : {r2_score(y, train_pred):.6f}")

TEST_FILE = "Datasets/BT2024210_test_var2.csv"

test = pd.read_csv(TEST_FILE)
X_test = scaler.transform(poly.transform(test[features].values))
test_pred = X_test @ coef + y_mean
predictions = pd.DataFrame({"y": test_pred})
predictions.to_csv("BT2024210_pred_var2.csv", index=False)
