import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # Pull the Boston house pricing dataset
    boston = fetch_openml(name="boston", version=1, as_frame=True)
    X = boston.data
    y = boston.target

    # Ensure all feature columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    # Fill any missing values with column means
    X = X.fillna(X.mean())
    # Ensure target is numeric
    y = y.astype(float)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the SVM regression model
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)

    # Predict house prices on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (SVM): {mse}")
    print("First 5 predictions:", y_pred[:5])

    # Visualization 1: Actual vs Predicted prices (scatter plot)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted House Prices (SVM)')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.show()

    # Visualization 2: Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Prices')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Prices (SVM)')
    plt.tight_layout()
    plt.show()

