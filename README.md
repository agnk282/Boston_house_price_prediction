# Boston House Price Prediction

This project demonstrates regression modeling and visualization for predicting house prices using the Boston housing dataset. It includes implementations of Linear Regression, Random Forest Regression, and Support Vector Machine (SVM) Regression, with code and visualizations for each model.

## Project Structure

- `linear_reg.py` — Linear Regression model and visualizations
- `random_forest.py` — Random Forest Regression model and visualizations
- `svm_reg.py` — SVM Regression model and visualizations
- `requirements.txt` — Python dependencies

## Features

- Loads the Boston housing dataset from OpenML
- Cleans and preprocesses the data
- Trains and evaluates three regression models:
  - Linear Regression
  - Random Forest Regression
  - SVM Regression
- Prints Mean Squared Error (MSE) for each model
- Visualizes:
  - Actual vs Predicted Prices (scatter plot)
  - Residuals vs Predicted Prices (scatter plot)
  - Feature Importance (for Random Forest and Linear Regression)

## Setup

1. **Clone the repository**
2. **Create and activate a Python virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run any of the model scripts to train, evaluate, and visualize results:

```bash
python linear_reg.py
python random_forest.py
python svm_reg.py
```

## Notes
- The Boston housing dataset is loaded using `sklearn.datasets.fetch_openml`.
- All features are converted to numeric and missing values are filled with column means.
- For SVM, feature importance is not available.
- Visualizations require `matplotlib`.



