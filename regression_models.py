import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import os

def create_regression_visualizations(X, y, output_dir='visualizations'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_lr, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Linear Regression: Actual vs Predicted')
    plt.savefig(f'{output_dir}/linear_regression.png')
    plt.close()
    
    # 2. Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    pr = LinearRegression()
    pr.fit(X_poly, y_train)
    y_pred_pr = pr.predict(X_test_poly)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_pr, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Polynomial Regression: Actual vs Predicted')
    plt.savefig(f'{output_dir}/polynomial_regression.png')
    plt.close()
    
    # 3. Decision Tree Regression
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_dt, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Decision Tree Regression: Actual vs Predicted')
    plt.savefig(f'{output_dir}/decision_tree_regression.png')
    plt.close()
    
    # Feature importance for Decision Tree
    plt.figure(figsize=(12, 6))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': dt.feature_importances_
    }).sort_values('importance', ascending=False)
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Decision Tree Feature Importance')
    plt.savefig(f'{output_dir}/decision_tree_feature_importance.png')
    plt.close()
    
    # 4. Random Forest Regression
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Random Forest Regression: Actual vs Predicted')
    plt.savefig(f'{output_dir}/random_forest_regression.png')
    plt.close()
    
    # Feature importance for Random Forest
    plt.figure(figsize=(12, 6))
    feature_importance_rf = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    sns.barplot(data=feature_importance_rf, x='importance', y='feature')
    plt.title('Random Forest Feature Importance')
    plt.savefig(f'{output_dir}/random_forest_feature_importance.png')
    plt.close()
    
    # 5. Support Vector Regression
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_svr, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Support Vector Regression: Actual vs Predicted')
    plt.savefig(f'{output_dir}/svr_regression.png')
    plt.close()
    
    # Model Comparison
    models = {
        'Linear Regression': (lr, y_pred_lr),
        'Polynomial Regression': (pr, y_pred_pr),
        'Decision Tree': (dt, y_pred_dt),
        'Random Forest': (rf, y_pred_rf),
        'SVR': (svr, y_pred_svr)
    }
    
    # Compare R² scores
    r2_scores = {name: r2_score(y_test, pred) for name, (_, pred) in models.items()}
    plt.figure(figsize=(10, 6))
    plt.bar(r2_scores.keys(), r2_scores.values())
    plt.title('Model Comparison - R² Scores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/regression_model_comparison.png')
    plt.close()
    
    return models, r2_scores
