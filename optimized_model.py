import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path, target_col='AQI'):
    """
    Load and preprocess the air quality data with advanced preprocessing.
    
    Args:
        file_path: Path to the CSV file
        target_col: Target column name
        
    Returns:
        Preprocessed data and related objects
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract temporal features
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Year'] = df['Date'].dt.year
    df['Season'] = df['Date'].dt.month.apply(get_season)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Create a copy of the dataframe for analysis
    print("Handling missing values...")
    
    # Drop rows with missing target values
    df_clean = df.dropna(subset=[target_col])
    
    # Use KNN imputation for missing values in features
    numerical_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    cols_to_impute = [col for col in numerical_cols if col not in [target_col, 'AQI_Bucket']]
    
    # KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    df_clean[cols_to_impute] = imputer.fit_transform(df_clean[cols_to_impute])
    
    # Drop non-feature columns for modeling
    X = df_clean.drop([target_col, 'Date', 'City', 'AQI_Bucket'], axis=1, errors='ignore')
    
    # Keep only numerical columns
    X = X.select_dtypes(include=['float64', 'int64'])
    
    # Get target
    y = df_clean[target_col]
    
    print(f"Original data shape: {df.shape}")
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Features shape: {X.shape}")
    
    return df, df_clean, X, y, imputer

def get_season(month):
    """
    Convert month to season.
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Season number (0-3)
    """
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall

def create_feature_interactions(X):
    """
    Create interaction features between important pollutants
    
    Args:
        X: Features DataFrame
        
    Returns:
        DataFrame with additional interaction features
    """
    # List of important pollutants based on domain knowledge
    important_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    
    # Create a copy of the original dataframe
    X_enhanced = X.copy()
    
    # Create interaction features for important pollutants
    for i in range(len(important_pollutants)):
        for j in range(i+1, len(important_pollutants)):
            if important_pollutants[i] in X.columns and important_pollutants[j] in X.columns:
                col_name = f"{important_pollutants[i]}_{important_pollutants[j]}_interaction"
                X_enhanced[col_name] = X[important_pollutants[i]] * X[important_pollutants[j]]
    
    # Create ratio features
    for i in range(len(important_pollutants)):
        for j in range(len(important_pollutants)):
            if i != j and important_pollutants[i] in X.columns and important_pollutants[j] in X.columns:
                col_name = f"{important_pollutants[i]}_{important_pollutants[j]}_ratio"
                # Avoid division by zero
                X_enhanced[col_name] = X[important_pollutants[i]] / (X[important_pollutants[j]] + 1e-5)
    
    return X_enhanced

def select_features(X, y, method='rfe', n_features=None):
    """
    Select the most important features using different methods
    
    Args:
        X: Features DataFrame
        y: Target Series
        method: Feature selection method ('rfe', 'model_based', or 'rfecv')
        n_features: Number of features to select (only for 'rfe')
        
    Returns:
        Selected features DataFrame and feature selector
    """
    if method == 'rfe':
        if n_features is None:
            n_features = max(int(X.shape[1] * 0.5), 10)  # Default to 50% of features or at least 10
        
        # Use Random Forest as the base estimator for RFE
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=1)
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.support_]
        X_selected = X[selected_features]
        
    elif method == 'model_based':
        # Use Random Forest for feature importance-based selection
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = SelectFromModel(estimator, threshold='median')
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()]
        X_selected = X[selected_features]
        
    elif method == 'rfecv':
        # Use RFECV for optimal feature selection with cross-validation
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFECV(estimator, step=1, cv=5, scoring='neg_mean_squared_error')
        selector.fit(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.support_]
        X_selected = X[selected_features]
    
    else:
        raise ValueError("Invalid method. Choose from 'rfe', 'model_based', or 'rfecv'")
    
    print(f"Selected {len(selected_features)} features using {method} method")
    print(f"Selected features: {', '.join(selected_features)}")
    
    return X_selected, selector

def build_optimized_model(X, y, output_dir='visualizations'):
    """
    Build and optimize multiple regression models for AQI prediction
    
    Args:
        X: Features DataFrame
        y: Target Series
        output_dir: Directory to save visualizations
        
    Returns:
        Best model, performance metrics, and feature importance
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Define base models
    base_models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate base models
    base_model_results = {}
    for name, model in base_models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate error rate (percentage of predictions with error > 10% of actual value)
        error_rate = np.mean(np.abs(y_test - y_pred) > 0.1 * y_test) * 100
        
        # Store results
        base_model_results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'error_rate': error_rate,
            'predictions': y_pred
        }
        
        print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, Error Rate: {error_rate:.2f}%")
    
    # Visualize base model performance
    metrics_df = pd.DataFrame({
        'Model': list(base_model_results.keys()),
        'RMSE': [results['rmse'] for results in base_model_results.values()],
        'MAE': [results['mae'] for results in base_model_results.values()],
        'R²': [results['r2'] for results in base_model_results.values()],
        'Error Rate (%)': [results['error_rate'] for results in base_model_results.values()]
    })
    
    # Sort by R² (higher is better)
    metrics_df = metrics_df.sort_values('R²', ascending=False)
    
    # Plot R² comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='R²', y='Model', data=metrics_df)
    plt.title('Model Comparison - R² Score (higher is better)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimized_model_r2_comparison.png')
    plt.close()
    
    # Plot RMSE comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='RMSE', y='Model', data=metrics_df)
    plt.title('Model Comparison - RMSE (lower is better)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimized_model_rmse_comparison.png')
    plt.close()
    
    # Plot Error Rate comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Error Rate (%)', y='Model', data=metrics_df)
    plt.title('Model Comparison - Error Rate % (lower is better)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimized_model_error_rate_comparison.png')
    plt.close()
    
    # Select top 3 performing models based on R²
    top_models = metrics_df.head(3)['Model'].tolist()
    print(f"\nTop performing models: {', '.join(top_models)}")
    
    # Fine-tune the best model
    best_model_name = top_models[0]
    best_base_model = base_models[best_model_name]
    
    print(f"\nFine-tuning {best_model_name}...")
    
    # Define hyperparameter grid based on the best model
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 3, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        
    elif best_model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'min_child_weight': [1, 3, 5, 7]
        }
        
    elif best_model_name == 'AdaBoost':
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
            'loss': ['linear', 'square', 'exponential']
        }
        
    elif best_model_name == 'SVR':
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
        
    elif best_model_name == 'KNN':
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 40, 50],
            'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
        }
        
    else:
        # Default to RandomForest if the best model doesn't match any of the above
        best_model_name = 'Random Forest'
        best_base_model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    
    # Use RandomizedSearchCV for faster tuning
    grid_search = RandomizedSearchCV(
        estimator=best_base_model,
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Evaluate the best model
    y_pred_best = best_model.predict(X_test_scaled)
    
    # Calculate metrics
    mse_best = mean_squared_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mse_best)
    mae_best = mean_absolute_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)
    
    # Calculate error rate (percentage of predictions with error > 10% of actual value)
    error_rate_best = np.mean(np.abs(y_test - y_pred_best) > 0.1 * y_test) * 100
    
    print(f"\nBest Model ({best_model_name} with tuned parameters):")
    print(f"RMSE: {rmse_best:.2f}")
    print(f"MAE: {mae_best:.2f}")
    print(f"R²: {r2_best:.4f}")
    print(f"Error Rate: {error_rate_best:.2f}%")
    
    # Create ensemble model
    print("\nBuilding ensemble model...")
    
    # Get top 3 models for ensemble
    ensemble_models = []
    for model_name in top_models:
        if model_name == best_model_name:
            # Use the tuned version of the best model
            ensemble_models.append((model_name, best_model))
        else:
            # Use the base version of other top models
            ensemble_models.append((model_name, base_models[model_name]))
    
    # Create voting ensemble
    voting_regressor = VotingRegressor(
        estimators=ensemble_models,
        weights=[0.5, 0.3, 0.2]  # Give more weight to the best model
    )
    
    # Train the ensemble
    voting_regressor.fit(X_train_scaled, y_train)
    
    # Evaluate the ensemble
    y_pred_ensemble = voting_regressor.predict(X_test_scaled)
    
    # Calculate metrics
    mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
    rmse_ensemble = np.sqrt(mse_ensemble)
    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
    r2_ensemble = r2_score(y_test, y_pred_ensemble)
    
    # Calculate error rate
    error_rate_ensemble = np.mean(np.abs(y_test - y_pred_ensemble) > 0.1 * y_test) * 100
    
    print(f"\nEnsemble Model:")
    print(f"RMSE: {rmse_ensemble:.2f}")
    print(f"MAE: {mae_ensemble:.2f}")
    print(f"R²: {r2_ensemble:.4f}")
    print(f"Error Rate: {error_rate_ensemble:.2f}%")
    
    # Compare best single model vs ensemble
    final_comparison = pd.DataFrame({
        'Model': ['Best Single Model', 'Ensemble Model'],
        'RMSE': [rmse_best, rmse_ensemble],
        'MAE': [mae_best, mae_ensemble],
        'R²': [r2_best, r2_ensemble],
        'Error Rate (%)': [error_rate_best, error_rate_ensemble]
    })
    
    # Plot final comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='R²', data=final_comparison)
    plt.title('Final Model Comparison - R² Score')
    plt.ylim(0.8, 1.0)  # Adjust as needed
    plt.savefig(f'{output_dir}/final_model_r2_comparison.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Error Rate (%)', data=final_comparison)
    plt.title('Final Model Comparison - Error Rate (%)')
    plt.savefig(f'{output_dir}/final_model_error_rate_comparison.png')
    plt.close()
    
    # Visualize actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_ensemble, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Ensemble Model: Actual vs Predicted AQI')
    plt.savefig(f'{output_dir}/ensemble_actual_vs_predicted.png')
    plt.close()
    
    # Determine the final model (choose the one with better performance)
    if r2_ensemble > r2_best and error_rate_ensemble < error_rate_best:
        final_model = voting_regressor
        final_model_name = 'Ensemble'
        final_rmse = rmse_ensemble
        final_r2 = r2_ensemble
        final_error_rate = error_rate_ensemble
    else:
        final_model = best_model
        final_model_name = f'Tuned {best_model_name}'
        final_rmse = rmse_best
        final_r2 = r2_best
        final_error_rate = error_rate_best
    
    print(f"\nFinal selected model: {final_model_name}")
    print(f"Final model R²: {final_r2:.4f}")
    print(f"Final model RMSE: {final_rmse:.2f}")
    print(f"Final model Error Rate: {final_error_rate:.2f}%")
    
    # Save the final model
    model_path = os.path.join(output_dir, 'final_aqi_model.pkl')
    scaler_path = os.path.join(output_dir, 'final_aqi_scaler.pkl')
    
    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nFinal model saved to {model_path}")
    
    # Get feature importance if available
    if hasattr(final_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': final_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    elif hasattr(final_model, 'coef_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(final_model.coef_)
        }).sort_values('Importance', ascending=False)
    elif final_model_name == 'Ensemble':
        # Try to get feature importance from the first model in the ensemble
        if hasattr(ensemble_models[0][1], 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': ensemble_models[0][1].feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            feature_importance = None
    else:
        feature_importance = None
    
    if feature_importance is not None:
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title(f'{final_model_name} - Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/final_model_feature_importance.png')
        plt.close()
    
    # Return results
    results = {
        'final_model': final_model,
        'final_model_name': final_model_name,
        'scaler': scaler,
        'metrics': {
            'rmse': final_rmse,
            'r2': final_r2,
            'error_rate': final_error_rate
        },
        'feature_importance': feature_importance,
        'model_path': model_path,
        'scaler_path': scaler_path
    }
    
    return results

def main():
    # Set output directory
    output_dir = 'optimized_models'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and preprocess data
    df, df_clean, X, y, imputer = load_and_preprocess_data('air quality data.csv')
    
    # Create feature interactions
    print("\nCreating feature interactions...")
    X_enhanced = create_feature_interactions(X)
    
    # Select features
    print("\nSelecting important features...")
    X_selected, selector = select_features(X_enhanced, y, method='rfecv')
    
    # Build optimized model
    print("\nBuilding optimized model...")
    results = build_optimized_model(X_selected, y, output_dir)
    
    # Save feature selector
    selector_path = os.path.join(output_dir, 'feature_selector.pkl')
    joblib.dump(selector, selector_path)
    
    # Save column names
    columns_path = os.path.join(output_dir, 'feature_names.pkl')
    joblib.dump(X_enhanced.columns, columns_path)
    
    # Save imputer
    imputer_path = os.path.join(output_dir, 'imputer.pkl')
    joblib.dump(imputer, imputer_path)
    
    print(f"\nModel building complete!")
    print(f"Final model: {results['final_model_name']}")
    print(f"R² Score: {results['metrics']['r2']:.4f}")
    print(f"RMSE: {results['metrics']['rmse']:.2f}")
    print(f"Error Rate: {results['metrics']['error_rate']:.2f}%")
    print(f"\nAll model files saved to {output_dir}/")

if __name__ == "__main__":
    main()
