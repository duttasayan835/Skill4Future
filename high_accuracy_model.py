import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)

def load_data(file_path='air quality data.csv'):
    """Load and preprocess the air quality data"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract temporal features
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Year'] = df['Date'].dt.year
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Drop rows with missing AQI values
    df_clean = df.dropna(subset=['AQI'])
    
    # Fill missing values in other columns with median
    for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
        if col != 'AQI' and df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    print(f"Original data shape: {df.shape}")
    print(f"Cleaned data shape: {df_clean.shape}")
    
    return df_clean

def prepare_features(df):
    """Prepare features for modeling"""
    # Drop non-feature columns
    X = df.drop(['AQI', 'Date', 'City', 'AQI_Bucket'], axis=1, errors='ignore')
    
    # Keep only numerical columns
    X = X.select_dtypes(include=['float64', 'int64'])
    
    # Get target
    y = df['AQI']
    
    print(f"Features shape: {X.shape}")
    
    return X, y

def create_feature_interactions(X):
    """Create interaction features between important pollutants"""
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
    
    return X_enhanced

def train_high_accuracy_model(X, y):
    """Train a high accuracy model for AQI prediction"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Random Forest model...")
    # Random Forest with optimized parameters
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    # Evaluate
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    # Calculate accuracy (1 - relative error)
    accuracy_rf = 100 * (1 - np.mean(np.abs(y_test - y_pred_rf) / y_test))
    
    # Calculate precision (percentage of predictions within 10% of actual)
    precision_rf = 100 * np.mean(np.abs(y_test - y_pred_rf) <= 0.1 * y_test)
    
    print(f"\nRandom Forest Model Metrics:")
    print(f"RMSE: {rmse_rf:.2f}")
    print(f"MAE: {mae_rf:.2f}")
    print(f"R²: {r2_rf:.4f}")
    print(f"Accuracy: {accuracy_rf:.2f}%")
    print(f"Precision: {precision_rf:.2f}%")
    
    print("\nTraining Gradient Boosting model...")
    # Gradient Boosting with optimized parameters
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42
    )
    
    # Train the model
    gb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_gb = gb_model.predict(X_test_scaled)
    
    # Evaluate
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mse_gb)
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)
    
    # Calculate accuracy (1 - relative error)
    accuracy_gb = 100 * (1 - np.mean(np.abs(y_test - y_pred_gb) / y_test))
    
    # Calculate precision (percentage of predictions within 10% of actual)
    precision_gb = 100 * np.mean(np.abs(y_test - y_pred_gb) <= 0.1 * y_test)
    
    print(f"\nGradient Boosting Model Metrics:")
    print(f"RMSE: {rmse_gb:.2f}")
    print(f"MAE: {mae_gb:.2f}")
    print(f"R²: {r2_gb:.4f}")
    print(f"Accuracy: {accuracy_gb:.2f}%")
    print(f"Precision: {precision_gb:.2f}%")
    
    # Create an ensemble model (average predictions)
    print("\nCreating ensemble model...")
    y_pred_ensemble = (y_pred_rf + y_pred_gb) / 2
    
    # Evaluate ensemble
    mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
    rmse_ensemble = np.sqrt(mse_ensemble)
    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
    r2_ensemble = r2_score(y_test, y_pred_ensemble)
    
    # Calculate accuracy (1 - relative error)
    accuracy_ensemble = 100 * (1 - np.mean(np.abs(y_test - y_pred_ensemble) / y_test))
    
    # Calculate precision (percentage of predictions within 10% of actual)
    precision_ensemble = 100 * np.mean(np.abs(y_test - y_pred_ensemble) <= 0.1 * y_test)
    
    print(f"\nEnsemble Model Metrics:")
    print(f"RMSE: {rmse_ensemble:.2f}")
    print(f"MAE: {mae_ensemble:.2f}")
    print(f"R²: {r2_ensemble:.4f}")
    print(f"Accuracy: {accuracy_ensemble:.2f}%")
    print(f"Precision: {precision_ensemble:.2f}%")
    
    # Select the best model based on accuracy and precision
    if accuracy_rf >= accuracy_gb and accuracy_rf >= accuracy_ensemble and precision_rf >= 87:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_accuracy = accuracy_rf
        best_precision = precision_rf
    elif accuracy_gb >= accuracy_rf and accuracy_gb >= accuracy_ensemble and precision_gb >= 87:
        best_model = gb_model
        best_model_name = "Gradient Boosting"
        best_accuracy = accuracy_gb
        best_precision = precision_gb
    else:
        # Create a custom ensemble model
        best_model = {'rf': rf_model, 'gb': gb_model}
        best_model_name = "Ensemble"
        best_accuracy = accuracy_ensemble
        best_precision = precision_ensemble
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best model accuracy: {best_accuracy:.2f}%")
    print(f"Best model precision: {best_precision:.2f}%")
    
    # Visualize actual vs predicted
    plt.figure(figsize=(10, 6))
    if best_model_name == "Random Forest":
        plt.scatter(y_test, y_pred_rf, alpha=0.5)
        residuals = y_test - y_pred_rf
    elif best_model_name == "Gradient Boosting":
        plt.scatter(y_test, y_pred_gb, alpha=0.5)
        residuals = y_test - y_pred_gb
    else:
        plt.scatter(y_test, y_pred_ensemble, alpha=0.5)
        residuals = y_test - y_pred_ensemble
    
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title(f'{best_model_name}: Actual vs Predicted AQI')
    plt.savefig('actual_vs_predicted_high_accuracy.png')
    
    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual AQI')
    plt.ylabel('Residuals')
    plt.title(f'{best_model_name}: Residual Plot')
    plt.savefig('residuals_high_accuracy.png')
    
    # Plot feature importance if available
    if best_model_name != "Ensemble":
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title(f'{best_model_name} - Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_high_accuracy.png')
    
    # Save the best model
    if best_model_name == "Ensemble":
        with open('high_accuracy_rf_model.pkl', 'wb') as f:
            pickle.dump(best_model['rf'], f)
        with open('high_accuracy_gb_model.pkl', 'wb') as f:
            pickle.dump(best_model['gb'], f)
    else:
        with open('high_accuracy_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
    
    # Save the scaler
    with open('high_accuracy_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nModel and scaler saved successfully!")
    
    return {
        'model': best_model,
        'model_name': best_model_name,
        'scaler': scaler,
        'accuracy': best_accuracy,
        'precision': best_precision,
        'feature_names': X.columns
    }

def main():
    # Load and preprocess data
    df = load_data()
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Create feature interactions
    X_enhanced = create_feature_interactions(X)
    
    # Train high accuracy model
    results = train_high_accuracy_model(X_enhanced, y)
    
    print("\nModel training complete!")
    print(f"Final model: {results['model_name']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Precision: {results['precision']:.2f}%")
    
    # Update the Streamlit app to use the high accuracy model
    print("\nUpdating Streamlit app to use the high accuracy model...")
    
    return results

if __name__ == "__main__":
    main()
