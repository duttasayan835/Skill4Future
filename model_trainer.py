import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

def train_rf_model(X_train_scaled, y_train):
    """Train a Random Forest model with optimized parameters"""
    print("\nTraining Random Forest model...")
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
    rf_model.fit(X_train_scaled, y_train)
    return rf_model

def train_gb_model(X_train_scaled, y_train):
    """Train a Gradient Boosting model with optimized parameters"""
    print("\nTraining Gradient Boosting model...")
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    return gb_model

def evaluate_model(model, X_test_scaled, y_test, model_name):
    """Evaluate a model and return metrics"""
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy (1 - relative error)
    accuracy = 100 * (1 - np.mean(np.abs(y_test - y_pred) / y_test))
    
    # Calculate precision (percentage of predictions within 10% of actual)
    precision = 100 * np.mean(np.abs(y_test - y_pred) <= 0.1 * y_test)
    
    print(f"\n{model_name} Model Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    
    return {
        'y_pred': y_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy,
        'precision': precision
    }

def train_models():
    """Train and evaluate models for AQI prediction"""
    # Load and preprocess data
    df = load_data()
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Create feature interactions
    X_enhanced = create_feature_interactions(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    rf_model = train_rf_model(X_train_scaled, y_train)
    rf_metrics = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    
    # Train Gradient Boosting model
    gb_model = train_gb_model(X_train_scaled, y_train)
    gb_metrics = evaluate_model(gb_model, X_test_scaled, y_test, "Gradient Boosting")
    
    # Create ensemble predictions (average of RF and GB)
    y_pred_ensemble = (rf_metrics['y_pred'] + gb_metrics['y_pred']) / 2
    
    # Evaluate ensemble
    mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
    rmse_ensemble = np.sqrt(mse_ensemble)
    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
    r2_ensemble = r2_score(y_test, y_pred_ensemble)
    
    # Calculate accuracy and precision for ensemble
    accuracy_ensemble = 100 * (1 - np.mean(np.abs(y_test - y_pred_ensemble) / y_test))
    precision_ensemble = 100 * np.mean(np.abs(y_test - y_pred_ensemble) <= 0.1 * y_test)
    
    print(f"\nEnsemble Model Metrics:")
    print(f"RMSE: {rmse_ensemble:.2f}")
    print(f"MAE: {mae_ensemble:.2f}")
    print(f"R²: {r2_ensemble:.4f}")
    print(f"Accuracy: {accuracy_ensemble:.2f}%")
    print(f"Precision: {precision_ensemble:.2f}%")
    
    # Select best model
    if rf_metrics['precision'] >= 87 and rf_metrics['precision'] >= gb_metrics['precision'] and rf_metrics['precision'] >= precision_ensemble:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_precision = rf_metrics['precision']
        best_accuracy = rf_metrics['accuracy']
    elif gb_metrics['precision'] >= 87 and gb_metrics['precision'] >= rf_metrics['precision'] and gb_metrics['precision'] >= precision_ensemble:
        best_model = gb_model
        best_model_name = "Gradient Boosting"
        best_precision = gb_metrics['precision']
        best_accuracy = gb_metrics['accuracy']
    else:
        # For ensemble, we need to save both models
        best_model = {'rf': rf_model, 'gb': gb_model}
        best_model_name = "Ensemble"
        best_precision = precision_ensemble
        best_accuracy = accuracy_ensemble
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best model accuracy: {best_accuracy:.2f}%")
    print(f"Best model precision: {best_precision:.2f}%")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('models/gb_model.pkl', 'wb') as f:
        pickle.dump(gb_model, f)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(list(X_enhanced.columns), f)
    
    # Save best model info
    with open('models/best_model_info.pkl', 'wb') as f:
        pickle.dump({
            'name': best_model_name,
            'accuracy': best_accuracy,
            'precision': best_precision
        }, f)
    
    print("\nModels saved successfully!")
    
    # Create visualizations
    create_visualizations(X_enhanced, y_test, rf_model, gb_model, 
                         rf_metrics['y_pred'], gb_metrics['y_pred'], y_pred_ensemble,
                         best_model_name)
    
    return {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'scaler': scaler,
        'feature_names': list(X_enhanced.columns),
        'best_model_name': best_model_name,
        'best_accuracy': best_accuracy,
        'best_precision': best_precision
    }

def create_visualizations(X, y_test, rf_model, gb_model, y_pred_rf, y_pred_gb, y_pred_ensemble, best_model_name):
    """Create visualizations for model performance"""
    os.makedirs('visualizations', exist_ok=True)
    
    # Actual vs Predicted for Random Forest
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Random Forest: Actual vs Predicted AQI')
    plt.savefig('visualizations/rf_actual_vs_predicted.png')
    plt.close()
    
    # Actual vs Predicted for Gradient Boosting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_gb, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Gradient Boosting: Actual vs Predicted AQI')
    plt.savefig('visualizations/gb_actual_vs_predicted.png')
    plt.close()
    
    # Actual vs Predicted for Ensemble
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_ensemble, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Ensemble: Actual vs Predicted AQI')
    plt.savefig('visualizations/ensemble_actual_vs_predicted.png')
    plt.close()
    
    # Feature importance for Random Forest
    feature_importance_rf = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_rf.head(15))
    plt.title('Random Forest - Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig('visualizations/rf_feature_importance.png')
    plt.close()
    
    # Feature importance for Gradient Boosting
    feature_importance_gb = pd.DataFrame({
        'Feature': X.columns,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_gb.head(15))
    plt.title('Gradient Boosting - Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig('visualizations/gb_feature_importance.png')
    plt.close()
    
    # Model comparison
    models = ['Random Forest', 'Gradient Boosting', 'Ensemble']
    accuracies = [
        100 * (1 - np.mean(np.abs(y_test - y_pred_rf) / y_test)),
        100 * (1 - np.mean(np.abs(y_test - y_pred_gb) / y_test)),
        100 * (1 - np.mean(np.abs(y_test - y_pred_ensemble) / y_test))
    ]
    precisions = [
        100 * np.mean(np.abs(y_test - y_pred_rf) <= 0.1 * y_test),
        100 * np.mean(np.abs(y_test - y_pred_gb) <= 0.1 * y_test),
        100 * np.mean(np.abs(y_test - y_pred_ensemble) <= 0.1 * y_test)
    ]
    
    # Accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies)
    plt.axhline(y=87, color='r', linestyle='--', label='Target (87%)')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.legend()
    plt.savefig('visualizations/model_accuracy_comparison.png')
    plt.close()
    
    # Precision comparison
    plt.figure(figsize=(10, 6))
    plt.bar(models, precisions)
    plt.axhline(y=87, color='r', linestyle='--', label='Target (87%)')
    plt.xlabel('Model')
    plt.ylabel('Precision (%)')
    plt.title('Model Precision Comparison')
    plt.legend()
    plt.savefig('visualizations/model_precision_comparison.png')
    plt.close()

if __name__ == "__main__":
    train_models()
