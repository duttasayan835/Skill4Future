# AQI Prediction Model using Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set the current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Loading data...")
# Load the dataset
df = pd.read_csv('air quality data.csv')
print("Data loaded successfully!")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Data information
print("\nDataset information:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Data preprocessing
print("\nPreprocessing data...")
# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create a copy of the dataframe for analysis
df1 = df.copy()

# Drop rows with missing AQI values
df1 = df1.dropna(subset=['AQI'])
print(f"Dataset shape after dropping rows with missing AQI: {df1.shape}")

# Summary statistics
print("\nSummary statistics:")
print(df1.describe().T)

# Univariate analysis
print("\nCreating visualizations...")
try:
    plt.figure(figsize=(10, 5))
    df1['Xylene'].plot(kind='hist')
    plt.title('Distribution of Xylene')
    plt.savefig('xylene_distribution.png')
    print("Saved Xylene distribution plot to 'xylene_distribution.png'")
except Exception as e:
    print(f"Error creating Xylene plot: {e}")

# Correlation analysis
try:
    # Handle date column for correlation
    df1_corr = df1.drop(['Date', 'City', 'AQI_Bucket'], axis=1, errors='ignore')
    
    # Fill NaN values with mean for correlation analysis
    df1_corr = df1_corr.fillna(df1_corr.mean())
    
    plt.figure(figsize=(12, 8))
    corr_matrix = df1_corr.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='Pastel1', annot_kws={"size": 8})
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    print("Saved correlation matrix to 'correlation_matrix.png'")
except Exception as e:
    print(f"Error creating correlation matrix: {e}")

# Feature selection
print("\nPreparing data for modeling...")
try:
    # Select features and target
    X = df1_corr.drop('AQI', axis=1)
    y = df1_corr['AQI']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train linear regression model
    print("\nTraining Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test_scaled)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.title('Actual vs Predicted AQI Values')
    plt.savefig('actual_vs_predicted.png')
    print("Saved actual vs predicted plot to 'actual_vs_predicted.png'")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(lr_model.coef_)
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance')
    plt.savefig('feature_importance.png')
    print("Saved feature importance plot to 'feature_importance.png'")
    
except Exception as e:
    print(f"Error in modeling: {e}")

print("\nAnalysis complete! Check the generated plots in the current directory.")
