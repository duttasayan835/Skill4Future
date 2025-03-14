import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the air quality data.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    # Load data
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
    
    # Create a copy of the dataframe for analysis
    print("Handling missing values...")
    
    # Fill missing values for numerical columns with median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='median')
    
    # Create a copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # First, handle missing values in the target variable (AQI)
    df_clean = df_clean.dropna(subset=['AQI'])
    
    # Then impute other numerical columns
    cols_to_impute = [col for col in numerical_cols if col not in ['AQI', 'AQI_Bucket']]
    df_clean[cols_to_impute] = imputer.fit_transform(df_clean[cols_to_impute])
    
    print(f"Original data shape: {df.shape}")
    print(f"Cleaned data shape (after removing NaN AQI): {df_clean.shape}")
    
    return df, df_clean

def get_season(month):
    """
    Convert month to season.
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Season name
    """
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def prepare_modeling_data(df, target_col='AQI'):
    """
    Prepare data for modeling by selecting features and target.
    
    Args:
        df: Preprocessed DataFrame
        target_col: Target column name
        
    Returns:
        X: Features DataFrame
        y: Target Series
        X_scaled: Scaled features
    """
    # Drop non-feature columns
    X = df.drop([target_col, 'Date', 'City', 'AQI_Bucket'], axis=1, errors='ignore')
    
    # Keep only numerical columns
    X = X.select_dtypes(include=['float64', 'int64'])
    
    # Get target
    y = df[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, scaler
