import pandas as pd
from data_preprocessing import load_and_preprocess_data, prepare_modeling_data
from regression_models import create_regression_visualizations
from classification_models import create_classification_visualizations
from clustering_and_pca import create_clustering_visualizations
import os

def main():
    # Create visualizations directory
    output_dir = 'visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df, df_clean = load_and_preprocess_data('air quality data.csv')
    
    # Prepare data for modeling
    print("Preparing data for modeling...")
    X, y, scaler = prepare_modeling_data(df_clean)
    
    # Generate regression visualizations
    print("Generating regression model visualizations...")
    regression_models, r2_scores = create_regression_visualizations(X, y, output_dir)
    print("\nRegression Model R² Scores:")
    for model, score in r2_scores.items():
        print(f"{model}: {score:.4f}")
    
    # Generate classification visualizations
    print("\nGenerating classification model visualizations...")
    classification_models, accuracies, classes = create_classification_visualizations(X, y, output_dir)
    print("\nClassification Model Accuracies:")
    for model, acc in accuracies.items():
        print(f"{model}: {acc:.4f}")
    
    # Generate clustering and PCA visualizations
    print("\nGenerating clustering and PCA visualizations...")
    clustering_results = create_clustering_visualizations(X, output_dir)
    
    print("\nAnalysis complete! All visualizations have been saved in the 'visualizations' directory.")
    print("\nVisualization files generated:")
    for file in os.listdir(output_dir):
        print(f"- {file}")

if __name__ == "__main__":
    main()
