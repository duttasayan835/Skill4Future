import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json
import nbformat
from nbconvert import HTMLExporter
from IPython.display import HTML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from data_preprocessing import load_and_preprocess_data, prepare_modeling_data
from regression_models import create_regression_visualizations
from classification_models import create_classification_visualizations
from clustering_and_pca import create_clustering_visualizations

# Set page configuration
st.set_page_config(
    page_title="AQI Prediction App",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for model training and saving
def train_and_save_model(X, y):
    """Train a Random Forest model and save it to disk"""
    model_path = "aqi_model.pkl"
    scaler_path = "aqi_scaler.pkl"
    feature_names_path = "feature_names.json"
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_names_path):
        st.info("Using existing trained model")
        return
    
    with st.spinner("Training model... This may take a moment."):
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Save model, scaler and feature names
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(feature_names_path, 'w') as f:
            json.dump(list(X.columns), f)
        
        st.success("Model trained and saved successfully!")

def load_model():
    """Load the trained model, scaler and feature names"""
    model_path = "aqi_model.pkl"
    scaler_path = "aqi_scaler.pkl"
    feature_names_path = "feature_names.json"
    
    # Check if model exists
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_names_path)):
        st.error("Model files not found. Please train the model first.")
        return None, None, None
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature names
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    return model, scaler, feature_names

def predict_aqi(input_data, model, scaler, feature_names):
    """Make AQI prediction using the trained model"""
    # Ensure input data has all required features
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return prediction

def get_aqi_category(aqi_value):
    """Convert AQI value to category"""
    if aqi_value <= 50:
        return "Good", "0, 228, 0"
    elif aqi_value <= 100:
        return "Moderate", "255, 255, 0"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "255, 126, 0"
    elif aqi_value <= 200:
        return "Unhealthy", "255, 0, 0"
    elif aqi_value <= 300:
        return "Very Unhealthy", "143, 63, 151"
    else:
        return "Hazardous", "126, 0, 35"

def render_notebook(notebook_path):
    """Render Jupyter notebook as HTML"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)
        
        html_exporter = HTMLExporter()
        html_exporter.exclude_input = False
        html_exporter.exclude_output_prompt = True
        html_exporter.exclude_input_prompt = True
        
        (body, _) = html_exporter.from_notebook_node(notebook_content)
        
        # Add custom CSS to make the notebook look better in Streamlit
        st.markdown(
            f"""
            <style>
                .jp-RenderedHTMLCommon {{ font-family: sans-serif; }}
                div.jp-RenderedHTMLCommon p {{ margin-top: 0.5em; margin-bottom: 0.5em; }}
                .jp-OutputArea-output pre {{ background-color: #f8f8f8; padding: 0.5em; }}
                .jp-RenderedMarkdown h1 {{ color: #1E88E5; }}
                .jp-RenderedMarkdown h2 {{ color: #0277BD; }}
                .jp-RenderedMarkdown h3 {{ color: #0288D1; }}
            </style>
            """, 
            unsafe_allow_html=True
        )
        
        st.components.v1.html(body, height=800, scrolling=True)
        
    except Exception as e:
        st.error(f"Error rendering notebook: {e}")

# Main app
def main():
    # Sidebar
    st.sidebar.title("AQI Prediction App")
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1113/1113895.png", width=100)
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Home", "Predict AQI", "Visualizations", "Notebook Viewer"])
    
    # Load data
    @st.cache_data
    def load_data():
        df, df_clean = load_and_preprocess_data('air quality data.csv')
        X, y, _ = prepare_modeling_data(df_clean)
        return df, df_clean, X, y
    
    try:
        df, df_clean, X, y = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Home page
    if page == "Home":
        st.title("Air Quality Index (AQI) Prediction")
        st.markdown("""
        This application helps predict Air Quality Index (AQI) values based on various pollutant measurements.
        
        ### Features:
        - **Predict AQI**: Input pollutant values to get AQI predictions
        - **Visualizations**: Explore data visualizations and model performance
        - **Notebook Viewer**: View the Jupyter notebook analysis
        
        ### Dataset Overview:
        """)
        
        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Records", df.shape[0])
            st.metric("Number of Features", df.shape[1])
        with col2:
            st.metric("Missing Values", df.isnull().sum().sum())
            st.metric("Time Period", f"{df['Date'].min().year} - {df['Date'].max().year}")
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Show data distribution
        st.subheader("AQI Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df_clean['AQI'], kde=True, ax=ax)
        plt.title("Distribution of AQI Values")
        plt.xlabel("AQI")
        plt.ylabel("Frequency")
        st.pyplot(fig)
        
        # Train model if not already trained
        train_and_save_model(X, y)
    
    # Prediction page
    elif page == "Predict AQI":
        st.title("Predict Air Quality Index")
        
        # Load the model
        model, scaler, feature_names = load_model()
        
        if model is None:
            st.warning("Please go to the Home page to train the model first.")
            return
        
        st.markdown("Enter pollutant values to predict the Air Quality Index (AQI).")
        
        # Create input form
        st.subheader("Input Pollutant Values")
        
        # Group features for better organization
        pollutants = [col for col in feature_names if col not in ['Month', 'Day', 'Year', 'DayOfWeek'] and 'Season' not in col]
        temporal = [col for col in feature_names if col in ['Month', 'Day', 'Year', 'DayOfWeek'] or 'Season' in col]
        
        # Create tabs for different input groups
        tab1, tab2 = st.tabs(["Pollutant Values", "Temporal Features"])
        
        input_data = {}
        
        with tab1:
            # Create columns for pollutants
            cols = st.columns(3)
            for i, pollutant in enumerate(pollutants):
                col_idx = i % 3
                # Get min and max values for the slider
                min_val = float(df_clean[pollutant].min())
                max_val = float(df_clean[pollutant].max())
                mean_val = float(df_clean[pollutant].mean())
                
                input_data[pollutant] = cols[col_idx].slider(
                    f"{pollutant}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    format="%.2f"
                )
        
        with tab2:
            # Create columns for temporal features
            cols = st.columns(3)
            for i, feature in enumerate(temporal):
                col_idx = i % 3
                
                if feature == 'Month':
                    input_data[feature] = cols[col_idx].slider(
                        "Month", 1, 12, 6
                    )
                elif feature == 'Day':
                    input_data[feature] = cols[col_idx].slider(
                        "Day", 1, 31, 15
                    )
                elif feature == 'Year':
                    input_data[feature] = cols[col_idx].slider(
                        "Year", 2015, 2023, 2022
                    )
                elif feature == 'DayOfWeek':
                    input_data[feature] = cols[col_idx].slider(
                        "Day of Week (0=Monday, 6=Sunday)", 0, 6, 3
                    )
                elif feature == 'Season':
                    season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
                    season = cols[col_idx].selectbox(
                        "Season", 
                        options=list(season_map.keys()),
                        index=2
                    )
                    input_data[feature] = season
        
        # Predict button
        if st.button("Predict AQI"):
            # Convert season to numeric if it's in the features
            if 'Season' in feature_names:
                season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
                input_data['Season'] = season_map[input_data['Season']]
            
            # Make prediction
            prediction = predict_aqi(input_data, model, scaler, feature_names)
            
            # Get AQI category
            category, color = get_aqi_category(prediction)
            
            # Display result
            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted AQI", f"{prediction:.2f}")
            
            with col2:
                st.markdown(
                    f"""
                    <div style="
                        background-color: rgba({color}, 0.3);
                        padding: 10px;
                        border-radius: 5px;
                        text-align: center;
                        font-weight: bold;
                    ">
                        AQI Category: {category}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Display AQI scale
            st.subheader("AQI Scale")
            aqi_scale = pd.DataFrame({
                "Category": ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"],
                "Range": ["0-50", "51-100", "101-150", "151-200", "201-300", "301+"],
                "Color": ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
            })
            
            # Create a horizontal bar chart for AQI scale
            fig, ax = plt.subplots(figsize=(10, 3))
            colors = aqi_scale['Color'].tolist()
            ax.barh(y=0, width=[50, 50, 50, 50, 100, 100], left=[0, 50, 100, 150, 200, 300], color=colors, height=0.5)
            
            # Add category labels
            for i, (category, range_val) in enumerate(zip(aqi_scale['Category'], aqi_scale['Range'])):
                if i < 4:
                    x_pos = i * 50 + 25
                elif i == 4:
                    x_pos = 200 + 50
                else:
                    x_pos = 300 + 50
                ax.text(x_pos, 0, f"{category}\n({range_val})", ha='center', va='center', fontsize=8)
            
            # Customize the plot
            ax.set_yticks([])
            ax.set_xlim(0, 500)
            ax.set_xlabel("AQI Value")
            ax.set_title("Air Quality Index (AQI) Scale")
            
            # Add a marker for the predicted value
            ax.axvline(x=prediction, color='black', linestyle='--', linewidth=2)
            ax.text(prediction, -0.1, f"{prediction:.1f}", ha='center', va='top', fontweight='bold')
            
            st.pyplot(fig)
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
            plt.title('Top 10 Feature Importance')
            st.pyplot(fig)
    
    # Visualizations page
    elif page == "Visualizations":
        st.title("Data Visualizations")
        
        # Create tabs for different visualization categories
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Data Exploration", "Regression Models", "Classification Models", "Clustering & PCA"]
        )
        
        if viz_type == "Data Exploration":
            st.subheader("Data Exploration Visualizations")
            
            # Correlation matrix
            st.markdown("### Correlation Matrix")
            if os.path.exists("correlation_matrix.png"):
                st.image("correlation_matrix.png")
            else:
                # Create correlation matrix
                df_corr = df_clean.drop(['Date', 'City', 'AQI_Bucket'], axis=1, errors='ignore')
                df_corr = df_corr.select_dtypes(include=['float64', 'int64'])
                
                fig, ax = plt.subplots(figsize=(12, 10))
                corr_matrix = df_corr.corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                plt.title('Correlation Matrix')
                st.pyplot(fig)
            
            # Distribution of pollutants
            st.markdown("### Pollutant Distributions")
            
            # Select pollutant
            pollutants = [col for col in df_clean.columns if col not in ['Date', 'City', 'AQI', 'AQI_Bucket']]
            selected_pollutant = st.selectbox("Select Pollutant", pollutants)
            
            # Plot distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df_clean[selected_pollutant], kde=True, ax=ax)
            plt.title(f"Distribution of {selected_pollutant}")
            plt.xlabel(selected_pollutant)
            plt.ylabel("Frequency")
            st.pyplot(fig)
            
            # Scatter plot with AQI
            st.markdown("### Pollutant vs AQI")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=selected_pollutant, y='AQI', data=df_clean, alpha=0.5, ax=ax)
            plt.title(f"{selected_pollutant} vs AQI")
            plt.xlabel(selected_pollutant)
            plt.ylabel("AQI")
            st.pyplot(fig)
        
        elif viz_type == "Regression Models":
            st.subheader("Regression Model Visualizations")
            
            # Check if visualizations directory exists
            if not os.path.exists("visualizations"):
                st.warning("Visualizations not found. Generating them now...")
                # Create visualizations
                create_regression_visualizations(X, y)
            
            # Display regression model visualizations
            regression_viz_files = [f for f in os.listdir("visualizations") if "regression" in f.lower()]
            
            if not regression_viz_files:
                st.warning("No regression visualizations found. Please run the main_analysis.py script to generate them.")
            else:
                # Group visualizations
                model_comparison = [f for f in regression_viz_files if "comparison" in f.lower()]
                model_viz = [f for f in regression_viz_files if "comparison" not in f.lower()]
                
                # Show model comparison first
                if model_comparison:
                    st.markdown("### Model Comparison")
                    for viz_file in model_comparison:
                        st.image(os.path.join("visualizations", viz_file))
                
                # Show individual model visualizations
                if model_viz:
                    st.markdown("### Individual Model Performance")
                    selected_viz = st.selectbox("Select Model Visualization", model_viz)
                    st.image(os.path.join("visualizations", selected_viz))
        
        elif viz_type == "Classification Models":
            st.subheader("Classification Model Visualizations")
            
            # Check if visualizations directory exists
            if not os.path.exists("visualizations"):
                st.warning("Visualizations not found. Generating them now...")
                # Create visualizations
                create_classification_visualizations(X, y)
            
            # Display classification model visualizations
            classification_viz_files = [f for f in os.listdir("visualizations") if "classification" in f.lower() or "cm" in f.lower()]
            
            if not classification_viz_files:
                st.warning("No classification visualizations found. Please run the main_analysis.py script to generate them.")
            else:
                # Group visualizations
                model_comparison = [f for f in classification_viz_files if "comparison" in f.lower()]
                model_viz = [f for f in classification_viz_files if "comparison" not in f.lower()]
                
                # Show model comparison first
                if model_comparison:
                    st.markdown("### Model Comparison")
                    for viz_file in model_comparison:
                        st.image(os.path.join("visualizations", viz_file))
                
                # Show individual model visualizations
                if model_viz:
                    st.markdown("### Individual Model Performance")
                    selected_viz = st.selectbox("Select Model Visualization", model_viz)
                    st.image(os.path.join("visualizations", selected_viz))
        
        elif viz_type == "Clustering & PCA":
            st.subheader("Clustering and PCA Visualizations")
            
            # Check if visualizations directory exists
            if not os.path.exists("visualizations"):
                st.warning("Visualizations not found. Generating them now...")
                # Create visualizations
                create_clustering_visualizations(X)
            
            # Display clustering visualizations
            clustering_viz_files = [f for f in os.listdir("visualizations") if "cluster" in f.lower() or "pca" in f.lower() or "dbscan" in f.lower() or "kmeans" in f.lower() or "hierarchical" in f.lower()]
            
            if not clustering_viz_files:
                st.warning("No clustering visualizations found. Please run the main_analysis.py script to generate them.")
            else:
                # Group visualizations by type
                kmeans_viz = [f for f in clustering_viz_files if "kmeans" in f.lower()]
                dbscan_viz = [f for f in clustering_viz_files if "dbscan" in f.lower()]
                hierarchical_viz = [f for f in clustering_viz_files if "hierarchical" in f.lower()]
                pca_viz = [f for f in clustering_viz_files if "pca" in f.lower() and "cluster" not in f.lower()]
                
                # Create tabs for different clustering methods
                cluster_tabs = st.tabs(["K-Means", "DBSCAN", "Hierarchical", "PCA"])
                
                with cluster_tabs[0]:
                    st.markdown("### K-Means Clustering")
                    if kmeans_viz:
                        for viz_file in kmeans_viz:
                            st.image(os.path.join("visualizations", viz_file))
                    else:
                        st.warning("No K-Means visualizations found.")
                
                with cluster_tabs[1]:
                    st.markdown("### DBSCAN Clustering")
                    if dbscan_viz:
                        for viz_file in dbscan_viz:
                            st.image(os.path.join("visualizations", viz_file))
                    else:
                        st.warning("No DBSCAN visualizations found.")
                
                with cluster_tabs[2]:
                    st.markdown("### Hierarchical Clustering")
                    if hierarchical_viz:
                        for viz_file in hierarchical_viz:
                            st.image(os.path.join("visualizations", viz_file))
                    else:
                        st.warning("No Hierarchical Clustering visualizations found.")
                
                with cluster_tabs[3]:
                    st.markdown("### PCA Analysis")
                    if pca_viz:
                        for viz_file in pca_viz:
                            st.image(os.path.join("visualizations", viz_file))
                    else:
                        st.warning("No PCA visualizations found.")
    
    # Notebook Viewer page
    elif page == "Notebook Viewer":
        st.title("Jupyter Notebook Viewer")
        
        # Check if notebook exists
        notebook_path = "AQI Prediction Model.ipynb"
        if os.path.exists(notebook_path):
            st.markdown("### AQI Prediction Model Notebook")
            render_notebook(notebook_path)
        else:
            st.error(f"Notebook file not found: {notebook_path}")

if __name__ == "__main__":
    main()
