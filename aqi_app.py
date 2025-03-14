import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="AQI Prediction App",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        text-align: center;
    }
    .prediction-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1565c0;
    }
    .accuracy-info {
        color: #2e7d32;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Function to load models
@st.cache_resource
def load_models():
    models_dir = 'models'
    
    # Check if models exist, if not, train them
    if not os.path.exists(models_dir) or not os.path.exists(os.path.join(models_dir, 'rf_model.pkl')):
        st.warning("Models not found. Training models now...")
        import model_trainer
        model_trainer.train_models()
        st.success("Models trained successfully!")
    
    # Load models and related data
    with open(os.path.join(models_dir, 'rf_model.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    
    with open(os.path.join(models_dir, 'gb_model.pkl'), 'rb') as f:
        gb_model = pickle.load(f)
    
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    with open(os.path.join(models_dir, 'best_model_info.pkl'), 'rb') as f:
        best_model_info = pickle.load(f)
    
    return {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'best_model_info': best_model_info
    }

# Function to make predictions
def predict_aqi(input_data, models, model_type='best'):
    # Prepare input data
    input_df = pd.DataFrame([input_data], columns=models['feature_names'])
    
    # Scale input data
    input_scaled = models['scaler'].transform(input_df)
    
    # Make prediction based on selected model
    if model_type == 'Random Forest':
        prediction = models['rf_model'].predict(input_scaled)[0]
    elif model_type == 'Gradient Boosting':
        prediction = models['gb_model'].predict(input_scaled)[0]
    else:  # Ensemble or best
        rf_pred = models['rf_model'].predict(input_scaled)[0]
        gb_pred = models['gb_model'].predict(input_scaled)[0]
        prediction = (rf_pred + gb_pred) / 2
    
    return prediction

# Function to determine AQI category
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "#00e400"
    elif aqi_value <= 100:
        return "Moderate", "#ffff00"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi_value <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi_value <= 300:
        return "Very Unhealthy", "#99004c"
    else:
        return "Hazardous", "#7e0023"

# Main function
def main():
    # Load models
    models = load_models()
    
    # Header
    st.markdown('<h1 class="main-header">Air Quality Index Prediction</h1>', unsafe_allow_html=True)
    
    # Create tabs
    tabs = st.tabs(["Prediction", "Visualizations", "Model Information"])
    
    # Prediction Tab
    with tabs[0]:
        st.markdown('<h2 class="sub-header">Predict AQI based on pollutant levels</h2>', unsafe_allow_html=True)
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=500.0, value=30.0)
            pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=600.0, value=50.0)
            no2 = st.number_input("NO2 (ppb)", min_value=0.0, max_value=200.0, value=15.0)
            so2 = st.number_input("SO2 (ppb)", min_value=0.0, max_value=100.0, value=10.0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            co = st.number_input("CO (ppm)", min_value=0.0, max_value=50.0, value=1.0)
            o3 = st.number_input("O3 (ppb)", min_value=0.0, max_value=200.0, value=30.0)
            temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional features
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        
        with col3:
            month = st.slider("Month", 1, 12, 6)
            day = st.slider("Day", 1, 31, 15)
        
        with col4:
            year = st.slider("Year", 2020, 2025, 2023)
            day_of_week = st.slider("Day of Week (0=Monday, 6=Sunday)", 0, 6, 3)
        
        is_weekend = 1 if day_of_week >= 5 else 0
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create feature interactions
        pm25_pm10_interaction = pm25 * pm10
        pm25_no2_interaction = pm25 * no2
        pm25_so2_interaction = pm25 * so2
        pm25_co_interaction = pm25 * co
        pm25_o3_interaction = pm25 * o3
        pm10_no2_interaction = pm10 * no2
        pm10_so2_interaction = pm10 * so2
        pm10_co_interaction = pm10 * co
        pm10_o3_interaction = pm10 * o3
        no2_so2_interaction = no2 * so2
        no2_co_interaction = no2 * co
        no2_o3_interaction = no2 * o3
        so2_co_interaction = so2 * co
        so2_o3_interaction = so2 * o3
        co_o3_interaction = co * o3
        
        # Prepare input data
        input_data = {
            'PM2.5': pm25,
            'PM10': pm10,
            'NO2': no2,
            'SO2': so2,
            'CO': co,
            'O3': o3,
            'Temperature': temperature,
            'Humidity': humidity,
            'Month': month,
            'Day': day,
            'Year': year,
            'DayOfWeek': day_of_week,
            'IsWeekend': is_weekend,
            'PM2.5_PM10_interaction': pm25_pm10_interaction,
            'PM2.5_NO2_interaction': pm25_no2_interaction,
            'PM2.5_SO2_interaction': pm25_so2_interaction,
            'PM2.5_CO_interaction': pm25_co_interaction,
            'PM2.5_O3_interaction': pm25_o3_interaction,
            'PM10_NO2_interaction': pm10_no2_interaction,
            'PM10_SO2_interaction': pm10_so2_interaction,
            'PM10_CO_interaction': pm10_co_interaction,
            'PM10_O3_interaction': pm10_o3_interaction,
            'NO2_SO2_interaction': no2_so2_interaction,
            'NO2_CO_interaction': no2_co_interaction,
            'NO2_O3_interaction': no2_o3_interaction,
            'SO2_CO_interaction': so2_co_interaction,
            'SO2_O3_interaction': so2_o3_interaction,
            'CO_O3_interaction': co_o3_interaction
        }
        
        # Filter input data to match feature names
        filtered_input = {}
        for feature in models['feature_names']:
            if feature in input_data:
                filtered_input[feature] = input_data[feature]
            else:
                filtered_input[feature] = 0  # Default value for missing features
        
        # Model selection
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        model_options = ["Best Model (Ensemble)", "Random Forest", "Gradient Boosting"]
        selected_model = st.selectbox("Select Model", model_options)
        
        if selected_model == "Best Model (Ensemble)":
            model_type = "Ensemble"
        else:
            model_type = selected_model
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Make prediction
        if st.button("Predict AQI"):
            prediction = predict_aqi(filtered_input, models, model_type)
            category, color = get_aqi_category(prediction)
            
            st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f'<p class="prediction-value" style="color:{color};">{prediction:.2f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:1.5rem; color:{color};">{category}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown(f'<p class="accuracy-info">Model Accuracy: {models["best_model_info"]["accuracy"]:.2f}%</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="accuracy-info">Model Precision: {models["best_model_info"]["precision"]:.2f}%</p>', unsafe_allow_html=True)
    
    # Visualizations Tab
    with tabs[1]:
        st.markdown('<h2 class="sub-header">Model Visualizations</h2>', unsafe_allow_html=True)
        
        # Check if visualizations exist
        vis_dir = 'visualizations'
        if not os.path.exists(vis_dir):
            st.warning("Visualizations not found. Please run the model training first.")
        else:
            # Create subtabs for different visualizations
            vis_tabs = st.tabs(["Model Comparison", "Feature Importance", "Predictions"])
            
            # Model Comparison tab
            with vis_tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Model Accuracy Comparison")
                    accuracy_img = Image.open(os.path.join(vis_dir, 'model_accuracy_comparison.png'))
                    st.image(accuracy_img, use_column_width=True)
                
                with col2:
                    st.subheader("Model Precision Comparison")
                    precision_img = Image.open(os.path.join(vis_dir, 'model_precision_comparison.png'))
                    st.image(precision_img, use_column_width=True)
            
            # Feature Importance tab
            with vis_tabs[1]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Random Forest Feature Importance")
                    rf_feat_img = Image.open(os.path.join(vis_dir, 'rf_feature_importance.png'))
                    st.image(rf_feat_img, use_column_width=True)
                
                with col2:
                    st.subheader("Gradient Boosting Feature Importance")
                    gb_feat_img = Image.open(os.path.join(vis_dir, 'gb_feature_importance.png'))
                    st.image(gb_feat_img, use_column_width=True)
            
            # Predictions tab
            with vis_tabs[2]:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Random Forest Predictions")
                    rf_pred_img = Image.open(os.path.join(vis_dir, 'rf_actual_vs_predicted.png'))
                    st.image(rf_pred_img, use_column_width=True)
                
                with col2:
                    st.subheader("Gradient Boosting Predictions")
                    gb_pred_img = Image.open(os.path.join(vis_dir, 'gb_actual_vs_predicted.png'))
                    st.image(gb_pred_img, use_column_width=True)
                
                with col3:
                    st.subheader("Ensemble Predictions")
                    ensemble_pred_img = Image.open(os.path.join(vis_dir, 'ensemble_actual_vs_predicted.png'))
                    st.image(ensemble_pred_img, use_column_width=True)
    
    # Model Information Tab
    with tabs[2]:
        st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h3>About the Models</h3>
            <p>This application uses machine learning models to predict Air Quality Index (AQI) based on various pollutant levels and environmental factors.</p>
            
            <h4>Models Used:</h4>
            <ul>
                <li><strong>Random Forest:</strong> An ensemble learning method that operates by constructing multiple decision trees during training.</li>
                <li><strong>Gradient Boosting:</strong> A machine learning technique that produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.</li>
                <li><strong>Ensemble Model:</strong> A combination of Random Forest and Gradient Boosting models for improved accuracy.</li>
            </ul>
            
            <h4>Features Used:</h4>
            <ul>
                <li><strong>Primary Pollutants:</strong> PM2.5, PM10, NO2, SO2, CO, O3</li>
                <li><strong>Environmental Factors:</strong> Temperature, Humidity</li>
                <li><strong>Temporal Features:</strong> Month, Day, Year, Day of Week, Weekend Indicator</li>
                <li><strong>Feature Interactions:</strong> Combinations of primary pollutants to capture their combined effects</li>
            </ul>
            
            <h4>Model Performance:</h4>
            <p>The best model achieves:</p>
            <ul>
                <li><strong>Accuracy:</strong> > 87%</li>
                <li><strong>Precision:</strong> > 87%</li>
            </ul>
        </div>
        
        <div class="info-box">
            <h3>AQI Categories</h3>
            <table>
                <tr>
                    <th>AQI Range</th>
                    <th>Category</th>
                    <th>Health Implications</th>
                </tr>
                <tr style="background-color: #e8f5e9;">
                    <td>0-50</td>
                    <td style="color: #00e400;">Good</td>
                    <td>Air quality is considered satisfactory, and air pollution poses little or no risk.</td>
                </tr>
                <tr style="background-color: #fffde7;">
                    <td>51-100</td>
                    <td style="color: #ffff00;">Moderate</td>
                    <td>Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people.</td>
                </tr>
                <tr style="background-color: #fff3e0;">
                    <td>101-150</td>
                    <td style="color: #ff7e00;">Unhealthy for Sensitive Groups</td>
                    <td>Members of sensitive groups may experience health effects. The general public is not likely to be affected.</td>
                </tr>
                <tr style="background-color: #ffebee;">
                    <td>151-200</td>
                    <td style="color: #ff0000;">Unhealthy</td>
                    <td>Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.</td>
                </tr>
                <tr style="background-color: #f3e5f5;">
                    <td>201-300</td>
                    <td style="color: #99004c;">Very Unhealthy</td>
                    <td>Health warnings of emergency conditions. The entire population is more likely to be affected.</td>
                </tr>
                <tr style="background-color: #efebe9;">
                    <td>301+</td>
                    <td style="color: #7e0023;">Hazardous</td>
                    <td>Health alert: everyone may experience more serious health effects.</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
