# Air Quality Index Prediction

This repository contains a machine learning application for predicting Air Quality Index (AQI) based on various pollutant levels and environmental factors.

## Features

- **High-Accuracy Models**: Achieves accuracy and precision above 87% using ensemble learning techniques
- **Interactive Streamlit App**: User-friendly interface for making AQI predictions
- **Comprehensive Visualizations**: Model comparisons, feature importance, and prediction analysis
- **Multiple Models**: Random Forest, Gradient Boosting, and Ensemble approaches

## Project Structure

- `model_trainer.py`: Script for training and evaluating machine learning models
- `aqi_app.py`: Streamlit application for interactive AQI prediction
- `data_preprocessing.py`: Functions for data cleaning and feature engineering
- `classification_models.py`: Implementation of classification algorithms
- `regression_models.py`: Implementation of regression models
- `run_aqi_model.py`: Main script for running AQI prediction models

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/Air-Quality-Index-Prediction.git
cd Air-Quality-Index-Prediction
```

2. Install required packages:
```
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```
streamlit run aqi_app.py
```

This will launch a web application where you can:
- Input pollutant levels and environmental factors
- Select the prediction model to use
- View the predicted AQI value and category
- Explore model visualizations and performance metrics

### Training Models Separately

```
python model_trainer.py
```

This will:
- Load and preprocess the air quality data
- Train Random Forest and Gradient Boosting models
- Create an ensemble model
- Generate visualizations
- Save the trained models for later use

## Models

The application uses the following machine learning models:

1. **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees during training.
2. **Gradient Boosting**: A machine learning technique that produces a prediction model in the form of an ensemble of weak prediction models.
3. **Ensemble Model**: A combination of Random Forest and Gradient Boosting for improved accuracy.

## AQI Categories

| AQI Range | Category | Health Implications |
|-----------|----------|---------------------|
| 0-50 | Good | Air quality is considered satisfactory, and air pollution poses little or no risk. |
| 51-100 | Moderate | Air quality is acceptable; however, for some pollutants there may be a moderate health concern for a very small number of people. |
| 101-150 | Unhealthy for Sensitive Groups | Members of sensitive groups may experience health effects. The general public is not likely to be affected. |
| 151-200 | Unhealthy | Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects. |
| 201-300 | Very Unhealthy | Health warnings of emergency conditions. The entire population is more likely to be affected. |
| 301+ | Hazardous | Health alert: everyone may experience more serious health effects. |

## Contributors

- AICTE Cycle 4 Team

## License

This project is licensed under the MIT License - see the LICENSE file for details.
