# Demand Forecasting System

This project is a **Demand Forecasting System** designed to predict demand for the top-selling products in a retail environment. The system uses historical transactional data to estimate future demand for the next 15 weeks, allowing businesses to optimize inventory and ensure efficient supply chain management.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [References](#references)

## Project Overview
The system uses various machine learning and time-series forecasting models (such as **Prophet** and **XGBoost**) to predict demand. The goal is to help retailers maintain optimal stock levels by forecasting demand for the top 10 products based on historical sales data.

### Key Use Case:
- **Forecasting future demand**: For the top 10 best-selling products, forecast demand for the next 15 weeks using historical data from December 2021 to December 2023.
  
### Datasets Used:
- **Transactional Data**: `Transactional_data_retail_01.csv`, `Transactional_data_retail_02.csv`
- **Product Information**: `ProductInfo.csv`
- **Customer Demographics**: `CustomerDemographics.csv`

## Features
- **Exploratory Data Analysis**: Provides customer, item, and transaction-level statistics.
- **Time Series Forecasting**: Uses the Prophet model to forecast future demand for the next 15 weeks.
- **XGBoost Non-Time Series Model**: Uses XGBoost to predict demand based on product pricing and customer country.
- **Error Visualization**: Displays training and testing error distributions.
- **Interactive Web App**: Allows users to select products and view demand forecasts through an interactive web application built with Streamlit.
- **CSV Download**: Users can download forecasted demand data as a CSV file.
  
## Requirements

Before running the project, ensure you have the following installed:

- Python 3.7+
- The following Python packages:
  streamlit pandas numpy prophet matplotlib scikit-learn xgboost statsmodels


You can install the required packages using the command:

```bash
pip install -r requirements.txt

Setup Instructions
1. Clone the Repository: Clone the project repository to your local machine:
git clone https://github.com/your-username/demand-forecasting-system.git
cd demand-forecasting-system

2. Place Datasets: Ensure the following dataset files are present in the datasets/ folder:

Transactional_data_retail_01.csv
Transactional_data_retail_02.csv
CustomerDemographics.csv
ProductInfo.csv

3. Install Dependencies: Install the required dependencies using the requirements.txt file:
pip install -r requirements.txt

4. Run the App: Launch the Streamlit app:
streamlit run app.py

The app will open in your web browser, where you can interactively forecast demand for the top products.

Project Structure
demand_forecasting/
│
├── app.py                      # Main Streamlit application
├── data_preprocessing.py        # Data preprocessing script
├── model_training.py            # Model training and forecasting logic
├── requirements.txt             # Required Python packages
├── README.md                    # Project documentation
├── datasets/                    # Directory to store your dataset CSV files
│   ├── Transactional_data_retail_01.csv
│   ├── Transactional_data_retail_02.csv
│   ├── CustomerDemographics.csv
│   ├── ProductInfo.csv
└── images/                      # Store any images needed (e.g., dashboards, examples)
    └── sample_dashboard.png
Files Description:
app.py: The main Streamlit app, which provides an interactive UI for demand forecasting.
data_preprocessing.py: Contains logic for loading and preprocessing the datasets.
model_training.py: Handles training the forecasting models and calculating errors.
requirements.txt: Lists the Python packages required to run the project.

How It Works
1. Data Preprocessing: The data_preprocessing.py script handles data loading and cleaning. It loads the transactional, product, and customer data, performing necessary transformations (such as grouping by date and summing quantities).

2. Forecasting:
Model Selection: The Prophet time-series model is used for demand forecasting.
Training: The model is trained on the historical sales data from December 2021 to December 2023.
Evaluation: The app computes error metrics (e.g., MAE, RMSE) and visualizes error distributions for both training and testing periods.

3. XGBoost Model:
Non-Time Series Model: The XGBoost model is trained on customer countries and product pricing to predict demand.

4. Visualization and Download:
Forecast Plot: Displays both historical and forecasted demand for the selected product.
Error Histograms: Visualizes the distribution of prediction errors for both the training and testing datasets.
CSV Download: Users can download forecasted demand values for the next 15 weeks as a CSV file.

Usage
1.Select a Stock Code: Use the sidebar to select a stock code (one of the top 10 best-selling products).
2.View the Forecast: The app will display the historical demand and forecast the demand for the next 15 weeks.
3.Analyze Errors: The app will show histograms of training and testing errors, helping you understand the accuracy of the model.
4.Download Forecasted Data: Download the forecasted data for the selected product as a CSV file.