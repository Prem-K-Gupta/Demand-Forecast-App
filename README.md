
# Demand Forecasting System

This is a **Demand Forecasting System** designed to optimize inventory and supply chain efficiency for a retail business. The application uses historical sales data to forecast demand for the next 15 weeks for the top 10 best-selling products. The primary goal is to help maintain optimal stock levels, ensuring that the supply chain remains efficient and meets customer demands.

## Project Overview

This project involves:
- **Exploratory Data Analysis (EDA)**: To understand sales patterns and customer behavior.
- **Demand Forecasting**: Using **Facebook Prophet**, a powerful forecasting model for time series, to predict the demand for top products.
- **Model Performance**: The app provides error metrics and visualizations for training and testing datasets.

The application is built using **Streamlit** for easy interactivity and visualization.

## Features

- **Exploratory Data Analysis**:
  - Customer, Product, and Transaction-Level Summary Statistics.
  - Visualization of top-selling products by quantity and revenue.

- **Demand Forecasting**:
  - Select a product from the top 10 best-selling items.
  - View historical and forecasted demand over the next 15 weeks.
  - Download the forecasted demand as a CSV file.
  - Error analysis using histograms for both training and testing periods.

## Technologies Used

- **Python** for programming.
- **Streamlit** for building an interactive application.
- **Facebook Prophet** for time series forecasting.
- **Pandas** for data manipulation.
- **Matplotlib** for plotting and visualization.

## Setup and Installation

To run this project on your local machine, follow these steps:

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository_url>
```

### 2. Navigate to the Project Directory

```bash
cd Demand-Forecasting-System
```

### 3. Install the Requirements

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

Start the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

```text
Demand-Forecasting-System/
  ├── app.py                    # The main Streamlit application script
  ├── datasets/                 # Folder containing transactional and product data
  │    ├── Transactional_data_retail_01.csv
  │    ├── Transactional_data_retail_02.csv
  │    ├── CustomerDemographics.csv
  │    └── ProductInfo.csv
  ├── requirements.txt          # List of all the dependencies for the project
  └── README.md                 # Project overview and setup instructions
```

## Usage

- Run the application using the steps mentioned above.
- Navigate between **Exploratory Data Analysis (EDA)** and **Demand Forecasting** using the sidebar.
- Select a stock code to view its **historical sales** and **forecasted demand**.
- **Download the forecast data** for further analysis.


