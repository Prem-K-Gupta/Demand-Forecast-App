import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
import numpy as np

@st.cache_data
def load_data():
    transactions_01 = pd.read_csv('datasets/Transactional_data_retail_01.csv')
    transactions_02 = pd.read_csv('datasets/Transactional_data_retail_02.csv')
    transactions = pd.concat([transactions_01, transactions_02], ignore_index=True)

    transactions['InvoiceDate'] = pd.to_datetime(transactions['InvoiceDate'], errors='coerce', dayfirst=True)

    transactions.dropna(subset=['InvoiceDate'], inplace=True)

    transactions['Revenue'] = transactions['Quantity'] * transactions['Price']

    return transactions

# Streamlit App
def main():
    st.sidebar.title("Input Options")
    
    transactions = load_data()

    analysis_type = st.sidebar.radio(
        "Choose Analysis Type:",
        ("Exploratory Data Analysis (EDA)", "Demand Forecasting")
    )

    # EDA Section
    if analysis_type == "Exploratory Data Analysis (EDA)":
        st.title("Exploratory Data Analysis (EDA)")

        st.subheader("Customer-Level Summary Statistics")
        customer_summary = transactions.groupby('Customer ID').agg({'Quantity': 'sum', 'Revenue': 'sum'}).reset_index()
        st.write(customer_summary.describe())

        st.subheader("Product-Level Summary Statistics")
        product_summary = transactions.groupby('StockCode').agg({'Quantity': 'sum', 'Revenue': 'sum'}).reset_index()
        st.write(product_summary.describe())

        # Visualizations
        st.subheader("Top 10 Products by Quantity Sold")
        top_10_products_quantity = product_summary.nlargest(10, 'Quantity')
        plt.figure(figsize=(10, 5))
        plt.bar(top_10_products_quantity['StockCode'], top_10_products_quantity['Quantity'], color='blue')
        plt.xlabel('Stock Code')
        plt.ylabel('Total Quantity Sold')
        plt.title('Top 10 Products by Quantity Sold')
        st.pyplot(plt)

        st.subheader("Top 10 Products by Revenue")
        top_10_products_revenue = product_summary.nlargest(10, 'Revenue')
        plt.figure(figsize=(10, 5))
        plt.bar(top_10_products_revenue['StockCode'], top_10_products_revenue['Revenue'], color='green')
        plt.xlabel('Stock Code')
        plt.ylabel('Total Revenue')
        plt.title('Top 10 Products by Revenue')
        st.pyplot(plt)

    # Demand Forecasting Section
    elif analysis_type == "Demand Forecasting":
        # Identify top 10 products by quantity sold
        top_10_stock_codes = transactions.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(10).index.tolist()

        stock_code = st.sidebar.selectbox('Select a Stock Code:', top_10_stock_codes)

        # Title and sub-title
        st.title('Demand Forecasting')
        st.subheader(f'Demand Overview for {stock_code}')

        # Load historical data for the selected product
        product_data = transactions[transactions['StockCode'] == stock_code].set_index('InvoiceDate')
        product_sales = product_data['Quantity'].resample('W').sum().fillna(0)

        # Prepare data for Prophet
        product_sales_df = product_sales.reset_index()
        product_sales_df.columns = ['ds', 'y']  

        train_data = product_sales_df[:-15]
        test_data = product_sales_df[-15:]
        
        model = Prophet(
            changepoint_prior_scale=0.5,   
            seasonality_mode='multiplicative' 
        )

        model.add_seasonality(name='weekly', period=7, fourier_order=3)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        model.fit(train_data)

        future_train = model.make_future_dataframe(periods=15, freq='W')

        forecast = model.predict(future_train)

        forecast_train = forecast[forecast['ds'] <= train_data['ds'].max()]
        forecast_test = forecast[forecast['ds'] > train_data['ds'].max()]

        # Plot Actual vs Predicted
        st.write("Actual vs Predicted Demand")
        plt.figure(figsize=(10, 6))
        plt.plot(train_data['ds'], train_data['y'], 'bo-', label='Train Actual Demand')
        plt.plot(forecast_train['ds'], forecast_train['yhat'], 'r-', label='Train Predicted Demand')
        plt.plot(test_data['ds'], test_data['y'], 'yo-', label='Test Actual Demand')
        plt.plot(forecast_test['ds'], forecast_test['yhat'], 'go-', label='Test Predicted Demand')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.title(f'Actual Vs Predicted Demand for {stock_code}')
        plt.legend()
        st.pyplot(plt)

        # Calculate Errors
        train_mae = mean_absolute_error(train_data['y'], forecast_train['yhat'][:len(train_data)])
        train_rmse = np.sqrt(root_mean_squared_error(train_data['y'], forecast_train['yhat'][:len(train_data)]))
        test_mae = mean_absolute_error(test_data['y'], forecast_test['yhat'][:len(test_data)])
        test_rmse = np.sqrt(root_mean_squared_error(test_data['y'], forecast_test['yhat'][:len(test_data)]))

        st.write(f"Training MAE: {train_mae:.2f}")
        st.write(f"Training RMSE: {train_rmse:.2f}")
        st.write(f"Testing MAE: {test_mae:.2f}")
        st.write(f"Testing RMSE: {test_rmse:.2f}")

        # Plot Training and Testing Error Distributions
        st.write("Error Distribution")

        # Training Error Histogram
        train_errors = train_data['y'] - forecast_train['yhat'][:len(train_data)]
        plt.figure(figsize=(6, 4))
        plt.hist(train_errors, bins=20, color='green', alpha=0.7)
        plt.title('Training Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        st.pyplot(plt)

        # Testing Error Histogram
        test_errors = test_data['y'] - forecast_test['yhat'][:len(test_data)]
        plt.figure(figsize=(6, 4))
        plt.hist(test_errors, bins=20, color='red', alpha=0.7)
        plt.title('Testing Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        st.pyplot(plt)

        st.write("Download Forecast Data")
        forecast_df = forecast[['ds', 'yhat']].tail(15)
        forecast_df.columns = ['Date', 'Forecasted Demand']
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name=f'forecast_{stock_code}.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
