import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from model_training import train_forecast_model, get_top_10_products, get_top_10_revenue_products, train_xgboost_model
from data_preprocessing import load_and_preprocess_data

st.title("Demand Forecasting System")
st.sidebar.title("Input Options")

@st.cache_data
def load_data():
    transactions1, transactions2, product_info, customer_info, customer_summary = load_and_preprocess_data()
    return transactions1, transactions2, product_info, customer_info, customer_summary

transactions1, transactions2, product_info, customer_info, customer_summary = load_data()

st.write("## Customer-Level Summary Statistics")
st.write(customer_summary.head())

top_10_products = get_top_10_products(transactions1, transactions2)

top_10_revenue_products = get_top_10_revenue_products(transactions1, transactions2, product_info)

selected_product = st.sidebar.selectbox("Select a Stock Code (Top 10 by Quantity):", top_10_products)
selected_revenue_product = st.sidebar.selectbox("Select a Stock Code (Top 10 by Revenue):", top_10_revenue_products)

st.write(f"## Demand Forecasting for {selected_product}")
model, forecast, train_residuals, test_residuals, y_true, y_pred = train_forecast_model(transactions1, selected_product)

st.write("### Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.write("### ACF and PACF Plots")
def plot_acf_pacf(data):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    sm.graphics.tsa.plot_acf(data, ax=axs[0])
    axs[0].set_title('ACF Plot')

    sm.graphics.tsa.plot_pacf(data, ax=axs[1])
    axs[1].set_title('PACF Plot')

    st.pyplot(fig)

plot_acf_pacf(pd.Series(forecast['yhat']))

st.write("### Error Distribution")
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].hist(train_residuals, bins=15, color='green', alpha=0.7)
axs[0].set_title('Training Error Distribution')

axs[1].hist(test_residuals, bins=15, color='red', alpha=0.7)
axs[1].set_title('Testing Error Distribution')

st.pyplot(fig)

xgb_model, rmse_xgb = train_xgboost_model(transactions1, customer_info, product_info)
st.write(f"XGBoost RMSE for demand prediction: {rmse_xgb:.2f}")

st.write("### Forecasted Demand for the Next 15 Weeks")
download_data = forecast[['ds', 'yhat']].tail(15)
st.write(download_data)

csv = download_data.to_csv(index=False)
st.download_button(label="Download Forecast Data as CSV", data=csv, file_name='forecast.csv', mime='text/csv')
