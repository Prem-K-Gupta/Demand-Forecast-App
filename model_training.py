from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
from sklearn.model_selection import train_test_split

def get_top_10_products(transactions1, transactions2):
    transactions = pd.concat([transactions1, transactions2])
    top_10_products = transactions.groupby('StockCode')['Quantity'].sum().nlargest(10).index
    return top_10_products

def get_top_10_revenue_products(transactions1, transactions2, product_info):
    transactions = pd.concat([transactions1, transactions2])
    
    # Join transactions with product info to get price information
    transactions = transactions.merge(product_info, on='StockCode', how='left')
    
    # Calculate revenue (quantity × price)
    transactions['Revenue'] = transactions['Quantity'] * transactions['Price']
    
    # Get top 10 products by revenue
    top_10_revenue_products = transactions.groupby('StockCode')['Revenue'].sum().nlargest(10).index
    
    return top_10_revenue_products

def train_forecast_model(transactions, selected_product):
    product_data = transactions[transactions['StockCode'] == selected_product]
    product_data = product_data.copy()
    product_data.loc[:, 'Date'] = pd.to_datetime(product_data['InvoiceDate'])
    product_data = product_data.groupby('Date')['Quantity'].sum().reset_index()
    product_data = product_data.rename(columns={'Date': 'ds', 'Quantity': 'y'})

    train_data = product_data.iloc[:-15]
    test_data = product_data.iloc[-15:]

    # Prophet Model
    model = Prophet()
    model.fit(train_data)

    # Forecast for 15 weeks
    future = model.make_future_dataframe(periods=15, freq='W')
    forecast = model.predict(future)

    # Evaluation
    y_true = test_data['y']
    y_pred = forecast['yhat'][-15:]
    train_residuals = train_data['y'] - forecast['yhat'][:-15]
    test_residuals = y_true - y_pred

    return model, forecast, train_residuals, test_residuals, y_true, y_pred

# Train an XGBoost model using customer country and product price
def train_xgboost_model(transactions, customer_info, product_info):
    # Merge transactions with customer and product info
    transactions = transactions.merge(customer_info, left_on='Customer ID', right_on='Customer ID', how='left')
    transactions = transactions.merge(product_info, on='StockCode', how='left')

    # Define the features and target
    X = transactions[['Country', 'Price']]  # Using 'Country' and 'Price' as features
    y = transactions['Quantity']  # Predicting 'Quantity'

    # Convert categorical variable 'Country' into numerical values using one-hot encoding
    X = pd.get_dummies(X, columns=['Country'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train XGBoost model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)

    # Predictions
    y_pred = xgb_model.predict(X_test)

    # Evaluation (RMSE)
    rmse = root_mean_squared_error(y_test, y_pred)
    return xgb_model, rmse
