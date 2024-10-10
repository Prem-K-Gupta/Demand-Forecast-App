import pandas as pd

def load_and_preprocess_data():
    transactions1 = pd.read_csv("datasets/Transactional_data_retail_01.csv")
    transactions2 = pd.read_csv("datasets/Transactional_data_retail_02.csv")
    product_info = pd.read_csv("datasets/ProductInfo.csv")
    customer_info = pd.read_csv("datasets/CustomerDemographics.csv")

    transactions1.dropna(inplace=True)
    transactions2.dropna(inplace=True)

    transactions = pd.concat([transactions1, transactions2])

    transactions = transactions.merge(customer_info, on='Customer ID', how='left')

    customer_summary = transactions.groupby('Customer ID').agg({
        'Quantity': 'sum', 
        'Price': 'mean'
    }).reset_index()

    return transactions1, transactions2, product_info, customer_info, customer_summary
