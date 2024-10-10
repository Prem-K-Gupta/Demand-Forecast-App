import pandas as pd

def load_and_preprocess_data():
    # Load data from CSV files
    transactions1 = pd.read_csv("datasets/Transactional_data_retail_01.csv")
    transactions2 = pd.read_csv("datasets/Transactional_data_retail_02.csv")
    product_info = pd.read_csv("datasets/ProductInfo.csv")
    customer_info = pd.read_csv("datasets/CustomerDemographics.csv")

    # Data cleaning steps (e.g., handling missing values, duplicates)
    transactions1.dropna(inplace=True)
    transactions2.dropna(inplace=True)

    # Merge the transactions data
    transactions = pd.concat([transactions1, transactions2])

    # Join transactions with customer demographics using the customer ID or identifier
    transactions = transactions.merge(customer_info, on='Customer ID', how='left')

    # Customer-level summary statistics (e.g., total purchases, average quantity per customer)
    customer_summary = transactions.groupby('Customer ID').agg({
        'Quantity': 'sum', 
        'Price': 'mean'
    }).reset_index()

    return transactions1, transactions2, product_info, customer_info, customer_summary
