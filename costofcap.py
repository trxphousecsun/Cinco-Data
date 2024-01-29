import os
import openpyxl
from openpyxl import Workbook
from pymongo import MongoClient
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
import yfinance as yf
import statsmodels.api as sm
import re

# MongoDB connection setup for company search
client = MongoClient('localhost', 27017)
db = client.history
collection = db.history

# Define a function to read in the financial data
def read_financial_data(path_to_financials):
    # Reads the financial data workbook and returns it as a pandas DataFrame
    return pd.read_excel(path_to_financials)

# Define a function to read in the company default spread
def read_company_default_spread():
    # You will need to replace 'path_to_spreadsheet' with the actual path to your spreadsheet
    spread_data = {
        "TIE Range": [(-100000, 0.199999), (0.2, 0.649999), (0.65, 0.799999), (0.8, 1.249999), (1.25, 1.499999), (1.5, 1.749999), (1.75, 1.999999),
                      (2, 2.2499999), (2.25, 2.49999), (2.5, 2.999999), (3, 4.249999), (4.25, 5.499999), (5.5, 6.499999), (6.5, 8.499999), (8.5, 100000)],
        "Spread (%)": [20.00, 17.50, 15.78, 11.57, 7.37, 5.26, 4.55, 3.13, 2.42, 2.00, 1.62, 1.42, 1.23, 0.85, 0.69]
    }
    spread_df = pd.DataFrame(data=spread_data)
    return spread_df

# Define a function to read in the T-bill rates
def read_tbill_rates(path_to_tbill_excel):
    # Load T-bill rates from Excel
    tbill_df = pd.read_excel(path_to_tbill_excel, index_col='Date')
    return tbill_df

# Define a function to get the risk-free rate based on the user's selected time horizon
def get_risk_free_rate(tbill_df, time_horizon):
    # Assuming the time_horizon is a string matching the column names in the T-bill rates DataFrame
    latest_rates = tbill_df.iloc[0]  # Assumes the most recent rates are in the first row
    risk_free_rate = latest_rates[time_horizon] / 100  # Convert percentage to decimal
    return risk_free_rate

# Define a function to get the company's default spread based on the TIE ratio
def get_company_default_spread(spread_df, tie_ratio):
    # Find the spread range that the TIE ratio falls into
    for index, row in spread_df.iterrows():
        if row['TIE Range'][0] < tie_ratio <= row['TIE Range'][1]:
            return row['Spread (%)'] / 100  # Convert percentage to decimal
    return None  # If no match found, return None

def calculate_tie_ratio(financial_data):
    operating_income_regex = r'^Operating Income.*'  # Matches strings that start with 'Operating Income'
    interest_expense_regex = r'^Interest Expense.*'  # Matches strings that start with 'Interest Expense'
    
    # Sort the financial data by 'Filed Date' in descending order to get the most recent entries first
    financial_data_sorted = financial_data.sort_values(by='Filed Date', ascending=False)

    # Find the most recent rows that match the regex patterns
    operating_income_row = financial_data_sorted[financial_data_sorted['Label'].str.match(operating_income_regex, na=False)]
    interest_expense_row = financial_data_sorted[financial_data_sorted['Label'].str.match(interest_expense_regex, na=False)]

    # Check if we found the rows, and then get the 'Value' for each label
    if not operating_income_row.empty and not interest_expense_row.empty:
        operating_income = operating_income_row['Value'].iloc[0]  # Get the first (most recent) match
        interest_expense = interest_expense_row['Value'].iloc[0]  # Get the first (most recent) match
    else:
        raise ValueError("Could not find the required labels in the financial data.")

    # Convert values to float if they are not already
    operating_income = float(str(operating_income).replace(',', '')) if isinstance(operating_income, str) else operating_income
    interest_expense = float(str(interest_expense).replace(',', '')) if isinstance(interest_expense, str) else interest_expense
    
    # Avoid division by zero
    if interest_expense == 0 or interest_expense is None:
        raise ValueError("Interest Expense is zero or not found, cannot calculate TIE ratio.")
    
    tie_ratio = operating_income / interest_expense
    return tie_ratio, operating_income_row.iloc[0], interest_expense_row.iloc[0]

# Helper function to calculate cost of debt for a single company
def calculate_cost_of_debt_single(financials_path, spread_df, tbill_df, time_horizon):
    financial_data = read_financial_data(financials_path)
    tie_ratio, _, _ = calculate_tie_ratio(financial_data)
    company_default_spread = get_company_default_spread(spread_df, tie_ratio)
    
    if company_default_spread is None:
        raise ValueError(f"No matching company default spread found for {financials_path}.")
    
    risk_free_rate = get_risk_free_rate(tbill_df, time_horizon)
    cost_of_debt = risk_free_rate + company_default_spread
    return cost_of_debt, None, None

# Helper function to calculate cost of equity for a single company
def calculate_cost_of_equity_single(stock_symbol, market_symbol, start_date, end_date, tbill_df, time_horizon):
    stock_beta, market_returns = get_stock_data(stock_symbol, market_symbol, start_date, end_date)
    risk_free_rate = get_risk_free_rate(tbill_df, time_horizon)
    market_return = market_returns.mean() * 252  # Annualize the market return
    expected_return = calculate_capm(stock_beta, market_return, risk_free_rate)
    return expected_return

# Modify the calculate_cost_of_debt function
def calculate_cost_of_debt_and_equity(folder_path, tbill_path, time_horizon):
    spread_df = read_company_default_spread()
    tbill_df = read_tbill_rates(tbill_path)
    
    # Initialize an empty list to store DataFrames for debt and equity results
    debt_results_df_list = []
    equity_results_df_list = []
    
    # Iterate over XLSX files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx"):
            financials_path = os.path.join(folder_path, filename)
            
            try:
                # Calculate cost of debt for each company
                cost_of_debt, _, _ = calculate_cost_of_debt_single(financials_path, spread_df, tbill_df, time_horizon)
                
                # Create a DataFrame for the current debt result
                company_name = filename.split(".xlsx")[0]
                debt_result_df = pd.DataFrame({"Company": [company_name], "Cost of Debt": [cost_of_debt]})
                
                # Append the DataFrame to the list
                debt_results_df_list.append(debt_result_df)
                
                # Calculate cost of equity for each company
                stock_symbol = input(f"Enter the stock symbol for {company_name}: ")
                market_symbol = input(f"Enter the market symbol for {company_name} (e.g., S&P 500): ")
                start_date = input("Enter the start date for stock data (YYYY-MM-DD): ")
                end_date = input("Enter the end date for stock data (YYYY-MM-DD): ")
                
                cost_of_equity = calculate_cost_of_equity_single(stock_symbol, market_symbol, start_date, end_date, tbill_df, time_horizon)
                
                # Create a DataFrame for the current equity result
                equity_result_df = pd.DataFrame({"Company": [company_name], "Cost of Equity": [cost_of_equity]})
                
                # Append the DataFrame to the list
                equity_results_df_list.append(equity_result_df)
                
            except Exception as e:
                print(f"An error occurred for {filename}: {e}")
    
    # Concatenate all DataFrames in the debt and equity lists
    debt_results_df = pd.concat(debt_results_df_list, ignore_index=True)
    equity_results_df = pd.concat(equity_results_df_list, ignore_index=True)
    
    # Merge debt and equity DataFrames based on the 'Company' column
    combined_df = pd.merge(debt_results_df, equity_results_df, on='Company')
    
    return combined_df

def find_companies(input_value):
    # Search by ticker symbol first
    results = list(collection.find({"tickers": input_value.upper()}, {"cik": 1, "name": 1, "tickers": 1}))
    
    # If no results found, search by company name
    if not results:
        results = list(collection.find({"name": {"$regex": input_value, "$options": "i"}}, {"cik": 1, "name": 1, "tickers": 1}))
    
    return results

def select_company(results):
    if not results:
        print("No company found for the given input.")
        return None

    print("\nSelect a company from the list:")
    for i, company in enumerate(results):
        print(f"{i+1}: {company['name']} (Tickers: {', '.join(company.get('tickers', []))})")
    
    choice = int(input("\nEnter your choice (number): "))
    return results[choice - 1] if 1 <= choice <= len(results) else None

def get_stock_data(stock_symbol, market_symbol, start_date, end_date):
    # Fetch the stock and market data
    yf.pdr_override()
    stock_data = pdr.get_data_yahoo(stock_symbol, start=start_date, end=end_date)
    market_data = pdr.get_data_yahoo(market_symbol, start=start_date, end=end_date)

    # Calculate returns for stock and market
    stock_returns = stock_data['Adj Close'].pct_change().dropna()
    market_returns = market_data['Adj Close'].pct_change().dropna()

    # Beta calculation using regression
    X = sm.add_constant(market_returns)  # Adding a constant for the regression model
    y = stock_returns
    model = sm.OLS(y, X).fit()
    stock_beta = model.params[1]  # Beta is the coefficient of the market return

    return stock_beta, market_returns

def get_risk_free_rate_from_csv(csv_path):
    # Read the CSV and find the most recent 10-year yield
    rates_df = pd.read_csv(csv_path)
    rates_df['Date'] = pd.to_datetime(rates_df['Date'])
    most_recent_rate = rates_df[rates_df['Date'] <= pd.Timestamp.now()].iloc[-1]
    risk_free_rate = most_recent_rate['5 Yr'] / 100  # Convert percentage to decimal
    return risk_free_rate

def calculate_capm(stock_beta, market_return, risk_free_rate):
    # CAPM calculation
    market_premium = market_return - risk_free_rate
    expected_return = risk_free_rate + stock_beta * market_premium
    return expected_return


def main():
    # User input for folder path
    folder_path = input("Enter the folder path containing XLSX files: ")
    
    if not os.path.exists(folder_path):
        print("Folder path does not exist.")
        return
    
    # User input for other parameters (tbill_path, time_horizon, etc.)
    tbill_path = '/Users/Jazzhashzzz/Documents/daily-treasury-rates.xlsx'  # Replace with the actual path
    time_horizon = '5 Yr'  # User-specified time horizon for T-bill rates

    try:
        # Calculate cost of debt and cost of equity for all companies in the folder
        debt_equity_df = calculate_cost_of_debt_and_equity(folder_path, tbill_path, time_horizon)
        
        # Save results to an Excel file
        output_file = 'cost_of_capital_results.xlsx'
        debt_equity_df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
