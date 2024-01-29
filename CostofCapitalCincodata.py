# Import necessary libraries
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


# Main function to calculate cost of debt
def calculate_cost_of_debt(financials_path, tbill_path, time_horizon):
    spread_df = read_company_default_spread()
    financial_data = read_financial_data(financials_path)
    tbill_df = read_tbill_rates(tbill_path)
    
    tie_ratio = calculate_tie_ratio(financial_data)
    tie_ratio, operating_income_info, interest_expense_info = calculate_tie_ratio(financial_data)
    company_default_spread = get_company_default_spread(spread_df, tie_ratio)
    if company_default_spread is None:
        raise ValueError("No matching company default spread found for the given TIE ratio.")
    
    risk_free_rate = get_risk_free_rate(tbill_df, time_horizon)
    cost_of_debt = risk_free_rate + company_default_spread
    return cost_of_debt, operating_income_info, interest_expense_info

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


# Combined main function
def main():
    # User input for company search
    input_value = input("Enter ticker symbol or company name: ")
    results = find_companies(input_value)
    selected_company = select_company(results)

    if not selected_company:
        print("No valid selection made.")
        return

    stock_symbol = selected_company['tickers'][0] if selected_company.get('tickers') else None
    if not stock_symbol:
        print("Ticker symbol not found for the selected company.")
        return

    # Financial data fetching
    market_symbol = 'SPY'  # Default market symbol for CAPM
    start_date = dt.datetime(2018, 1, 1)
    end_date = dt.datetime(2023, 12, 18)


    try:
        # CAPM calculation for cost of equity
        stock_beta, market_returns = get_stock_data(stock_symbol, market_symbol, start_date, end_date)
        risk_free_rate = get_risk_free_rate_from_csv('/Users/jazzhashzzz/Desktop/CincoData/Tbills/treasuryrates.csv')  # Corrected function call
        market_return = market_returns.mean() * 252  # Annualize the market return
        expected_return = calculate_capm(stock_beta, market_return, risk_free_rate)


        # Cost of debt calculation
        financials_path = '/Users/Jazzhashzzz/Desktop/testfiles/tech/inteldsfasdfds.xlsx'  # Replace with the actual path
        tbill_path = '/Users/jazzhashzzz/Desktop/CincoData/Tbills/treasuryrates.csv'  # Replace with the actual path to the CSV file
        time_horizon = '5 Yr'  # User-specified time horizon for T-bill rates

        cost_of_debt, operating_income_info, interest_expense_info = calculate_cost_of_debt(financials_path, tbill_path, time_horizon)

        # Display results
        print(f"\nCompany: {selected_company['name']}")
        print(f"Expected Return using CAPM (Cost of Equity): {expected_return:.2%}")
        print(f"Cost of Debt: {cost_of_debt:.2%}")

        info_df = pd.DataFrame([operating_income_info, interest_expense_info])
        print("\nFinancial Information Used for Cost of Debt Calculation:")
        print(info_df)

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
