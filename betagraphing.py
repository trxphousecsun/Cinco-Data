from pymongo import MongoClient
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# MongoDB connection setup for company search
client = MongoClient('localhost', 27017)
db = client.history
collection = db.history

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
    beta = model.params[1]
    alpha = model.params[0]
    r_squared = model.rsquared
    std_err = model.bse[1]  # Standard error of beta
    t_stat = model.tvalues[1]  # t-statistic of beta
    # Additional calculations
    num_years = (end_date - start_date).days / 365.25
    annualized_return = np.power((stock_data['Adj Close'].iloc[-1] / stock_data['Adj Close'].iloc[0]), (1/num_years)) - 1
    
    # Create a scatter plot
    plt.scatter(market_returns, stock_returns)
    plt.xlabel('Market Returns')
    plt.ylabel('Stock Returns')
    plt.title(f'Scatter Plot for {stock_symbol}')

    # Calculate and plot the regression line
    regression_line = alpha + beta * market_returns
    plt.plot(market_returns, regression_line, color='red')  # Line of best fit
    
    # Save the plot as an image
    plot_filename = f'/Users/jazzhashzzz/Desktop/testfiles/tech/{stock_symbol}_scatter_plot.png'
    plt.savefig(plot_filename)
    plt.close()
    
    return stock_data, market_data, beta, alpha, r_squared, std_err, t_stat, annualized_return, plot_filename

def main():
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
    market_symbol = 'SPY'
    start_date = dt.datetime(2018, 1, 1)
    end_date = dt.datetime.now()
    try:
        stock_data, market_data, beta, alpha, r_squared, std_err, t_stat, annualized_return, plot_filename = get_stock_data(stock_symbol, market_symbol, start_date, end_date)

        # Prepare results for Excel output
        results_df = pd.DataFrame({
            'Beta': [beta],
            'Alpha': [alpha],
            'R-squared': [r_squared],
            'Std. Error of Beta': [std_err],
            'T-stat of Beta': [t_stat],
            'Annualized Return': [annualized_return]
        })
        
        # Save scatter plot to Excel file
        with pd.ExcelWriter(f'/Users/jazzhashzzz/Desktop/testfiles/tech/{stock_symbol}_financial_data.xlsx', engine='xlsxwriter') as writer:
            stock_data.to_excel(writer, sheet_name=f'{stock_symbol} Data')
            market_data.to_excel(writer, sheet_name='Market Data')
            results_df.to_excel(writer, sheet_name='Statistics')

            # Insert the scatter plot
            workbook  = writer.book
            worksheet = workbook.add_worksheet('Scatter Plot')
            worksheet.insert_image('A1', plot_filename)
        
        print(f"Data and scatter plot successfully saved to Excel file at /Users/jazzhashzzz/Desktop/testfiles/tech/{stock_symbol}_financial_data.xlsx.")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()