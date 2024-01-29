from pymongo import MongoClient
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

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
    return stock_data, market_data, beta, alpha, r_squared, std_err, t_stat, annualized_return

def plot_data(stock_data, market_data, beta, alpha):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Scatter plot for beta calculation
    stock_returns = stock_data['Adj Close'].pct_change().dropna()
    market_returns = market_data['Adj Close'].pct_change().dropna()
    ax[0].scatter(market_returns, stock_returns, color='orange')
    ax[0].plot(market_returns, beta*market_returns + alpha, color='red')  # Regression line
    ax[0].set_xlabel('Market Returns')
    ax[0].set_ylabel('Stock Returns')
    ax[0].set_title('Stock Returns vs Market Returns')

    # Time series plot for stock prices
    ax[1].plot(stock_data.index, stock_data['Adj Close'], color='blue')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Adjusted Close Price')
    ax[1].set_title('Stock Price Over Time')

    plt.tight_layout()
    return fig

def create_gui(stock_data, market_data, beta, alpha, r_squared, std_err, t_stat, annualized_return):
    root = tk.Tk()
    root.title("Stock Data Visualization")

    # Create a figure and a canvas to show it on the GUI
    fig = plot_data(stock_data, market_data, beta, alpha)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Additional statistics as labels
    stats_text = f'Beta: {beta}\nAlpha: {alpha}\nR-squared: {r_squared}\nStd. Error of Beta: {std_err}\nT-stat of Beta: {t_stat}\nAnnualized Return: {annualized_return}'
    label = tk.Label(root, text=stats_text, justify='left')
    label.pack()

    # Run the Tkinter event loop
    root.mainloop()

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
        stock_data, market_data, beta, alpha, r_squared, std_err, t_stat, annualized_return = get_stock_data(stock_symbol, market_symbol, start_date, end_date)

        if stock_data.empty or market_data.empty:
            print("Stock or market data is empty. Exiting.")
            return

        # Prepare results for Excel output
        results_df = pd.DataFrame({
            'Beta': [beta],
            'Alpha': [alpha],
            'R-squared': [r_squared],
            'Std. Error of Beta': [std_err],
            'T-stat of Beta': [t_stat],
            'Annualized Return': [annualized_return]
        })
        
        # Save data to Excel file
        with pd.ExcelWriter(f'/Users/jazzhashzzz/Desktop/{stock_symbol}_financial_data.xlsx') as writer:
            stock_data.to_excel(writer, sheet_name=f'{stock_symbol} Data')
            market_data.to_excel(writer, sheet_name='Market Data')
            results_df.to_excel(writer, sheet_name='Statistics')
        
        print(f"Data successfully saved to Excel file at /Users/jazzhashzzz/Desktop/tech/{stock_symbol}_financial_data.xlsx.")
        
        # Instead of printing, launch the GUI
        create_gui(stock_data, market_data, beta, alpha, r_squared, std_err, t_stat, annualized_return)

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
