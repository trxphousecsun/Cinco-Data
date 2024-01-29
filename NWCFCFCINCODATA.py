import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import csv
import pandas_datareader.data as pdr
import yfinance as yf
import statsmodels.api as sm
import numpy as np
import datetime as dt

# MongoDB connection setup for 'history' and 'sec' databases
client = MongoClient('localhost', 27017)
db_history = client.history
collection_history = db_history.history

db_sec = client.sec
collection_sec = db_sec.sec

db_sec = client.sec
collection_sec = db_sec.sec

def find_company_by_ticker(ticker_symbol):
    result = collection_history.find_one({"tickers": ticker_symbol}, {"cik": 1, "name": 1, "tickers": 1})
    if result:
        return str(result['cik']), result['name']  # Convert CIK to string
    else:
        return None, None

def fetch_company_section_data(cik, section):
    try:
        cik_int = int(cik)  # Convert CIK to integer for querying sec.sec
    except ValueError:
        print("Invalid CIK format.")
        return None

    query = {'cik': cik_int, f'facts.{section}': {'$exists': True}}
    pipeline = [
        {'$match': query},
        {'$project': {f'facts.{section}': 1}}
    ]
    result = list(collection_sec.aggregate(pipeline))

    if not result:
        print(f"No data found for CIK: {cik_int} in section: {section}")
        return None

    return result[0]['facts'][section]

def search_for_keywords(section_data, keywords):
    results = {}
    for key, value in section_data.items():
        if keywords == ['all']:
            results[key] = value
        else:
            for keyword in keywords:
                if key and isinstance(key, str) and keyword.lower() in key.lower():
                    results[key] = value
                elif value and isinstance(value, dict):
                    label = key if 'ifrs-full' in section_data else value.get('label', '')
                    if label and isinstance(label, str) and keyword.lower() in label.lower():
                        results[key] = value
    return results

def preprocess_data_for_gpt(data, section, company_name):
    processed_data = []
    for key, value in data.items():
        label = key if section == 'ifrs-full' else value.get('label', 'N/A')
        if 'units' in value:
            for currency, items in value['units'].items():
                for item in items:
                    filed_date = item.get('filed', 'N/A')
                    end_date = item.get('end', 'N/A')
                    val = item.get('val', 'N/A')
                    fy = item.get('fy', 'N/A')
                    form = item.get('form', 'N/A')
                    fp = item.get('fp', 'N/A') if 'fp' in item else determine_fp(filed_date, form, item)
                    processed_data.append({
                        'Company Name': company_name,
                        'Label': label,
                        'Value': val,
                        'End Date': end_date,
                        'Form Type': form,
                        'Filing Period': fp,
                        'Filing Year': fy,
                        'Filed Date': filed_date
                    })
    return processed_data

def determine_fp(filed_date, form, item):
    if '10-Q' in form:
        filed_month = datetime.strptime(filed_date, "%Y-%m-%d").month
        if 1 <= filed_month <= 4:
            return 'Q1'
        elif 5 <= filed_month <= 8:
            return 'Q2'
        elif 9 <= filed_month <= 12:
            return 'Q3'
    elif 'frame' in item:
        frame = item['frame']
        if 'H1' in frame:
            return 'H1'
        elif 'H2' in frame:
            return 'H2'
    return 'N/A'

def get_date_filter_options(processed_data):
    unique_entries = sorted({(entry['Filed Date'], entry['Form Type'], entry['Filing Period'], entry['Filing Year'])
                             for entry in processed_data if entry['Filed Date'] != 'N/A'})
    return [{'label': f"{date} | {form_type} | {filing_period} | {filing_year}", 
             'value': date} 
            for date, form_type, filing_period, filing_year in unique_entries]

def style_data_conditional(processed_data):
    styles = []
    for i in range(len(processed_data) - 1):
        if processed_data[i]['Filing Year'] != processed_data[i + 1]['Filing Year']:
            styles.append({'if': {'row_index': i}, 'borderBottom': '2px solid black'})
    return styles

def get_user_input(prompt, options):
    for idx, option in enumerate(options):
        print(f"{idx + 1}: {option}")
    choice = int(input(prompt)) - 1
    return options[choice]

def extract_date(period_label):
    return period_label.split(' | ')[0]  # Assuming the date is the first part of the label

def apply_end_dates_limit(processed_data, limit):
    # Group data by a combination of Label, Filing Year, Form Type, and Filing Period
    grouped_data = {}
    for entry in processed_data:
        group_key = (entry['Label'], entry['Filing Year'], entry['Form Type'], entry['Filing Period'])
        if group_key not in grouped_data:
            grouped_data[group_key] = []
        grouped_data[group_key].append(entry)

    # Sort each group by 'End Date' and apply the limit to take the most recent entry/entries
    limited_data = []
    for group_key, entries in grouped_data.items():
        entries.sort(key=lambda x: datetime.strptime(x['End Date'], "%Y-%m-%d"), reverse=True)
        limited_data.extend(entries[:limit])

    return limited_data

def fetch_stock_data(ticker_symbol, start_date, end_date):
    yf.pdr_override()
    stock_data = pdr.get_data_yahoo(ticker_symbol, start=start_date, end=end_date)
    return stock_data

def calculate_stock_metrics(stock_data):
    stock_returns = stock_data['Adj Close'].pct_change().dropna()
    market_returns = pdr.get_data_yahoo('SPY', start=stock_data.index[0], end=stock_data.index[-1])['Adj Close'].pct_change().dropna()

    stock_returns = stock_returns.reindex(market_returns.index)
    market_returns = market_returns.reindex(stock_returns.index)

    X = sm.add_constant(market_returns)
    model = sm.OLS(stock_returns, X).fit()
    alpha, beta = model.params

    total_return = (stock_data['Adj Close'][-1] / stock_data['Adj Close'][0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(stock_returns)) - 1
    average_annual_return = stock_returns.mean() * 252

    return beta, total_return, annualized_return, average_annual_return


def main():
    ticker_symbol = input("Enter ticker symbol: ").upper()
    cik, company_name = find_company_by_ticker(ticker_symbol)
    
    if not cik:
        print("Company not found for the given ticker symbol.")
        return

    print(f"Selected Company: {company_name} (CIK: {cik})")

    sections = ['us-gaap', 'dei', 'ifrs-full']
    chosen_section = get_user_input("Choose a section: ", sections)

    section_data = fetch_company_section_data(cik, chosen_section)
    if not section_data:
        print("No data found for this section.")
        return
    
    keyword = input("Enter keyword (or 'all' for all data): ")
    keywords = keyword.split(',') if keyword != 'all' else ['all']
    filtered_data = search_for_keywords(section_data, keywords)

    # Use company_name instead of chosen_company
    processed_data = preprocess_data_for_gpt(filtered_data, chosen_section, company_name)

    # Period, Form Type, and Filing Period Selection
    date_options = get_date_filter_options(processed_data)
    date_labels = [option['label'] for option in date_options]
    start_period = get_user_input("Select start period: ", date_labels)
    end_period = get_user_input("Select end period: ", date_labels)

    form_options = {entry['Form Type'] for entry in processed_data}
    chosen_form = get_user_input("Select form type: ", list(form_options))

    filing_periods = {entry['Filing Period'] for entry in processed_data}
    chosen_filing_period = get_user_input("Select filing period: ", list(filing_periods))

    start_date = extract_date(start_period)
    end_date = extract_date(end_period)

    filtered_data = [entry for entry in processed_data if start_date <= entry['Filed Date'] <= end_date]
    filtered_data = [entry for entry in filtered_data if entry['Form Type'] == chosen_form]
    filtered_data = [entry for entry in filtered_data if entry['Filing Period'] == chosen_filing_period]
    
    # End Dates Limit
    end_dates_limit = int(input("Enter end dates limit: "))

    # Apply the end dates limit
    limited_data = apply_end_dates_limit(filtered_data, end_dates_limit)

    # Display and CSV export
    df = pd.DataFrame(limited_data)
    print(df)

    if input("Do you want to save the data to an Excel file? (yes/no): ").lower() == 'yes':
        file_name = input("Enter the filename for the Excel file (without extension): ")
        file_path = f'/Users/jazzhashzzz/Desktop/testfiles/tech/{file_name}.xlsx'
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Financial Statements', index=False)
            
            # Fetch and calculate stock metrics
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            stock_data = fetch_stock_data(ticker_symbol, start_date, end_date)
            beta, total_return, annualized_return, average_annual_return = calculate_stock_metrics(stock_data)

            # Save stock metrics to separate sheets
            stock_metrics_df = pd.DataFrame({
                'Company Name': [company_name],
                'Ticker Symbol': [ticker_symbol],
                'Stock Beta': [beta],
                'Total Return': [total_return],
                'Annualized Return': [annualized_return],
                'Average Annual Return': [average_annual_return]
            })
            stock_metrics_df.to_excel(writer, sheet_name='Stock Metrics', index=False)

            # Save market returns and stock returns as separate sheets
            market_returns = pdr.get_data_yahoo('SPY', start=stock_data.index[0], end=stock_data.index[-1])['Adj Close'].pct_change().dropna()
            market_returns.to_excel(writer, sheet_name='Market Returns', index=False)

            stock_returns = stock_data['Adj Close'].pct_change().dropna()
            stock_returns.to_excel(writer, sheet_name='Stock Returns', index=False)

        print(f"Data saved to {file_path}")

        return file_path 
    
if __name__ == "__main__":
    file_path = main() 

# Read the Excel file
data = pd.read_excel(file_path)



# Define new labels
Operating_Income_labels = [
    "Operating Income (Loss)",
    "Disposal Group, Including Discontinued Operation, Operating Income (Loss)",
    "Other Operating Income",
    "Other Operating Income (Expense), Net",
    "Noninterest Income, Other Operating Income",
    "Segment Reporting Information, Operating Income (Loss) (Deprecated 2011-01-31)",
    "Variable Interest Entity, Measure of Activity, Operating Income or Loss"
]

tax_labels = [
    "Deferred Federal Income Tax Expense (Benefit)",
    "Deferred Income Tax Expense (Benefit)",
    "Income Tax Expense (Benefit)",
    "Current Federal Tax Expense (Benefit)",
    "Current Income Tax Expense (Benefit)",
    "Current State and Local Tax Expense (Benefit)",
    "Current Foreign Tax Expense (Benefit)",
    "Deferred Foreign Income Tax Expense (Benefit)",
    "Deferred State and Local Income Tax Expense (Benefit)",
    "Other Tax Expense (Benefit)",
    "Federal Income Tax Expense (Benefit), Continuing Operations",
    "Foreign Income Tax Expense (Benefit), Continuing Operations",
    "Income Tax Expense (Benefit), Continuing Operations",
    "Discontinued Operation, Tax (Expense) Benefit from Provision for (Gain) Loss on Disposal",
    "Income Tax (Expense) Benefit, Continuing Operations, Government Grants"
]

amort_labels = [
    "Amortization of Deferred Loan Origination Fees, Net",
    "Amortization of Debt Issuance Costs",
    "Amortization",
    "Amortization of Intangible Assets",
    "Capitalized Computer Software, Amortization",
    "Capitalized Contract Cost, Amortization"
]

depreciation_labels = [
    "Depreciation",
    "Capital Leases, Lessee Balance Sheet, Assets by Major Class, Accumulated Depreciation",
    "Operating Leases, Income Statement, Depreciation Expense on Property Subject to or Held-for-lease",
    "Property, Plant and Equipment, Other, Accumulated Depreciation",
    "Property, Plant, and Equipment, Owned, Accumulated Depreciation",
    "Restructuring and Related Cost, Accelerated Depreciation",
    "Depreciation Expense",
    "Depreciation Expense on Reclassified Assets",
    "Flight Equipment, Accumulated Depreciation",
]

capex_label = [
    "Capital Expenditures Incurred but Not yet Paid",
    "Capital Expenditure, Discontinued Operations"
]
current_assets_label = [
    "Assets, Current",
    "Intangible Assets, Current"

]

current_liabilities_label = [
    "Liabilities, Current"
]

# adding negtives to negative things
def get_data_for_label(df, label):
    if isinstance(label, list):
        data = df[df['Label'].isin(label)]
        for lbl in label:
            if "(Loss)" in lbl or "(Decrease)" in lbl:
                data.loc[data['Label'] == lbl, 'Value'] *= -1
        return data
    else:
        data = df[df['Label'] == label] if label in df['Label'].values else pd.DataFrame(columns=df.columns)
        if "(Loss)" in label or "(Decrease)" in label:
            data.loc[:, 'Value'] *= -1
        return data

# Adjusted function to calculate Net Working Capital (NWC) and adjust scale
def calculate_nwc(group):
    current_assets = get_data_for_label(group, current_assets_label)['Value'].sum()
    current_liabilities = get_data_for_label(group, current_liabilities_label)['Value'].sum()
    nwc = current_assets - current_liabilities
    return nwc / 1e6 if abs(nwc) >= 1e9 else nwc / 1e3

def calculate_tax_rate(group, tax_labels):
    tax_expense = sum(get_data_for_label(group, label)['Value'].sum() for label in tax_labels)
    operating_income = sum(get_data_for_label(group, label)['Value'].sum() for label in Operating_Income_labels)
    
    # Avoid division by zero
    if operating_income == 0:
        return 0
    return tax_expense / operating_income

def calculate_fcff(group, previous_nwc):
    # Operating Income (EBIT)
    operating_income = sum(get_data_for_label(group, label)['Value'].sum() for label in Operating_Income_labels)

    # Calculate Effective Tax Rate
    tax_rate = calculate_tax_rate(group, tax_labels)

    # Depreciation & Amortization
    depreciation_amortization = sum(get_data_for_label(group, label)['Value'].sum() for label in depreciation_labels)
    depreciation_amortization += sum(get_data_for_label(group, label)['Value'].sum() for label in amort_labels)

    # Capital Expenditures
    capex = sum(get_data_for_label(group, label)['Value'].sum() for label in capex_label)

    # Change in Non-Cash Working Capital
    current_nwc = calculate_nwc(group)
    delta_nwc = current_nwc - previous_nwc

    # Free Cash Flow to Firm Calculation
    fcff = (operating_income * (1 - tax_rate)) + (capex - depreciation_amortization) - delta_nwc

    # Scaling the FCFF value if it's in billions or millions
    return fcff / 1e6 if abs(fcff) >= 1e9 else fcff / 1e3




# Group the data by 'Company Name' and 'Filing Year'
grouped_data = data.groupby(['Company Name', 'Filing Year'])

# Initialize a list to store the results and a dictionary for previous NWC values
results = []
previous_nwc_values = {}

# Additional list to store detailed data
detailed_data = []

# Updated calculations with FCFF instead of FCF
for (name, year), group in grouped_data:
    previous_nwc = previous_nwc_values.get(name, 0)
    nwc = calculate_nwc(group)
    fcff = calculate_fcff(group, previous_nwc)
    previous_nwc_values[name] = nwc
    results.append({
        'Company Name': name,
        'Filing Year': year,
        'NWC (Thousands/Millions)': nwc,
        'FCFF (Thousands/Millions)': fcff
    })

    # Extract and store detailed data for each label
    for label in Operating_Income_labels + tax_labels + depreciation_labels + amort_labels + capex_label + current_assets_label + current_liabilities_label:
        label_value = get_data_for_label(group, label)['Value'].sum()
        # Determine which formula the label belongs to
        if label in Operating_Income_labels:
            formula = 'Operating Income'
        elif label in tax_labels:
            formula = 'Tax'
        elif label in depreciation_labels or label in amort_labels:
            formula = 'Depreciation & Amortization'
        elif label in capex_label:
            formula = 'CAPEX'
        elif label in current_assets_label + current_liabilities_label:
            formula = 'NWC'
        else:
            formula = 'Other'
        detailed_data.append({
            'Company Name': name,
            'Filing Year': year,
            'Label': label,
            'Value': label_value,
            'Formula': formula
        })

# Convert the results list to a DataFrame
result_df = pd.DataFrame(results)


# Make sure to change 'FCF (Thousands/Millions)' to 'FCFF (Thousands/Millions)'
result_df['NWC % Change'] = result_df.groupby('Company Name')['NWC (Thousands/Millions)'].pct_change() * 100
result_df['FCFF % Change'] = result_df.groupby('Company Name')['FCFF (Thousands/Millions)'].pct_change() * 100

# Sort the DataFrame and reset index
result_df.sort_values(by=['Company Name', 'Filing Year'], inplace=True)
result_df.reset_index(drop=True, inplace=True)
# Convert the detailed data list to a DataFrame
detailed_data_df = pd.DataFrame(detailed_data)

# Filter out rows where the value is 0 for the 'Detailed Label Data' sheet
detailed_data_df = detailed_data_df[detailed_data_df['Value'] != 0]

# Using ExcelWriter to write to new sheets
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
    result_df.to_excel(writer, sheet_name='Calculated Data')
    detailed_data_df.to_excel(writer, sheet_name='Detailed Label Data')

print("Data and detailed label/value mapping successfully saved to new sheets in the Excel file.")
