import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import csv

# MongoDB connection setup
client = MongoClient('localhost', 27017)
db_history = client.history
collection_history = db_history.history
db_sec = client.sec
collection_sec = db_sec.sec

def fetch_industries():
    industries = collection_history.distinct("sicDescription")
    return industries

def find_companies_by_industry(industry):
    companies = collection_history.find({"sicDescription": industry}, {"cik": 1, "name": 1})
    return [(str(company['cik']), company['name']) for company in companies]

def fetch_company_section_data(cik, section, keywords):
    try:
        cik_int = int(cik)
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

    return search_for_keywords(result[0]['facts'][section], keywords)

def search_for_keywords(section_data, keywords):
    results = {}
    for key, value in section_data.items():
        if keywords == ['all']:
            results[key] = value
        else:
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if key and isinstance(key, str) and keyword_lower in key.lower():
                    results[key] = value
                elif value and isinstance(value, dict):
                    label = key if 'ifrs-full' in section_data else value.get('label', '')
                    if label and isinstance(label, str) and keyword_lower in label.lower():
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

# ... [Earlier parts of your script] ...

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


# ... [Previous parts of the script] ...

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
def select_multiple_options(prompt, options, all_option=True):
    """
    Function to allow multiple selections from a list of options.
    """
    print(prompt)
    for idx, option in enumerate(options):
        print(f"{idx + 1}: {option}")
    if all_option:
        print(f"{len(options) + 1}: Select All")

    selections = input("Enter your choices separated by commas (e.g., 1,3,5): ").split(',')
    selected_options = []
    for selection in selections:
        if all_option and int(selection) == len(options) + 1:
            return options
        selected_options.append(options[int(selection) - 1])

    return selected_options

def main():
    industries = fetch_industries()
    chosen_industry = get_user_input("Choose an industry: ", industries)
    companies_in_industry = find_companies_by_industry(chosen_industry)

    company_options = [f"{cik}: {name}" for cik, name in companies_in_industry]
    selected_companies = select_multiple_options("Select companies (or select 'All'):", company_options)

    selected_cik_names = [(cik.split(':')[0], cik.split(': ')[1]) for cik in selected_companies]

    all_processed_data = []
    for cik, company_name in selected_cik_names:
        for section in ['us-gaap', 'ifrs-full']:
            keyword = input(f"Enter keyword(s) for {company_name} in {section} (comma-separated, 'all' for all data): ")
            keywords = keyword.split(',') if keyword != 'all' else ['all']
            section_data = fetch_company_section_data(cik, section, keywords)

            if section_data:
                processed_data = preprocess_data_for_gpt(section_data, section, company_name)
                all_processed_data.extend(processed_data)

    # User selects Form Type
    form_types = set(entry['Form Type'] for entry in all_processed_data)
    chosen_form_type = get_user_input("Select a Form Type: ", list(form_types))

    # User selects Filing Period
    filing_periods = set(entry['Filing Period'] for entry in all_processed_data)
    chosen_filing_period = get_user_input("Select a Filing Period: ", list(filing_periods))

    # Filter data by chosen Form Type and Filing Period
    filtered_data = [entry for entry in all_processed_data if entry['Form Type'] == chosen_form_type and entry['Filing Period'] == chosen_filing_period]

    # Apply end dates limit
    end_dates_limit = int(input("Enter end dates limit: "))
    limited_data = apply_end_dates_limit(filtered_data, end_dates_limit)

    df = pd.DataFrame(limited_data)
    print(df)

    # Save data option
    if input("Do you want to save the data to an Excel file? (yes/no): ").lower() == 'yes':
        file_name = input("Enter the filename for the Excel file (without extension): ") + '.xlsx'
        path = f'/Users/jazzhashzzz/Desktop/testfiles/{file_name}'
        df.to_excel(path, index=False)
        print(f"Data saved to {path}")

if __name__ == "__main__":
    main()
