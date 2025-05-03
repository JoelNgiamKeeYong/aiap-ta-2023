# src/clean_data.py

import time
import pandas as pd

def clean_data(
    df, save_data=False, 
    remove_irrelevant_features=False, irrelevant_features=[], 
    columns_to_clean=[]
):
    """
    Cleans and preprocesses the raw dataset to prepare it for machine learning modeling.

    Parameters:
        df (pd.DataFrame): The raw dataset.
        save_data (bool, optional): Whether to save the cleaned data to a CSV file. Defaults to False.
        remove_irrelevant_features (bool, optional): Whether to remove irrelevant features. Defaults to False.
        irrelevant_features (list, optional): List of column names to remove if `remove_irrelevant_features` is True. Defaults to [].
        columns_to_clean (list, optional): List of column names to clean. Defaults to [].

    Returns:
        pd.DataFrame: A cleaned DataFrame ready for preprocessing and modeling.

    Raises:
        ValueError: If a specified column to clean does not exist in the DataFrame.
        RuntimeError: If an error occurs during the cleaning process.
    """
    try:
        print("\n🧼 Cleaning the dataset...")
        start_time = time.time()

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.copy()

        # Step 1: Remove irrelevant features
        if remove_irrelevant_features and len(irrelevant_features) > 0:
            print(f"   └── Removing irrelevant columns: {irrelevant_features}...")
            df_cleaned = df_cleaned.drop(columns=irrelevant_features, errors='ignore')

        # Step 2: Apply cleaning steps to specified columns
        if columns_to_clean:
            print("   └── Cleaning specified columns...")
            
            # Define a mapping of column names to their respective cleaning functions
            cleaners = {
                "final_test": clean_final_test_function,
                "branch": clean_branch_function,
                "booking_month": clean_booking_month_function,
                "arrival_month": clean_arrival_month_function,
                "arrival_day": clean_arrival_day_function,
                "checkout_month": clean_checkout_month_function,
                "checkout_day": clean_checkout_day_function,
                "country": clean_country_function,
                "first_time": clean_first_time_function,
                "room": clean_room_function,
                "price": clean_price_function,
                "platform": clean_platform_function,
                "num_adults": clean_num_adults_function,
                "num_children": clean_num_children_function,
            }

            # Iterate over the columns to clean
            for column_name in columns_to_clean:
                if column_name in cleaners:
                    print(f"\n   🫧  Cleaning {column_name} column...")
                    
                    # Ensure the column exists in the DataFrame
                    if column_name not in df_cleaned.columns:
                        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
                    
                    # Apply the cleaning function
                    df_cleaned = cleaners[column_name](df_cleaned, column_name)
                else:
                    print(f"⚠️  No cleaning function defined for column '{column_name}'. Skipping...")

        # Step 3: Save the cleaned data to a CSV file
        if save_data:
            output_path = "./data/cleaned_data.csv"
            print(f"\n💾 Saving cleaned data to {output_path}...")
            df_cleaned.to_csv(output_path, index=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n✅ Data cleaning completed in {elapsed_time:.2f} seconds!")

        # Return cleaned dataset
        return df_cleaned

    except Exception as e:
        print(f"\n❌ An error occurred during data cleaning: {e}")
        raise RuntimeError("Data cleaning process failed.") from e


def clean_final_test_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()
    
    # Step 1: Convert the column to integer type and then to categorical type
    print (f"      └── Converting {column_name} column to categorical type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('int64')
    
    return df_cleaned

def clean_branch_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Convert the column to categorical type
    print(f"      └── Converting {column_name} column to categorical type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('category')
    
    return df_cleaned

def clean_booking_month_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Convert the column to categorical type
    print(f"      └── Converting {column_name} column to categorical type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('category')
    
    return df_cleaned

def clean_arrival_month_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Define a mapping from lowercase month names to their proper forms
    month_mapping = {
        'january': 'January',
        'february': 'February',
        'march': 'March',
        'april': 'April',
        'may': 'May',
        'june': 'June',
        'july': 'July',
        'august': 'August',
        'september': 'September',
        'october': 'October',
        'november': 'November',
        'december': 'December'
    }

    # Convert the column values to lowercase for case-insensitive comparison
    print(f"      └── Converting {column_name} values to lowercase...")
    df_cleaned[column_name] = df_cleaned[column_name].str.lower()

    # Map the lowercase values to their proper forms using the dictionary
    print(f"      └── Standardizing {column_name} values...")
    df_cleaned[column_name] = df_cleaned[column_name].map(month_mapping).fillna(df_cleaned[column_name])

    # Convert to categorical type for better memory efficiency and performance
    print(f"      └── Converting {column_name} column to categorical type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('category')

    return df_cleaned

def clean_arrival_day_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Convert the column to integer type
    print(f"      └── Converting {column_name} column to integer type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('int64')

    return df_cleaned

def clean_checkout_month_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Convert the column to categorical type
    print(f"      └── Converting {column_name} column to categorical type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('category')

    return df_cleaned

def clean_checkout_day_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Convert negative values to positive using abs()
    print(f"      └── Converting negative values in {column_name} to positive...")
    df_cleaned[column_name] = df_cleaned[column_name].apply(abs)

    # Convert the column to integer type
    print(f"      └── Converting {column_name} column to integer type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('int64')

    return df_cleaned

def clean_country_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Convert the column to categorical type
    print(f"      └── Converting {column_name} column to categorical type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('category')

    return df_cleaned

def clean_first_time_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Map "Yes" to 1 and "No" to 0
    print(f"      └── Mapping 'Yes' to 1 and 'No' to 0 in {column_name} column...")
    df_cleaned[column_name] = df_cleaned[column_name].map({'Yes': 1, 'No': 0})

    # Convert the column to integer type
    print(f"      └── Converting {column_name} column to integer type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('int64')

    # Convert the column to categorical type for better memory efficiency
    print(f"      └── Converting {column_name} column to categorical type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('category')

    return df_cleaned

def clean_room_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Step 1: Impute missing values with "Missing"
    print(f"      └── Imputing missing values in {column_name} column with 'Missing'...")
    initial_missing = df_cleaned[column_name].isnull().sum()
    df_cleaned[column_name] = df_cleaned[column_name].fillna("Missing")
    print(f"      └── Imputed {initial_missing} missing values with 'Missing'.")

    # Step 2: Convert the column to categorical type
    print(f"      └── Converting {column_name} column to categorical type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('category')

    return df_cleaned

def clean_price_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()
    exchange_rate = 1.35  # Exchange rate for USD to SGD

    # Step 1: Impute missing values with 0
    print(f"      └── Imputing missing values in {column_name} column with 0...")
    initial_missing = df_cleaned[column_name].isnull().sum()
    df_cleaned[column_name] = df_cleaned[column_name].fillna(0)
    remaining_missing = df_cleaned[column_name].isnull().sum()
    print(f"      └── Imputed {initial_missing} missing values with 0. Remaining missing values: {remaining_missing}")

    # Step 2: Add a new column to track the original currency type
    print(f"      └── Extracting currency type from {column_name} column...")
    print(f"      └── Creating new currency_type column...")
    df_cleaned['currency_type'] = df_cleaned[column_name].str.extract(r'^(\w+)\$', expand=False).fillna('SGD').astype('category')

    # Step 3: Remove currency prefixes (e.g., "SGD$" or "USD$")
    print(f"      └── Removing currency prefixes (e.g., 'SGD$', 'USD$')...")
    df_cleaned[column_name] = df_cleaned[column_name].str.replace(r'^\w+\$', '', regex=True)

    # Step 4: Convert the cleaned price column to numeric
    print(f"      └── Converting {column_name} column to numeric...")
    df_cleaned[column_name] = pd.to_numeric(df_cleaned[column_name], errors='coerce')

    # Step 5: Convert USD to SGD
    print(f"      └── Converting USD prices to SGD using an exchange rate of {exchange_rate}...")
    df_cleaned['price_in_sgd'] = df_cleaned.apply(
        lambda row: row[column_name] * exchange_rate if row['currency_type'] == 'USD' else row[column_name],
        axis=1
    )

    # Step 6: Handle missing values in the new column (e.g., replace with 0)
    print(f"      └── Replacing missing values in price_in_sgd column with 0...")
    df_cleaned['price_in_sgd'] = df_cleaned['price_in_sgd'].fillna(0)

    # Step 7: Drop the original 'price' column
    print(f"      └── Dropping the original '{column_name}' column...")
    df_cleaned = df_cleaned.drop(columns=[column_name])

    # Step 8: Reorder columns to place 'currency_type' before 'price_in_sgd'
    print(f"      └── Reordering columns to place 'currency_type' before price_in_sgd...")
    cols = list(df_cleaned.columns)
    price_in_sgd_index = cols.index('price_in_sgd')
    cols.insert(price_in_sgd_index, 'currency_type')  # Insert 'currency_type' before 'price_in_sgd'

    # Remove duplicate columns (if any) while preserving order
    cols = list(dict.fromkeys(cols))
    df_cleaned = df_cleaned[cols]

    return df_cleaned

def clean_platform_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Convert the column to categorical type
    print(f"      └── Converting {column_name} column to categorical type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('category')

    return df_cleaned

def clean_num_adults_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Define the mapping for textual representations to integers
    mapping = {"one": 1, "two": 2}

    # Map textual representations to integers
    print(f"      └── Mapping textual values in {column_name} to integers...")
    df_cleaned[column_name] = df_cleaned[column_name].replace(mapping)

    # Convert the column to integer type
    print(f"      └── Converting {column_name} column to integer type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('int64')

    # Convert the column to categorical type for better memory efficiency
    print(f"      └── Converting {column_name} column to categorical type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('category')

    return df_cleaned

def clean_num_children_function(df_cleaned, column_name):
    df_cleaned = df_cleaned.copy()

    # Convert the column to integer type
    print(f"      └── Converting {column_name} column to integer type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('int64')

    # Convert the column to categorical type for better memory efficiency
    print(f"      └── Converting {column_name} column to categorical type...")
    df_cleaned[column_name] = df_cleaned[column_name].astype('category')

    return df_cleaned