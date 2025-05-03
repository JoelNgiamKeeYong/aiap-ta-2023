# src/utils/analyse_missing_values.py

import pandas as pd
from IPython.display import display

def analyse_missing_values(df, column_name, exclude_columns=None):
    """
    Perform a comprehensive analysis of missing values in a specific column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column to analyze for missing values.
        exclude_columns (list, optional): List of columns to exclude from the analysis. Default is None.

    Returns:
        dict: A dictionary containing summary statistics and insights about missing values.
    """
    # Validate inputs
    if column_name not in df.columns:
        raise ValueError(f"❌ Column '{column_name}' not found in the DataFrame.")

    # Handle exclude_columns
    if exclude_columns is None:
        exclude_columns = []  # Default to an empty list if no columns are excluded
    else:
        # Validate that all excluded columns exist in the DataFrame
        invalid_columns = [col for col in exclude_columns if col not in df.columns]
        if invalid_columns:
            raise ValueError(f"❌ Invalid columns in exclude_columns: {invalid_columns}")

    # Step 1: Calculate basic missing value statistics
    total_count = len(df)
    missing_count = df[column_name].isnull().sum()
    non_missing_count = total_count - missing_count
    missing_proportion = missing_count / total_count

    print(f"📊 Missing Value Analysis for Column: '{column_name}'")
    print(f"   └── Total Rows: {total_count}")
    print(f"   └── Missing Rows: {missing_count} ({missing_proportion:.2%})")
    print(f"   └── Non-Missing Rows: {non_missing_count} ({1 - missing_proportion:.2%})\n")

    # Step 3: Analyze patterns in missingness across other columns
    missing_patterns = {}
    for col in df.columns:
        if col == column_name or col in exclude_columns:
            continue  # Skip the target column and excluded columns
        
        if df[col].dtype in ['object', 'category']:
            # For categorical columns, count missing values by category
            pattern = (
                df.groupby(col)[column_name]
                .apply(lambda x: x.isnull().mean())
                .reset_index(name='Missing_Proportion')
            )
            missing_patterns[col] = pattern
            print(f"Correlation with '{col}':")
            display(pattern.style.format({'Missing_Proportion': '{:.2%}'}))

        elif pd.api.types.is_numeric_dtype(df[col]):
            # For numerical columns, bin the data into intervals
            num_bins = min(10, len(df[col].unique()))  # Use at most 10 bins
            binned_col = pd.cut(df[col], bins=num_bins)
            pattern = (
                df.assign(Binned=binned_col)
                .groupby('Binned')[column_name]
                .apply(lambda x: x.isnull().mean())
                .reset_index(name='Missing_Proportion')
            )
            pattern['Binned'] = pattern['Binned'].astype(str)  # Convert intervals to strings for display
            missing_patterns[col] = pattern
            print(f"Correlation with binned '{col}':")
            display(pattern.style.format({'Missing_Proportion': '{:.2%}'}))