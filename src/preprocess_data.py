# src/preprocess_data.py

import datetime
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(
    df_cleaned, target, save_data=False, 
    test_size=0.1, random_state=42
):
    """
    Preprocess the cleaned dataset and train multiple machine learning models.

    Parameters:
        df_cleaned (pd.DataFrame): The cleaned dataset.
        target (str): The name of the target variable.
        models (dict): Dictionary of models and their hyperparameter grids.
        n_jobs (int): Number of parallel jobs for GridSearchCV.
        cv_folds (int): Number of cross-validation folds.
        scoring_metric (str): Scoring metric for GridSearchCV.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - trained_models (list): List of tuples containing model name, best model, training time, and space required.
            - X_train (pd.DataFrame): Transformed training feature matrix.
            - X_test (pd.DataFrame): Transformed testing feature matrix.
            - y_train (pd.Series): Training target variable.
            - y_test (pd.Series): Testing target variable.
    """
    try:
        print("\n🔧 Preprocessing the dataset...")
        start_time = time.time()

        # Separate features (X) and target (y)
        print("\n   ⚙️  Separating the features and the target...")
        y = df_cleaned[target]  # Target variable
        X = df_cleaned.drop(columns=[target])  # Feature matrix

        # Split the data into training and testing sets
        print("   ⚙️  Splitting the data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            # stratify=y,                  # Ensures the same class distribution in train and test sets, important for imbalanced datasets
            test_size=test_size,         # Set in the config.yaml
            random_state=random_state    # To ensure reproducibility as per EDA
        )
        # df_train = pd.concat([X_train, y_train], axis=1)  # Combine X_train and y_train into a single DataFrame
        # df_test = pd.concat([X_test, y_test], axis=1)  # Combine X_test and y_test into a single DataFrame

        # # Carry out additional cleaning
        # print("   ⚙️  Carrying out additional dataset cleaning...")
        # print(f"\n      └── Cleaning training set...")
        # df_train = impute_missing_values_price_in_sgd(df=df_train)
        # df_train = remove_outliers_price_in_sgd(df=df_train)
        # df_train = remove_missing_values_room(df=df_train)
        # print(f"\n      └── Cleaning test set...")
        # df_test = impute_missing_values_price_in_sgd(df=df_test)
        # df_test = remove_outliers_price_in_sgd(df=df_test)
        # df_test = remove_missing_values_room(df=df_test)

        # # Create new features
        # print("\n   ⚙️  Creating new features via feature engineering...")
        # print(f"\n      └── Engineering features in training set...")
        # df_train = create_months_to_arrival_feature(df=df_train)
        # df_train = create_stay_duration_days_feature(df=df_train)
        # df_train = create_stay_category_feature(df=df_train)
        # df_train = create_has_children_feature(df=df_train)
        # df_train = create_total_pax_feature(df=df_train)
        # print(f"\n      └── Engineering features in test set...")
        # df_test = create_months_to_arrival_feature(df=df_test)
        # df_test = create_stay_duration_days_feature(df=df_test)
        # df_test = create_stay_category_feature(df=df_test)
        # df_test = create_has_children_feature(df=df_test)
        # df_test = create_total_pax_feature(df=df_test)

        # # Dropping irrelevant features
        # print("\n   ⚙️  Dropping irrelevant features...")
        # features_to_drop = ["arrival_day", "checkout_day", "stay_duration_days","num_children"]
        # print("      └── Dropping features in training set...")
        # df_train = df_train.drop(columns=features_to_drop, errors='ignore')
        # print("      └── Dropping features in test set...")
        # df_test = df_test.drop(columns=features_to_drop, errors='ignore')
        # print(f"      └── Dropped irrelevant features: {features_to_drop}")

        # # Checking data integrity after partial preprocessing
        # print("\n   ⚙️  Checking data integrity after preprocessing...")
        # print(f"      └── Training set shape: {df_train.shape} ...")
        # print(f"      └── Test set shape: {df_test.shape} ...")
        # total_rows = df_train.shape[0] + df_test.shape[0]
        # test_ratio = (df_test.shape[0] / total_rows) * 100
        # print(f"      └── Test set constitutes {test_ratio:.2f}% of the total dataset. Original split: {(test_size*100):.2f}%")

        # # Separate features (X) and target (y) after partial preprocessing
        # X_train = df_train.drop(columns=[target])
        # y_train = df_train[target] 
        # X_test = df_test.drop(columns=[target])
        # y_test = df_test[target] 
        
        # Define numerical and categorical features
        print("\n   ⚙️  Defining numerical and categorical features...")
        numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        # Apply feature scaling and encoding
        print("   ⚙️  Applying feature scaling and encoding...")
        print("      └── Defining the feature transformer pipeline...")
        print("      └── Standard scaler for numerical features...")
        print("      └── One hot encoder for categorical features...")
        transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),  # Scale numerical features
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # Encode categorical features
            ]
        )
        print("      └── Transforming training and test data...")
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)
        feature_names = [name.split('__', 1)[-1] for name in transformer.get_feature_names_out()]  # Remove the prefixes from feature names
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)

        # # Removing features based on feature selection findings in EDA
        # print("\n   ⚙️  Removing features based on feature selection findings in EDA...")
        # features_to_remove = [
        #     "checkout_month_January", "checkout_month_February", "checkout_month_March",
        #     "checkout_month_April", "checkout_month_May", "checkout_month_June",
        #     "checkout_month_July", "checkout_month_August", "checkout_month_September",
        #     "checkout_month_October", "checkout_month_November", "checkout_month_December",  
        # ]
        # X_train = X_train.drop(columns=features_to_remove, errors='ignore')
        # X_test = X_test.drop(columns=features_to_remove, errors='ignore')
        # feature_names = X_train.columns.tolist()  # Update feature_names to reflect the remaining columns
        # print(f"      └── Removed features: {features_to_remove}")

        # Combine features and target into a single DataFrame
        X_combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        y_combined = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
        df_preprocessed = pd.concat([X_combined, y_combined], axis=1)

        # Save the preprocessed data to a CSV file
        if save_data:
            output_path = "./data/preprocessed_data.csv"
            print(f"\n💾 Saving preprocessed data to {output_path}...")
            df_preprocessed.to_csv(output_path, index=False)

        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"\n✅ Data preprocessing completed in {elapsed_time:.2f} seconds!")

        return X_train, X_test, y_train, y_test, df_preprocessed, feature_names

    except Exception as e:
        print(f"❌ An error occurred during data preprocessing: {e}")
        raise RuntimeError("Data preprocessing process failed.") from e
    

def impute_missing_values_price_in_sgd(df):
    print("      └── Imputing missing values in price_in_sgd feature with median of currency_type group......")

    # Replace 0 values in the 'price_in_sgd' column with NaN
    df['price_in_sgd'] = df['price_in_sgd'].replace(0, np.nan)
    
    # Group by currency_type and impute missing prices with group-specific median
    df['price_in_sgd'] = df.groupby('currency_type', observed=True)['price_in_sgd'].transform(lambda x: x.fillna(x.median()))

    return df


def remove_outliers_price_in_sgd(df):
    print("      └── Removing outliers in price_in_sgd feature...")

    is_president_suite = df['room'] == 'President Suite'
    is_low_price = df['price_in_sgd'] < 1500
    is_outlier = is_president_suite & is_low_price
    is_not_outlier = ~is_outlier
    df = df[is_not_outlier]

    return df


def remove_missing_values_room(df):
    print("      └── Removing missing values in room feature...")

    # Count the initial number of rows
    initial_rows = len(df)

    # Filter out rows where 'room' is "Missing"
    df = df[df['room'] != 'Missing']

    # Count the number of rows after dropping missing values
    final_rows = len(df)

    # Calculate the number of rows dropped
    rows_dropped = initial_rows - final_rows

    # Print the number of rows dropped
    print(f"      └── Removed {rows_dropped} rows with missing values in 'room'.")

    return df


# Define a mapping from month names to numeric values
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}


def create_months_to_arrival_feature(df):
    print("      └── Creating months_to_arrival feature...")

    # Map the month names to numeric values
    df['booking_month_numeric'] = df['booking_month'].map(month_mapping).astype('int64')
    df['arrival_month_numeric'] = df['arrival_month'].map(month_mapping).astype('int64')

    # Calculate the difference in months (accounting for year wrap-around)
    df['months_to_arrival'] = (df['arrival_month_numeric'] - df['booking_month_numeric']) % 12

    # Drop intermediate columns if not needed
    df.drop(columns=['booking_month_numeric', 'arrival_month_numeric'], inplace=True)

    return df


def create_stay_duration_days_feature(df):
    print("      └── Creating stay_duration_days feature...")

    # Map month names to numeric values for arrival and checkout months
    df['arrival_month_numeric'] = df['arrival_month'].map(month_mapping)
    df['checkout_month_numeric'] = df['checkout_month'].map(month_mapping)

    # Function to validate dates
    def validate_date(row, month_col, day_col):
        try:
            datetime.datetime(year=2023, month=row[month_col], day=row[day_col])
            return True
        except ValueError:
            return False

    # Validate arrival and checkout dates
    df['is_valid_arrival'] = df.apply(validate_date, axis=1, month_col='arrival_month_numeric', day_col='arrival_day')
    df['is_valid_checkout'] = df.apply(validate_date, axis=1, month_col='checkout_month_numeric', day_col='checkout_day')

    # Filter out rows with invalid dates
    valid_df = df[df['is_valid_arrival'] & df['is_valid_checkout']].copy()

    # Create datetime objects for arrival and checkout dates
    valid_df['arrival_date'] = pd.to_datetime(
        valid_df[['arrival_month_numeric', 'arrival_day']].astype(int).apply(
            lambda x: f"{x.iloc[0]:02d}-{x.iloc[1]:02d}-2023", axis=1
        ), format='%m-%d-%Y'
    )

    # Handle year change for checkout dates
    valid_df['checkout_date'] = valid_df.apply(
        lambda row: pd.to_datetime(f"{row['checkout_month_numeric']:02d}-{row['checkout_day']:02d}-2023", format='%m-%d-%Y')
        if row['checkout_month_numeric'] >= row['arrival_month_numeric']
        else pd.to_datetime(f"{row['checkout_month_numeric']:02d}-{row['checkout_day']:02d}-2024", format='%m-%d-%Y'),
        axis=1
    )

    # Calculate the difference in days
    valid_df['stay_duration_days'] = (valid_df['checkout_date'] - valid_df['arrival_date']).dt.days.astype('int64')

    # Drop intermediate columns if not needed
    intermediate_columns = [
        'arrival_month_numeric', 'checkout_month_numeric',
        'arrival_date', 'checkout_date',
        'is_valid_arrival', 'is_valid_checkout'
    ]
    valid_df.drop(columns=intermediate_columns, inplace=True)

    # Print summary of changes
    print(f"      └── Created 'stay_duration_days' feature for {len(valid_df)} valid rows.")
    print(f"      └── Removed {len(df) - len(valid_df)} rows due to invalid dates.")

    return valid_df


def create_stay_category_feature(df):
    print("      └── Creating stay_category feature...")

    # Define bins for short-term, mid-term, and long-term stays
    bins = [0, 3, 14, df['stay_duration_days'].max()]
    labels = ['Short-term', 'Mid-term', 'Long-term']

    # Create a new column for stay categories
    df['stay_category'] = pd.cut(
        df['stay_duration_days'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df


def create_has_children_feature(df):
    print("      └── Creating has_children feature...")

     # Create the new binary column
    df['has_children'] = df['num_children'].apply(lambda x: 1 if x > 0 else 0)

    # Convert the new column to categorical type for memory efficiency
    df['has_children'] = df['has_children'].astype('category')

    return df


def create_total_pax_feature(df):
    print("      └── Creating total_pax feature...")

    # Create the new feature by summing `num_adults` and `num_children`
    df['total_pax'] = df['num_adults'].astype('int') + df['num_children'].astype('int')

    # Optionally convert the new column to integer type for consistency
    df['total_pax'] = df['total_pax'].astype('category')

    return df
