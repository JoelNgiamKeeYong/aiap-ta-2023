# src/pipeline.py
# This script orchestrates the entire pipeline for the project.

import os
import time
import math
import yaml
import argparse
import matplotlib
from tabulate import tabulate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Importing custom modules
from utils import compare_dataframes
from load_data import load_data 
from clean_data import clean_data
from preprocess_data import preprocess_data
from train_models import train_models
from evaluate_models import evaluate_models

def main():
    # Start timer
    start_time = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the ML prediction pipeline")
    parser.add_argument("--lite", action="store_true", help="Run the pipeline in lite mode: uses a simpler model.")
    args = parser.parse_args()

    # Load configuration from YAML file
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration values
    LOKY_MAX_CPU_COUNT = config["LOKY_MAX_CPU_COUNT"]
    DB_PATH = config["db_path"]
    DB_TABLE_NAME = config["db_table_name"]
    TARGET = config["target"]
    TEST_SIZE = config["test_size"]
    RANDOM_STATE = config["random_state"]
    N_JOBS = config["n_jobs"]
    CV_FOLDS = config["cv_folds"]
    SCORING_METRIC = config["scoring_metric"]
    LR_PARAMS = config["model_configuration"]["Linear Regression"]["params"]
    RF_PARAMS = config["model_configuration"]["Random Forest"]["params"]
    XG_PARAMS = config["model_configuration"]["XGBoost"]["params"]
    LGBM_PARAMS = config["model_configuration"]["LightGBM"]["params"]

    # Set environment variables for parallel processing and matplotlib backend
    os.environ['LOKY_MAX_CPU_COUNT'] = LOKY_MAX_CPU_COUNT 
    matplotlib.use('Agg')  # Non-interactive backend for plotting

    # Step 1: Load the dataset
    df = load_data(db_path=DB_PATH, db_table_name=DB_TABLE_NAME)  

    # Step 2: Clean the dataset
    # irrelevant_features = ['booking_id']
    # df_cleaned = clean_data(
    #     df=df, 
    #     save_data=True,
    #     remove_irrelevant_features=True, irrelevant_features=irrelevant_features,
    #     columns_to_clean=[col for col in df.columns if col not in irrelevant_features]
    # )
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned.drop_duplicates(subset='student_id', keep='last')
    df_cleaned = df_cleaned.drop(["index", "student_id", "sleep_time", "wake_time", "mode_of_transport", "bag_color"], axis=1)
    compare_dataframes(df_original=df, df_new=df_cleaned, original_name_string="raw", new_name_string="cleaned")

    # Step 3: Perform preprocessing (e.g. removing outliers, feature selection, feature engineering) based on training data
    # - Excludes test data information (to avoid data leakage)
    X_train, X_test, y_train, y_test, df_preprocessed, feature_names = preprocess_data(
        df_cleaned=df_cleaned,
        save_data=True,
        target=TARGET, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    compare_dataframes(df_original=df_cleaned, df_new=df_preprocessed, original_name_string="cleaned", new_name_string="preprocessed", show_verbose=False)

    # Step 4: Define the models and their respective hyperparameter grids
    if args.lite:
        models = {
            "LightGBM": {  
                "model": LGBMRegressor(verbose=-1, force_row_wise=True, random_state=RANDOM_STATE),
                "params": LGBM_PARAMS
            }
        }
    else:
        models = {
            "Linear Regression": {
                "model": LinearRegression(),
                "params": LR_PARAMS
            },
            "Random Forest": {
                "model": RandomForestRegressor(random_state=RANDOM_STATE),
                "params": RF_PARAMS
            },
            "XGBoost": {
                "model": XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE),
                "params": XG_PARAMS
            },
            "LightGBM": {  
                "model": LGBMRegressor(verbose=-1, force_row_wise=True, random_state=RANDOM_STATE),
                "params": LGBM_PARAMS
            }
        }

    # Step 5: Train the models
    trained_models = train_models(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        models=models, 
        n_jobs=N_JOBS, cv_folds=CV_FOLDS, scoring_metric=SCORING_METRIC,
    )

    # Step 6: Evaluate the models
    trained_models = evaluate_models(
        trained_models=trained_models,
        feature_names=feature_names,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    # Print a summary table of training times and model sizes
    print("\nPipeline Summary Table:")
    table_data = [
        [model_name, f"{model_size_kb:.2f}", f"{training_time:.2f}", f"{evaluation_time:.2f}"]
        for model_name, best_model, training_time, model_size_kb, evaluation_time in trained_models
    ]
    headers = ["Model Name", "Model Size (KB)", "Training Time (s)", "Evaluation Time (s)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Record time taken for pipeline execution
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"\n✅ Completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()