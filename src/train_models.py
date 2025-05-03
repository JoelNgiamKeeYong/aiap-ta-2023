# src/train_models.py

import joblib
import os
import time
from sklearn.model_selection import GridSearchCV

def train_models(
    X_train, X_test, y_train, y_test,
    models,
    n_jobs, cv_folds, scoring_metric
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
        print("\n🤖 Training the candidate models...")
        start_time = time.time()

        # Ensure the models and output directory exists
        os.makedirs("models", exist_ok=True)
        os.makedirs("output", exist_ok=True)

        trained_models = []

        # Loop through each model and perform training and evaluation
        for model_name, model_info in models.items():
            print(f"\n   ⛏️  Training {model_name} model...")
            start_time = time.time()

            # Perform hyperparameter tuning using GridSearchCV
            print(f"      └── Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                estimator=model_info["model"],
                param_grid=model_info["params"],
                scoring=scoring_metric,
                cv=cv_folds,
                n_jobs=n_jobs
            )
            grid_search.fit(X_train, y_train)

            # Measure training time
            end_time = time.time()
            training_time = end_time - start_time
            print(f"      └── Model trained successfully in {training_time:.2f} seconds.")

            # Extract the best model and parameters
            best_model = grid_search.best_estimator_
            print(f"      └── Best parameters: {grid_search.best_params_}")

            # Save the trained model permanently
            model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
            print(f"      └── Saving trained model to {model_path}...")
            joblib.dump(best_model, model_path)
            model_size_kb = round(os.path.getsize(model_path) / 1024, 2)  # Size in KB
            print(f"      └── Model size: {model_size_kb} KB")
            
            # Store the trained model details in a list for later use
            trained_models.append([model_name, best_model, training_time, model_size_kb])


        return trained_models

    except Exception as e:
        print(f"❌ An error occurred during model training: {e}")
        raise RuntimeError("Model training process failed.") from e