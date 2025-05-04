# src/evaluate_models.py

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    mean_squared_log_error,
    explained_variance_score,
    max_error,
)
from sklearn.model_selection import learning_curve


def evaluate_models(trained_models, feature_names, X_train, X_test, y_train, y_test, n_jobs, cv_folds, scoring_metric):
    """
    Evaluate a list of trained models, compute performance metrics, generate visualizations, and save results.

    This function evaluates each model in the provided list by computing performance metrics (e.g., RMSE, MAE),
    extracting feature importances, generating diagnostic plots (e.g., residual plots, learning curves), and saving
    all results to a consolidated output directory. It also measures and records the evaluation time for each model.

    Args:
        trained_models (list): A list of tuples containing the trained models to evaluate.
        feature_names (list): List of feature names corresponding to the columns in `X_train` and `X_test`.
        X_train (pd.DataFrame or np.ndarray): Training feature matrix used to fit the models.
        X_test (pd.DataFrame or np.ndarray): Testing feature matrix used for evaluation.
        y_train (pd.Series or np.ndarray): True target values for the training set.
        y_test (pd.Series or np.ndarray): True target values for the testing set.
        n_jobs (int): Number of parallel jobs to use during learning curve generation.
        cv_folds (int): Number of cross-validation folds to use for learning curve evaluation.
        scoring_metric (str): Scoring metric to use for evaluating learning curves (e.g., 'neg_mean_squared_error').

    Returns:
        list: Updated list of tuples containing the evaluated models. Each tuple includes:
              - model_name (str): Name of the model.
              - best_model: The trained model object.
              - training_time (float): Time taken to train the model (in seconds).
              - model_size_kb (float): Size of the trained model in kilobytes.
              - evaluation_time (float): Time taken to evaluate the model (in seconds).

    Output:
        - Metrics and visualizations are saved to the `output` directory.
        - Consolidated evaluation metrics are saved as a summary file.
    """
    print(f"\n📊 Evaluating best models...")

    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Lists to store results
    results = []  # Evaluation metrics for each model

    # Loop through each trained model and evaluate
    for i, (model_name, best_model, training_time, model_size_kb) in enumerate(trained_models):
        print(f"\n   📋 Evaluating {model_name} model...")
        start_time = time.time()

        # Predict values
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Evaluate the model on training and test sets
        formatted_metrics = calculate_evaluation_metrics(model_name, y_train, y_train_pred, y_test, y_test_pred)
        results.append(formatted_metrics)

        # Generate and store evaluation scores and metrics
        extract_feature_importances(best_model, model_name, feature_names, output_dir)
        generate_residual_plot(model_name, y_test, y_test_pred, output_dir)
        generate_prediction_vs_actual_plot(model_name, y_test, y_test_pred, output_dir)
        generate_learning_curves(best_model, model_name, X_train, y_train, n_jobs, cv_folds, scoring_metric, output_dir)

        # Record and store evaluation time
        end_time = time.time()
        evaluation_time = end_time - start_time
        trained_models[i] = (model_name, best_model, training_time, model_size_kb, evaluation_time)
        print(f"      └── Evaluation completed in {evaluation_time:.2f} seconds!")

    # Save consolidated metrics results
    save_consolidated_metrics(results, output_dir)
    print(f"\n💾 Saved evaluation metrics and charts to {output_dir} folder!")

    return trained_models


def calculate_evaluation_metrics(model_name, y_train, y_train_pred, y_test, y_test_pred):
    """
    Calculate and format evaluation metrics for a regression model.

    Args:
        model_name (str): Name of the model.
        y_train: True target values for the training set.
        y_train_pred: Predicted target values for the training set.
        y_test: True target values for the test set.
        y_test_pred: Predicted target values for the test set.

    Returns:
        dict: Dictionary of formatted evaluation metrics.
    """
    # Calculate training metrics
    train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_MAE = mean_absolute_error(y_train, y_train_pred)
    train_R_Squared = r2_score(y_train, y_train_pred)
    train_MedAE = median_absolute_error(y_train, y_train_pred)
    train_MSLE = mean_squared_log_error(y_train, y_train_pred) if all(y_train > 0) else None
    train_Explained_Variance = explained_variance_score(y_train, y_train_pred)
    train_Max_Error = max_error(y_train, y_train_pred)

    # Calculate test metrics
    test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_MAE = mean_absolute_error(y_test, y_test_pred)
    test_R_Squared = r2_score(y_test, y_test_pred)
    test_MedAE = median_absolute_error(y_test, y_test_pred)
    test_MSLE = mean_squared_log_error(y_test, y_test_pred) if all(y_test > 0) else None
    test_Explained_Variance = explained_variance_score(y_test, y_test_pred)
    test_Max_Error = max_error(y_test, y_test_pred)

    # Format evaluation metrics
    formatted_metrics = {
        "Model": model_name,
        "RMSE": f"{test_RMSE:.2f} ({train_RMSE:.2f})",
        "MAE": f"{test_MAE:.2f} ({train_MAE:.2f})",
        "MedAE": f"{test_MedAE:.2f} ({train_MedAE:.2f})",
        "MSLE": f"{test_MSLE:.2f} ({train_MSLE:.2f})",
        "R²": f"{test_R_Squared:.2f} ({train_R_Squared:.2f})",
        "Explained Variance": f"{test_Explained_Variance:.2f} ({train_Explained_Variance:.2f})",
        "Max Error": f"{test_Max_Error:.2f} ({train_Max_Error:.2f})",
    }

    return formatted_metrics


def extract_feature_importances(model, model_name, feature_names, output_dir):
    """
    Extract and save feature importance scores for a trained model.

    Args:
        model: Trained machine learning model.
        model_name (str): Name of the model.
        feature_names (list): List of feature names corresponding to the model's features.
        output_dir (str): Directory to save the feature importance file.

    Notes:
        - Supports models with `feature_importances_` (e.g., tree-based models) or `coef_` (e.g., linear models).
        - Saves feature importance scores to a text file in the specified directory.
    """
    try:
        # Check if the model supports feature importances or coefficients
        if hasattr(model, "feature_importances_"):
            feature_importances = pd.Series(
                model.feature_importances_, index=feature_names
            ).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            feature_importances = pd.Series(
                model.coef_, index=feature_names
            ).sort_values(ascending=False)
        else:
            raise AttributeError("Model does not have feature_importances_ or coef_.")

        # Save feature importance scores to a file
        os.makedirs(f"{output_dir}/feature_importance", exist_ok=True)
        file_path = f"{output_dir}/feature_importance/feature_importances_{model_name.replace(' ', '_').lower()}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"- 📊 Feature Importance Scores for {model_name}:-\n")
            f.write(feature_importances.to_string())
        print(f"      └── Computing and saving feature importance scores for {model_name}...")

    except Exception as e:
        print(f"      └── ⚠️  Could not compute or save feature importance for {model_name}: {str(e)}")


def generate_residual_plot(model_name, y_test, y_test_pred, output_dir):
    """
    Generate and save residual plots for a regression model.

    This function creates two subplots:
    1. A scatter plot of residuals vs predicted values to identify patterns.
    2. A histogram with a KDE of residuals to assess their distribution.

    Args:
        model_name (str): Name of the model.
        y_test (pd.Series or np.ndarray): True target values for the test set.
        y_test_pred (pd.Series or np.ndarray): Predicted target values for the test set.
        output_dir (str): Directory to save residual plots.

    Notes:
        - Residuals are calculated as `y_test - y_pred`.
        - The output file is saved in PNG format with a resolution of 300 DPI.
    """
    try:
        # Calculate residuals
        residuals = y_test - y_test_pred

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

        # Subplot 1: Residual Plot (Scatter plot of residuals vs predicted values)
        axes[0].scatter(y_test_pred, residuals, alpha=0.5, color="blue")
        axes[0].axhline(y=0, color="red", linestyle="--")
        axes[0].set_title(f"Residual Plot for {model_name}", fontsize=14)
        axes[0].set_xlabel("Predicted Values", fontsize=12)
        axes[0].set_ylabel("Residuals", fontsize=12)
        axes[0].grid(True)

        # Subplot 2: Residual Distribution Plot (Histogram with KDE)
        sns.histplot(residuals, kde=True, bins=30, color="blue", ax=axes[1])
        axes[1].axvline(0, color="red", linestyle="--", label="Zero Residual Line")
        axes[1].set_title(f"Distribution of Residuals for {model_name}", fontsize=14)
        axes[1].set_xlabel("Residuals", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].legend()
        axes[1].grid(True)

        # Adjust layout and save the combined plot
        plt.tight_layout()
        os.makedirs(f"{output_dir}/residual_plots", exist_ok=True)
        file_path = f"{output_dir}/residual_plots/residual_plot_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"      └── Generated and saved residual plot for {model_name}.")

    except Exception as e:
        print(f"      └── ⚠️  Error generating residual plot for {model_name}: {str(e)}")


def generate_prediction_vs_actual_plot(model_name, y_test, y_test_pred, output_dir):
    """
    Generate and save a prediction vs actual plot for a regression model.

    This function creates a scatter plot comparing predicted values against actual values,
    along with a diagonal line representing perfect predictions.

    Args:
        model_name (str): Name of the model.
        y_test (pd.Series or np.ndarray): True target values for the test set.
        y_test_pred (pd.Series or np.ndarray): Predicted target values for the test set.
        output_dir (str): Directory to save the prediction vs actual plot.
    """
    try:
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.5, label="Predictions")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Perfect Fit")
        plt.title(f"Prediction vs Actual for {model_name}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)

        # Save the plot
        os.makedirs(f"{output_dir}/prediction_vs_actual", exist_ok=True)
        file_path = f"{output_dir}/prediction_vs_actual/prediction_vs_actual_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"      └── Generated and saved prediction vs actual plot for {model_name}.")

    except Exception as e:
        print(f"      └── ⚠️  Error generating prediction vs actual plot for {model_name}: {str(e)}")


def generate_learning_curves(model, model_name, X_train, y_train, n_jobs, cv_folds, scoring_metric, output_dir):
    """
    Generate and save learning curves for a regression model.

    This function creates a plot showing how the model's training and validation performance 
    changes with the size of the training set.

    Args:
        model: Trained regression model.
        model_name (str): Name of the model.
        X_train: Training feature matrix.
        y_train: Training target variable.
        n_jobs (int): Number of parallel jobs for learning curve computation.
        cv_folds (int): Number of cross-validation folds.
        scoring_metric (str): Scoring metric for learning curve evaluation (e.g., 'neg_mean_squared_error').
        output_dir (str): Directory to save learning curves.
    """
    try:
        # Calculate learning curves
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X_train, y_train,
            n_jobs=n_jobs, cv=cv_folds, scoring=scoring_metric,
        )

        # Convert scores to RMSE if using 'neg_mean_squared_error'
        train_rmse = np.sqrt(-train_scores.mean(axis=1))
        test_rmse = np.sqrt(-test_scores.mean(axis=1))

        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_rmse, 'o-', label="Training RMSE", color="blue")
        plt.plot(train_sizes, test_rmse, 'o-', label="Validation RMSE", color="red")
        plt.fill_between(train_sizes, train_rmse - train_rmse.std(), train_rmse + train_rmse.std(), alpha=0.1, color="blue")
        plt.fill_between(train_sizes, test_rmse - test_rmse.std(), test_rmse + test_rmse.std(), alpha=0.1, color="red")

        # Add labels, title, and legend
        plt.title(f"Learning Curves for {model_name}")
        plt.xlabel("Training Examples")
        plt.ylabel("RMSE")
        plt.legend(loc="best")
        plt.grid(True)

        # Save the plot
        os.makedirs(f"{output_dir}/learning_curves", exist_ok=True)
        file_path = f"{output_dir}/learning_curves/learning_curve_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"      └── Generated and saved learning curves for {model_name}.")

    except Exception as e:
        print(f"      └── ⚠️  Error generating learning curves for {model_name}: {str(e)}")
 

def save_consolidated_metrics(results, output_dir):
    """
    Save all evaluation results to files.

    Args:
        results (list): List of evaluation metrics for each model.
        feature_importance_results (list): List of feature importance results.
        residual_plots_data (list): List of residual plot data.
        output_dir (str): Directory to save consolidated results.
    """
    # Save consolidated metrics summary
    metrics_file_path = os.path.join(output_dir, "evaluation_metrics_summary.txt")
    with open(metrics_file_path, "w", encoding="utf-8") as f:
        # Write metrics
        f.write("📊 Consolidated Regression Evaluation Metrics:\n")
        f.write("(Note: Test metrics are shown first, followed by training metrics in brackets.)\n\n")
        f.write(tabulate(results, headers="keys", tablefmt="grid"))
        f.write("\n\n")

        # Add explanations for each metric
        f.write("📝 Explanation of Metrics:\n\n")

        f.write("1. RMSE (Root Mean Squared Error):\n")
        f.write("   - Measures the average magnitude of prediction errors, penalizing larger errors more heavily.\n")
        f.write("   - Lower values indicate better performance.\n\n")
        
        f.write("2. MAE (Mean Absolute Error):\n")
        f.write("   - Measures the average absolute difference between true and predicted values.\n")
        f.write("   - Robust to outliers compared to RMSE.\n\n")

        f.write("3. MedAE (Median Absolute Error):\n")
        f.write("   - Measures the median of absolute differences between true and predicted values.\n")
        f.write("   - Robust to outliers and provides a central tendency of errors.\n\n")

        f.write("4. MSLE (Mean Squared Logarithmic Error):\n")
        f.write("   - Penalizes underestimations more than overestimations and works well for positive target values.\n")
        f.write("   - Useful when relative errors are more important than absolute errors.\n\n")
        
        f.write("5. R² (R-squared):\n")
        f.write("   - Indicates the proportion of variance in the target variable explained by the model.\n")
        f.write("   - Values closer to 1 indicate better performance.\n\n")
        
        f.write("6. Explained Variance:\n")
        f.write("   - Shows how much of the variation in the target variable is captured by the model.\n")
        f.write("   - Values closer to 1 indicate better performance.\n\n")
        
        f.write("7. Max Error:\n")
        f.write("   - Captures the largest residual error, which can be useful for identifying extreme cases.\n")
        f.write("   - Lower values indicate better performance.\n\n")