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


def evaluate_models(trained_models, feature_names, X_train, X_test, y_train, y_test):
    """
    Evaluate a list of trained models and save evaluation results.

    Args:
        trained_models (list): List of trained models to evaluate.
        X_train (pd.DataFrame or np.ndarray): Training feature matrix.
        X_test (pd.DataFrame or np.ndarray): Testing feature matrix.
        y_train (pd.Series or np.ndarray): Training target variable.
        y_test (pd.Series or np.ndarray): Testing target variable.

    Returns:
        dict: Dictionary containing consolidated evaluation results.
    """
    print(f"\n📊 Evaluating best models...")

    # Ensure the output directory exists
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for each type of output
    feature_importance_dir = os.path.join(output_dir, "feature_importance")
    residual_plots_dir = os.path.join(output_dir, "residual_plots")
    prediction_vs_actual_dir = os.path.join(output_dir, "prediction_vs_actual")
    learning_curves_dir = os.path.join(output_dir, "learning_curves")
    os.makedirs(feature_importance_dir, exist_ok=True)
    os.makedirs(residual_plots_dir, exist_ok=True)
    os.makedirs(prediction_vs_actual_dir, exist_ok=True)
    os.makedirs(learning_curves_dir, exist_ok=True)

    # Lists to store results
    results = []  # Evaluation metrics for each model

    # Loop through each trained model and evaluate
    for i, (model_name, best_model, training_time, model_size_kb) in enumerate(trained_models):
        print(f"\n   📋 Evaluating {model_name} model...")
        start_time = time.time()

        # Evaluate the model on training and test sets
        formatted_metrics = calculate_evaluation_metrics(model_name, best_model, X_train, X_test, y_train, y_test)
        results.append(formatted_metrics)

        # Generate and store evaluation scores and metrics
        extract_feature_importances(best_model, model_name, feature_names, feature_importance_dir)
        generate_residual_plot(best_model, X_test, y_test, model_name, residual_plots_dir)
        generate_prediction_vs_actual_plot(best_model, X_test, y_test, model_name, prediction_vs_actual_dir)
        generate_learning_curves(best_model, model_name, X_train, y_train, learning_curves_dir)

        # Record and store evaluation time
        end_time = time.time()
        evaluation_time = end_time - start_time
        trained_models[i] = (model_name, best_model, training_time, model_size_kb, evaluation_time)
        print(f"      └── Evaluation completed in {evaluation_time:.2f} seconds!")

    # Save consolidated metrics results
    save_consolidated_metrics(results, output_dir)
    print(f"\n💾 Saved evaluation metrics and charts to {output_dir} folder!")

    return trained_models


def calculate_evaluation_metrics(model_name, model, X_train, X_test, y_train, y_test):
    """
    Calculate evaluation metrics for a regression model.

    Args:
        model: Trained regression model.
        X: Feature matrix.
        y_true: True target values.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Calculate training metrics
    y_train_pred = model.predict(X_train)
    train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_MAE = mean_absolute_error(y_train, y_train_pred)
    train_R_Squared = r2_score(y_train, y_train_pred)
    train_MedAE = median_absolute_error(y_train, y_train_pred)
    train_MSLE = mean_squared_log_error(y_train, y_train_pred) if all(y_train > 0) else None
    train_Explained_Variance = explained_variance_score(y_train, y_train_pred)
    train_Max_Error = max_error(y_train, y_train_pred)

    # Calculate test metrics
    y_test_pred = model.predict(X_test)
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
        "R²": f"{test_R_Squared:.2f} ({train_R_Squared:.2f})"
        if train_MSLE is not None and test_MSLE is not None
        else "N/A (Negative Values)",
        "Explained Variance": f"{test_Explained_Variance:.2f} ({train_Explained_Variance:.2f})",
        "Max Error": f"{test_Max_Error:.2f} ({train_Max_Error:.2f})",
    }

    return formatted_metrics


def extract_feature_importances(model, model_name, feature_names, feature_importance_dir):
    """
    Extract and store feature importance scores.

    Args:
        model: Trained regression model.
        model_name (str): Name of the model.
        feature_names (list): List of feature names.
        feature_importance_results (list): List to store feature importance results.
        feature_importance_dir (str): Directory to save feature importance files.
    """
    try:
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

        file_path = os.path.join(feature_importance_dir, f"feature_importances_{model_name.replace(' ', '_').lower()}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"- 📊 Feature Importance Scores for {model_name}:-\n")
            f.write(feature_importances.to_string())
        print(f"      └── Computing and saving feature importance scores for {model_name}...")

    except Exception as e:
        print(f"      └── ⚠️  Could not compute or save feature importance for {model_name}: {str(e)}")


def generate_residual_plot(model, X_test, y_test, model_name, residual_plots_dir):
    """
    Generate and save residual plots.

    Args:
        model: Trained regression model.
        X_test: Test feature matrix.
        y_test: True target values for the test set.
        model_name (str): Name of the model.
        residual_plots_data (list): List to store residual plot data.
        residual_plots_dir (str): Directory to save residual plots.
    """
    try:
        # Calculate predictions and residuals
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

        # Subplot 1: Residual Plot (Scatter plot of residuals vs predicted values)
        axes[0].scatter(y_pred, residuals, alpha=0.5, color="blue")
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
        file_path = os.path.join(residual_plots_dir, f"residual_plot_{model_name.replace(' ', '_').lower()}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"      └── Generated and saved residual plot for {model_name}.")

    except Exception as e:
        print(f"      └── ⚠️  Error generating residual plot for {model_name}: {str(e)}")


def generate_prediction_vs_actual_plot(model, X_test, y_test, model_name, prediction_vs_actual_dir):
    """
    Generate and save prediction vs actual plot.

    Args:
        model: Trained regression model.
        X_test: Test feature matrix.
        y_test: True target values for the test set.
        model_name (str): Name of the model.
        prediction_vs_actual_dir (str): Directory to save prediction vs actual plots.
    """
    try:
        y_pred = model.predict(X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
        plt.title(f"Prediction vs Actual for {model_name}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)

        file_path = os.path.join(prediction_vs_actual_dir, f"prediction_vs_actual_{model_name.replace(' ', '_').lower()}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"      └── Generated and saved prediction vs actual plot for {model_name}.")
    except Exception as e:
        print(f"      └── ⚠️ Error generating prediction vs actual plot for {model_name}: {str(e)}")


def generate_learning_curves(model, model_name, X_train, y_train, learning_curves_dir):
    """
    Generate and save learning curves.

    Args:
        model: Trained regression model.
        model_name (str): Name of the model.
        X_train: Training feature matrix.
        y_train: Training target variable.
        learning_curves_dir (str): Directory to save learning curves.
    """
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
        )

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

        file_path = os.path.join(learning_curves_dir, f"learning_curves_{model_name.replace(' ', '_').lower()}.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"      └── Generated and saved learning curves for {model_name}.")
    except Exception as e:
        print(f"      └── ⚠️ Error generating learning curves for {model_name}: {str(e)}")
 

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