# src/utils/perform_multivariate_analysis.py


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency, f_oneway

def perform_multivariate_analysis(df, col1, col2, col3):
    """
    Perform multivariate analysis between three columns in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col1 (str): The first column to analyze.
    col2 (str): The second column to analyze.
    col3 (str): The third column to analyze.
    
    Returns:
    None: Displays visualizations, statistics, and insights.
    """
    # Validate inputs
    if not all(col in df.columns for col in [col1, col2, col3]):
        missing_cols = [col for col in [col1, col2, col3] if col not in df.columns]
        raise ValueError(f"❌ Column(s) {', '.join(missing_cols)} not found in the DataFrame.")
    
    print(f"📊 Performing multivariate analysis on columns: {col1}, {col2}, {col3}")
    
    # Determine column types
    col_types = {
        col1: "categorical" if df[col1].dtype in ['object', 'category'] else "numerical",
        col2: "categorical" if df[col2].dtype in ['object', 'category'] else "numerical",
        col3: "categorical" if df[col3].dtype in ['object', 'category'] else "numerical"
    }
    print(" └── Column Types:")
    for col, col_type in col_types.items():
        print(f"     └── {col}: {col_type}")
    
    # Count numerical and categorical columns
    num_cols = [col for col, col_type in col_types.items() if col_type == "numerical"]
    cat_cols = [col for col, col_type in col_types.items() if col_type == "categorical"]
    
    # Case 1: All Numerical Variables
    if len(num_cols) == 3:
        print("\n📊 Analyzing all numerical variables...")

        # Create subplots with a grid layout
        fig = plt.figure(figsize=(14,6))
        fig.suptitle("Multivariate Analysis of Numerical Variables", fontsize=16, fontweight="bold")

        # Correlation Heatmap
        ax1 = fig.add_subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
        corr_matrix = df[num_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax1)
        ax1.set_title("Correlation Heatmap", fontsize=12, fontweight="bold")

        # 3D Scatter Plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')  # 1 row, 2 columns, 2nd subplot
        sc = ax2.scatter(
            df[num_cols[0]], 
            df[num_cols[1]], 
            df[num_cols[2]], 
            c=df[num_cols[0]], 
            cmap='viridis', 
            alpha=0.7
        )
        ax2.set_xlabel(num_cols[0], fontsize=10)
        ax2.set_ylabel(num_cols[1], fontsize=10)
        ax2.set_zlabel(num_cols[2], fontsize=10)
        ax2.set_title("3D Scatter Plot", fontsize=12, fontweight="bold")
        fig.colorbar(sc, ax=ax2, shrink=0.5, aspect=10, label=num_cols[0])  # Add color bar

        # Adjust layout for compactness
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
        plt.show()

    # Case 2: All Categorical Variables
    elif len(cat_cols) == 3:
        print("\n📊 Analyzing all categorical variables...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Multivariate Analysis of Categorical Variables", fontsize=16, fontweight="bold")
        
        # Crosstab and Chi-Square Test
        crosstab = pd.crosstab([df[col1], df[col2]], df[col3])
        chi2, p_value, _, _ = chi2_contingency(crosstab)
        cramers_v = np.sqrt(chi2 / len(df))
        print(f"🧪 Chi-Square Test: χ² = {chi2:.2f}, p-value = {p_value:.4f}")
        print(f"🧪 Cramér's V: {cramers_v:.2f}")
        
        # Mosaic Plot (Heatmap of Proportions)
        mosaic_data = pd.crosstab([df[col1], df[col2], df[col3]], normalize=True)
        sns.heatmap(mosaic_data, annot=True, fmt=".2%", cmap="flare", cbar=False, ax=axes[0])
        axes[0].set_title("Mosaic Plot (Proportions)", fontsize=12, fontweight="bold")
        
        # Stacked Bar Chart
        grouped = pd.crosstab([df[col1], df[col2]], df[col3]).stack().reset_index(name="count")
        grouped.plot(kind="bar", stacked=True, colormap="viridis", ax=axes[1])
        axes[1].set_title("Stacked Bar Chart", fontsize=12, fontweight="bold")
        axes[1].tick_params(axis='x', rotation=45)
        
        # Heatmap of Proportions
        heatmap_data = pd.crosstab([df[col1], df[col2]], df[col3], normalize="index")
        sns.heatmap(heatmap_data, annot=True, fmt=".2%", cmap="flare", cbar=False, ax=axes[2])
        axes[2].set_title("Heatmap of Proportions", fontsize=12, fontweight="bold")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title
        plt.show()
    
    # Case 3: Two Categorical + One Numerical
    if len(cat_cols) == 2 and len(num_cols) == 1:
        print("\n📊 Analyzing two categorical variables and one numerical variable...")
        
        # Create subplots for all plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Multivariate Analysis of Two Categorical + One Numerical Variables", fontsize=16, fontweight="bold")
        
        # Extract columns
        cat_col1, cat_col2 = cat_cols
        num_col = num_cols[0]
        
        # Plot 1: Grouped Box Plot
        sns.boxplot(data=df, x=cat_col1, y=num_col, hue=cat_col2, palette="inferno", ax=axes[0, 0])
        axes[0, 0].set_title(f"Box Plot: {num_col} Across {cat_col1} by {cat_col2}", fontsize=12, fontweight="bold")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Violin Plot
        sns.violinplot(data=df, x=cat_col1, y=num_col, hue=cat_col2, split=True, palette="inferno", ax=axes[0, 1])
        axes[0, 1].set_title(f"Violin Plot: {num_col} Across {cat_col1} by {cat_col2}", fontsize=12, fontweight="bold")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Heatmap of Aggregated Values
        heatmap_data = df.groupby([cat_col1, cat_col2])[num_col].mean().unstack()
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[1, 0])
        axes[1, 0].set_title(f"Heatmap of Mean {num_col} by {cat_col1} and {cat_col2}", fontsize=12, fontweight="bold")
        
        # Plot 4: Stacked Bar Chart
        stacked_data = pd.crosstab([df[cat_col1], df[cat_col2]], df[num_col] > df[num_col].mean())
        stacked_data.plot(kind="bar", stacked=True, colormap="viridis", ax=axes[1, 1])
        axes[1, 1].set_title(f"Stacked Bar Chart: Proportions of {num_col}", fontsize=12, fontweight="bold")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # Case 4: Two Numerical + One Categorical
    elif len(num_cols) == 2 and len(cat_cols) == 1:
        print("\n📊 Analyzing two numerical variables and one categorical variable...")
        
        # Create subplots for all plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Multivariate Analysis of Two Numerical + One Categorical Variables", fontsize=16, fontweight="bold")
        
        # Extract columns
        num_col1, num_col2 = num_cols
        cat_col = cat_cols[0]
        
        # Plot 1: Faceted Scatter Plot
        g = sns.FacetGrid(df, col=cat_col, hue=cat_col, palette="inferno", height=4, aspect=1.2)
        g.map(sns.scatterplot, num_col1, num_col2, alpha=0.7).add_legend()
        g.fig.suptitle(f"Faceted Scatter Plot: {num_col1} vs {num_col2} by {cat_col}", fontsize=12, fontweight="bold", y=1.02)
        plt.show()
        
        # Plot 2: Grouped Regression Lines
        sns.lmplot(data=df, x=num_col1, y=num_col2, hue=cat_col, palette="inferno", height=5, aspect=1.5)
        plt.title(f"Grouped Regression Lines: {num_col1} vs {num_col2} by {cat_col}", fontsize=12, fontweight="bold")
        plt.show()
        
        # Plot 3: Violin Plot
        sns.violinplot(data=df, x=cat_col, y=num_col1, palette="inferno", ax=axes[0, 0])
        axes[0, 0].set_title(f"Violin Plot: {num_col1} Across {cat_col}", fontsize=12, fontweight="bold")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Box Plot
        sns.boxplot(data=df, x=cat_col, y=num_col2, palette="inferno", ax=axes[0, 1])
        axes[0, 1].set_title(f"Box Plot: {num_col2} Across {cat_col}", fontsize=12, fontweight="bold")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()