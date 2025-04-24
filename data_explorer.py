# data_explorer.py
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any, Dict, List # Ensure all types are imported

# Note: This module does NOT import Streamlit.
# It provides data/results to the Streamlit app, which handles display.

def get_duplicates_count(df: pd.DataFrame) -> int:
    """
    Calculates the number of duplicate rows in the DataFrame (based on all columns).

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        int: The count of duplicate rows. Returns 0 if df is None or on error.
    """
    if df is None:
        print("Warning: Input DataFrame is None (get_duplicates_count).")
        return 0
    try:
        return df.duplicated().sum()
    except Exception as e:
        print(f"Error counting duplicates: {e}")
        return 0


def get_descriptive_stats(df: pd.DataFrame, include_type: Any = 'all') -> pd.DataFrame:
    """
    Calculates descriptive statistics for the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        include_type (Any): Type filter for pd.describe (e.g., 'all', np.number, ['object', 'category', 'bool']).

    Returns:
        pd.DataFrame: A DataFrame containing descriptive statistics. Returns empty DataFrame on error or if df is None.
    """
    if df is None:
        print("Warning: Input DataFrame is None (get_descriptive_stats).")
        return pd.DataFrame()
    try:
        return df.describe(include=include_type)
    except Exception as e:
        print(f"Error getting descriptive stats: {e}")
        return pd.DataFrame()


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary of missing values (count and percentage) for columns with missing data.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Count' and 'Percentage (%)', indexed by column name.
                      Only includes columns that have missing values. Returns empty DataFrame if no missing values or df is None/empty.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['Count', 'Percentage (%)'])

    try:
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]

        if missing_cols.empty:
            return pd.DataFrame(columns=['Count', 'Percentage (%)'])
        else:
            total_rows = len(df)
            missing_percent = (missing_cols / total_rows) * 100
            missing_df = pd.DataFrame({
                'Count': missing_cols,
                'Percentage (%)': missing_percent.round(2)
            })
            missing_df = missing_df.sort_values(by='Count', ascending=False)
            missing_df.index.name = 'Column'
            return missing_df
    except Exception as e:
         print(f"Error generating missing summary: {e}")
         return pd.DataFrame(columns=['Count', 'Percentage (%)'])


def get_value_counts(df: pd.DataFrame, column: str, top_n: int = 20) -> Tuple[Optional[pd.DataFrame], int]:
    """
    Gets value counts for a specified column, including counts for NaN values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to analyze.
        top_n (int): The maximum number of unique values to return counts for.
                     If negative or zero, returns all unique values.

    Returns:
        Tuple[Optional[pd.DataFrame], int]:
            - A DataFrame with 'Value', 'Count', and 'Percentage (%)' columns for the top_n values (or all),
              sorted by Count descending. Returns None on error.
            - An integer representing the total number of unique values (including NaN).
    """
    empty_result = (pd.DataFrame(columns=['Value', 'Count', 'Percentage (%)']), 0)
    if df is None or column not in df.columns:
        print(f"Error: DataFrame is None or column '{column}' not found (get_value_counts).")
        return empty_result

    try:
        counts_series = df[column].value_counts(dropna=False)
        total_unique = len(counts_series)

        if top_n > 0 and total_unique > top_n:
            counts_series_display = counts_series.head(top_n)
        else:
            counts_series_display = counts_series

        counts_df = counts_series_display.reset_index()
        counts_df.columns = ['Value', 'Count']
        counts_df['Percentage (%)'] = ((counts_df['Count'] / len(df)) * 100).round(2)

        return counts_df, total_unique

    except Exception as e:
        print(f"Error getting value counts for column '{column}': {e}")
        return empty_result


def calculate_correlation(df: pd.DataFrame, method: str = 'pearson') -> Optional[pd.DataFrame]:
    """
    Calculates the pairwise correlation matrix for numeric columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str): Correlation method ('pearson', 'kendall', 'spearman'). Defaults to 'pearson'.

    Returns:
        Optional[pd.DataFrame]: The correlation matrix (DataFrame), or None if insufficient numeric columns or an error occurs.
    """
    if df is None:
        print("Warning: Input DataFrame is None (calculate_correlation).")
        return None

    try:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            return None

        correlation_matrix = numeric_df.corr(method=method)
        return correlation_matrix
    except Exception as e:
        print(f"Error calculating {method} correlation: {e}")
        return None


def get_outlier_summary(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """
    Identifies potential outliers in numeric columns using the specified method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str): The outlier detection method ('iqr' currently supported).

    Returns:
        pd.DataFrame: A DataFrame summarizing outliers ('Column', 'Outlier Count', 'Percentage (%)').
                      Returns empty DataFrame if no numeric columns, df is None/empty, or on error.
                      Sorted by Outlier Count descending.
    """
    summary_cols = ['Column', 'Outlier Count', 'Percentage (%)']
    if df is None or df.empty:
        print("Warning: Input DataFrame is None or empty for outlier detection.")
        return pd.DataFrame(columns=summary_cols)

    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.any():
        print("Warning: No numeric columns found for outlier detection.")
        return pd.DataFrame(columns=summary_cols)

    outlier_data: List[Dict[str, Any]] = []
    total_rows = len(df)

    if method.lower() == 'iqr':
        try:
            for col in numeric_cols:
                col_data_dropna = df[col].dropna()
                if len(col_data_dropna) < 3:
                     continue

                Q1 = col_data_dropna.quantile(0.25)
                Q3 = col_data_dropna.quantile(0.75)
                IQR = Q3 - Q1

                if IQR <= 0:
                    continue

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_series = df.loc[df[col].notna(), col]
                outlier_count = outliers_series[(outliers_series < lower_bound) | (outliers_series > upper_bound)].count()

                if outlier_count > 0:
                    outlier_percentage = (outlier_count / total_rows) * 100
                    outlier_data.append({
                        'Column': col,
                        'Outlier Count': outlier_count,
                        'Percentage (%)': outlier_percentage
                    })

            summary_df = pd.DataFrame(outlier_data)

            if not summary_df.empty:
                 summary_df['Percentage (%)'] = summary_df['Percentage (%)'].round(2)
                 summary_df = summary_df.sort_values(by='Outlier Count', ascending=False)

            return summary_df if not summary_df.empty else pd.DataFrame(columns=summary_cols)

        except Exception as e:
            print(f"Error during IQR outlier detection: {e}")
            return pd.DataFrame(columns=summary_cols)

    else:
        print(f"Warning: Outlier detection method '{method}' not currently supported.")
        return pd.DataFrame(columns=summary_cols)