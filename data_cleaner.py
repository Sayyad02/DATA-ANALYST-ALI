# data_cleaner.py
import pandas as pd
import numpy as np
import streamlit as st # Used only for st.error/warning/info feedback
from typing import Optional, List, Any

def handle_missing_values(df: pd.DataFrame, strategy: str, columns: Optional[List[str]] = None, fill_value: Optional[Any] = None) -> Optional[pd.DataFrame]:
    """
    Handles missing values in a DataFrame based on the chosen strategy.

    Args:
        df (pd.DataFrame): The DataFrame to clean (a copy should be passed).
        strategy (str): Method to use ('drop_row', 'fill_mean', 'fill_median', 'fill_mode', 'fill_value').
        columns (Optional[List[str]]): List of columns to apply the strategy to. If None, applies based on strategy logic.
        fill_value (Optional[Any]): The value to use when strategy is 'fill_value'.

    Returns:
        Optional[pd.DataFrame]: The cleaned DataFrame, or the original if no changes made/needed, or None if a fatal error occurs.
                                Shows st.error/warning/info for feedback within the Streamlit app.
    """
    if df is None:
        st.error("Input DataFrame is None (handle_missing_values).", icon="⚠️")
        return None

    df_cleaned = df # Work on the passed copy directly

    try:
        # Determine target columns: specified columns or all columns
        target_cols = columns if columns else df_cleaned.columns.tolist()
        # Filter target_cols to only those present in the DataFrame
        valid_target_cols = [col for col in target_cols if col in df_cleaned.columns]
        if len(valid_target_cols) != len(target_cols):
             missing_user_cols = set(target_cols) - set(valid_target_cols)
             st.warning(f"Columns specified but not found: {missing_user_cols}. Applying only to valid columns.", icon="⚠️")
             if not valid_target_cols:
                  st.warning("No valid columns selected or found to apply the strategy.", icon="⚠️")
                  return df # Return original if no valid columns

        # --- Strategy Implementation ---
        if strategy == 'drop_row':
            original_rows = len(df_cleaned)
            # Drop rows where NA appears in ANY of the valid_target_cols
            df_cleaned.dropna(subset=valid_target_cols, inplace=True)
            rows_dropped = original_rows - len(df_cleaned)
            if rows_dropped > 0 :
                 st.info(f"Dropped {rows_dropped} rows containing missing values in columns: {', '.join(valid_target_cols)}.")
            else:
                 st.info("No rows dropped (no missing values found in selected columns).")

        elif strategy in ['fill_mean', 'fill_median']:
            # Apply only to numeric columns within the valid target columns
            numeric_target_cols = df_cleaned[valid_target_cols].select_dtypes(include=np.number).columns
            if not numeric_target_cols.any():
                 st.warning(f"Strategy '{strategy}' requires numeric columns, but none found in the selection: {', '.join(valid_target_cols)}.", icon="⚠️")
                 return df # Return original df
            filled_cols_count = 0
            for col in numeric_target_cols:
                 if df_cleaned[col].isnull().any(): # Only fill if needed
                    fill_val = df_cleaned[col].mean() if strategy == 'fill_mean' else df_cleaned[col].median()
                    df_cleaned[col].fillna(fill_val, inplace=True)
                    filled_cols_count += 1
            if filled_cols_count > 0:
                 st.info(f"Filled missing values in {filled_cols_count} numeric column(s) using {strategy}.")
            else:
                 st.info(f"No missing values found in the selected numeric columns to fill with {strategy}.")

        elif strategy == 'fill_mode':
             filled_cols_count = 0
             for col in valid_target_cols:
                 if df_cleaned[col].isnull().any():
                    mode_val = df_cleaned[col].mode()
                    if not mode_val.empty:
                        # Handle potential multiple modes - consistently use the first one
                        df_cleaned[col].fillna(mode_val[0], inplace=True)
                        filled_cols_count += 1
                    else:
                        st.warning(f"Could not find a mode for column '{col}'. Missing values remain.", icon="⚠️")
             if filled_cols_count > 0:
                  st.info(f"Filled missing values in {filled_cols_count} column(s) using mode.")
             else:
                  st.info("No missing values found in the selected columns to fill with mode.")

        elif strategy == 'fill_value':
            if fill_value is None:
                 st.error("Strategy 'fill_value' requires a value to be entered.", icon="❌")
                 return None # Return None as it's a fatal configuration error
            filled_cols_count = 0
            for col in valid_target_cols:
                if df_cleaned[col].isnull().any():
                     try:
                          df_cleaned[col].fillna(fill_value, inplace=True)
                          filled_cols_count += 1
                     except Exception as convert_err:
                          st.warning(f"Could not reliably fill column '{col}' with value '{fill_value}'. Potential type mismatch? Error: {convert_err}", icon="⚠️")
            if filled_cols_count > 0:
                 st.info(f"Filled missing values in {filled_cols_count} column(s) using the specified value.")
            else:
                 st.info("No missing values found in the selected columns to fill.")

        else:
            st.error(f"Unknown missing value strategy: '{strategy}'", icon="❌")
            return None

        return df_cleaned

    except Exception as e:
        st.error(f"Error handling missing values: {e}", icon="❌")
        return None # Return None on unexpected failure


def remove_duplicates(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Removes duplicate rows from the DataFrame based on all columns.

    Args:
        df (pd.DataFrame): The input DataFrame (a copy should be passed).

    Returns:
        Optional[pd.DataFrame]: DataFrame with duplicates removed, or None on error.
    """
    if df is None:
        print("Error: Input DataFrame is None (remove_duplicates).")
        return None
    try:
        df_cleaned = df.drop_duplicates(keep='first')
        return df_cleaned
    except Exception as e:
        st.error(f"Error removing duplicates: {e}", icon="❌")
        return None


def change_data_type(df: pd.DataFrame, column: str, target_type: str) -> Optional[pd.DataFrame]:
    """
    Changes the data type of a specified column.

    Args:
        df (pd.DataFrame): The input DataFrame (a copy should be passed).
        column (str): The name of the column to modify.
        target_type (str): The desired data type ('int', 'float', 'str', 'datetime', 'bool', 'category').

    Returns:
        Optional[pd.DataFrame]: DataFrame with modified type, or the original if conversion fails/no change, or None on fatal error.
                                Shows st.error/warning for feedback.
    """
    if df is None:
        print("Error: Input DataFrame is None (change_data_type).")
        return None
    if column not in df.columns:
        st.error(f"Column '{column}' not found in DataFrame.", icon="❌")
        return None

    df_modified = df # Work on the passed copy
    original_type = df_modified[column].dtype

    try:
        converted_series = None
        if target_type == 'datetime':
            converted_series = pd.to_datetime(df_modified[column], errors='coerce')
            if converted_series.isnull().sum() > df[column].isnull().sum():
                 st.warning(f"Some values in '{column}' could not be converted to datetime and became NaT.", icon="⚠️")
        elif target_type == 'int':
             numeric_series = pd.to_numeric(df_modified[column], errors='coerce')
             if numeric_series.isnull().sum() > df[column].isnull().sum():
                  st.warning(f"Some values in '{column}' could not be interpreted as numeric and became NaN.", icon="⚠️")
             converted_series = numeric_series.astype('Int64') # Use nullable integer
        elif target_type == 'float':
            converted_series = pd.to_numeric(df_modified[column], errors='coerce')
            if converted_series.isnull().sum() > df[column].isnull().sum():
                 st.warning(f"Some values in '{column}' could not be converted to float (e.g., non-numeric text) and became NaN.", icon="⚠️")
        elif target_type == 'bool':
             # Flexible boolean mapping
             map_dict = {'true': True, 'yes': True, '1': True, 't': True, 'y': True,
                         'false': False, 'no': False, '0': False, 'f': False, 'n': False,
                         1: True, 0: False, True: True, False: False}
             # Apply mapping to lowercased string version, keep original if not mappable -> becomes NaN then False?
             converted_series = df_modified[column].astype(str).str.lower().map(map_dict).astype('boolean') # Use nullable boolean
             # Check if unexpected NaNs were created
             if converted_series.isnull().sum() > df[column].isnull().sum():
                 st.warning(f"Some values in '{column}' could not be interpreted as boolean (e.g., 'maybe', empty strings) and became NA.", icon="⚠️")
        elif target_type in ['str', 'category']:
            converted_series = df_modified[column].astype(target_type)
        else:
             st.error(f"Unsupported target data type '{target_type}' specified.", icon="❌")
             return None

        df_modified[column] = converted_series

        if df_modified[column].dtype == original_type and str(target_type) != str(original_type).lower():
             st.warning(f"Attempted conversion of '{column}' to '{target_type}', but dtype remained '{original_type}'. Data might be incompatible or already conformant.", icon="⚠️")
        elif df_modified[column].dtype != original_type:
             st.success(f"Successfully converted column '{column}' to type '{df_modified[column].dtype}'.")

        return df_modified

    except ValueError as ve:
         st.error(f"Conversion Error for column '{column}' to '{target_type}': {ve}. Ensure data is compatible.", icon="❌")
         return None
    except TypeError as te:
         st.error(f"Type Error during conversion for column '{column}' to '{target_type}': {te}.", icon="❌")
         return None
    except Exception as e:
        st.error(f"Unexpected error changing data type for column '{column}': {e}", icon="❌")
        return None


def drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> Optional[pd.DataFrame]:
    """
    Drops specified columns from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame (a copy should be passed).
        columns_to_drop (List[str]): A list of column names to remove.

    Returns:
        Optional[pd.DataFrame]: DataFrame with columns dropped, or None on error.
                                Shows st.warning/info for feedback.
    """
    if df is None:
        print("Error: Input DataFrame is None (drop_columns).")
        return None
    if not columns_to_drop:
        st.warning("No columns selected to drop.", icon="⚠️")
        return df

    df_dropped = df # Work on the passed copy

    existing_cols_to_drop = [col for col in columns_to_drop if col in df_dropped.columns]
    missing_cols = [col for col in columns_to_drop if col not in df_dropped.columns]

    if missing_cols:
         st.warning(f"Columns not found and cannot be dropped: {', '.join(missing_cols)}", icon="⚠️")

    if not existing_cols_to_drop:
         st.info("No valid columns specified to drop.")
         return df

    try:
        df_dropped = df_dropped.drop(columns=existing_cols_to_drop)
        st.info(f"Dropped columns: {', '.join(existing_cols_to_drop)}")
        return df_dropped
    except Exception as e:
        st.error(f"Error dropping columns: {e}", icon="❌")
        return None