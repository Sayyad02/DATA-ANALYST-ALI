# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import io
import matplotlib.pyplot as plt
from datetime import datetime
import pytz # For timezone handling
import os
from typing import Optional, List, Tuple, Any # Ensure typing imports are here

# --- helper modules ---
try:
    import data_loader
    import data_cleaner
    import data_explorer
    import data_visualizer
    import data_analyzer
except ImportError as e:
    st.error(f"Error importing helper modules: {e}. Please ensure all .py files (loader, cleaner, explorer, visualizer, analyzer) are in the same directory and required libraries (pandas, numpy, matplotlib, seaborn, scipy, sklearn, wordcloud, openpyxl, pytz) are installed.", icon="🚨")
    st.stop()

# Import specific components if needed (though direct calls are used here)
try:
    from sklearn.decomposition import PCA # Specifically for type hinting if needed elsewhere
except ImportError:
    PCA = None # Define as None if sklearn is missing, checks later will handle it

# --- Page Configuration ---
st.set_page_config(
    page_title="DATA ANALYSIS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
MAX_HISTORY_SIZE = 10 # Limit the number of undo steps to prevent memory issues

# --- Initialize Session State ---
# Using st.session_state to preserve data and settings across reruns
default_values = {
    'df': None,
    'df_name': "Untitled",
    'corr_matrix': None,
    'corr_method_used': "Pearson",
    'numeric_cols': [],
    'categorical_cols': [],
    'all_cols': [],
    'main_action': "View Data Info",
    'plot_button_clicked': False, # Track if a plot button was clicked in the current run
    'report_content': None,
    'sample_size': 5, # Default sample size
    # Add state for K-Means results display after rerun
    'kmeans_results_display': None,
    # --- History / Undo-Redo State ---
    'df_history': [],       # List to store previous DataFrame states (copies)
    'history_pointer': -1, # Index of the current state in df_history
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper function to update column lists in session state ---
def update_column_lists(df: Optional[pd.DataFrame]):
    """Updates lists of column names based on dtype in session state."""
    if df is not None:
        st.session_state.all_cols = df.columns.tolist()
        st.session_state.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.session_state.categorical_cols = df.select_dtypes(
            include=['object', 'category', 'bool'],
            exclude=[np.number, 'datetime64[ns]', 'timedelta64[ns]']
        ).columns.tolist()
    else:
        st.session_state.all_cols = []
        st.session_state.numeric_cols = []
        st.session_state.categorical_cols = []

# --- Helper function for Timestamp ---
def get_current_timestamp(timezone='Asia/Kolkata'):
    """Returns a formatted timestamp string in the specified timezone."""
    try:
        tz = pytz.timezone(timezone)
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as tz_err:
        print(f"Timezone error ({timezone}): {tz_err}. Falling back.")
        try: # Try local timezone if pytz failed
             local_tz_name = datetime.now().astimezone().tzname()
             return datetime.now().strftime(f"%Y-%m-%d %H:%M:%S ({local_tz_name})")
        except Exception: # Fallback to UTC if local fails
             return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# --- History Management Functions ---
def save_state_for_undo(current_df: pd.DataFrame):
    """Saves the current DataFrame state to history for undo functionality."""
    if current_df is None:
        return # Don't save None state

    # Truncate history if we branched off after an undo
    if st.session_state.history_pointer < len(st.session_state.df_history) - 1:
        st.session_state.df_history = st.session_state.df_history[:st.session_state.history_pointer + 1]

    # Append a copy of the current state
    st.session_state.df_history.append(current_df.copy())

    # Limit history size
    if len(st.session_state.df_history) > MAX_HISTORY_SIZE:
        st.session_state.df_history.pop(0) # Remove the oldest state

    # Update pointer to the new latest state
    st.session_state.history_pointer = len(st.session_state.df_history) - 1
    # Debug: st.sidebar.caption(f"History size: {len(st.session_state.df_history)}, Pointer: {st.session_state.history_pointer}")

def undo_action():
    """Reverts to the previous state in the history."""
    if st.session_state.history_pointer > 0:
        st.session_state.history_pointer -= 1
        st.session_state.df = st.session_state.df_history[st.session_state.history_pointer].copy()
        update_column_lists(st.session_state.df)
        # Clear potentially invalid results from the undone state
        st.session_state.corr_matrix = None
        st.session_state.report_content = None
        st.session_state.kmeans_results_display = None
        st.toast("Undo successful!", icon="⏪")
        # Debug: st.sidebar.caption(f"Undo! Pointer: {st.session_state.history_pointer}")

def redo_action():
    """Reapplies the next state in the history."""
    if st.session_state.history_pointer < len(st.session_state.df_history) - 1:
        st.session_state.history_pointer += 1
        st.session_state.df = st.session_state.df_history[st.session_state.history_pointer].copy()
        update_column_lists(st.session_state.df)
        # Clear potentially invalid results from the redone state (might need recalculation)
        st.session_state.corr_matrix = None
        st.session_state.report_content = None
        st.session_state.kmeans_results_display = None
        st.toast("Redo successful!", icon="⏩")
        # Debug: st.sidebar.caption(f"Redo! Pointer: {st.session_state.history_pointer}")

def reset_history(initial_df: Optional[pd.DataFrame]):
    """Resets the history, optionally starting with an initial DataFrame."""
    if initial_df is not None:
        st.session_state.df_history = [initial_df.copy()]
        st.session_state.history_pointer = 0
    else:
        st.session_state.df_history = []
        st.session_state.history_pointer = -1
    # Debug: st.sidebar.caption("History reset!")

# --- Sidebar ---
with st.sidebar:
    st.title("📊 Data Analyst Ali")

    uploaded_file = st.file_uploader(
        "1. Upload your data (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="file_uploader",
        help="Upload a CSV or Excel file to start analysis."
    )

    # --- Logic to handle file upload and data loading ---
    if uploaded_file is not None:
        # Load data only if it's a new file or no data is currently loaded
        if st.session_state.df is None or st.session_state.df_name != uploaded_file.name:
            with st.spinner(f"Loading '{uploaded_file.name}'..."):
                df_loaded = data_loader.load_data_from_upload(uploaded_file)
                if df_loaded is not None:
                    st.session_state.df = df_loaded
                    st.session_state.df_name = uploaded_file.name
                    # --- Clear previous analysis results AND HISTORY on new file upload ---
                    st.session_state.corr_matrix = None
                    st.session_state.report_content = None
                    st.session_state.corr_method_used = "Pearson" # Reset default
                    st.session_state.kmeans_results_display = None # Clear temp kmeans results
                    update_column_lists(st.session_state.df) # Update column lists based on loaded data
                    reset_history(st.session_state.df) # Reset history with the new df
                    st.success(f"Loaded '{st.session_state.df_name}' successfully!")
                    st.session_state.main_action = "View Data Info" # Reset action
                    st.rerun()
                else:
                    # Error message should be shown by the loader function via st.error
                    st.session_state.df = None # Ensure df is None on failure
                    st.session_state.df_name = "Untitled"
                    update_column_lists(None) # Clear column lists
                    reset_history(None) # Reset history

    # --- Actions (only show if data is loaded) ---
    if st.session_state.df is not None:
        st.markdown("---")
        st.header("2. Choose an Action")
        actions = (
            "View Data Info",
            "Explore Data",
            "Clean Data",
            "Analyze & Visualize",
            "Generate Detailed Report"
        )

        def clear_temp_results_on_action_change():
            """Clears temporary results like report previews when changing main action."""
            st.session_state.report_content = None
            st.session_state.kmeans_results_display = None # Also clear temp results display

        st.radio(
            "Select Task:",
            actions,
            key="main_action",
            on_change=clear_temp_results_on_action_change # Clear previews if user changes action
        )
        st.markdown("---")

        # --- Undo/Redo Section ---
        st.header("3. History")
        undo_disabled = st.session_state.history_pointer <= 0
        redo_disabled = st.session_state.history_pointer >= len(st.session_state.df_history) - 1

        col_undo, col_redo = st.columns(2)
        with col_undo:
            st.button("⏪ Undo", key="undo_button", on_click=undo_action, disabled=undo_disabled, use_container_width=True, help="Revert the last data modification.")
        with col_redo:
            st.button("Redo ⏩", key="redo_button", on_click=redo_action, disabled=redo_disabled, use_container_width=True, help="Reapply the next data modification.")
        # Display current history position (optional)
        st.caption(f"History position: {st.session_state.history_pointer + 1} of {len(st.session_state.df_history)}")


        # --- Download Button for Current Data (as Excel) ---
        st.markdown("---") # Separator
        st.header("4. Download Data")
        try:
            base_name, ext = os.path.splitext(st.session_state.df_name)
            processed_filename = f"{base_name}_processed_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

            output_excel = io.BytesIO()
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                st.session_state.df.to_excel(writer, index=False, sheet_name='Processed_Data')
            excel_data = output_excel.getvalue()

            st.download_button(
                label="⬇️ Download Current Data (Excel)",
                data=excel_data,
                file_name=processed_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel_button",
                help="Download the current state of the DataFrame as an Excel file."
            )
        except Exception as e:
            st.error(f"Error preparing Excel download: {e}", icon="⚠️")

# --- Main Panel ---
st.header(f" Output: {st.session_state.get('main_action', 'No Action Selected')}") # Show current action title

# Display welcome message if no data is loaded
if st.session_state.df is None:
    st.info("👋 Welcome! Please upload a CSV or Excel file using the sidebar to begin analysis.")
# Proceed with selected action if data is loaded
else:
    df = st.session_state.df # Local variable for easier access in this block
    current_action = st.session_state.main_action

    # ========================================
    # --- Action: View Data Info ---
    # ========================================
    if current_action == "View Data Info":
        st.subheader(f"Basic Information for '{st.session_state.df_name}'")
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

        st.subheader("DataFrame Head")
        st.dataframe(df.head())

        if st.button("Show Tail (Last 5 Rows)", key="show_tail"):
             st.subheader("DataFrame Tail")
             st.dataframe(df.tail())

        st.subheader("Random Sample")
        st.session_state.sample_size = st.number_input(
             "Number of random rows to show:", min_value=1, max_value=len(df),
             value=st.session_state.sample_size, key="sample_rows_input" )
        if st.button("Show Random Sample", key="show_sample"):
             # Ensure sample size isn't larger than dataframe length
             n_sample = min(st.session_state.sample_size, len(df))
             st.dataframe(df.sample(n=n_sample, random_state=42)) # Use random_state for reproducibility if desired

        st.subheader("Data Types & Non-Null Counts")
        buffer = io.StringIO()
        try:
            df.info(buf=buffer, verbose=True, show_counts=True)
            st.text(buffer.getvalue())
        except Exception as e:
            st.error(f"Could not retrieve DataFrame info: {e}", icon="⚠️")

        st.subheader("Duplicate Rows Check")
        try:
            num_duplicates = data_explorer.get_duplicates_count(df)
            perc_duplicates = (num_duplicates / len(df) * 100) if len(df) > 0 else 0
            if num_duplicates == 0:
                st.success("✅ No duplicate rows found.")
            else:
                st.warning(f"⚠️ Found {num_duplicates} duplicate rows ({perc_duplicates:.2f}%). Consider using 'Clean Data' -> 'Handle Duplicates'.")
        except AttributeError:
             st.error("Helper function `get_duplicates_count` not found.", icon="🚨")
        except Exception as e:
             st.error(f"Could not check for duplicates: {e}", icon="⚠️")

    # ========================================
    # --- Action: Explore Data ---
    # ========================================
    elif current_action == "Explore Data":
        st.subheader("Descriptive Statistics")
        try:
            stats_df = data_explorer.get_descriptive_stats(df, include_type='all')
            if stats_df is not None and not stats_df.empty:
                 st.dataframe(stats_df.round(3)) # Round for display
            else:
                 st.warning("Could not generate descriptive statistics or DataFrame is empty.")
        except AttributeError:
            st.error("Helper function `get_descriptive_stats` not found.", icon="🚨")
        except Exception as e:
           st.error(f"Could not generate descriptive statistics: {e}", icon="⚠️")

        st.subheader("Missing Values Summary")
        try:
            missing_df = data_explorer.get_missing_summary(df)
            if missing_df is None: # Handle None return on error
                 st.error("Failed to retrieve missing value summary.", icon="🚨")
            elif missing_df.empty:
                 st.success("✅ No missing values found.")
            else:
                 st.dataframe(missing_df)
                 st.caption("Consider using 'Clean Data' -> 'Handle Missing Values'.")
        except AttributeError:
            st.error("Helper function `get_missing_summary` not found.", icon="🚨")
        except Exception as e:
           st.error(f"Could not generate missing values summary: {e}", icon="⚠️")

        st.subheader("Outlier Summary (IQR Method)")
        st.markdown("Identifies potential outliers in numeric columns using the Interquartile Range (IQR) method (value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR).")
        if not st.session_state.numeric_cols:
            st.warning("No numeric columns available for outlier detection.")
        else:
            try:
                 outlier_summary_df = data_explorer.get_outlier_summary(df[st.session_state.numeric_cols], method='iqr')
                 if outlier_summary_df is None: # Handle None return on error
                       st.error("Failed to retrieve outlier summary.", icon="🚨")
                 elif outlier_summary_df.empty:
                        st.success("✅ No potential outliers detected based on IQR method in numeric columns.")
                 else:
                        st.warning("⚠️ Potential outliers detected:")
                        st.dataframe(outlier_summary_df)
                        st.caption("Note: Outlier detection is heuristic. Investigate these values further.")
            except AttributeError:
                 st.error("Helper function `get_outlier_summary` not found.", icon="🚨")
            except Exception as e:
                 st.error(f"Could not generate outlier summary: {e}", icon="⚠️")

        st.subheader("Value Counts for a Column")
        if not st.session_state.all_cols:
            st.warning("No columns available to analyze.")
        else:
            # Suggest categorical columns first, then others
            vc_options = sorted(st.session_state.categorical_cols) + sorted([c for c in st.session_state.all_cols if c not in st.session_state.categorical_cols])
            selected_col_vc = st.selectbox("Select column for value counts:", vc_options, key="vc_col")

            if selected_col_vc:
                try:
                    max_unique_slider = min(100, df[selected_col_vc].nunique(dropna=False)) # Include NaN in unique count
                    default_slider_val = min(10, max_unique_slider if max_unique_slider > 0 else 1)

                    top_n_vc = st.slider( f"Number of top unique values to show for '{selected_col_vc}':", min_value=1,
                        max_value=max_unique_slider if max_unique_slider > 0 else 1, # Ensure max_value is at least 1
                        value=default_slider_val, key="vc_top_n",
                        help="Adjust the slider to see more or fewer unique values. Includes NaN if present." )

                    if st.button("Show Value Counts", key="show_vc_btn"):
                        with st.spinner(f"Calculating value counts for '{selected_col_vc}'..."):
                            counts_df, total_unique = data_explorer.get_value_counts(df, selected_col_vc, top_n=top_n_vc)
                        st.write(f"Total unique values in '{selected_col_vc}' (incl. NaN): **{total_unique}**")
                        if counts_df is not None and not counts_df.empty:
                             st.dataframe(counts_df)
                        elif counts_df is not None: # Empty dataframe returned
                             st.info("No values to display (column might be empty or only NaNs).")
                        else: # None returned
                             st.error("Could not retrieve value counts due to an error.")

                except AttributeError:
                    st.error("Helper function `get_value_counts` not found.", icon="🚨")
                except Exception as e:
                    st.error(f"An error occurred while getting value counts for '{selected_col_vc}': {e}", icon="⚠️")

    # ========================================
    # --- Action: Clean Data ---
    # ========================================
    elif current_action == "Clean Data":
        st.subheader("Data Cleaning Operations")
        st.caption("Note: Cleaning operations modify the DataFrame. Use Undo/Redo in the sidebar to revert changes. Download the processed data using the sidebar button.")
        tab1, tab2, tab3, tab4 = st.tabs(["Handle Missing Values", "Handle Duplicates", "Change Data Type", "Drop Columns"])

        # --- Missing Values Tab ---
        with tab1:
            st.markdown("#### Handle Missing Values")
            try:
                total_missing = df.isnull().sum().sum()
                if total_missing == 0:
                    st.success("✅ No missing values detected in the current data.")
                else:
                    missing_summary_df_clean = data_explorer.get_missing_summary(df)
                    if missing_summary_df_clean is not None: st.dataframe(missing_summary_df_clean)
                    cols_with_na = missing_summary_df_clean.index.tolist() if missing_summary_df_clean is not None else [] # Get columns with NA from summary

                    col1_na, col2_na = st.columns(2)
                    with col1_na:
                        strategy_na = st.selectbox("Choose strategy:", ['drop_row', 'fill_mean', 'fill_median', 'fill_mode', 'fill_value'], key="na_strategy",
                            help="Drop rows with NA, or fill with mean/median (numeric only), mode (most frequent), or a specific value.")
                    with col2_na:
                        columns_na = st.multiselect("Select columns to apply strategy:", options=st.session_state.all_cols, default=cols_with_na, key="na_cols",
                            help="Select columns. If blank, applies to columns with NA based on strategy (drop_row affects rows with NA in *any* column if blank).")

                    fill_value_na = None
                    if strategy_na == 'fill_value':
                        fill_value_na = st.text_input("Enter the value to fill missing entries with:", key="na_fill_val", help="The entered value will be used to fill NaNs in the selected columns.")

                    if st.button("Apply Missing Value Handling", key="apply_na", type="primary"):
                        target_cols_na = columns_na if columns_na else None
                        with st.spinner("Applying cleaning..."):
                            # Pass a COPY of the dataframe to the cleaning function
                            cleaned_df_na = data_cleaner.handle_missing_values(df.copy(), strategy_na, columns=target_cols_na, fill_value=fill_value_na)

                        # Update session state only if cleaning was successful and changed the DataFrame
                        if cleaned_df_na is not None and not df.equals(cleaned_df_na):
                            save_state_for_undo(st.session_state.df) # Save previous state BEFORE updating
                            st.session_state.df = cleaned_df_na
                            update_column_lists(st.session_state.df) # Update lists after potential type changes/drops
                            st.success("Missing values handled! DataFrame updated.")
                            st.rerun()
                        elif cleaned_df_na is not None: # Function returned dataframe, but it was identical
                             st.info("DataFrame unchanged. Strategy might not have been applicable or no changes occurred.")
                        # else: Error message should be shown by the cleaner function via st.error/warning

            except AttributeError:
                 st.error("Helper function `handle_missing_values` or `get_missing_summary` not found.", icon="🚨")
            except Exception as e:
                 st.error(f"An error occurred during missing value handling setup: {e}", icon="⚠️")

        # --- Duplicates Tab ---
        with tab2:
             st.markdown("#### Handle Duplicates")
             try:
                num_duplicates = data_explorer.get_duplicates_count(df)
                perc_duplicates = (num_duplicates / len(df) * 100) if len(df) > 0 else 0
                if num_duplicates > 0:
                    st.warning(f"⚠️ Found {num_duplicates} duplicate rows ({perc_duplicates:.2f}%).")
                    if st.button("Remove Duplicate Rows", key="remove_dupes", type="primary"):
                        with st.spinner("Removing duplicates..."):
                            cleaned_df_dupes = data_cleaner.remove_duplicates(df.copy())
                        # Update session state only if successful and changed
                        if cleaned_df_dupes is not None and not df.equals(cleaned_df_dupes):
                                save_state_for_undo(st.session_state.df) # Save previous state
                                st.session_state.df = cleaned_df_dupes
                                update_column_lists(st.session_state.df) # Row removal doesn't change cols, but good practice
                                st.success(f"{num_duplicates} duplicate rows removed! DataFrame updated.")
                                st.rerun() # Refresh UI
                        elif cleaned_df_dupes is not None:
                                st.info("No duplicates removed or DataFrame unchanged.")
                        else: # Function returned None
                           st.error("Failed to remove duplicates due to an error.", icon="🚨")
                else:
                    st.success("✅ No duplicate rows found.")
             except AttributeError:
                 st.error("Helper function `get_duplicates_count` or `remove_duplicates` not found.", icon="🚨")
             except Exception as e:
                 st.error(f"An error occurred during duplicate handling setup: {e}", icon="⚠️")

        # --- Data Type Tab ---
        with tab3:
            st.markdown("#### Change Column Data Type")
            if not st.session_state.all_cols: st.warning("No columns available.")
            else:
                try:
                    col_to_change = st.selectbox("Select column to change type:", st.session_state.all_cols, key="dtype_col")
                    current_type = df[col_to_change].dtype if col_to_change else "N/A"
                    st.write(f"Current data type: `{current_type}`")
                    target_type = st.selectbox("Select target data type:", ['int', 'float', 'str', 'datetime', 'bool', 'category'], key="dtype_target")

                    if st.button("Apply Type Change", key="apply_dtype", type="primary"):
                         if col_to_change and target_type:
                             with st.spinner(f"Attempting to convert '{col_to_change}' to {target_type}..."):
                                 modified_df_dtype = data_cleaner.change_data_type(df.copy(), col_to_change, target_type)

                             if modified_df_dtype is not None:
                                 if not df.equals(modified_df_dtype):
                                     save_state_for_undo(st.session_state.df) # Save previous state
                                     st.session_state.df = modified_df_dtype
                                     update_column_lists(st.session_state.df) # Type change might alter numeric/categorical lists
                                     new_type_check = modified_df_dtype[col_to_change].dtype
                                     if new_type_check != current_type:
                                          st.success(f"Column '{col_to_change}' type changed to `{new_type_check}`! DataFrame updated.")
                                     else:
                                          st.info(f"Values in '{col_to_change}' may have been modified (e.g., coercion to NaN), but dtype remained `{current_type}`. DataFrame updated.")
                                     st.rerun()
                                 else:
                                     st.info(f"Data type for '{col_to_change}' remains `{current_type}`. No changes applied.")
                             # else: Error message shown by cleaner function via st.error/warning

                except AttributeError:
                    st.error("Helper function `change_data_type` not found.", icon="🚨")
                except Exception as e:
                    st.error(f"An error occurred during data type change setup: {e}", icon="⚠️")

        # --- Drop Columns Tab ---
        with tab4:
            st.markdown("#### Drop Columns")
            if not st.session_state.all_cols: st.warning("No columns available to drop.")
            else:
                 try:
                     cols_to_drop = st.multiselect("Select columns to drop:", options=st.session_state.all_cols, key="drop_cols_select")
                     if st.button("Drop Selected Columns", key="apply_drop_cols", type="primary", disabled=not cols_to_drop):
                          if cols_to_drop:
                              with st.spinner(f"Dropping columns: {', '.join(cols_to_drop)}..."):
                                   cleaned_df_dropped = data_cleaner.drop_columns(df.copy(), cols_to_drop)

                              if cleaned_df_dropped is not None and not df.columns.equals(cleaned_df_dropped.columns):
                                  save_state_for_undo(st.session_state.df) # Save previous state
                                  st.session_state.df = cleaned_df_dropped
                                  update_column_lists(st.session_state.df) # Column lists definitely change here
                                  st.success(f"Columns dropped successfully! DataFrame updated.")
                                  st.rerun()
                              elif cleaned_df_dropped is not None: # No change occurred
                                   st.info("Columns not dropped or DataFrame unchanged.")
                              # else: Error message shown by cleaner function via st.error/warning
                          else:
                              st.warning("Please select at least one column to drop.")

                 except AttributeError:
                     st.error("Helper function `drop_columns` not found.", icon="🚨")
                 except Exception as e:
                     st.error(f"An error occurred during column dropping setup: {e}", icon="⚠️")


    # ========================================
    # --- Action: Analyze & Visualize ---
    # ========================================
    elif current_action == "Analyze & Visualize":
        st.subheader("Analysis and Visualization")
        st.session_state.plot_button_clicked = False # Reset flag for plot display logic

        # --- Tabbed Interface ---
        tab_corr, tab_plots, tab_advanced = st.tabs([
            "📊 Correlation & Heatmap",
            "📈 Generate Standard Plots",
            "🔬 Advanced Analysis"
        ])

        # --- Correlation Tab ---
        with tab_corr:
            st.markdown("#### Correlation Analysis")
            try:
                if len(st.session_state.numeric_cols) < 2:
                    st.warning("⚠️ Need at least two numeric columns to calculate correlation.")
                else:
                    corr_method = st.selectbox("Select correlation method:", ['pearson', 'kendall', 'spearman'], key="corr_method")
                    if st.button("Calculate Correlation Matrix", key="calc_corr", type="primary"):
                        with st.spinner(f"Calculating {corr_method} correlation..."):
                            corr_mat = data_explorer.calculate_correlation(df, method=corr_method)
                        if corr_mat is not None:
                            st.session_state.corr_matrix = corr_mat
                            st.session_state.corr_method_used = corr_method.capitalize() # Store method used
                            st.success(f"{st.session_state.corr_method_used} correlation matrix calculated.")
                            # Display nicely formatted correlation matrix
                            st.dataframe(st.session_state.corr_matrix.style.format("{:.3f}")
                                         .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1))
                        else:
                            st.error("Failed to calculate correlation matrix (check if numeric columns exist).", icon="🚨")
                            st.session_state.corr_matrix = None # Ensure it's None on failure

                    # --- Plot Heatmap ---
                    if st.session_state.corr_matrix is not None:
                        st.markdown("##### Correlation Heatmap")
                        if st.button("Plot Heatmap", key="plot_heatmap_btn"):
                            st.session_state.plot_button_clicked = True # Mark button clicked for this specific plot type
                            with st.spinner("Generating heatmap..."):
                                fig_heatmap = data_visualizer.plot_correlation_heatmap(st.session_state.corr_matrix)
                            # Display logic specific to this plot
                            if fig_heatmap:
                                st.pyplot(fig_heatmap)
                                plt.close(fig_heatmap) # Close figure to free memory
                            elif st.session_state.plot_button_clicked: # Check if button clicked but fig is None
                                 st.error("❌ Could not generate heatmap.", icon="🚨")

            except AttributeError:
                 st.error("Helper function `calculate_correlation` or `plot_correlation_heatmap` not found.", icon="🚨")
            except Exception as e:
                 st.error(f"An error occurred during correlation analysis setup: {e}", icon="⚠️")


        # --- Standard Plots Tab ---
        with tab_plots:
            st.markdown("#### Generate Standard Plots")
            plot_col1, plot_col2 = st.columns([1, 2])
            plot_fig = None # Initialize figure variable for this tab

            with plot_col1:
                plot_types = ["Histogram", "Box Plot", "Scatter Plot", "Bar Plot", "Pair Plot",
                              "Violin Plot", "Joint Plot", "Q-Q Plot", "Word Cloud"]
                plot_type = st.selectbox("Select plot type:", plot_types, key="plot_type_selector") # Unique key
                button_key_suffix = plot_type.lower().replace(" ", "_").replace("-", "_")
                generate_plot_button = st.button(f"📊 Generate {plot_type}", key=f"gen_{button_key_suffix}", type="primary")
                if generate_plot_button:
                     st.session_state.plot_button_clicked = True # Mark that *a* plot button was clicked in this run

            # --- Plotting Options Logic (inside plot_col2) ---
            with plot_col2:
                st.markdown("##### Plot Options")
                # --- Histogram ---
                if plot_type == "Histogram":
                    if st.session_state.numeric_cols:
                        hist_col = st.selectbox("Select numeric column:", st.session_state.numeric_cols, key="hist_col")
                        hist_kde = st.checkbox("Show density curve (KDE)", value=True, key="hist_kde")
                        if generate_plot_button and hist_col:
                            with st.spinner(f"Generating {plot_type}..."): plot_fig = data_visualizer.plot_histogram(df, hist_col, kde=hist_kde)
                    else: st.warning("⚠️ No numeric columns available.")
                # --- Box Plot ---
                elif plot_type == "Box Plot":
                    if st.session_state.numeric_cols:
                        box_col_y = st.selectbox("Select numeric column (Y-axis):", st.session_state.numeric_cols, key="box_y")
                        group_by_box = st.checkbox("Group by another column (X-axis)?", key="box_group_check", value=False)
                        box_col_x = None
                        if group_by_box:
                            box_x_options = sorted(st.session_state.categorical_cols) + sorted([c for c in st.session_state.all_cols if c != box_col_y and c not in st.session_state.categorical_cols])
                            if box_x_options: box_col_x = st.selectbox("Select grouping column (X-axis):", box_x_options, key="box_x")
                            else: st.warning("⚠️ No other columns available for grouping."); group_by_box = False
                        if generate_plot_button and box_col_y:
                            grouping_col = box_col_x if group_by_box and box_col_x else None
                            with st.spinner(f"Generating {plot_type}..."): plot_fig = data_visualizer.plot_boxplot(df, box_col_y, group_by=grouping_col)
                    else: st.warning("⚠️ No numeric columns available.")
                # --- Scatter Plot ---
                elif plot_type == "Scatter Plot":
                    if len(st.session_state.numeric_cols) >= 2:
                        scat_x = st.selectbox("Select X-axis column:", st.session_state.numeric_cols, key="scat_x")
                        scat_y_options = [col for col in st.session_state.numeric_cols if col != scat_x]
                        if scat_y_options: scat_y = st.selectbox("Select Y-axis column:", scat_y_options, key="scat_y")
                        else: st.warning("⚠️ Need at least two distinct numeric columns."); scat_y = None
                        scat_hue = None
                        if scat_x and scat_y:
                            hue_scat_check = st.checkbox("Color points by another column (Hue)?", key="scat_hue_check", value = False)
                            if hue_scat_check:
                                hue_options = sorted(st.session_state.categorical_cols) + sorted([c for c in st.session_state.all_cols if c != scat_x and c != scat_y and c not in st.session_state.categorical_cols])
                                if hue_options: scat_hue = st.selectbox("Select Hue column:", hue_options, key="scat_hue")
                                else: st.warning("⚠️ No other columns available for Hue.")
                        if generate_plot_button and scat_x and scat_y:
                            with st.spinner(f"Generating {plot_type}..."): plot_fig = data_visualizer.plot_scatterplot(df, scat_x, scat_y, hue_col=scat_hue)
                    else: st.warning("⚠️ Need at least two numeric columns.")
                # --- Bar Plot ---
                elif plot_type == "Bar Plot":
                    if st.session_state.all_cols:
                        suggested_x_cols_bar = sorted(st.session_state.categorical_cols) + sorted([c for c in st.session_state.all_cols if c not in st.session_state.categorical_cols])
                        bar_x = st.selectbox("Select X-axis column (Categorical Suggested):", suggested_x_cols_bar, key="bar_x")
                        is_agg_bar = st.checkbox("Aggregate a numeric column on Y-axis? (If unchecked, shows counts)", key="bar_agg_check", value = False)
                        bar_y = None; bar_est = 'count'
                        if is_agg_bar:
                            if st.session_state.numeric_cols:
                                bar_y_options = [col for col in st.session_state.numeric_cols if col != bar_x]
                                if bar_y_options:
                                    bar_y = st.selectbox("Select numeric Y-axis column:", bar_y_options, key="bar_y")
                                    bar_est = st.selectbox("Select aggregation:", ['mean', 'sum', 'median'], index=0, key="bar_est")
                                else: st.warning("⚠️ No suitable numeric columns for Y-axis aggregation."); is_agg_bar = False
                            else: st.warning("⚠️ No numeric columns available."); is_agg_bar = False
                        if generate_plot_button and bar_x:
                             y_val_for_func = bar_y if is_agg_bar and bar_y else None
                             est_for_func = bar_est if is_agg_bar and bar_y else 'count'
                             with st.spinner(f"Generating {plot_type}..."): plot_fig = data_visualizer.plot_barplot(df, bar_x, y_col=y_val_for_func, estimator=est_for_func)
                    else: st.warning("⚠️ No columns available.")
                # --- Pair Plot ---
                elif plot_type == "Pair Plot":
                     st.markdown("Scatter plots for pairs of numeric columns, plus distributions on diagonal.")
                     if len(st.session_state.numeric_cols) >= 2:
                          default_pair_cols = st.session_state.numeric_cols[:min(len(st.session_state.numeric_cols), 5)]
                          pair_cols = st.multiselect("Select numeric columns for pair plot (2+):", st.session_state.numeric_cols, default=default_pair_cols, key="pair_cols_select")
                          pair_hue = None
                          hue_pair_check = st.checkbox("Color points by another column (Hue)?", key="pair_hue_check", value = False)
                          if hue_pair_check:
                              hue_options_pair = sorted(st.session_state.categorical_cols) + sorted([col for col in st.session_state.all_cols if col not in pair_cols and col not in st.session_state.categorical_cols])
                              if hue_options_pair: pair_hue = st.selectbox("Select Hue column:", hue_options_pair, key="pair_hue")
                              else: st.warning("⚠️ No suitable columns available for Hue.")
                          if generate_plot_button and pair_cols and len(pair_cols) >= 2:
                                with st.spinner(f"Generating {plot_type}... (May take time)"): plot_fig = data_visualizer.plot_pairplot(df, columns=pair_cols, hue_col=pair_hue)
                          elif generate_plot_button: st.warning("⚠️ Please select at least two numeric columns.")
                     else: st.warning("⚠️ Need at least two numeric columns.")
                # --- Violin Plot ---
                elif plot_type == "Violin Plot":
                    if st.session_state.numeric_cols:
                        violin_y = st.selectbox("Select numeric column (Y-axis, shows distribution):", st.session_state.numeric_cols, key="violin_y")
                        violin_group = st.checkbox("Group by another column (X-axis)?", key="violin_group_check", value=False)
                        violin_x = None
                        if violin_group:
                            x_opts_v = sorted(st.session_state.categorical_cols) + sorted([c for c in st.session_state.all_cols if c != violin_y and c not in st.session_state.categorical_cols])
                            if x_opts_v: violin_x = st.selectbox("Select grouping column (X-axis):", x_opts_v, key="violin_x")
                            else: st.warning("⚠️ No other columns available for grouping."); violin_group = False
                        if generate_plot_button and violin_y:
                             with st.spinner(f"Generating {plot_type}..."): plot_fig = data_visualizer.plot_violin(df, violin_y, x_col=violin_x)
                    else: st.warning("⚠️ No numeric columns available.")
                # --- Joint Plot ---
                elif plot_type == "Joint Plot":
                    st.markdown("Relationship between two numerics + individual distributions.")
                    if len(st.session_state.numeric_cols) >= 2:
                        joint_x = st.selectbox("Select X-axis column:", st.session_state.numeric_cols, key="joint_x")
                        joint_y_opts = [c for c in st.session_state.numeric_cols if c != joint_x]
                        if joint_y_opts:
                             joint_y = st.selectbox("Select Y-axis column:", joint_y_opts, key="joint_y")
                             joint_kind = st.selectbox("Select plot kind:", ['scatter', 'kde', 'hist', 'reg'], key="joint_kind")
                             if generate_plot_button and joint_x and joint_y:
                                 with st.spinner(f"Generating {plot_type}..."): plot_fig = data_visualizer.plot_joint(df, joint_x, joint_y, kind=joint_kind)
                        else: st.warning("⚠️ Need at least two distinct numeric columns.")
                    else: st.warning("⚠️ Need at least two numeric columns.")
                # --- Q-Q Plot ---
                elif plot_type == "Q-Q Plot":
                     st.markdown("Visually check if data follows a Normal distribution.")
                     if st.session_state.numeric_cols:
                          qq_col = st.selectbox("Select numeric column:", st.session_state.numeric_cols, key="qq_col")
                          if generate_plot_button and qq_col:
                              with st.spinner(f"Generating {plot_type}..."):
                                   series_qq = df[qq_col].dropna()
                                   if len(series_qq) > 2: plot_fig = data_visualizer.plot_qq(series_qq)
                                   else: st.warning("⚠️ Not enough non-missing data points (>2).")
                     else: st.warning("⚠️ No numeric columns available.")
                # --- Word Cloud ---
                elif plot_type == "Word Cloud":
                     st.markdown("Generates a Word Cloud from a text column.")
                     text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                     if text_cols:
                          wc_col = st.selectbox("Select text column:", text_cols, key="wc_col")
                          if generate_plot_button and wc_col:
                              with st.spinner(f"Generating {plot_type}..."): plot_fig = data_visualizer.plot_wordcloud(df[wc_col])
                     else: st.warning("⚠️ No text (object/string) columns found.")


            # --- Display the generated plot in tab_plots ---
            st.markdown("##### Plot Output")
            if plot_fig:
                if isinstance(plot_fig, sns.PairGrid):
                     st.pyplot(plot_fig.fig) # Display the underlying figure of the PairGrid
                     try: plt.close(plot_fig.fig) # Close the underlying figure
                     except Exception: pass
                else: # Assume it's a standard matplotlib figure
                     st.pyplot(plot_fig)
                     try: plt.close(plot_fig)
                     except Exception: pass
            elif st.session_state.plot_button_clicked: # Check if button clicked but fig is None/False
                 st.error("❌ Could not generate the selected plot. Please check selections, data types, and logs.", icon="🚨")
            elif not st.session_state.plot_button_clicked: # No button clicked yet in this run
                 st.info("Select plot options above and click 'Generate Plot'.")


        # --- Advanced Analysis Tab ---
        with tab_advanced:
            st.markdown("#### Advanced Analysis Tools")
            st.caption("Apply statistical tests, clustering, or dimensionality reduction. Modifications here are added to the Undo/Redo history.")

            # --- 1. Normality Test ---
            st.markdown("##### Normality Test (Shapiro-Wilk)")
            if st.session_state.numeric_cols:
                 norm_col = st.selectbox("Select numeric column to test for normality:", st.session_state.numeric_cols, key="norm_col_select")
                 if st.button("Run Normality Test", key="run_norm_test"):
                     if norm_col:
                         with st.spinner(f"Running Shapiro-Wilk test on '{norm_col}'..."):
                             series_to_test = df[norm_col].dropna() # Drop NA before testing
                             if len(series_to_test) >= 3:
                                  norm_result = data_analyzer.perform_normality_test(series_to_test, method='shapiro')
                                  if norm_result:
                                      stat, p_val, interp = norm_result
                                      st.write(f"**Result:** {interp}")
                                      st.write(f"* Statistic: `{stat:.4f}`")
                                      st.write(f"* P-value: `{p_val:.4g}`")
                                      st.markdown("Consider using `Generate Standard Plots -> Q-Q Plot` to visually inspect.")
                                  # else: Error shown by analyzer function via st.error
                             else:
                                  st.warning("Not enough non-missing data points (>=3) for normality test.", icon="⚠️")
                     else: st.warning("Please select a column.")
            else: st.info("No numeric columns available for normality testing.")
            st.markdown("---")

            # --- 2. Independent Samples T-test ---
            st.markdown("##### Independent Samples T-test")
            st.caption("Compare means of a numeric variable between **two** groups.")
            suitable_group_cols_ttest = [c for c in st.session_state.all_cols if df[c].dropna().nunique()==2]
            if suitable_group_cols_ttest and st.session_state.numeric_cols:
                 col_ttest_group = st.selectbox("Select Grouping Column (must have exactly 2 groups):", suitable_group_cols_ttest, key="ttest_group")
                 col_ttest_value = st.selectbox("Select Value Column (Numeric) :", st.session_state.numeric_cols, key="ttest_value_selector") # Unique key
                 equal_var_ttest = st.checkbox("Assume equal variances (Student's t-test)?", value=True, key="ttest_eq_var", help="Uncheck to use Welch's t-test if variances might be unequal.")

                 if st.button("Run T-test", key="run_ttest"):
                     if col_ttest_group and col_ttest_value:
                         with st.spinner("Running T-test..."):
                             ttest_result = data_analyzer.perform_ttest_ind(df, col_ttest_group, col_ttest_value, equal_var=equal_var_ttest)
                             if ttest_result:
                                 stat, p_val, interp = ttest_result
                                 st.write(f"**Result:** {interp}")
                                 st.write(f"* Statistic: `{stat:.4f}`")
                                 st.write(f"* P-value: `{p_val:.4g}`")
                             # else: Error shown by analyzer function via st.error
                     else: st.warning("Please select both columns.")
            else: st.info("Need at least one numeric column and one suitable categorical column (with exactly 2 unique non-missing groups).")
            st.markdown("---")

            # --- 3. One-Way ANOVA ---
            st.markdown("##### One-Way ANOVA")
            st.caption("Compare means of a numeric variable across **two or more** groups.")
             # Find columns with 2 to 50 unique non-NA values
            suitable_group_cols_anova = [c for c in st.session_state.all_cols if 2 <= df[c].dropna().nunique() <= 50]
            if suitable_group_cols_anova and st.session_state.numeric_cols:
                 col_anova_group = st.selectbox("Select Grouping Column (2-50 groups):", suitable_group_cols_anova, key="anova_group")
                 col_anova_value = st.selectbox("Select Value Column (Numeric) :", st.session_state.numeric_cols, key="anova_value_selector") # Unique key

                 if st.button("Run ANOVA", key="run_anova"):
                     if col_anova_group and col_anova_value:
                          with st.spinner("Running ANOVA..."):
                              anova_result = data_analyzer.perform_anova(df, col_anova_group, col_anova_value)
                              if anova_result:
                                  stat, p_val, interp = anova_result
                                  st.write(f"**Result:** {interp}")
                                  st.write(f"* F-Statistic: `{stat:.4f}`")
                                  st.write(f"* P-value: `{p_val:.4g}`")
                              # else: Error shown by analyzer function via st.error
                     else: st.warning("Please select both columns.")
            else: st.info("Need at least one numeric column and one suitable categorical column (with 2 to 50 unique non-missing groups).")
            st.markdown("---")

            # --- 4. Chi-Squared Test ---
            st.markdown("##### Chi-Squared Test of Independence")
            st.caption("Test for association between **two categorical** variables.")
             # Find non-numeric columns with > 1 unique level
            cat_cols_chi = [c for c in st.session_state.all_cols if not pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().nunique() > 1]
            if len(cat_cols_chi) >= 2:
                 col_chi1 = st.selectbox("Select First Categorical Variable:", cat_cols_chi, key="chi1_col")
                 col_chi2_options = [c for c in cat_cols_chi if c != col_chi1]
                 if col_chi2_options:
                     col_chi2 = st.selectbox("Select Second Categorical Variable:", col_chi2_options, key="chi2_col")
                     if st.button("Run Chi-Squared Test", key="run_chi2"):
                         if col_chi1 and col_chi2:
                             with st.spinner("Running Chi-Squared test..."):
                                 chi2_result = data_analyzer.perform_chi_squared(df, col_chi1, col_chi2)
                                 if chi2_result:
                                     stat, p_val, dof, interp = chi2_result
                                     st.write(f"**Result:** {interp}")
                                     st.write(f"* Chi2 Statistic: `{stat:.4f}`")
                                     st.write(f"* P-value: `{p_val:.4g}`")
                                     st.write(f"* Degrees of Freedom: `{dof}`")
                                     with st.expander("Show Contingency Table"):
                                          try: st.dataframe(pd.crosstab(df[col_chi1], df[col_chi2]))
                                          except Exception as ct_err: st.error(f"Could not display contingency table: {ct_err}")
                                 # else: Error shown by analyzer function via st.error
                         else: st.warning("Please select two different categorical variables.")
                 else: st.info("Need at least two distinct suitable categorical columns.")
            else: st.info("Need at least two categorical columns (with >1 level each) for Chi-Squared test.")
            st.markdown("---")

            # --- 5. K-Means Clustering ---
            st.markdown("##### K-Means Clustering")
            st.caption("Group data points into K clusters based on selected numeric features (data is automatically scaled). Adds a 'Cluster' column to the DataFrame.")

            # Display results stored from previous successful run (cleared on action change/undo/redo)
            if 'kmeans_results_display' in st.session_state and st.session_state.kmeans_results_display:
                results = st.session_state.kmeans_results_display
                st.write(f"**Last Run Results (K={results['k']}):**")
                st.write(f"* Inertia (lower is better): `{results['inertia']:.2f}`")
                st.write(f"* Silhouette Score (higher is better): `{results['score']:.3f}`")
                st.write("**Cluster Sizes (Current Data):**")
                if 'Cluster' in st.session_state.df.columns:
                     st.dataframe(st.session_state.df['Cluster'].value_counts().reset_index().rename(columns={'index':'Cluster', 'Cluster':'Size'}))
                else: st.warning("Could not find 'Cluster' column to display sizes (was it dropped, renamed, or undone?).")
                # Do not delete here, let it persist until next action/undo/redo/load
                st.markdown("---") # Separator after displaying results


            # --- UI for K-Means ---
            if st.session_state.numeric_cols:
                default_kmeans_cols = st.session_state.numeric_cols[:min(len(st.session_state.numeric_cols), 5)]
                kmeans_features = st.multiselect("Select Numeric Features for Clustering:", st.session_state.numeric_cols, default=default_kmeans_cols, key="kmeans_features")
                k_clusters = st.number_input("Number of Clusters (K):", min_value=2, max_value=20, value=3, step=1, key="kmeans_k")

                if st.button("Run K-Means", key="run_kmeans"):
                     if kmeans_features and k_clusters:
                         with st.spinner(f"Running K-Means with K={k_clusters}..."):
                             # Pass the current df state copy
                             kmeans_result = data_analyzer.perform_kmeans(st.session_state.df.copy(), kmeans_features, k_clusters)

                         if kmeans_result:
                             df_clustered, centers, inertia, score = kmeans_result
                             # Check if DataFrame actually changed (e.g., 'Cluster' added or updated)
                             if not st.session_state.df.equals(df_clustered):
                                 # Store results temporarily for display AFTER rerun
                                 st.session_state.kmeans_results_display = {
                                     'k': k_clusters, 'inertia': inertia, 'score': score
                                 }
                                 save_state_for_undo(st.session_state.df) # Save previous state
                                 st.session_state.df = df_clustered # Update the main DataFrame state
                                 update_column_lists(st.session_state.df) # Update column lists
                                 st.success("K-Means complete! DataFrame updated with 'Cluster' column. Refreshing...")
                                 st.rerun() # Rerun to show updated df and results message
                             else:
                                 st.info("K-Means ran but did not result in a change to the DataFrame (perhaps the 'Cluster' column already existed with the same assignments).")
                         else:
                             # Analyzer function should show error via st.error
                             st.warning("K-Means execution failed or returned no result. Check console/logs if necessary.")
                             # Clear any old results display state if it failed
                             if 'kmeans_results_display' in st.session_state: del st.session_state['kmeans_results_display']

                     else:
                         st.warning("Please select at least one feature and set K >= 2.")
            else:
                st.info("Need numeric columns for K-Means clustering.")
            st.markdown("---")


            # --- 6. Principal Component Analysis (PCA) ---
            st.markdown("##### Principal Component Analysis (PCA)")
            st.caption("Reduce dimensionality of selected numeric features (data is automatically scaled). Results can be added to the DataFrame.")
            if st.session_state.numeric_cols:
                default_pca_cols = st.session_state.numeric_cols[:min(len(st.session_state.numeric_cols), 10)]
                pca_features = st.multiselect("Select Numeric Features for PCA:", st.session_state.numeric_cols, default=default_pca_cols, key="pca_features")
                max_components = len(pca_features) if pca_features else 1
                if max_components < 1: st.info("Select at least one feature for PCA.")
                else:
                    n_components_pca = st.number_input("Number of Principal Components:", min_value=1, max_value=max_components, value=min(2, max_components), step=1, key="pca_n")

                    if st.button("Run PCA", key="run_pca"):
                         if pca_features and n_components_pca:
                             with st.spinner(f"Running PCA for {n_components_pca} components..."):
                                 # Pass a copy to avoid modifying original if PCA fails midway
                                 pca_result = data_analyzer.perform_pca(st.session_state.df.copy(), pca_features, n_components_pca)

                             if pca_result:
                                 pca_obj, pca_df_results = pca_result
                                 st.write("**PCA Results:**")
                                 variance_ratios = pca_obj.explained_variance_ratio_
                                 st.write(f"* Explained Variance Ratio per Component: {[f'{r:.3f}' for r in variance_ratios]}")
                                 st.write(f"* **Total Explained Variance ({pca_obj.n_components_} components): {variance_ratios.sum():.3f}**")
                                 st.write("First 5 rows of Principal Components:")
                                 st.dataframe(pca_df_results.head())

                                 # Option to add components back to main DataFrame
                                 # Use a separate button to confirm adding columns, preventing accidental modification on PCA run
                                 st.markdown("---")
                                 if st.button("Add PCA Components to DataFrame", key="add_pca_to_df_btn"):
                                     try:
                                         # Make a copy to modify
                                         df_with_pca = st.session_state.df.copy()
                                         cols_added_pca = []
                                         # Ensure no column name collision before joining
                                         existing_cols = set(df_with_pca.columns)
                                         new_pca_cols_map = {}
                                         for col in pca_df_results.columns:
                                             new_name = col; suffix = 1
                                             while new_name in existing_cols: new_name = f"{col}_{suffix}"; suffix += 1
                                             new_pca_cols_map[col] = new_name
                                             cols_added_pca.append(new_name)

                                         pca_df_renamed = pca_df_results.rename(columns=new_pca_cols_map)
                                         # Use join - safer for potential duplicate indices
                                         df_updated_with_pca = df_with_pca.join(pca_df_renamed)

                                         # Check if successful and update state
                                         if not st.session_state.df.equals(df_updated_with_pca):
                                             save_state_for_undo(st.session_state.df) # Save previous state
                                             st.session_state.df = df_updated_with_pca
                                             update_column_lists(st.session_state.df)
                                             st.success(f"Added PCA components ({', '.join(cols_added_pca)}) to DataFrame. Refreshing...")
                                             st.rerun()
                                         else:
                                             st.info("Adding PCA columns did not change the DataFrame (columns might already exist).")

                                     except Exception as e_join: st.error(f"Could not add PCA components to DataFrame: {e_join}")


                                 # Offer visualization of the calculated components
                                 st.write("**Visualization (Calculated Components):**")
                                 with st.spinner("Generating PCA plots..."):
                                     pca_fig = data_visualizer.plot_pca_results(pca_df_results, pca_obj)
                                     if pca_fig:
                                          st.pyplot(pca_fig)
                                          plt.close(pca_fig)
                                     # else: error shown by visualizer function

                             # else: Error shown by analyzer function via st.error
                         else: st.warning("Please select features and set number of components >= 1.")
            else: st.info("Need numeric columns for PCA.")

# ========================================
    # --- Action: Generate Detailed Report ---
    # ========================================
    elif current_action == "Generate Detailed Report":
        st.subheader(f"Detailed Report Generation for '{st.session_state.df_name}'")
        st.markdown("Click the button below to generate a comprehensive markdown report based on the **current state** of the data. This does not modify the data or history.")

        # Initialize flag if it doesn't exist
        if 'gen_report_btn_clicked' not in st.session_state:
             st.session_state.gen_report_btn_clicked = False

        if st.button("🚀 Generate Full Report", key="gen_report_btn", type="primary", help="Generates a markdown report summarizing the data."):
            st.session_state.gen_report_btn_clicked = True # Mark button clicked for this run
            st.session_state.report_content = None # Clear previous before generating new
            report_parts = []
            df_report = st.session_state.df.copy() # Use a copy for report generation safety
            MAX_UNIQUE_FOR_VALUE_COUNTS = 25 # Limit for showing full value counts
            TOP_N_VALUE_COUNTS = 10 # Show top N for high cardinality cols
            SAMPLE_ROWS_IN_REPORT = 5

            with st.spinner("Generating extensive report sections... Please wait."):
                try:
                    # --- Pre-calculate components needed for summary and sections ---
                    report_shape = df_report.shape
                    report_numeric_cols = df_report.select_dtypes(include=np.number).columns.tolist()
                    report_cat_cols = df_report.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                    num_duplicates_report = data_explorer.get_duplicates_count(df_report)
                    missing_summary_report = data_explorer.get_missing_summary(df_report)
                    total_missing_report = missing_summary_report['Count'].sum() if missing_summary_report is not None and not missing_summary_report.empty else 0
                    num_cols_with_missing = len(missing_summary_report) if missing_summary_report is not None else 0
                    perc_missing_total = (total_missing_report / (report_shape[0] * report_shape[1]) * 100) if report_shape[0] > 0 and report_shape[1] > 0 else 0
                    perc_duplicates_report = (num_duplicates_report / report_shape[0] * 100) if report_shape[0] > 0 else 0

                    # Correlation (use existing or calculate Pearson - needed for summary)
                    report_corr_matrix = None
                    high_corr_found = False
                    if st.session_state.corr_matrix is not None:
                        report_corr_matrix = st.session_state.corr_matrix
                    elif len(report_numeric_cols) >= 2:
                         try:
                             report_corr_matrix = data_explorer.calculate_correlation(df_report, method='pearson')
                         except Exception:
                             report_corr_matrix = None # Calculation failed

                    if report_corr_matrix is not None:
                         try:
                             corr_unstacked = report_corr_matrix.unstack()
                             if not corr_unstacked[ (abs(corr_unstacked) >= 0.8) & (abs(corr_unstacked) < 1.0) ].empty:
                                 high_corr_found = True
                         except Exception: pass # Ignore error in checking for high corr

                    # --- Basic Textual Summary Generation ---
                    def generate_basic_text_summary(shape, num_numeric, num_cat, total_missing, missing_cols_count, missing_perc, duplicates_count, duplicates_perc, high_corr_flag):
                        summary_lines = []
                        rows, cols = shape
                        summary_lines.append(f"The dataset contains **{rows}** rows and **{cols}** columns.")
                        summary_lines.append(f"It includes **{num_numeric}** numeric and **{num_cat}** categorical/object type columns.")

                        if total_missing > 0:
                            summary_lines.append(f"There are **{total_missing}** missing values ({missing_perc:.2f}% of all entries), affecting **{missing_cols_count}** column(s).")
                        else:
                            summary_lines.append("✅ No missing values were found.")

                        if duplicates_count > 0:
                            summary_lines.append(f"⚠️ **{duplicates_count}** duplicate rows ({duplicates_perc:.2f}%) were detected.")
                        else:
                             summary_lines.append("✅ No complete duplicate rows were found.")

                        if high_corr_flag:
                             summary_lines.append("Evidence of high correlation (>= 0.8 or <= -0.8) between some numeric variables was observed.")

                        return " ".join(summary_lines)
                    # --- End of Text Summary Generation ---


                    # --- Build Report Sections ---
                    section_counter = 1

                    # Section: Overview
                    report_parts.append(f"# Data Analysis Report for: {st.session_state.df_name}")
                    report_time = get_current_timestamp()
                    report_parts.append(f"\n*Report Generated: {report_time}*")
                    report_parts.append(f"\n## {section_counter}. Executive Summary") # Renamed first section
                    section_counter += 1
                    # --- Insert Textual Summary Here ---
                    text_summary = generate_basic_text_summary(
                        shape=report_shape,
                        num_numeric=len(report_numeric_cols),
                        num_cat=len(report_cat_cols),
                        total_missing=total_missing_report,
                        missing_cols_count=num_cols_with_missing,
                        missing_perc=perc_missing_total,
                        duplicates_count=num_duplicates_report,
                        duplicates_perc=perc_duplicates_report,
                        high_corr_flag=high_corr_found
                    )
                    report_parts.append(f"\n{text_summary}\n")
                    # --- End Textual Summary ---

                    report_parts.append(f"\n## {section_counter}. Data Overview Details") # Renamed second section
                    section_counter += 1
                    report_parts.append(f"* **Shape:** {report_shape[0]} rows, {report_shape[1]} columns")
                    try:
                        mem_usage_total_mb = df_report.memory_usage(deep=True).sum() / (1024**2)
                        report_parts.append(f"* **Total Memory Usage:** {mem_usage_total_mb:.3f} MB")
                    except Exception: report_parts.append("* **Total Memory Usage:** Could not calculate.")
                    report_parts.append(f"* **Column Names:**\n    ```\n    {', '.join(df_report.columns.tolist())}\n    ```")

                    # Section: Data Types & Non-Null Counts
                    report_parts.append(f"\n## {section_counter}. Data Types & Non-Null Counts")
                    section_counter += 1
                    buffer_info = io.StringIO(); df_report.info(buf=buffer_info, verbose=True, show_counts=True)
                    report_parts.append("\n```\n" + buffer_info.getvalue() + "\n```")

                    # Section: Memory Usage Per Column
                    report_parts.append(f"\n## {section_counter}. Memory Usage per Column")
                    section_counter += 1
                    try:
                        mem_usage_series = df_report.memory_usage(deep=True) / (1024**2) # In MB
                        report_parts.append("\n```\n" + mem_usage_series.round(4).to_string() + "\n```")
                    except Exception as mem_err:
                        report_parts.append(f"\n*Could not calculate memory usage per column: {mem_err}*")

                    # Section: Missing Values
                    report_parts.append(f"\n## {section_counter}. Missing Values Analysis")
                    section_counter += 1
                    # Use pre-calculated values
                    report_parts.append(f"* **Total Missing Entries:** {total_missing_report}")
                    if missing_summary_report is not None and not missing_summary_report.empty:
                        report_parts.append("* **Columns with Missing Values:**\n")
                        report_parts.append(missing_summary_report.to_markdown(index=True))
                    elif missing_summary_report is not None: report_parts.append("* ✅ No missing values found.")
                    else: report_parts.append("* Error retrieving missing value summary.")

                    # Section: Duplicate Rows
                    report_parts.append(f"\n## {section_counter}. Duplicate Row Analysis")
                    section_counter += 1
                    # Use pre-calculated values
                    report_parts.append(f"* **Number of Complete Duplicate Rows Found:** {num_duplicates_report} ({perc_duplicates_report:.2f}%)")

                    # Section: Unique Value Counts
                    report_parts.append(f"\n## {section_counter}. Unique Value Counts per Column")
                    section_counter += 1
                    try:
                        unique_counts = df_report.nunique(dropna=False).reset_index() # Include NaN in count
                        unique_counts.columns = ['Column', 'UniqueCount']
                        report_parts.append(unique_counts.to_markdown(index=False))
                    except Exception as unique_err:
                        report_parts.append(f"*Could not calculate unique counts: {unique_err}*")

                    # Section: Descriptive Statistics
                    report_parts.append(f"\n## {section_counter}. Descriptive Statistics")
                    section_counter += 1
                    numeric_desc_report = data_explorer.get_descriptive_stats(df_report, include_type=np.number)
                    cat_desc_report = data_explorer.get_descriptive_stats(df_report, include_type=['object', 'category', 'bool'])

                    if numeric_desc_report is not None and not numeric_desc_report.empty:
                        report_parts.append("\n### Numeric Columns\n```\n" + numeric_desc_report.round(3).to_string() + "\n```")
                        # Add Skewness and Kurtosis
                        try:
                            skewness = df_report[report_numeric_cols].skew().reset_index()
                            skewness.columns = ['Column', 'Skewness']
                            report_parts.append("\n#### Skewness\n" + skewness.round(3).to_markdown(index=False))
                        except Exception as skew_err: report_parts.append(f"\n*Could not calculate skewness: {skew_err}*")
                        try:
                            kurt = df_report[report_numeric_cols].kurtosis().reset_index()
                            kurt.columns = ['Column', 'Kurtosis']
                            report_parts.append("\n#### Kurtosis\n" + kurt.round(3).to_markdown(index=False))
                        except Exception as kurt_err: report_parts.append(f"\n*Could not calculate kurtosis: {kurt_err}*")
                    else: report_parts.append("\n* No numeric columns found or error fetching stats.")

                    if cat_desc_report is not None and not cat_desc_report.empty:
                        report_parts.append("\n### Categorical / Object / Boolean Columns\n```\n" + cat_desc_report.to_string() + "\n```")
                    else: report_parts.append("\n* No categorical/object/boolean columns found or error fetching stats.")

                    # Section: Outlier Summary
                    report_parts.append(f"\n## {section_counter}. Outlier Summary (IQR Method)")
                    section_counter += 1
                    if not report_numeric_cols: report_parts.append("* No numeric columns available for outlier detection.")
                    else:
                        outlier_summary_report = data_explorer.get_outlier_summary(df_report[report_numeric_cols], method='iqr')
                        if outlier_summary_report is None: report_parts.append("* Error retrieving outlier summary.")
                        elif outlier_summary_report.empty: report_parts.append("* ✅ No potential outliers detected based on IQR method.")
                        else:
                            report_parts.append("* ⚠️ Potential outliers detected in the following columns:\n")
                            report_parts.append(outlier_summary_report.to_markdown(index=False))

                    # Section: Correlation Matrix
                    report_parts.append(f"\n## {section_counter}. Correlation Matrix Analysis")
                    section_counter += 1
                    # Use pre-calculated matrix
                    report_corr_method_used_in_report = "Not Available" # Default
                    if report_corr_matrix is not None:
                         # Determine which method was used based on where matrix came from
                         if st.session_state.corr_matrix is not None:
                              report_corr_method_used_in_report = st.session_state.corr_method_used
                              report_parts.append(f"\n*(Using previously calculated **{report_corr_method_used_in_report}** correlation)*\n")
                         else:
                              report_corr_method_used_in_report = "Pearson" # Assumed default calculation
                              report_parts.append(f"\n*(Using **{report_corr_method_used_in_report}** correlation calculated for report)*\n")

                         report_parts.append(f"**Method Used:** {report_corr_method_used_in_report}\n")
                         report_parts.append("```\n" + report_corr_matrix.round(3).to_string() + "\n```")
                         # Use pre-calculated high_corr_found flag
                         if high_corr_found:
                             report_parts.append(f"\n* **Note ({report_corr_method_used_in_report}):** At least one pair with high absolute correlation (>= 0.8) was found.")
                             # Optionally add back the detailed list if needed, but summary just notes the presence
                    elif len(report_numeric_cols) < 2:
                        report_parts.append("\n* Not enough numeric columns (need 2+) to calculate correlation.")
                    else:
                        report_parts.append("\n* Correlation matrix could not be calculated or retrieved.")


                    # Section: Value Counts Summary
                    report_parts.append(f"\n## {section_counter}. Value Counts Summary (Categorical / Low Cardinality)")
                    section_counter += 1
                    # Use pre-calculated cat cols
                    numeric_low_cardinality = [c for c in report_numeric_cols if df_report[c].nunique(dropna=False) <= MAX_UNIQUE_FOR_VALUE_COUNTS]
                    cols_for_value_counts = sorted(list(set(report_cat_cols + numeric_low_cardinality)))

                    if not cols_for_value_counts:
                        report_parts.append("\n*No suitable columns found for value counts summary.*")
                    else:
                        report_parts.append(f"\n*Showing full counts for columns with up to {MAX_UNIQUE_FOR_VALUE_COUNTS} unique values, or top {TOP_N_VALUE_COUNTS} for others.*")
                        for col in cols_for_value_counts:
                            try:
                                report_parts.append(f"\n### Value Counts: `{col}`")
                                n_unique = df_report[col].nunique(dropna=False)
                                value_counts_series = df_report[col].value_counts(dropna=False)
                                if n_unique <= MAX_UNIQUE_FOR_VALUE_COUNTS:
                                    counts_df = value_counts_series.reset_index()
                                    counts_df.columns = ['Value', 'Count']
                                    report_parts.append("\n" + counts_df.to_markdown(index=False))
                                else:
                                    counts_df = value_counts_series.head(TOP_N_VALUE_COUNTS).reset_index()
                                    counts_df.columns = ['Value', 'Count']
                                    report_parts.append(f"\n*(Showing Top {TOP_N_VALUE_COUNTS} of {n_unique} unique values)*")
                                    report_parts.append("\n" + counts_df.to_markdown(index=False))
                            except Exception as vc_err:
                                report_parts.append(f"\n*Could not calculate value counts for {col}: {vc_err}*")


                    # Section: Data Samples
                    report_parts.append(f"\n## {section_counter}. Data Samples")
                    section_counter += 1
                    try:
                        report_parts.append(f"\n### Head ({SAMPLE_ROWS_IN_REPORT} Rows)\n```\n{df_report.head(SAMPLE_ROWS_IN_REPORT).to_string()}\n```")
                    except Exception as head_err: report_parts.append(f"\n*Could not get head sample: {head_err}*")
                    try:
                        report_parts.append(f"\n### Tail ({SAMPLE_ROWS_IN_REPORT} Rows)\n```\n{df_report.tail(SAMPLE_ROWS_IN_REPORT).to_string()}\n```")
                    except Exception as tail_err: report_parts.append(f"\n*Could not get tail sample: {tail_err}*")
                    try:
                        n_sample = min(SAMPLE_ROWS_IN_REPORT, len(df_report))
                        report_parts.append(f"\n### Random Sample ({n_sample} Rows)\n```\n{df_report.sample(n=n_sample, random_state=42).to_string()}\n```")
                    except Exception as sample_err: report_parts.append(f"\n*Could not get random sample: {sample_err}*")


                    # Section: Advanced Analysis Summary
                    report_parts.append(f"\n## {section_counter}. Advanced Analysis Summary")
                    section_counter += 1
                    report_parts.append("\n### K-Means Clustering")
                    if 'Cluster' in df_report.columns:
                        report_parts.append(f"* K-Means results ('Cluster' column) present in the current data state ({df_report['Cluster'].nunique()} clusters found).")
                        try:
                            cluster_size_report_str = df_report['Cluster'].value_counts().reset_index().rename(columns={'index':'Cluster', 'Cluster':'Size'}).to_markdown(index=False)
                            report_parts.append("* Cluster sizes:\n" + cluster_size_report_str)
                        except Exception as cluster_disp_err:
                            report_parts.append(f"* Could not display cluster sizes: {cluster_disp_err}")
                    else: report_parts.append("* K-Means 'Cluster' column not found in the current data state.")

                    report_parts.append("\n### Principal Component Analysis (PCA)")
                    pca_cols_present_report = [col for col in df_report.columns if col.startswith('PC')]
                    if pca_cols_present_report: report_parts.append(f"* PCA results ({len(pca_cols_present_report)} components: {', '.join(pca_cols_present_report)}) appear to be present in the current data state.")
                    else: report_parts.append("* PCA component columns ('PC1', 'PC2', etc.) not found in the current data state.")
                    report_parts.append("\n*Note: Detailed results for Hypothesis Tests (T-test, ANOVA, Chi-Squared) are shown interactively in the 'Advanced Analysis' tab but are not included in this summary report.*")

                    # --- End of Report ---
                    report_parts.append("\n\n---\n*End of Automated Report*\n---")
                    st.session_state.report_content = "\n".join(report_parts)
                    st.success("✅ Report generated successfully!")

                except AttributeError as attr_err:
                    st.error(f"Report generation failed: Helper function not found ({attr_err}). Ensure all helper modules are up-to-date.", icon="🚨")
                    st.session_state.report_content = None
                    st.session_state.gen_report_btn_clicked = False # Reset flag on error
                except Exception as e:
                    st.error(f"❌ An error occurred during report generation: {e}", icon="🚨")
                    st.session_state.report_content = None # Clear on error
                    st.session_state.gen_report_btn_clicked = False # Reset flag on error


        # --- Display Report Preview and Download Button ---
        # (Display logic remains the same)
        if st.session_state.report_content:
            st.markdown("---"); st.subheader("Generated Report Preview")
            report_container = st.container()
            with report_container: # Use markdown with CSS for scrollable div
                 # Using black background and white text as per previous request
                 st.markdown(f'<div style="height: 500px; overflow-y: scroll; border: 1px solid #555; padding: 15px; border-radius: 5px; background-color: #000000; color: #ffffff;">'
                            f'{st.session_state.report_content}'
                            f'</div>', unsafe_allow_html=True)

            st.markdown("---")
            report_base_name, _ = os.path.splitext(st.session_state.df_name)
            report_filename = f"report_{report_base_name}_{datetime.now().strftime('%Y%m%d')}.md"
            try:
                 report_bytes = st.session_state.report_content.encode('utf-8')
                 st.download_button(
                    label="⬇️ Download Full Report (Markdown)", data=report_bytes,
                    file_name=report_filename, mime="text/markdown", key="download_report_md_btn" )
            except Exception as download_err:
                 st.error(f"Error preparing report for download: {download_err}")

        elif st.session_state.gen_report_btn_clicked and not st.session_state.report_content:
            st.warning("⚠️ Report could not be generated. Check error messages above or data logs.")

        st.session_state.gen_report_btn_clicked = False

    # ... (rest of your app code, e.g., Footer) ...

    # --- Footer ---
    st.markdown("---")
    st.caption(f"Interactive Data Analyst Ali | {get_current_timestamp()}")