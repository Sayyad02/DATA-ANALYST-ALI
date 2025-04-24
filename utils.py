# utils.py
import pandas as pd
import os

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def save_dataframe(df, file_path):
    """
    Saves the DataFrame to a specified file (CSV or Excel).
    The file will be saved inside the 'output' directory.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        file_path (str): The desired filename (e.g., 'cleaned_data.csv').
    """
    if df is None:
        print("Error: No DataFrame to save.")
        return

    base_filename = os.path.basename(file_path)
    full_path = os.path.join(output_dir, base_filename)
    file_extension = os.path.splitext(base_filename)[1].lower()

    try:
        if file_extension == '.csv':
            df.to_csv(full_path, index=False)
            print(f"DataFrame successfully saved to: {full_path}")
        elif file_extension in ['.xls', '.xlsx']:
            # Requires openpyxl for .xlsx
            df.to_excel(full_path, index=False)
            print(f"DataFrame successfully saved to: {full_path}")
        else:
            print(f"Error: Unsupported file format '{file_extension}'. Please use .csv or .xlsx.")
    except Exception as e:
        print(f"Error saving DataFrame to '{full_path}': {e}")


def display_help():
    """Prints the help message listing available commands."""
    print("\n--- Data Analyst Assistant Commands ---")
    print("Core:")
    print("  load <file_path>       : Load data from CSV or Excel (e.g., load data/my_data.csv)")
    print("  save <filename.csv|xlsx> : Save the current DataFrame to the 'output' folder.")
    print("  help                   : Show this help message.")
    print("  exit                   : Exit the assistant.")
    print("\nInspect Data:")
    print("  info                   : Show basic info (shape, head, data types, non-nulls).")
    print("  describe               : Show descriptive statistics for all columns.")
    print("  dtypes                 : Show column data types.")
    print("  missing                : Show count and percentage of missing values per column.")
    print("  duplicates             : Show the number of duplicate rows.")
    print("  counts <column> [top_n]: Show value counts for a column (default top 20).")
    print("\nClean Data:")
    print("  clean_missing          : Interactively handle missing values (prompts for strategy/columns).")
    # print("  clean_missing <strategy> [columns...] : (Alternative syntax - less interactive)")
    print("  drop_duplicates        : Remove duplicate rows (keeps first).")
    print("  change_dtype <col> <type>: Change data type (int, float, str, datetime, bool).")
    print("\nAnalyze & Visualize:")
    print("  corr [method]          : Calculate and show correlation matrix (methods: pearson, kendall, spearman).")
    print("  plot hist <col> [savename.png] : Plot histogram for a numeric column.")
    print("  plot box <col> [by <group_col>] [savename.png] : Plot boxplot (optionally grouped).")
    print("  plot scatter <x_col> <y_col> [hue <hue_col>] [savename.png]: Plot scatter plot.")
    print("  plot bar <x_col> [y <y_col>] [est <estimator>] [savename.png]: Plot barplot (counts if no y_col, else aggregate y_col).")
    print("                                     Estimators: mean (default), sum, median.")
    print("  plot heatmap [savename.png] : Plot heatmap of the calculated correlation matrix.")
    print("-------------------------------------\n")

# Example of how command parsing could be structured if it gets complex,
# but for now, it's handled directly in main.py's if/elif block.
# def parse_command(user_input):
#     parts = user_input.strip().split()
#     if not parts:
#         return None, None
#     command = parts[0].lower()
#     args = parts[1:]
#     return command, args