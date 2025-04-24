# Interactive Data Analyst Ali 📊

An interactive web application built with Streamlit for easy data loading, cleaning, exploration, visualization, analysis, and report generation.

## Description

"Data Analyst Ali" provides a user-friendly interface to perform common data analysis tasks without writing code directly. Upload your CSV or Excel file, and interactively explore, clean, visualize, and analyze your data through various tabs and options. The application also features undo/redo functionality for data modification steps and allows downloading the processed data and generated analysis reports.

## Features

* **⬆️ Data Loading:** Upload data from CSV or Excel files (`.csv`, `.xlsx`, `.xls`).
* **ℹ️ View Data Info:**
    * Display DataFrame head, tail, and random samples.
    * Show data types, non-null counts, and memory usage (`df.info()`).
    * Check for and report the number of duplicate rows.
* **探索 Explore Data:**
    * Calculate and display descriptive statistics for numeric and categorical columns.
    * Summarize missing values (count and percentage).
    * Identify potential outliers using the IQR method.
    * Show value counts for selected columns.
* **✨ Clean Data:**
    * Handle missing values (drop rows, fill with mean, median, mode, or a specific value).
    * Remove duplicate rows.
    * Change column data types (numeric, string, datetime, boolean, category).
    * Drop selected columns.
* **⏪ Undo/Redo:** Revert or reapply data cleaning and analysis steps (like clustering or PCA component addition).
* **📈 Analyze & Visualize:**
    * **Correlation:** Calculate Pearson, Kendall, or Spearman correlation matrices and display a heatmap.
    * **Standard Plots:** Generate various plots using Matplotlib and Seaborn:
        * Histogram (with optional KDE)
        * Box Plot (with optional grouping)
        * Scatter Plot (with optional hue)
        * Bar Plot (counts or aggregated values)
        * Pair Plot (for numeric column relationships, with optional hue)
        * Violin Plot (distribution comparison, with optional grouping)
        * Joint Plot (bivariate relationship with marginal distributions)
        * Q-Q Plot (normality check)
        * Word Cloud (for text columns)
    * **Advanced Analysis:**
        * Normality Test (Shapiro-Wilk)
        * Independent Samples T-test
        * One-Way ANOVA
        * Chi-Squared Test of Independence
        * K-Means Clustering (adds 'Cluster' column)
        * Principal Component Analysis (PCA) (optionally add components to data)
* **📝 Generate Detailed Report:**
    * Create a comprehensive markdown report summarizing the current state of the data.
    * Includes a textual summary, overview, data types, missing values, duplicates, unique counts, descriptive stats (with skew/kurtosis), outliers, correlation, value counts, data samples, and advanced analysis presence.
* **⬇️ Download:**
    * Download the *current state* of the processed DataFrame as an Excel file.
    * Download the generated detailed analysis report as a Markdown file.

## Screenshots

*(Optional: Add screenshots of the application here to showcase the UI)*

*Example:*
![Screenshot of Data Analyst Ali Report Preview](Screenshot%202025-04-24%20031623.png)
*(Replace the above with actual paths/links to your screenshots)*

## Technology Stack

* **Framework:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, WordCloud
* **Statistical Analysis:** SciPy, Scikit-learn
* **File Handling:** openpyxl (for Excel)
* **Utilities:** pytz (for timezones)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    Create a file named `requirements.txt` in the project root directory with the following content:
    ```txt
    streamlit
    pandas
    numpy
    seaborn
    matplotlib
    scipy
    scikit-learn
    openpyxl
    pytz
    wordcloud
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
2.  The application will open in your web browser.
3.  Use the sidebar to **upload** your CSV or Excel data file.
4.  Once the data is loaded, select an **Action** from the sidebar (View Data Info, Explore Data, Clean Data, Analyze & Visualize, Generate Detailed Report).
5.  Interact with the options presented in the main panel based on the selected action.
6.  Use the **Undo** and **Redo** buttons in the sidebar to navigate through the history of data modifications.
7.  Use the **Download** buttons in the sidebar to save your processed data or the generated report.

## File Structure

.
├── app.py             # Main Streamlit application script
├── data_loader.py     # Helper module for loading data
├── data_cleaner.py    # Helper module for data cleaning functions
├── data_explorer.py   # Helper module for data exploration functions
├── data_visualizer.py # Helper module for generating plots
├── data_analyzer.py   # Helper module for statistical tests and advanced analysis
├── requirements.txt   # List of Python dependencies
├── README.md          # This file



## License


This project is licensed under the MIT License.

## Author / Contact

Application developed as "Data Analyst Ali".
         *("SHAFIN ALI SYED")*