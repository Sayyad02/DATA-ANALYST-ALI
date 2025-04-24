# data_analyzer.py
import pandas as pd
import numpy as np
import streamlit as st # For feedback only
from typing import Optional, List, Tuple, Dict, Any

# Import necessary libraries for analysis (ensure installed: pip install scipy scikit-learn)
try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score # For K-Means evaluation
except ImportError as e:
    # This error is critical for this module's functionality
    st.error(f"Required libraries (scipy, scikit-learn) not found. Please install them: `pip install scipy scikit-learn`. Cannot perform advanced analysis. Error: {e}", icon="🚨")
    stats = None
    StandardScaler = None
    KMeans = None
    PCA = None
    silhouette_score = None

def perform_normality_test(series: pd.Series, method: str = 'shapiro') -> Optional[Tuple[float, float, str]]:
    """
    Performs a normality test (Shapiro-Wilk or KS) on a numeric series.

    Args:
        series (pd.Series): The numeric data series (should have NaNs dropped by caller).
        method (str): 'shapiro' (default) or 'kstest'.

    Returns:
        Optional[Tuple[float, float, str]]: (statistic, p_value, interpretation) or None on error/invalid input.
    """
    if stats is None: print("Scipy.stats not available."); return None # Check if import failed
    if series is None or not pd.api.types.is_numeric_dtype(series) or series.nunique() < 2:
        print("Warning: Series is not numeric or has < 2 unique values (perform_normality_test).")
        return None
    if series.isnull().any():
        print("Error: Series contains NaNs. Must be handled before normality test.")
        st.error("Series contains NaNs. Must be handled before normality test.", icon="❌") # Feedback to user
        return None
    if len(series) < 3 : # Shapiro needs at least 3 samples, KS test might work with fewer but less reliably
         print(f"Warning: Not enough samples ({len(series)}) for normality test '{method}'. Minimum 3 required.")
         st.warning(f"Not enough samples ({len(series)}) for normality test '{method}'. Minimum 3 required.", icon="⚠️")
         return None

    alpha = 0.05 # Significance level
    try:
        stat, p_value = -1.0, -1.0 # Initialize
        test_name = "Unknown"
        if method == 'shapiro':
            if len(series) > 5000:
                 print("Warning: Shapiro-Wilk test performs best on samples < 5000. Consider KS test.")
            stat, p_value = stats.shapiro(series)
            test_name = "Shapiro-Wilk"
        elif method == 'kstest':
            stat, p_value = stats.kstest(stats.zscore(series), 'norm') # Test standardized data against N(0,1)
            test_name = "Kolmogorov-Smirnov (vs Normal)"
        else:
            print(f"Error: Unknown normality test method '{method}'.")
            st.error(f"Unknown normality test method '{method}'.") # Feedback to user
            return None

        # Interpretation based on p-value and alpha
        if p_value > alpha:
            interpretation = f"Sample looks Gaussian (fail to reject H0 at alpha={alpha})"
        else:
            interpretation = f"Sample does not look Gaussian (reject H0 at alpha={alpha})"

        return stat, p_value, f"{test_name}: {interpretation}"

    except Exception as e:
        print(f"Error during {method} normality test: {e}")
        st.error(f"Error during {method} normality test: {e}", icon="❌") # Feedback to user
        return None


def perform_ttest_ind(df: pd.DataFrame, group_col: str, value_col: str, equal_var: bool = True) -> Optional[Tuple[float, float, str]]:
    """
    Performs an Independent Samples t-test to compare means of two groups.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        group_col (str): Categorical column defining the two groups.
        value_col (str): Numeric column whose means are compared.
        equal_var (bool): Assume equal variances (True, Student's) or not (False, Welch's).

    Returns:
        Optional[Tuple[float, float, str]]: (statistic, p_value, interpretation) or None on error.
                                            Shows st.error/warning for feedback.
    """
    if stats is None: st.error("Scipy.stats not available."); return None
    # Input validation
    if df is None or group_col not in df.columns or value_col not in df.columns:
         st.error("Invalid DataFrame or columns for t-test.", icon="❌"); return None
    if not pd.api.types.is_numeric_dtype(df[value_col]):
         st.error(f"Value column '{value_col}' must be numeric for t-test.", icon="❌"); return None
    unique_groups = df[group_col].dropna().unique()
    if len(unique_groups) != 2:
         st.error(f"Grouping column '{group_col}' must have exactly two unique non-missing groups for t-test. Found {len(unique_groups)}.", icon="❌"); return None

    try:
        group1_data = df[df[group_col] == unique_groups[0]][value_col].dropna()
        group2_data = df[df[group_col] == unique_groups[1]][value_col].dropna()
        if len(group1_data) < 2 or len(group2_data) < 2:
             st.error("Each group must have at least 2 non-missing values for t-test.", icon="❌"); return None

        stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var, nan_policy='raise')

        alpha = 0.05
        if p_value > alpha:
            interpretation = f"No significant difference in means of '{value_col}' found between groups '{unique_groups[0]}' and '{unique_groups[1]}' (fail to reject H0 at alpha={alpha})"
        else:
            interpretation = f"Significant difference in means of '{value_col}' found between groups '{unique_groups[0]}' and '{unique_groups[1]}' (reject H0 at alpha={alpha})"

        test_type = "Student's t-test (Equal Variances Assumed)" if equal_var else "Welch's t-test (Unequal Variances)"
        return stat, p_value, f"{test_type}: {interpretation}"

    except Exception as e:
        st.error(f"Error during Independent t-test: {e}", icon="❌")
        return None


def perform_anova(df: pd.DataFrame, group_col: str, value_col: str) -> Optional[Tuple[float, float, str]]:
    """
    Performs a One-Way ANOVA test to compare means across multiple groups.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        group_col (str): Categorical column defining the groups (2 or more expected).
        value_col (str): Numeric column whose means are compared across groups.

    Returns:
        Optional[Tuple[float, float, str]]: (F-statistic, p_value, interpretation) or None on error.
                                            Shows st.error/warning for feedback.
    """
    if stats is None: st.error("Scipy.stats not available."); return None
    # Input validation
    if df is None or group_col not in df.columns or value_col not in df.columns:
         st.error("Invalid DataFrame or columns for ANOVA.", icon="❌"); return None
    if not pd.api.types.is_numeric_dtype(df[value_col]):
         st.error(f"Value column '{value_col}' must be numeric for ANOVA.", icon="❌"); return None
    unique_groups = df[group_col].dropna().unique()
    num_groups = len(unique_groups)
    if num_groups < 2:
         st.error(f"Grouping column '{group_col}' must have at least two unique non-missing groups for ANOVA. Found {num_groups}.", icon="❌"); return None
    if num_groups == 2: st.warning("Only 2 groups found. ANOVA is valid, but T-test is typically used.", icon="⚠️")

    try:
        group_data_list = [df[df[group_col] == group][value_col].dropna() for group in unique_groups]
        if any(len(data) < 2 for data in group_data_list):
             st.error("Each group must have at least 2 non-missing values in the value column for ANOVA.", icon="❌"); return None

        f_stat, p_value = stats.f_oneway(*group_data_list)

        alpha = 0.05
        if p_value > alpha:
            interpretation = f"No significant difference in means of '{value_col}' found across the groups in '{group_col}' (fail to reject H0 at alpha={alpha})"
        else:
            interpretation = f"Significant difference in means of '{value_col}' found across at least some groups in '{group_col}' (reject H0 at alpha={alpha}). Post-hoc tests needed to identify which specific groups differ."

        return f_stat, p_value, f"One-Way ANOVA: {interpretation}"

    except Exception as e:
        st.error(f"Error during ANOVA: {e}", icon="❌")
        return None


def perform_chi_squared(df: pd.DataFrame, col1: str, col2: str) -> Optional[Tuple[float, float, float, str]]:
    """
    Performs a Chi-squared test of independence between two categorical columns.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        col1 (str): Name of the first categorical column.
        col2 (str): Name of the second categorical column.

    Returns:
        Optional[Tuple[float, float, float, str]]: (Chi2 statistic, p_value, degrees_of_freedom, interpretation) or None on error.
                                                   Shows st.error/warning for feedback.
    """
    if stats is None: st.error("Scipy.stats not available."); return None
    # Input validation
    if df is None or col1 not in df.columns or col2 not in df.columns:
         st.error("Invalid DataFrame or columns for Chi-squared test.", icon="❌"); return None
    if col1 == col2:
         st.error("Cannot perform Chi-squared test with the same column for both variables.", icon="❌"); return None
    if df[col1].nunique() < 2 or df[col2].nunique() < 2:
         st.error(f"Both columns ('{col1}', '{col2}') must have at least 2 unique levels for Chi-squared test.", icon="❌"); return None

    try:
        contingency_table = pd.crosstab(df[col1], df[col2])
        if contingency_table.sum().sum() == 0:
             st.error("Contingency table is empty. Cannot perform test.", icon="❌"); return None

        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        low_expected_freq_count = (expected < 5).sum()
        if low_expected_freq_count > 0:
             percent_low = (low_expected_freq_count / expected.size) * 100
             st.warning(f"Chi-squared assumption violation: {low_expected_freq_count} cell(s) ({percent_low:.1f}%) have expected frequencies < 5. Results may be less reliable.", icon="⚠️")

        alpha = 0.05
        if p > alpha:
            interpretation = f"No significant association found between '{col1}' and '{col2}' (variables appear independent, fail to reject H0 at alpha={alpha})"
        else:
            interpretation = f"Significant association found between '{col1}' and '{col2}' (variables appear dependent, reject H0 - independence at alpha={alpha})"

        return chi2, p, dof, f"Chi-Squared Test: {interpretation}"

    except ValueError as ve:
         st.error(f"Could not perform Chi-squared test. Check data variance and levels. Error: {ve}", icon="❌"); return None
    except Exception as e:
        st.error(f"Error during Chi-squared test: {e}", icon="❌")
        return None


def perform_kmeans(df: pd.DataFrame, feature_cols: List[str], n_clusters: int) -> Optional[Tuple[pd.DataFrame, np.ndarray, float, float]]:
    """
    Performs K-Means clustering on selected numeric features after scaling.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        feature_cols (List[str]): List of numeric columns to use for clustering.
        n_clusters (int): The number of clusters (k).

    Returns:
        Optional[Tuple[pd.DataFrame, np.ndarray, float, float]]:
            - DataFrame with added 'Cluster' column (assigned based on index).
            - Cluster centers (in the original scaled space).
            - Inertia (within-cluster sum of squares).
            - Silhouette score (-1 to 1, higher is better).
            Returns None on error. Shows st.error/warning for feedback.
    """
    if KMeans is None or StandardScaler is None or silhouette_score is None:
         st.error("Scikit-learn library not available (KMeans/StandardScaler/silhouette_score).", icon="🚨"); return None
    # Input validation
    if df is None or not feature_cols:
         st.error("DataFrame is None or no feature columns selected for K-Means.", icon="❌"); return None
    if n_clusters < 2:
         st.error("Number of clusters (k) must be at least 2.", icon="❌"); return None

    invalid_cols = [col for col in feature_cols if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col])]
    if invalid_cols:
         st.error(f"Invalid or non-numeric feature columns selected for K-Means: {invalid_cols}", icon="❌"); return None

    data_to_cluster = df[feature_cols].dropna()
    if data_to_cluster.empty:
         st.error("No data remaining after dropping rows with missing values in selected features.", icon="❌"); return None
    if len(data_to_cluster) < n_clusters:
         st.error(f"Not enough non-missing data points ({len(data_to_cluster)}) to form {n_clusters} clusters.", icon="❌"); return None

    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_cluster)
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(scaled_data)

        cluster_labels = kmeans.labels_
        cluster_centers_scaled = kmeans.cluster_centers_
        inertia = kmeans.inertia_

        score = -1.0 # Default score if calculation fails
        try:
             if n_clusters <= len(data_to_cluster) and len(np.unique(cluster_labels)) > 1: # Need > 1 unique label
                  score = silhouette_score(scaled_data, cluster_labels)
             elif len(np.unique(cluster_labels)) <= 1:
                  st.warning("Silhouette score requires at least 2 distinct clusters to be formed.", icon="⚠️")
        except ValueError as sil_err:
             st.warning(f"Could not compute Silhouette Score: {sil_err}", icon="⚠️")

        df_clustered = df.copy()
        df_clustered['Cluster'] = pd.Series(dtype='Int64') # Initialize nullable integer
        df_clustered.loc[data_to_cluster.index, 'Cluster'] = cluster_labels
        df_clustered['Cluster'] = df_clustered['Cluster'].astype('category')

        st.info(f"K-Means clustering completed. Added 'Cluster' column.")
        return df_clustered, cluster_centers_scaled, inertia, score

    except Exception as e:
        st.error(f"Error during K-Means clustering: {e}", icon="❌")
        return None


def perform_pca(df: pd.DataFrame, feature_cols: List[str], n_components: Optional[int] = None) -> Optional[Tuple[PCA, pd.DataFrame]]:
    """
    Performs Principal Component Analysis (PCA) on selected numeric features after scaling.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        feature_cols (List[str]): List of numeric columns to use for PCA.
        n_components (Optional[int]): Number of principal components to compute.
                                       If None, computes all possible components. Defaults to None.

    Returns:
        Optional[Tuple[PCA, pd.DataFrame]]:
            - Fitted PCA object from scikit-learn (contains explained variance, components, etc.).
            - DataFrame containing the principal components (PC1, PC2, ...), indexed like original non-NaN rows.
            Returns None on error. Shows st.error/warning for feedback.
    """
    if PCA is None or StandardScaler is None:
         st.error("Scikit-learn library not available (PCA/StandardScaler).", icon="🚨"); return None
    # Input validation
    if df is None or not feature_cols:
         st.error("DataFrame is None or no feature columns selected for PCA.", icon="❌"); return None

    invalid_cols = [col for col in feature_cols if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col])]
    if invalid_cols:
         st.error(f"Invalid or non-numeric feature columns selected for PCA: {invalid_cols}", icon="❌"); return None

    data_to_reduce = df[feature_cols].dropna()
    original_index = data_to_reduce.index

    max_possible_components = min(len(feature_cols), len(data_to_reduce))
    if max_possible_components < 1:
         st.error("Not enough data points or features for PCA after dropping NaNs.", icon="❌"); return None

    if n_components is not None:
        if n_components < 1: st.error("Number of PCA components must be >= 1.", icon="❌"); return None
        if n_components > max_possible_components:
             st.warning(f"Requested components ({n_components}) > max possible ({max_possible_components}). Setting n_components={max_possible_components}.", icon="⚠️")
             n_components = max_possible_components
    else:
         # If None, let PCA decide (min(n_samples, n_features))
         n_components = max_possible_components # Be explicit for internal logic

    if data_to_reduce.empty or len(data_to_reduce) < n_components:
         st.error(f"Not enough non-missing data points ({len(data_to_reduce)}) for {n_components} PCA components.", icon="❌"); return None

    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_reduce)
        pca = PCA(n_components=n_components, random_state=42)
        principal_components = pca.fit_transform(scaled_data)

        actual_n_components = pca.n_components_ # Get the actual number computed
        pc_col_names = [f'PC{i+1}' for i in range(actual_n_components)]
        pca_df = pd.DataFrame(data=principal_components, columns=pc_col_names, index=original_index)

        st.success(f"PCA completed, computed {actual_n_components} principal components.")
        return pca, pca_df

    except Exception as e:
        st.error(f"Error during PCA: {e}", icon="❌")
        return None