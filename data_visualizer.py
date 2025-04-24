# data_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st # Import Streamlit only for st.error/warning/info feedback
from typing import Optional, List, Tuple, Any # Ensure typing imports are here

# Import specific components needed for type hints or functions
from sklearn.decomposition import PCA # For type hint in plot_pca_results

# Import optional libraries with error handling
try:
    from scipy import stats # Needed for Q-Q plot
except ImportError:
    st.warning("Library 'scipy' not found. Some advanced plots (e.g., Q-Q plot) may be disabled. Install with `pip install scipy`.", icon="⚠️")
    stats = None # Set to None so checks later will fail gracefully
try:
    from wordcloud import WordCloud # Needed for Word Cloud
except ImportError:
     st.warning("Library 'wordcloud' not found. Word Cloud visualization disabled. Install with `pip install wordcloud`.", icon="⚠️")
     WordCloud = None # Set to None so checks later will fail gracefully

# --- Plotting Configuration ---
# Set a default plotting style for seaborn
sns.set_style("whitegrid")
# Set default DPI for figures for better resolution in Streamlit
plt.rcParams['figure.dpi'] = 100
# Set default figure size (can be overridden)
plt.rcParams['figure.figsize'] = (8, 5)

# --- Helper Function ---
def _create_figure(figsize: Optional[Tuple[float, float]] = None) -> Tuple[Optional[plt.Figure], Optional[plt.Axes]]:
    """Helper to create a matplotlib figure and axes with default or specified size."""
    # Use try-except for robustness in figure creation if needed
    try:
        if figsize is None:
            fig, ax = plt.subplots() # Use default size from rcParams
        else:
            fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    except Exception as fig_err:
        st.error(f"Failed to create plot figure: {fig_err}", icon="🚨")
        return None, None # Return None tuple if creation fails

# --- Plotting Functions ---

def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Plots a heatmap for the given correlation matrix.

    Args:
        corr_matrix (pd.DataFrame): A square DataFrame containing correlation values.

    Returns:
        Optional[plt.Figure]: A matplotlib Figure object, or None on error. Shows st.error on failure.
    """
    if corr_matrix is None or corr_matrix.empty:
        st.error("Correlation matrix is empty or None. Cannot plot heatmap.", icon="⚠️")
        return None
    # Check if it's reasonably square (allow for floating point inaccuracies if needed)
    if not np.isclose(corr_matrix.shape[0], corr_matrix.shape[1]):
         st.error("Input matrix is not square. Cannot plot heatmap.", icon="❌")
         return None
    if corr_matrix.shape[0] < 2:
         st.warning("Need at least a 2x2 matrix for a meaningful heatmap.", icon="⚠️")
         # Optionally plot anyway, or return None
         # return None

    fig, ax = _create_figure(figsize=None) # Use helper, start with default size
    if fig is None: return None # Check if figure creation failed

    try:
        # Adjust figsize based on matrix size for better readability
        num_vars = len(corr_matrix.columns)
        # Heuristic sizing: start at 6x5, increase by ~0.5 inch per variable beyond ~5-6 vars
        width = max(6, 4 + num_vars * 0.4)
        height = max(5, 3 + num_vars * 0.4)
        fig.set_size_inches(width, height) # Adjust size after creation

        # Use seaborn heatmap
        sns.heatmap(corr_matrix,
                    annot=True,          # Show correlation values on the heatmap
                    cmap='coolwarm',     # Color map (diverging blue-red is common for correlation)
                    fmt=".2f",           # Format annotations to 2 decimal places
                    linewidths=.5,       # Add lines between cells
                    ax=ax,               # Plot on the created axes
                    vmin=-1, vmax=1,     # Ensure color scale covers full range -1 to 1
                    annot_kws={"size": 8} # Adjust annotation font size if needed
                   )
        ax.set_title('Correlation Matrix Heatmap', fontsize=14, pad=20) # Add padding to title
        # Improve tick label readability
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout() # Adjust layout automatically
        return fig
    except Exception as e:
        st.error(f"Error plotting correlation heatmap: {e}", icon="❌")
        if fig: plt.close(fig) # Close figure if created before error
        return None


def plot_histogram(df: pd.DataFrame, column: str, kde: bool = True) -> Optional[plt.Figure]:
    """
    Plots a histogram for a specified numeric column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the numeric column to plot.
        kde (bool): Whether to overlay a Kernel Density Estimate curve. Defaults to True.

    Returns:
        Optional[plt.Figure]: A matplotlib Figure object, or None on error. Shows st.error on failure.
    """
    if df is None or column not in df.columns:
        st.error(f"Column '{column}' not found or DataFrame is None.", icon="❌")
        return None
    if not pd.api.types.is_numeric_dtype(df[column]):
        st.error(f"Column '{column}' is not numeric. Cannot plot histogram.", icon="❌")
        return None
    if df[column].dropna().empty:
         st.warning(f"Column '{column}' contains only missing values. Cannot plot histogram.", icon="⚠️")
         return None

    fig, ax = _create_figure()
    if fig is None: return None
    try:
        sns.histplot(data=df, x=column, kde=kde, ax=ax, bins='auto', color='skyblue', edgecolor='black') # Specify color/edge
        ax.set_title(f'Distribution of {column}', fontsize=14)
        ax.set_xlabel(column, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting histogram for '{column}': {e}", icon="❌")
        if fig: plt.close(fig)
        return None


def plot_boxplot(df: pd.DataFrame, y_col: str, group_by: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plots a box plot for a numeric column, optionally grouped by another column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        y_col (str): The numeric column for the Y-axis.
        group_by (Optional[str]): The column to group by for the X-axis.

    Returns:
        Optional[plt.Figure]: A matplotlib Figure object, or None on error. Shows st.error on failure.
    """
    if df is None or y_col not in df.columns:
        st.error(f"Y-axis column '{y_col}' not found or DataFrame is None.", icon="❌")
        return None
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        st.error(f"Y-axis column '{y_col}' must be numeric for box plot.", icon="❌")
        return None
    if df[y_col].dropna().empty:
         st.warning(f"Y-axis column '{y_col}' has no non-missing values. Cannot plot boxplot.", icon="⚠️")
         return None
    if group_by:
         if group_by not in df.columns:
              st.error(f"Grouping column '{group_by}' not found.", icon="❌")
              return None
         if df[group_by].nunique() > 50: # Limit number of boxes for readability
             st.warning(f"Grouping column '{group_by}' has too many unique values (>50). Box plot may be unreadable.", icon="⚠️")
             # Consider alternative plot or subsampling if this happens often

    fig = None
    try:
        # Adjust figsize: wider if grouped
        fig_width = 10 if group_by else 6
        fig, ax = _create_figure(figsize=(fig_width, 5))
        if fig is None: return None

        plot_params = {'data': df, 'ax': ax, 'palette': 'viridis'}
        title = f'Box Plot of {y_col}'
        x_label = None

        if group_by:
            plot_params['y'] = y_col
            plot_params['x'] = group_by
            title += f' grouped by {group_by}'
            x_label = group_by
            n_groups = df[group_by].nunique()
            # Add ordering by median if desired and feasible (can be slow for many groups)
            # median_order = df.groupby(group_by)[y_col].median().sort_values().index
            # plot_params['order'] = median_order[:50] # Limit order calculation too

            sns.boxplot(**plot_params)
            if n_groups > 10: # Rotate labels if many groups
                 ax.tick_params(axis='x', rotation=45, labelsize=9)
                 # plt.xticks(rotation=45, ha='right', fontsize=9) # Alternate way
            else:
                 ax.tick_params(axis='x', labelsize=9)
        else:
            # If not grouped, plot horizontally for potentially better use of space
            plot_params['x'] = y_col # Plot single box plot horizontally
            sns.boxplot(**plot_params)
            x_label = y_col # X axis now represents the value
            ax.set_ylabel(None) # No Y axis label needed


        ax.set_title(title, fontsize=14)
        ax.set_xlabel(x_label, fontsize=10)
        ax.tick_params(axis='y', labelsize=9) # Ensure y tick labels are readable

        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting box plot for '{y_col}': {e}", icon="❌")
        if fig: plt.close(fig)
        return None


def plot_scatterplot(df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plots a scatter plot between two numeric columns, optionally colored by a third column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The numeric column for the X-axis.
        y_col (str): The numeric column for the Y-axis.
        hue_col (Optional[str]): The column to use for coloring points.

    Returns:
        Optional[plt.Figure]: A matplotlib Figure object, or None on error. Shows st.error on failure.
    """
    required_cols = [x_col, y_col]
    if hue_col: required_cols.append(hue_col)

    if df is None: st.error("DataFrame is None.", icon="❌"); return None
    missing_req = [col for col in required_cols if col not in df.columns]
    if missing_req: st.error(f"Required columns not found: {missing_req}", icon="❌"); return None
    if not pd.api.types.is_numeric_dtype(df[x_col]): st.error(f"X-axis column '{x_col}' must be numeric.", icon="❌"); return None
    if not pd.api.types.is_numeric_dtype(df[y_col]): st.error(f"Y-axis column '{y_col}' must be numeric.", icon="❌"); return None

    fig, ax = _create_figure()
    if fig is None: return None
    try:
        valid_hue = hue_col if hue_col and hue_col in df.columns else None
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=valid_hue, ax=ax,
                        palette='viridis', # Color palette for hue
                        s=30,              # Marker size
                        alpha=0.7          # Marker transparency
                       )
        hue_title_part = f" colored by {valid_hue}" if valid_hue else ""
        ax.set_title(f'Scatter Plot of {y_col} vs {x_col}{hue_title_part}', fontsize=14)
        ax.set_xlabel(x_col, fontsize=10)
        ax.set_ylabel(y_col, fontsize=10)

        # Improve legend handling
        if valid_hue:
             num_hues = df[valid_hue].nunique()
             if num_hues > 15: # Hide legend if too many categories
                  ax.legend().set_visible(False)
                  st.caption("Legend hidden due to large number of hue categories (>15).")
             # Can add code here to place legend outside plot if needed

        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting scatter plot ({y_col} vs {x_col}): {e}", icon="❌")
        if fig: plt.close(fig)
        return None


def plot_barplot(df: pd.DataFrame, x_col: str, y_col: Optional[str] = None, estimator: str = 'count') -> Optional[plt.Figure]:
    """
    Plots a bar plot. Shows counts if y_col is None or estimator is 'count'.
    Otherwise, shows aggregated value (mean, sum, median) of y_col for each category in x_col.

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): The column for the X-axis (usually categorical).
        y_col (Optional[str]): The numeric column for the Y-axis (if aggregating).
        estimator (str): Aggregation function ('count', 'mean', 'sum', 'median'). Defaults to 'count'.

    Returns:
        Optional[plt.Figure]: A matplotlib Figure object, or None on error. Shows st.error on failure.
    """
    if df is None or x_col not in df.columns: st.error(f"X-axis column '{x_col}' missing or DataFrame is None.", icon="❌"); return None
    if df[x_col].dropna().empty: st.warning(f"X-axis column '{x_col}' has no non-missing values.", icon="⚠️"); return None

    # Validate y_col and estimator if aggregation is requested
    use_countplot = (y_col is None) or (estimator == 'count')
    valid_estimator = estimator
    if not use_countplot:
        if y_col not in df.columns: st.error(f"Y-axis column '{y_col}' not found.", icon="❌"); return None
        if not pd.api.types.is_numeric_dtype(df[y_col]): st.error(f"Y-axis column '{y_col}' must be numeric for aggregation.", icon="❌"); return None
        if estimator not in ['mean', 'sum', 'median']: st.warning(f"Unsupported estimator '{estimator}'. Defaulting to 'mean'.", icon="⚠️"); valid_estimator = 'mean'

    fig, ax = _create_figure(figsize=(10, 6))
    if fig is None: return None
    try:
        n_groups = df[x_col].nunique()
        max_bars = 30 # Limit number of bars shown for readability
        order = None # Initialize order

        if use_countplot:
            # Get top N categories by count
            order = df[x_col].value_counts().iloc[:max_bars].index
            sns.countplot(data=df, x=x_col, ax=ax, palette='viridis', order=order)
            ax.set_title(f'Count Plot of {x_col}', fontsize=14)
            ax.set_ylabel('Count', fontsize=10)
        else:
             # For aggregated plots, order by the aggregated value
             try:
                  # Calculate aggregation to determine order
                  grouped_data = df.groupby(x_col)[y_col].agg(valid_estimator).sort_values(ascending=False)
                  order = grouped_data.iloc[:max_bars].index
                  sns.barplot(data=df, x=x_col, y=y_col, estimator=valid_estimator, ax=ax, palette='viridis', ci=None, order=order) # ci=None suppresses confidence intervals
                  ax.set_title(f'Bar Plot: {valid_estimator.capitalize()} of {y_col} by {x_col}', fontsize=14)
                  ax.set_ylabel(f'{valid_estimator.capitalize()} of {y_col}', fontsize=10)
             except Exception as agg_err:
                   st.error(f"Could not aggregate '{y_col}' by '{x_col}' using '{valid_estimator}': {agg_err}", icon="❌"); plt.close(fig); return None

        ax.set_xlabel(x_col, fontsize=10)
        if n_groups > max_bars: st.caption(f"Note: Displaying top {max_bars} categories only.")
        # Rotate labels if many categories are actually plotted
        if len(order) > 10: ax.tick_params(axis='x', rotation=45, labelsize=9)
        else: ax.tick_params(axis='x', labelsize=9)

        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting bar plot for '{x_col}': {e}", icon="❌")
        if fig: plt.close(fig)
        return None


def plot_qq(series: pd.Series) -> Optional[plt.Figure]:
    """
    Generates a Q-Q plot for a numeric series against a normal distribution.

    Args:
        series (pd.Series): The numeric data series (NaNs will be dropped).

    Returns:
        Optional[plt.Figure]: A matplotlib Figure object, or None on error. Shows st.error on failure.
    """
    if stats is None: # Check if scipy.stats was imported
        st.error("Scipy library needed for Q-Q plot not available. Please install it (`pip install scipy`).", icon="🚨")
        return None
    if series is None or not pd.api.types.is_numeric_dtype(series):
        st.error("Input series must be numeric for Q-Q plot.", icon="❌")
        return None
    series_dropna = series.dropna() # Drop NA for the plot
    if len(series_dropna) < 3:
         st.warning("Need at least 3 non-missing data points for a meaningful Q-Q plot.", icon="⚠️")
         return None

    fig, ax = _create_figure(figsize=(6, 6)) # Square plot is typical
    if fig is None: return None
    try:
        # Create the Q-Q plot using scipy.stats.probplot
        stats.probplot(series_dropna, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot for {series.name} (vs Normal Distribution)', fontsize=12)
        ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=10)
        ax.set_ylabel('Sample Quantiles', fontsize=10)
        # Enhance line visibility
        ax.get_lines()[0].set_markerfacecolor('lightblue') # Points
        ax.get_lines()[0].set_markersize(4.0)
        ax.get_lines()[1].set_color('red') # Fit line
        ax.get_lines()[1].set_linewidth(1.5)
        ax.grid(True, linestyle='--', alpha=0.6) # Add grid
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error generating Q-Q plot for '{series.name}': {e}", icon="❌")
        if fig: plt.close(fig)
        return None


def plot_violin(df: pd.DataFrame, y_col: str, x_col: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Generates a violin plot for a numeric column, optionally grouped by a categorical column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        y_col (str): The numeric column for the Y-axis (distribution shown).
        x_col (Optional[str]): The categorical column for the X-axis (grouping).

    Returns:
        Optional[plt.Figure]: A matplotlib Figure object, or None on error. Shows st.error on failure.
    """
    if df is None or y_col not in df.columns: st.error(f"Y-col '{y_col}' missing or DataFrame None.", icon="❌"); return None
    if not pd.api.types.is_numeric_dtype(df[y_col]): st.error(f"Y-col '{y_col}' must be numeric.", icon="❌"); return None
    if df[y_col].dropna().empty: st.warning(f"Y-col '{y_col}' has no non-missing values.", icon="⚠️"); return None
    if x_col and x_col not in df.columns: st.error(f"X-col '{x_col}' not found.", icon="❌"); return None

    fig = None
    try:
        fig_width = 10 if x_col else 6
        fig, ax = _create_figure(figsize=(fig_width, 5))
        if fig is None: return None

        plot_params = {'data': df, 'ax': ax, 'palette': 'viridis', 'inner': 'quartile'} # Show quartiles inside violin
        title = f'Violin Plot of {y_col}'
        x_label = None; order = None; n_groups = 0

        if x_col:
            plot_params['y'] = y_col; plot_params['x'] = x_col
            title += f' grouped by {x_col}'; x_label = x_col
            n_groups = df[x_col].nunique()
            max_groups = 15
            if n_groups > max_groups:
                 top_groups = df[x_col].value_counts().nlargest(max_groups).index
                 plot_params['data'] = df[df[x_col].isin(top_groups)]
                 order = top_groups # Plot only top N groups
                 st.caption(f"Note: Displaying violin plot for top {max_groups} groups only based on frequency.")

            plot_params['order'] = order # Pass order (can be None)
            sns.violinplot(**plot_params)
            if n_groups > 10: ax.tick_params(axis='x', rotation=45, labelsize=9)
            else: ax.tick_params(axis='x', labelsize=9)
        else:
             # Plot single violin horizontally
             plot_params['x'] = y_col
             plot_params.pop('y') # Remove 'y' if plotting horizontally
             sns.violinplot(**plot_params)
             x_label = y_col
             ax.set_ylabel(None)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(x_label, fontsize=10)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6) # Add horizontal grid lines
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error plotting violin plot for '{y_col}': {e}", icon="❌")
        if fig: plt.close(fig)
        return None


def plot_joint(df: pd.DataFrame, x_col: str, y_col: str, kind: str = 'scatter') -> Optional[plt.Figure]:
    """
    Generates a joint distribution plot (e.g., scatter with marginal histograms).

    Args:
        df (pd.DataFrame): The input DataFrame.
        x_col (str): Numeric column for X-axis.
        y_col (str): Numeric column for Y-axis.
        kind (str): Type of joint plot ('scatter', 'kde', 'hist', 'reg'). Defaults to 'scatter'.

    Returns:
        Optional[plt.Figure]: A Seaborn JointGrid object's figure, or None on error. Shows st.error on failure.
    """
    # --- Input Validation ---
    if df is None or x_col not in df.columns or y_col not in df.columns:
        st.error(f"Columns '{x_col}', '{y_col}' not found or DataFrame is None.", icon="❌")
        return None
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
         st.error(f"Both columns '{x_col}' and '{y_col}' must be numeric for joint plot.", icon="❌")
         return None
    if df[[x_col, y_col]].dropna().empty:
         st.warning(f"No non-missing data pairs found for columns '{x_col}' and '{y_col}'. Cannot plot.", icon="⚠️")
         return None
    # --- End Validation ---

    joint_grid = None # Initialize
    try:
        # --- Define keyword arguments conditionally based on 'kind' ---
        current_joint_kws = {}
        current_marginal_kws = {'bins': 'auto'} # Common settings for marginal histograms

        if kind == 'scatter':
            current_joint_kws = {'alpha': 0.6, 's': 20} # Add alpha and maybe size
        elif kind == 'reg':
             current_joint_kws = {'scatter_kws': {'alpha': 0.4, 's': 20}} # Pass alpha via scatter_kws
             # Can add line_kws={'color':'red'} etc.
        elif kind == 'kde':
             current_joint_kws = {'fill': True, 'thresh': 0.05, 'levels': 5}
             current_marginal_kws = {'fill': True, 'bw_adjust': 0.75} # Adjust bandwidth for marginal KDEs
        elif kind == 'hist':
             # Example: Calculate bins based on data range, limit number of bins
             xbins = min(df[x_col].nunique(), 50)
             ybins = min(df[y_col].nunique(), 50)
             current_joint_kws = {'bins': (xbins, ybins), 'pmax': 0.9} # Use pmax to handle outliers in color scale
             current_marginal_kws = {'bins': 'auto'}

        # --- Create the Joint Plot ---
        joint_grid = sns.jointplot(
            data=df,
            x=x_col,
            y=y_col,
            kind=kind,
            height=6,            # Control overall figure size
            color='steelblue',     # Base color for univariate plots
            joint_kws=current_joint_kws,  # Pass conditional kwargs for the main plot
            marginal_kws=current_marginal_kws # Pass conditional kwargs for marginal plots
        )

        # --- Customize Appearance ---
        joint_grid.fig.suptitle(f'Joint Distribution ({kind.capitalize()}) of {y_col} vs {x_col}', y=1.02, fontsize=14)
        joint_grid.ax_joint.set_xlabel(x_col, fontsize=10)
        joint_grid.ax_joint.set_ylabel(y_col, fontsize=10)

        if kind in ['scatter', 'reg']:
             joint_grid.ax_joint.grid(True, linestyle='--', alpha=0.6)

        return joint_grid.fig # Return the underlying figure object

    except Exception as e:
        st.error(f"Error plotting joint plot for '{x_col}' vs '{y_col}' (kind='{kind}'): {e}", icon="❌")
        if joint_grid and hasattr(joint_grid, 'fig'):
             try: plt.close(joint_grid.fig)
             except Exception: pass # Ignore errors during closing
        return None


def plot_clusters(df: pd.DataFrame, pc1_col: str, pc2_col: str, cluster_col: str) -> Optional[plt.Figure]:
    """
    Visualizes K-Means clustering results using specified columns (e.g., first two principal components).

    Args:
        df (pd.DataFrame): DataFrame containing principal components (or other features) and cluster labels.
        pc1_col (str): Name of the column for the X-axis (e.g., 'PC1').
        pc2_col (str): Name of the column for the Y-axis (e.g., 'PC2').
        cluster_col (str): Name of the column containing the cluster labels (e.g., 'Cluster').

    Returns:
        Optional[plt.Figure]: A matplotlib Figure object, or None on error. Shows st.error on failure.
    """
    required_cols = [pc1_col, pc2_col, cluster_col]
    if df is None or any(col not in df.columns for col in required_cols):
         missing = [col for col in required_cols if col not in df.columns]
         st.error(f"Required columns for cluster plot missing: {missing}", icon="❌")
         return None

    fig, ax = _create_figure(figsize=(8, 6))
    if fig is None: return None
    try:
        # Ensure cluster column is treated as categorical for hue mapping
        plot_df = df.copy()
        plot_df[cluster_col] = plot_df[cluster_col].astype('category')

        unique_clusters = plot_df[cluster_col].cat.categories.tolist() # Get categories in order
        if not unique_clusters:
             st.warning("No valid cluster labels found in the specified column.", icon="⚠️"); plt.close(fig); return None
        palette = sns.color_palette('viridis', n_colors=len(unique_clusters))

        sns.scatterplot(data=plot_df, x=pc1_col, y=pc2_col, hue=cluster_col, ax=ax,
                        palette=palette, s=50, alpha=0.8, legend='full')

        ax.set_title(f'Clusters (colored by {cluster_col})', fontsize=14)
        ax.set_xlabel(pc1_col, fontsize=10)
        ax.set_ylabel(pc2_col, fontsize=10)
        ax.legend(title=f'{cluster_col}', bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        return fig
    except Exception as e:
        st.error(f"Error plotting clusters using columns '{pc1_col}', '{pc2_col}', '{cluster_col}': {e}", icon="❌")
        if fig: plt.close(fig)
        return None


def plot_pca_results(pca_df: pd.DataFrame, pca_obj: PCA) -> Optional[plt.Figure]:
    """
    Visualizes PCA results: scatter plot of first two components and explained variance ratio plot.

    Args:
        pca_df (pd.DataFrame): DataFrame containing the principal components ('PC1', 'PC2', ...).
        pca_obj (PCA): The fitted sklearn PCA object.

    Returns:
        Optional[plt.Figure]: A matplotlib Figure object containing two subplots, or None on error. Shows st.error on failure.
    """
    if pca_df is None or pca_obj is None or 'PC1' not in pca_df.columns:
         st.error("Invalid PCA data or object provided.", icon="❌"); return None
    n_components = pca_obj.n_components_
    if n_components < 1: st.error("PCA object has no components.", icon="❌"); return None

    fig = None # Initialize fig
    try:
        # Create a figure with two subplots (1 row, 2 columns)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # Adjust size as needed
        if fig is None: return None # Check if subplot creation failed

        # --- Plot 1: Scatter or Hist ---
        if n_components >= 2 and 'PC2' in pca_df.columns:
            sns.scatterplot(data=pca_df, x='PC1', y='PC2', ax=axes[0], s=30, alpha=0.7, color='darkblue')
            axes[0].set_title('Data Projected onto First Two PCs', fontsize=12)
            axes[0].set_xlabel('Principal Component 1', fontsize=10); axes[0].set_ylabel('Principal Component 2', fontsize=10)
            axes[0].grid(True, linestyle='--', alpha=0.6)
        elif n_components == 1:
             sns.histplot(pca_df['PC1'], kde=True, ax=axes[0], color='darkblue')
             axes[0].set_title('Distribution of Principal Component 1', fontsize=12)
             axes[0].set_xlabel('Principal Component 1', fontsize=10); axes[0].set_ylabel('Frequency', fontsize=10)
        else: # Should not happen if n_components >= 1 check passed, but defensive
             axes[0].text(0.5, 0.5, 'PC Data Not Available', ha='center', va='center', transform=axes[0].transAxes)
             axes[0].set_title('PCA Component Plot', fontsize=12)

        # --- Plot 2: Explained Variance ---
        explained_variance_ratio = pca_obj.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        pc_labels = [f'PC{i+1}' for i in range(n_components)]

        sns.barplot(x=pc_labels, y=explained_variance_ratio, ax=axes[1], color='dodgerblue', edgecolor='black')
        axes[1].set_title('Explained Variance per Component', fontsize=12)
        axes[1].set_xlabel('Principal Component', fontsize=10); axes[1].set_ylabel('Explained Variance Ratio', fontsize=10)
        axes[1].tick_params(axis='x', rotation=45, labelsize=9)
        # Add text labels for variance ratio on bars
        for index, value in enumerate(explained_variance_ratio): axes[1].text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=8)

        ax2 = axes[1].twinx()
        ax2.plot(pc_labels, cumulative_variance, color='crimson', marker='o', linestyle='--', markersize=5)
        ax2.set_ylabel('Cumulative Explained Variance', color='crimson', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='crimson', labelsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.grid(False) # Turn off grid for secondary axis

        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Error plotting PCA results: {e}", icon="❌")
        if fig: plt.close(fig)
        return None


def plot_wordcloud(series: pd.Series) -> Optional[plt.Figure]:
    """
    Generates a word cloud from a text series.

    Args:
        series (pd.Series): A pandas Series containing text data.

    Returns:
        Optional[plt.Figure]: A matplotlib Figure object displaying the word cloud, or None on error.
                               Shows st.error/warning if 'wordcloud' library is missing or on failure.
    """
    # Check if WordCloud library was imported successfully
    if 'WordCloud' not in globals() or WordCloud is None:
         st.error("WordCloud library not installed/imported. Cannot generate plot.", icon="🚨")
         return None

    if series is None or series.dropna().empty:
        st.warning("Input series empty or contains only missing values. Cannot generate word cloud.", icon="⚠️")
        return None
    # Ensure data is string type
    if not pd.api.types.is_string_dtype(series.dropna()):
         st.warning(f"Column '{series.name}' is not primarily text data. Word cloud might be nonsensical. Attempting conversion to string.", icon="⚠️")

    fig, ax = _create_figure(figsize=(10, 5)) # Adjust size
    if fig is None: return None
    try:
        # Attempt conversion, drop non-strings/NaNs
        text_data = series.dropna().astype(str)
        if text_data.empty:
             st.warning("No valid text data found after handling missing values.", icon="⚠️"); plt.close(fig); return None
        full_text = ' '.join(text_data)
        if not full_text.strip():
             st.warning("Text content is empty or only whitespace after joining.", icon="⚠️"); plt.close(fig); return None

        # Generate word cloud object
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              max_words=150,          # Limit number of words
                              collocations=False,     # Avoid bigrams/trigrams
                              contour_width=1,
                              contour_color='steelblue').generate(full_text)

        # Plot using matplotlib
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off') # Hide axes
        ax.set_title(f'Word Cloud for {series.name}', fontsize=14, pad=10)
        plt.tight_layout(pad=0)
        return fig

    except ValueError as ve:
         # Catch specific error if all words are filtered out (e.g., only stopwords)
         if "zero words were placed" in str(ve):
              st.warning(f"Could not generate word cloud for '{series.name}'. All words might have been filtered out (e.g., only stopwords or very short words).", icon="⚠️")
         else:
              st.error(f"Error generating word cloud for '{series.name}': {ve}", icon="❌")
         if fig: plt.close(fig)
         return None
    except Exception as e:
        st.error(f"Error generating word cloud for '{series.name}': {e}", icon="❌")
        if fig: plt.close(fig)
        return None