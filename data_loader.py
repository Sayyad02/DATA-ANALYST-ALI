# data_loader.py
import pandas as pd
import streamlit as st
import io
from typing import Optional

# Required for reading Excel files - ensure it's installed (pip install openpyxl)
try:
    import openpyxl
except ImportError:
    st.error("The 'openpyxl' library is required for Excel support. Please install it (`pip install openpyxl`).", icon="🚨")
    # st.stop() # Uncomment this if you want the app to stop if openpyxl is missing

def load_data_from_upload(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[pd.DataFrame]:
    """
    Loads data from a Streamlit UploadedFile object (CSV or Excel).

    Args:
        uploaded_file: The file object uploaded via st.file_uploader.

    Returns:
        A pandas DataFrame if loading is successful, otherwise None.
        Shows st.error message on failure within the Streamlit app.
    """
    if uploaded_file is None:
        print("Error: No file provided to load_data_from_upload.") # Use print for internal log
        return None

    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            return df
        elif file_extension in ['xlsx', 'xls']:
            # Ensure 'openpyxl' is available for .xlsx
            if file_extension == 'xlsx' and 'openpyxl' not in globals():
                 st.error("Need 'openpyxl' library to read .xlsx files. Please install it (`pip install openpyxl`).", icon="🚨")
                 return None
            df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else None)
            return df
        else:
            st.error(f"Unsupported file type: '.{file_extension}'. Please upload a CSV or Excel file.", icon="❌")
            return None
    except pd.errors.EmptyDataError:
        st.error(f"Error: The uploaded file '{file_name}' is empty or has no data to parse.", icon="❌")
        return None
    except ImportError as imp_err:
         if 'openpyxl' in str(imp_err):
              st.error("Need 'openpyxl' library to read Excel files. Please install it (`pip install openpyxl`).", icon="🚨")
         else:
              st.error(f"Import error while loading data: {imp_err}", icon="❌")
         return None
    except Exception as e:
        st.error(f"Error loading data from '{file_name}': {e}", icon="❌")
        return None