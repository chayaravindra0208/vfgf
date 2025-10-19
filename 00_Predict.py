import streamlit as st
import json
import pandas as pd
import pickle
from io import StringIO
import random

from constants import CATEGORICAL_COLUMNS, ALL_COLUMNS

# constants
JSONS = ["example_one.json", "example_two.json", "example_three.json"]


@st.cache_resource
def load_pickle_model(model_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# load models
model = load_pickle_model("random_forest_regression")
scaler_model = load_pickle_model("scaler_model")

# Authentication check
if not st.user.is_logged_in:
    st.error("You must be logged in to access this page.", icon="⚠️")
    st.stop()

st.title("🌌 Galaxy Redshift Predictor")
st.subheader("Unlock the Secrets of the Cosmos")
# Main content
st.write("""
Welcome to the Galaxy Redshift Prediction Platform. Upload your galaxy observation data in JSON format 
to predict redshifts and analyze cosmic expansion patterns. Our advanced algorithms help astronomers 
and researchers understand the large-scale structure of the universe.
""")

# Sidebar for file upload
st.header("📤 Upload JSON Data")
uploaded_file = st.file_uploader(
    "Upload your galaxy data (JSON only)",
    type=["json"],
    accept_multiple_files=False,
    help="Upload a JSON file containing galaxy observation data"
)

with st.sidebar:
    st.subheader("Download Example Json")
    json_name = st.selectbox(
        "Select Example Json",
        JSONS,
    )
    with open(json_name, "r") as f:
        json_data = json.load(f)
    # disply the json if needed
    with st.expander("Example Json"):
        st.json(json_data)
    json_download_data = json.dumps(json_data, indent=4)
    st.download_button(
        label="Download Example Json",
        data=json_download_data,
        file_name=json_name,
        mime="application/json"
    )

# File processing logic
if uploaded_file is not None:
    try:
        # Read the uploaded file
        json_data = json.load(uploaded_file)
        
        # Display success message
        st.success("✅ File successfully uploaded and processed!")
        
        # Show a preview of the data
        with st.expander("📊 View Uploaded Data"):
            st.json(json_data)
            
        # Convert to DataFrame for better display (if the JSON structure is compatible)
        try:
            df = pd.json_normalize(json_data)
            columns_not_available = False
            for col in ALL_COLUMNS:
                if col not in df.columns:
                    columns_not_available = True
                    break
            if columns_not_available:
                st.error("❌ The uploaded JSON file does not contain all the required columns.", icon="⚠️")
                st.stop()
            
            # Add prediction button (functionality to be implemented)
            if st.button("🚀 Predict Redshifts", type="primary"):
                with st.spinner("Analyzing galaxy data..."):
                    # Placeholder for prediction logic
                    continuous_columns = [col for col in ALL_COLUMNS if col not in CATEGORICAL_COLUMNS]
                    df[continuous_columns] = scaler_model.transform(df[continuous_columns])
                    predictions = model.predict(df.values)
                    st.header("Predicted Redshifts: {}".format(predictions[0]))
                    
        except Exception as error:
            print(str(error))
            st.warning("Error in processing the file.", icon="⚠️")
            
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON file. Please upload a valid JSON file.", icon="⚠️")
    except Exception as error:
        st.error(f"An error occurred: {str(error)}", icon="❌")
