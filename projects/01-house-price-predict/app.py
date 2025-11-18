import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib 

# --- Configuration ---
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load the model and data
@st.cache_resource
def load_resources():
    try:

        with open('linearmodel.pkl', 'rb') as f:
            model_pipeline = joblib.load(f)
        
        # Access the ColumnTransformer (pre_processor)
        preprocessor = model_pipeline.named_steps['columntransformer']
        

        one_hot_encoder = preprocessor.named_transformers_['onehotencoder']
        
        # Get the categories for the 'location' column (the first categorical column)
        all_locations = one_hot_encoder.categories_[0].tolist()
        
        return model_pipeline, all_locations
    except FileNotFoundError:
        st.error("Error: The model file 'linearmodel.pkl' was not found. Please ensure it is in the same directory.")
        return None, None
    except KeyError:
        st.error("Error: Could not extract locations from the pipeline. Check the pipeline structure.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        return None, None


# --- Load Model and Data ---
model_pipeline, locations = load_resources()

if model_pipeline is None or locations is None:
    st.stop() # Stop the app if resources couldn't load

# Filter out 'other' and potentially 'nan' from the display list, and sort them
locations_display = sorted([loc for loc in locations if loc not in ['other', 'nan', None]])

# --- Streamlit UI ---

st.title("🏡 Bengaluru House Price Prediction App")
st.markdown("""
This application predicts the price (in Lakhs INR) of a house in Bengaluru based on its key features.
The prediction is powered by a **Random Forest Regressor** model trained on your processed dataset.
""")

st.divider()

# Input Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Property Location")
    # Location selection
    selected_location = st.selectbox(
        "Select Area",
        locations_display,
        index=locations_display.index('Sarjapur  Road') if 'Sarjapur  Road' in locations_display else 0,
        help="Choose the micro-location of the property."
    )

with col2:
    st.subheader("Area & Rooms")
    # Total Sqft input
    total_sqft = st.number_input(
        "Total Square Feet (Sqft)",
        min_value=300,
        max_value=10000,
        value=1500,
        step=50,
        help="The area of the property. Must be > 300 sqft per BHK."
    )
    # BHK input
    bhk = st.number_input(
        "Number of Bedrooms (BHK)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="The number of bedrooms (BHK)."
    )

with col3:
    st.subheader("Bathroom Count")
    # Bath input
    bath = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="The number of bathrooms. Cannot be less than 1."
    )

# --- Prediction Logic ---
st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):
 
    try:
        input_data = pd.DataFrame({
            'location': [selected_location],
            'total_sqft': [total_sqft],
            'bath': [bath],
            'bhk': [bhk]
        })

        # 2. Predict using the pipeline
        prediction = model_pipeline.predict(input_data)[0]

        # 3. Display Result
        st.success(
            f"**The estimated house price is: ₹ {prediction:,.2f} Lakhs**"
        )
        st.balloons()
        
        # Optional: Show the input features used for the prediction
        st.markdown("---")
        st.markdown("### 🔍 Prediction Details")
        st.json(input_data.to_dict('records')[0])

    except Exception as e:
        st.error(f"An error occurred during prediction. Please check your inputs. Error: {e}")

st.markdown("""
<style>
    .stButton>button {
        font-size: 1.2rem;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stSuccess {
        font-size: 1.5rem !important;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)