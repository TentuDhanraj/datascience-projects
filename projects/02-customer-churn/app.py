import streamlit as st
import pandas as pd
import pickle
#from io import BytesIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#import matplotlib.pyplot as plt

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

#load model
@st.cache_resource
def load_pipeline():
    with open('pipeline_churn.pkl','rb') as f:
        model = pickle.load(f)
    return model


# load datset

@st.cache_data
def load_df(path='churn_cleaned_dataset.csv'):
    """Load sample cleaned CSV to infer features """
    try:
        df =pd.read_csv(path)
        return df
    except Exception as e:
        st.warning(f"Path could not be found {path} : {e}")

        

pipe = load_pipeline()
sample_df = load_df()

# derive feature list
if sample_df is not None:
    if 'Churn' in sample_df.columns:
        feature_cols = [c for c in sample_df.columns if c != 'Churn']
    else:
        feature_cols = sample_df.columns.to_list()
else:
    feature_cols = None

st.title("Telco Customer Churn Predictor")
st.markdown(
    "Predict whether a customer will churn. "
    "Use the sidebar to choose single prediction, batch upload, or diagnostics."
)



# Sidebar controls

st.sidebar.header("Controls")
mode = st.sidebar.radio("choose mode",["Single prediction", "Batch prediction", "Diagnostics"])
threshold = st.sidebar.slider("Decision Threshold", min_value=0.0, max_value=1.0,value=0.40,step=0.01)

# Layout
if mode == 'Prediction':
    st.header("Single-customer prediction")

    #inputs left , result right
    left_col, right_col = st.columns([2,1])

    # left: input
    with left_col:
        with st.form("Single form"):
            st.subheader("Customer Details")
            tenure = st.number_input("tenure (months)", min_value=0, max_value=200, value=12)
            monthly_charges = st.number_input("MonthlyCharges", min_value=0.0, value=70.0, step=0.1, format="%.2f")
            senior = st.selectbox("SeniorCitizen", options=['No', 'Yes'], index=0)
            partner = st.selectbox("Partner", options=["Yes", "No"], index=1)
            contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])

if mode == "Single prediction":
    st.header("Single-customer prediction")
    if pipe is None:
        st.error("Model not loaded. Fix model path and restart.")
    elif sample_df is None:
        st.error("Sample CSV not loaded. Place `Churn_Cleaned_dataset.csv` in project root.")
    else:
        st.markdown("Fill customer details below and click **Predict**.")
        # Build an input form dynamically from sample_df's columns
        with st.form("single_form"):
            input_data = {}
            # iterate features and choose appropriate widget
            for col in feature_cols:
                # infer type from sample df
                ser = sample_df[col]
                # treat binary 0/1 as checkbox
                unique_vals = pd.unique(ser.dropna())
                if pd.api.types.is_integer_dtype(ser) and set(unique_vals).issubset({0, 1}):
                    # show checkbox (1=True, 0=False)
                    input_data[col] = st.checkbox(col, value=bool(int(ser.mode().iloc[0])))
                elif pd.api.types.is_numeric_dtype(ser) and ser.nunique() > 10:
                    # numeric input
                    min_v = int(max(0, ser.min()))
                    max_v = int(ser.max() + 10)
                    default = int(ser.median())
                    input_data[col] = st.number_input(col, min_value=min_v, max_value=max_v, value=default)
                else:
                    # categorical: show selectbox with sample categories
                    opts = [str(x) for x in sorted(unique_vals.astype(str))]
                    default = opts[0] if opts else ""
                    input_data[col] = st.selectbox(col, opts, index=0, key=col)
            submit = st.form_submit_button("Predict")

        if submit:
            # Build a single-row DataFrame consistent with training columns
            row = {}
            for k, v in input_data.items():
                # convert bool to int for binary columns
                if isinstance(v, bool):
                    row[k] = int(v)
                else:
                    row[k] = v
            X_single = pd.DataFrame([row], columns=feature_cols)

            try:
                proba = pipe.predict_proba(X_single)[:, 1][0]
                pred = int(proba >= threshold)
                st.metric("Churn probability", f"{proba:.3f}")
                st.success("Predicted: CHURN" if pred == 1 else "Predicted: NO CHURN")
                # optional: short advice
                if pred == 1:
                    st.info("Suggested action: prioritize this customer for retention offers / outreach.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


