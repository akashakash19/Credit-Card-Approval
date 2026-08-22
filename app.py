import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. LOAD SAVED MODEL FILES
# ============================================================

MODEL_PATH = "credit_card_rf_model.pkl"
PREPROCESSOR_PATH = "credit_card_preprocessor.pkl"
THRESHOLD_PATH = "credit_card_threshold.pkl"


@st.cache_resource
def load_model_files():
    """Load trained model, preprocessor and threshold."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing file: {MODEL_PATH}"
        )

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Missing file: {PREPROCESSOR_PATH}"
        )

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            f"Missing file: {THRESHOLD_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, preprocessor, threshold


try:
    model, preprocessor, threshold = load_model_files()

except Exception as error:
    st.error(f"Unable to load model files: {error}")
    st.stop()


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }

    .stButton > button {
        width: 100%;
        height: 50px;
        font-size: 17px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💳 Credit Card Risk Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning based customer credit-risk prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Model Information")

    st.write(
        """
        This application uses a trained **Random Forest
        classification model** to estimate the customer's
        credit-risk status.
        """
    )

    st.subheader("Model Details")

    st.write("🤖 Model: Random Forest")
    st.write(f"🎯 Decision Threshold: {threshold:.2f}")
    st.write("📈 ROC-AUC: 0.781")
    st.write("🔢 Input Features: 16")

    st.divider()

    st.info(
        """
        The prediction represents a **credit-risk proxy**
        derived from historical credit behaviour. It is not
        an actual bank approval decision.
        """
    )


# ============================================================
# 6. PERSONAL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Personal Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["F", "M"]
    )

with col2:
    age = st.number_input(
        "Age (years)",
        min_value=18.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )

with col3:
    children_count = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )


# ============================================================
# 7. FINANCIAL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">💰 Financial Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input(
        "Annual Income",
        min_value=27000.0,
        max_value=2000000.0,
        value=150000.0,
        step=5000.0
    )

with col2:
    income_type = st.selectbox(
        "Income Type",
        [
            "Working",
            "Commercial associate",
            "Pensioner",
            "State servant",
            "Student"
        ]
    )

with col3:
    years_employed = st.number_input(
        "Years Employed",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )


# ============================================================
# 8. EDUCATION & FAMILY INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">🎓 Education & Family</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    education = st.selectbox(
        "Education",
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree"
        ]
    )

with col2:
    family_status = st.selectbox(
        "Family Status",
        [
            "Married",
            "Single / not married",
            "Civil marriage",
            "Separated",
            "Widow"
        ]
    )

with col3:
    family_members_count = st.number_input(
        "Family Members",
        min_value=1.0,
        max_value=20.0,
        value=2.0,
        step=1.0
    )


# ============================================================
# 9. HOUSING & OCCUPATION
# ============================================================

st.markdown(
    '<div class="section-title">🏠 Housing & Employment</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    housing_type = st.selectbox(
        "Housing Type",
        [
            "House / apartment",
            "With parents",
            "Municipal apartment",
            "Rented apartment",
            "Office apartment",
            "Co-op apartment"
        ]
    )

with col2:
    occupation = st.selectbox(
        "Occupation",
        [
            "Unknown",
            "Laborers",
            "Core staff",
            "Sales staff",
            "Managers",
            "Drivers",
            "High skill tech staff",
            "Accountants",
            "Medicine staff",
            "Cooking staff",
            "Security staff",
            "Cleaning staff",
            "Private service staff",
            "Low-skill Laborers",
            "Waiters/barmen staff",
            "Secretaries",
            "HR staff",
            "Realty agents",
            "IT staff"
        ]
    )

with col3:
    has_car = st.selectbox(
        "Has Car",
        ["Y", "N"]
    )


# ============================================================
# 10. CONTACT & PROPERTY DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">📱 Contact & Property</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    has_realty = st.selectbox(
        "Has Property",
        ["Y", "N"]
    )

with col2:
    has_work_phone = st.selectbox(
        "Has Work Phone",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:
    has_phone = st.selectbox(
        "Has Phone",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col4:
    has_email = st.selectbox(
        "Has Email",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


# ============================================================
# 11. CREATE CUSTOMER DATA
# ============================================================

def create_customer_data():
    """Create a dictionary matching the training features."""

    return {
        "Gender": gender,
        "Has_car": has_car,
        "Has_realty": has_realty,
        "Children_count": children_count,
        "Income": income,
        "Income_type": income_type,
        "Education": education,
        "Family_status": family_status,
        "Housing_type": housing_type,
        "Age": age,
        "Years_employed": years_employed,
        "Has_work_phone": has_work_phone,
        "Has_phone": has_phone,
        "Has_email": has_email,
        "Occupation": occupation,
        "Family_members_count": family_members_count
    }


# ============================================================
# 12. PREDICTION FUNCTION
# ============================================================

def predict_customer(customer_data):
    """Generate prediction and Bad probability."""

    customer_df = pd.DataFrame([customer_data])

    # Transform using the SAME preprocessor used during training
    customer_processed = preprocessor.transform(customer_df)

    # Probability of class 1 = Bad
    bad_probability = model.predict_proba(
        customer_processed
    )[0, 1]

    # Apply saved decision threshold
    prediction = (
        1 if bad_probability >= threshold else 0
    )

    result = "Bad" if prediction == 1 else "Good"

    return result, bad_probability


# ============================================================
# 13. PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Credit Risk"
)


# ============================================================
# 14. DISPLAY RESULT
# ============================================================

if predict_button:

    customer_data = create_customer_data()

    try:
        result, probability = predict_customer(
            customer_data
        )

        probability_percent = probability * 100

        st.divider()

        if result == "Bad":

            st.error(
                "⚠️ High Credit Risk"
            )

        else:

            st.success(
                "✅ Low Credit Risk"
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Prediction",
                result
            )

        with col2:

            st.metric(
                "Bad Probability",
                f"{probability_percent:.2f}%"
            )

        with col3:

            st.metric(
                "Decision Threshold",
                f"{threshold * 100:.0f}%"
            )

        st.progress(
            min(probability, 1.0)
        )

        if result == "Bad":

            st.warning(
                f"""
                The model estimated a {probability_percent:.2f}%
                probability of the customer belonging to the
                Bad-risk class.
                """
            )

        else:

            st.info(
                f"""
                The model estimated a {probability_percent:.2f}%
                probability of the customer belonging to the
                Bad-risk class.
                """
            )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Card Risk Predictor | Machine Learning Project"
)
