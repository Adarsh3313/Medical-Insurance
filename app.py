import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Train model (cached)
# -----------------------------
@st.cache_resource
def train_model():
    # Public dataset with "charges" column
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)

    df_encoded = pd.get_dummies(df, drop_first=True)

    # Target column is "charges" (not expenses)
    X = df_encoded.drop("charges", axis=1)
    y = df_encoded["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏥 Medical Insurance Cost Prediction")
st.write("Enter details below to estimate your medical expenses:")

age = st.number_input("Age", min_value=18, max_value=100, step=1)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Convert input into dataframe
input_data = pd.DataFrame(
    {
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0],
    }
)

# Prediction
if st.button("Predict Expense"):
    prediction_usd = model.predict(input_data)[0]
    
    # Convert USD → INR (approx. 1 USD = 83 INR)
    prediction_inr = prediction_usd * 83  
    
    st.success(f"Estimated Medical Expense: ₹{prediction_inr:,.2f}")
