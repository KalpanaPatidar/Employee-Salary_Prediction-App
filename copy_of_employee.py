import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Streamlit Configuration
# -------------------------------
st.set_page_config(page_title="Employee Salary Classification App", layout="wide")
st.title("💼 Employee Salary Classification App")
st.write("Predict whether an employee earns >50K or <=50K based on input features.")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("adult.csv")
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

data = load_data()

# Label encoding
label_encoders = {}
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature selection
X = data.drop("income", axis=1)
y = data["income"]

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=23, stratify=y)

# Train KNN model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# -------------------------------
# Sidebar Input for Individual Prediction
# -------------------------------
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", label_encoders["education"].classes_)
occupation = st.sidebar.selectbox("Job Role", label_encoders["occupation"].classes_)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Encode and structure input
input_dict = {
    "age": age,
    "education": label_encoders["education"].transform([education])[0],
    "occupation": label_encoders["occupation"].transform([occupation])[0],
    "hours-per-week": hours_per_week,
    "experience": experience
}

input_df = pd.DataFrame([input_dict])
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]
input_scaled = scaler.transform(input_df)

# -------------------------------
# Display Input Data
# -------------------------------
st.subheader("📌 Input Data")
display_df = pd.DataFrame({
    "age": [age],
    "education": [education],
    "occupation": [occupation],
    "hours-per-week": [hours_per_week],
    "experience": [experience]
})
st.dataframe(display_df)

# -------------------------------
# Predict Salary Class
# -------------------------------
if st.button("Predict Salary Class"):
    pred = model.predict(input_scaled)[0]
    label = ">50K" if pred == 1 else "<=50K"
    st.success(f"🎯 Predicted Salary Class: **{label}**")

# -------------------------------
# Batch Prediction Section
# -------------------------------
st.subheader("📂 Batch Prediction")
st.write("Upload a CSV file for batch prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

def safe_label_encode(le, series):
    known = set(le.classes_)
    series_safe = series.apply(lambda x: x if x in known else None)
    if series_safe.isnull().any():
        st.warning(f"⚠️ Column '{series.name}' has unseen labels. Replacing with most frequent known value: '{le.classes_[0]}'")
        series_safe = series_safe.fillna(le.classes_[0])
    return le.transform(series_safe)

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    original_df = batch_df.copy()

    for col in label_encoders:
        if col in batch_df.columns:
            batch_df[col] = safe_label_encode(label_encoders[col], batch_df[col])

    for col in X.columns:
        if col not in batch_df.columns:
            batch_df[col] = 0

    batch_df = batch_df[X.columns]
    batch_scaled = scaler.transform(batch_df)
    batch_preds = model.predict(batch_scaled)
    original_df["Predicted Salary"] = [">50K" if p == 1 else "<=50K" for p in batch_preds]

    st.subheader("📋 Prediction Results")
    st.dataframe(original_df)

    csv_output = original_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results", csv_output, "salary_predictions.csv", "text/csv")

# -------------------------------
# Optional Accuracy Display
# -------------------------------
st.subheader("📊 Model Accuracy")
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Accuracy: **{accuracy:.4f}**")
