import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Employee Salary Classification App", layout="wide")
st.title("💼 Employee Salary Classification App")
st.write("Predict whether an employee earns >50K or <=50K based on input features.")

# -------------------------------
# Load and preprocess data
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("adult.csv")
    data.replace("?", np.nan, inplace=True)
    data.dropna(inplace=True)
    return data

data = load_data()

# Label Encoding
label_encoders = {}
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature-label split
X = data.drop("income", axis=1)
y = data["income"]

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=23, stratify=y)

# Model training
model = KNeighborsClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# -------------------------------
# Display Model Accuracy
# -------------------------------
st.subheader("✅ Model Accuracy")
st.write(f"Accuracy Score: **{accuracy:.4f}**")

# -------------------------------
# Sidebar - Input for Single Prediction
# -------------------------------
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", label_encoders['education'].classes_)
occupation = st.sidebar.selectbox("Job Role", label_encoders['occupation'].classes_)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Construct input DataFrame
input_dict = {
    'age': age,
    'education': label_encoders['education'].transform([education])[0],
    'occupation': label_encoders['occupation'].transform([occupation])[0],
    'hours-per-week': hours_per_week,
    'experience': experience
}

input_df = pd.DataFrame([input_dict])

# Add missing columns with zeros (if any)
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X.columns]

# Scale input
input_scaled = scaler.transform(input_df)

# Show Input Data
st.subheader("📥 Input Data")
st.write(input_df)

# Predict
if st.button("Predict Salary Class"):
    prediction = model.predict(input_scaled)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Predicted Salary Class: **{result}**")

# -------------------------------
# Batch Prediction
# -------------------------------
st.subheader("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])

def safe_label_encode(le, series):
    known_classes = set(le.classes_)
    safe_series = series.apply(lambda val: val if val in known_classes else None)
    if safe_series.isnull().any():
        st.warning(f"⚠️ Unseen labels found in column. Replacing with most frequent known value.")
        safe_series = safe_series.fillna(le.classes_[0])
    return le.transform(safe_series)

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    original_df = batch_df.copy()

    # Encode categorical columns safely
    for col in label_encoders:
        if col in batch_df.columns:
            batch_df[col] = safe_label_encode(label_encoders[col], batch_df[col])

    # Fill missing columns if necessary
    for col in X.columns:
        if col not in batch_df.columns:
            batch_df[col] = 0

    batch_df = batch_df[X.columns]
    batch_scaled = scaler.transform(batch_df)
    batch_preds = model.predict(batch_scaled)
    original_df['Predicted Salary'] = [">50K" if p == 1 else "<=50K" for p in batch_preds]

    st.write("📊 Batch Prediction Results")
    st.dataframe(original_df)

    # Download button
    csv = original_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "predicted_salaries.csv", "text/csv")

# -------------------------------
# Visualizations
# -------------------------------
st.subheader("📊 Data Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.write("Boxplot of Capital Gain")
    fig1, ax1 = plt.subplots()
    ax1.boxplot(data['capital-gain'])
    st.pyplot(fig1)

with col2:
    st.write("Boxplot of Age")
    fig2, ax2 = plt.subplots()
    ax2.boxplot(data['age'])
    st.pyplot(fig2)
