import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Set Streamlit page configuration
st.set_page_config(page_title="Employee Salary Classification App", layout="wide")

st.title("💼 Employee Salary Classification App")
st.write("Predict whether an employee earns >50K or <=50K based on input features.")

# Load and preprocess the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("adult.csv")
    data.replace("?", np.nan, inplace=True)
    data.dropna(inplace=True)
    return data

data = load_data()

# Encode categorical columns
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split features and labels
x = data.drop('income', axis=1)
y = data['income']

# Scale features
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.2, random_state=23, stratify=y)

# Train model
model = KNeighborsClassifier()
model.fit(xtrain, ytrain)
acc = accuracy_score(ytest, model.predict(xtest))

st.subheader("✅ KNN Model Accuracy")
st.write(f"Accuracy Score: **{acc:.4f}**")

# Sidebar - User input
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", label_encoders['education'].classes_)
job = st.sidebar.selectbox("Job Role", label_encoders['occupation'].classes_)
hours = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Prepare single prediction input
input_dict = {
    'age': age,
    'education': label_encoders['education'].transform([education])[0],
    'occupation': label_encoders['occupation'].transform([job])[0],
    'hours-per-week': hours,
    'experience': experience,
}

# Create a DataFrame with dummy columns to match the original dataset
input_df = pd.DataFrame([input_dict])
for col in x.columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[x.columns]
input_scaled = scaler.transform(input_df)

st.subheader("📥 Input Data")
st.write(input_df)

if st.button("Predict Salary Class"):
    pred = model.predict(input_scaled)
    pred_label = ">50K" if pred[0] == 1 else "<=50K"
    st.success(f"Predicted Salary Class: **{pred_label}**")

# Batch prediction
st.subheader("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    original_data = batch_data.copy()

    # Apply label encoding
    for col in label_encoders:
        if col in batch_data.columns:
            batch_data[col] = label_encoders[col].transform(batch_data[col])

    batch_data_scaled = scaler.transform(batch_data[x.columns])
    batch_preds = model.predict(batch_data_scaled)
    batch_labels = [">50K" if pred == 1 else "<=50K" for pred in batch_preds]

    original_data['Predicted Salary'] = batch_labels
    st.write("📋 Prediction Results:")
    st.dataframe(original_data)

    # Optionally download results
    csv_download = original_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Prediction Results", csv_download, "batch_predictions.csv", "text/csv")

# Visualization section
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
