import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Employee Income Classification App", layout="wide")

# Title
st.title("💼 Employee Income Classification App")

# Load and clean data
@st.cache_data
def load_data():
    data = pd.read_csv("adult.csv")
    data.replace("?", pd.NA, inplace=True)
    data.dropna(inplace=True)
    return data

data = load_data()

# Show preview
st.subheader("📊 Preview of Cleaned Dataset")
st.dataframe(data.head())

# Visualizations
st.subheader("📈 Data Visualizations")

# Boxplot - Capital Gain
st.write("Boxplot of Capital Gain")
fig1, ax1 = plt.subplots()
ax1.boxplot(data['capital-gain'])
st.pyplot(fig1)

# Boxplot - Age
st.write("Boxplot of Age")
fig2, ax2 = plt.subplots()
ax2.boxplot(data['age'])
st.pyplot(fig2)

# ✅ Encode all object columns (categorical columns)
for col in data.select_dtypes(include=['object']).columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# ✅ Split features and label
x = data.drop('income', axis=1)
y = data['income']

# ✅ Scale the numeric features safely
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# ✅ Train-Test Split
xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.2, random_state=23, stratify=y)

# ✅ Train KNN Model
knn = KNeighborsClassifier()
knn.fit(xtrain, ytrain)
predictions = knn.predict(xtest)

# ✅ Accuracy
acc = accuracy_score(ytest, predictions)
st.subheader("✅ KNN Model Accuracy")
st.write(f"Accuracy Score: {acc:.4f}")
