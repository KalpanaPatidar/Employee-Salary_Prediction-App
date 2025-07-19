import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Set Streamlit page config
st.set_page_config(page_title="Employee Income Classification App", layout="wide")

# Title
st.title("💼 Employee Income Classification App")

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv("adult.csv")
    data.replace("?", pd.NA, inplace=True)
    data.dropna(inplace=True)
    return data

data = load_data()

# Display dataset preview
st.subheader("📊 Preview of Cleaned Dataset")
st.dataframe(data.head())

# Visualizations
st.subheader("📈 Data Visualizations")

# Boxplot for capital-gain
st.write("Boxplot of Capital Gain")
fig1, ax1 = plt.subplots()
ax1.boxplot(data['capital-gain'])
st.pyplot(fig1)

# Boxplot for Age
st.write("Boxplot of Age")
fig2, ax2 = plt.subplots()
ax2.boxplot(data['age'])
st.pyplot(fig2)

# Encode categorical features
label_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
encoder = LabelEncoder()
for col in label_cols:
    data[col] = encoder.fit_transform(data[col])

# Split the data
x = data.drop('income', axis=1)
y = data['income']

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=23, stratify=y)

# Train KNN model
knn = KNeighborsClassifier()
knn.fit(xtrain, ytrain)
predictions = knn.predict(xtest)

# Show accuracy
acc = accuracy_score(ytest, predictions)
st.subheader("✅ KNN Model Accuracy")
st.write(f"Accuracy Score: {acc:.4f}")
