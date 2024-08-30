
#pip install streamlit pandas plotly seaborn matplotlib

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target


# Create a title for the app
st.title("Iris Dataset Exploratory Data Analysis")


# Display dataset summary
st.write("### Dataset Summary")
st.write(df.describe())

# Display dataset head
st.write("### Dataset Head")
st.write(df.head())

# Plot 1: Histogram of Sepal Length
st.write("### Histogram of Sepal Length")
fig = px.histogram(df, x="sepal length (cm)", title="Sepal Length Distribution")
st.plotly_chart(fig)

# Plot 2: Scatter Plot of Sepal Length vs Sepal Width
st.write("### Scatter Plot of Sepal Length vs Sepal Width")
fig = px.scatter(df, x="sepal length (cm)", y="sepal width (cm)", title="Sepal Length vs Sepal Width")
st.plotly_chart(fig)

# Plot 5: Heatmap of Correlation Matrix
st.write("### Heatmap of Correlation Matrix")
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
st.pyplot(fig)








