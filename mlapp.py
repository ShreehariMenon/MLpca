import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Wine PCA Explorer", layout="wide")

# Title
st.title("🍷 PCA on Wine Dataset")

# Info sections
with st.expander("ℹ️ About PCA (Principal Component Analysis)", expanded=True):
    st.markdown("""
    **Principal Component Analysis (PCA)** is a technique for reducing the dimensionality of large datasets by transforming them into a new set of variables called **principal components**.
    
    - These components are **uncorrelated** and capture the **maximum variance** in the data.
    - PCA helps to **visualize** high-dimensional data in 2D or 3D space.
    - It is widely used to **simplify data**, **reduce noise**, and improve the performance of machine learning models.

    **Steps in PCA:**
    1. Standardize the features
    2. Compute the covariance matrix
    3. Calculate eigenvectors and eigenvalues
    4. Project the data onto principal components
    """)

with st.expander("📄 About the Wine Dataset", expanded=True):
    st.markdown("""
    The **Wine Dataset** is a well-known dataset from the UCI Machine Learning Repository.

    - It contains the results of a chemical analysis of **178 wine samples** from three different cultivars in Italy.
    - Each sample is described using **13 numeric features** like:
        - Alcohol
        - Malic acid
        - Ash
        - Flavanoids
        - Color intensity
        - Hue, and more.
    - The goal is to classify wines into **3 classes** based on these features.

    This makes the dataset a perfect candidate for PCA as it has **many correlated features**.
    """)

# Load the dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="Wine Class")
df = pd.concat([X, y], axis=1)

# Sidebar
st.sidebar.header("⚙️ PCA Settings")
n_components = st.sidebar.slider("Select number of PCA components", min_value=1, max_value=13, value=2)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

# Create DataFrame of PCA results
pca_columns = [f'PC{i+1}' for i in range(n_components)]
pca_df = pd.DataFrame(X_pca, columns=pca_columns)
pca_df["Wine Class"] = y

# Scree plot (explained variance)
st.subheader("🔍 Explained Variance (Scree Plot)")
fig, ax = plt.subplots()
ax.plot(np.cumsum(explained_var), marker='o', linestyle='-')
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.set_title("Explained Variance by PCA Components")
st.pyplot(fig)

# Scatter plot
if n_components >= 2:
    st.subheader("📉 PCA Scatter Plot")
    if n_components >= 3:
        fig3d = px.scatter_3d(
            pca_df, x='PC1', y='PC2', z='PC3', color='Wine Class',
            title="3D PCA Scatter Plot", labels={'Wine Class': 'Class'}
        )
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        fig2d = px.scatter(
            pca_df, x='PC1', y='PC2', color='Wine Class',
            title="2D PCA Scatter Plot", labels={'Wine Class': 'Class'}
        )
        st.plotly_chart(fig2d, use_container_width=True)

# Show transformed data
st.subheader("🧾 PCA Transformed Data")
st.dataframe(pca_df)

# Download button
csv = pca_df.to_csv(index=False)
st.download_button("📥 Download Transformed Data", data=csv, file_name="wine_pca.csv", mime='text/csv')
