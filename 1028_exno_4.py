import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Step 1: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Step 3: Display explained variance (correlation strength)
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Step 4: Display PCA components (feature correlations)
print("\nPCA Components (Feature Correlation Matrix):")
pca_components = pd.DataFrame(
    pca.components_,
    columns=X.columns,
    index=[f"PC{i+1}" for i in range(len(X.columns))]
)
print(pca_components)