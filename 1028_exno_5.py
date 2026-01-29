import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
# Load dataset
data = load_iris()
X = data.data[:, :2]   # Set 1 features
Y = data.data[:, 2:]   # Set 2 features

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)
cca = CCA(n_components=2)
X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)
canonical_df = pd.DataFrame(
    np.corrcoef(X_c.T, Y_c.T)[:2, 2:],
    columns=["Y_Canon1", "Y_Canon2"],
    index=["X_Canon1", "X_Canon2"]
)
plt.figure(figsize=(6,4))
sns.heatmap(
    canonical_df,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Heatmap of Canonical Covariates")
plt.show()
print("Canonical Correlation Coefficients:\n", canonical_df)