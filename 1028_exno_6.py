
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Sample Dataset
# -------------------------------
data = {
    'Age': [22, 25, 47, 52, 46],
    'Salary': [25000, 32000, 47000, 52000, 46000],
    'Experience': [1, 2, 10, 15, 9],
    'Purchased': [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# -------------------------------
# 1. FEATURE SELECTION
# -------------------------------
print("\n--- Feature Selection ---")

X = df[['Age', 'Salary', 'Experience']]
y = df['Purchased']

# Select top 2 features using Chi-Square
selector = SelectKBest(score_func=chi2, k=2)
X_selected = selector.fit_transform(X, y)

print("Selected Feature Indices:", selector.get_support(indices=True))
print("Selected Features:\n", X_selected)

# -------------------------------
# 2. FEATURE TRANSFORMATION
# -------------------------------
print("\n--- Feature Transformation ---")

# (a) Normalization (Min-Max Scaling)
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)
print("Normalized Data:\n", X_normalized)

# (b) Standardization (Z-score Scaling)
standard_scaler = StandardScaler()
X_standardized = standard_scaler.fit_transform(X)
print("Standardized Data:\n", X_standardized)

# (c) Encoding Categorical Feature
cat_data = {'City': ['Chennai', 'Delhi', 'Mumbai', 'Delhi']}
df_cat = pd.DataFrame(cat_data)

label_encoder = LabelEncoder()
df_cat['City_Encoded'] = label_encoder.fit_transform(df_cat['City'])

print("Encoded Categorical Data:\n", df_cat)

# -------------------------------
# 3. FEATURE EXTRACTION
# -------------------------------
print("\n--- Feature Extraction ---")

# (a) PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)

print("PCA Result:\n", X_pca)

# (b) Text Feature Extraction using TF-IDF
documents = [
    "Machine learning is powerful",
    "Feature engineering is important",
    "Machine learning uses features"
]

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(documents)

print("TF-IDF Feature Names:\n", tfidf.get_feature_names_out())
print("TF-IDF Matrix:\n", X_tfidf.toarray())
