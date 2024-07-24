import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

df = pd.read_csv('./data/raw/admission.csv')

# Separating features and target variable
X = df.drop(columns=['admit'])
y = df['admit']

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Creating a DataFrame with PCA results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['admit'] = y.values

df_pca.to_csv(os.path.join('data','processed','admission_pca.csv'), index=False)