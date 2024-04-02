import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('resampled.csv')
X = df.iloc[:, :4]
y = df.iloc[:,4]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_reduced = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
last_column_of_df1 = df.iloc[:, -1]
target_df = pd.DataFrame(last_column_of_df1)
print(last_column_of_df1)
print(target_df)
X_reduced = pd.concat([X_reduced, target_df], axis=1)

X_reduced.to_csv('reduced_data.csv', index=False)
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

import matplotlib.pyplot as plt


plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of 4 Features to 2 Features')
plt.show()