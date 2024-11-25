import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Create synthetic data
data = {
    'CustomerID': range(1, 101),
    'AnnualIncome': np.random.randint(20000, 120000, size=100),  # Annual Income in USD
    'SpendingScore': np.random.randint(1, 100, size=100)         # Spending Score from 1 to 100
}

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Data Preprocessing
# We will scale the data for better performance of K-means
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['AnnualIncome', 'SpendingScore']])

# Step 3: Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 5))
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(K)
plt.grid()
plt.show()

# Step 4: Fit K-means with the optimal number of clusters (let's say 4 for this example)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Step 5: Analyze the clusters
print(df.groupby('Cluster').mean())

# Step 6: Visualize the clusters
plt.figure(figsize=(10, 5))
plt.scatter(df['AnnualIncome'], df['SpendingScore'], c=df['Cluster'], cmap='viridis', marker='o')
centers = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers)  # Inverse transform to original scale
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()