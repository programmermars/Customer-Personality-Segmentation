import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D  # Explicit import for 3D plotting

# --- 1. Load Cleaned Data ---
# Ensure the preprocessing script has been executed and the file exists
file_path = "customer_dataset/cleaned_customer_data.csv"
try:
    df = pd.read_csv(file_path)
    print(f"✅ Successfully loaded cleaned data. Total samples: {len(df)}")
except FileNotFoundError:
    print("❌ Error: 'cleaned_customer_data.csv' not found. Please run the preprocessing script first!")
    exit()

# --- 2. Prepare Clustering Features ---
# Select core business metrics for clustering
features = ["Income", "Age", "Spent", "Customer_For", "Children", "Family_Size", "Recency"]

# Data Standardization - Critical for K-Means performance
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# --- 3. PCA Dimensionality Reduction ---
# Reduce 7 features to 3 dimensions for visualization and noise reduction
pca = PCA(n_components=3)
pca_data = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(pca_data, columns=["col1", "col2", "col3"])
print("✅ PCA Dimensionality Reduction complete.")

# --- 4. K-Means Clustering (K=4) ---
# Using K=4 based on previous Elbow Method analysis
print("Executing K-Means Clustering (K=4)...")

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(pca_df)
pca_df["Cluster"] = df["Cluster"]

# Calculate Silhouette Score - Validates the separation distance between clusters
score = silhouette_score(pca_df[["col1", "col2", "col3"]], df["Cluster"])
print(f"🚀 Clustering complete! Silhouette Score: {score:.4f}")

# --- 5. Visualization: 3D Cluster Plot ---
# Used for README demonstration
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define color mapping for clusters
colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}

ax.scatter(pca_df['col1'], pca_df['col2'], pca_df['col3'], 
           c=pca_df['Cluster'].map(colors), s=40, alpha=0.6)

ax.set_title(f'3D Customer Clusters (K=4)\nSilhouette Score: {score:.2f}')
ax.set_xlabel('PCA Dimension 1 (Income/Spent)')
ax.set_ylabel('PCA Dimension 2 (Age/Tenure)')
ax.set_zlabel('PCA Dimension 3 (Family Size)')

# Add Legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}', 
                          markerfacecolor=c, markersize=10) for i, c in colors.items()]
ax.legend(handles=legend_elements, title="Customer Segments")

plt.tight_layout()
plt.savefig("customer_dataset/3d_cluster_plot.png")
print("💾 3D Cluster Plot saved as '3d_cluster_plot.png'")
# plt.show() # Uncomment to display window

# --- 6. Visualization: Business Insights Scatter Plot (Income vs Spent) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Income', y='Spent', hue='Cluster', palette=['red', 'green', 'blue', 'yellow'])
plt.title('Customer Segments: Income vs Total Spending')
plt.xlabel('Annual Income ($)')
plt.ylabel('Total Spent ($)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("customer_dataset/income_vs_spent.png")
print("💾 Business Insights Chart saved as 'income_vs_spent.png'")
# plt.show() # Uncomment to display window

# --- 7. Generate Marketing Strategy Report ---
# Calculate mean values for each cluster to identify characteristics
summary = df.groupby("Cluster")[["Income", "Spent", "Age", "Children"]].mean().reset_index()
print("\n📊 Cluster Profiles (Mean Values):")
print(summary)

# Save the final labeled dataset for Power BI or further analysis
df.to_csv("customer_dataset/segmented_customers.csv", index=False)
print("💾 Final labeled dataset saved as 'segmented_customers.csv'")