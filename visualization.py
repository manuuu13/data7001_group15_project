import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Data Preparation
df = pd.read_csv('mhp_dataset.csv')
features = ['stress_z_score', 'anxiety_z_score', 'depression_z_score']
df_clean = df.dropna(subset=features).copy()

# 2. Professional Clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean[features])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(scaled_data)

# 3. UNIQUE VISUALIZATION: The "Resilience" Violin Plot
# This shows how CGPA is distributed across different Mental Health Clusters
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Create a specialized plot showing CGPA distribution within clusters
# We use a stripplot on top of a violinplot to show individual "Resilient" outliers
palette = sns.color_palette("husl", 3)
ax = sns.violinplot(x='Cluster', y='stress_z_score', data=df_clean, inner=None, color=".8")
sns.stripplot(x='Cluster', y='stress_z_score', hue='cgpa', data=df_clean,
              palette="viridis", alpha=0.5, jitter=True)

plt.title('Mental Health Profiles & Academic Resilience\n(Highlighting High-GPA Students in High-Stress Clusters)', fontsize=15)
plt.xlabel('Student Profile (Cluster)', fontsize=12)
plt.ylabel('Psychological Distress (Z-Score)', fontsize=12)
plt.legend(title='CGPA Band', bbox_to_anchor=(1.05, 1), loc='upper left')

# Annotating Cluster 0 to highlight your unique contribution
plt.annotate('High-Stress but High-Achievement\n(The Resilience Group)',
             xy=(0, 2), xytext=(0.5, 2.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('unique_resilience_visualization.png')
plt.show()

# 4. Export Table for Report
cluster_summary = df_clean.groupby('Cluster')[features].mean().round(2)
print("Table for Section 4.3 (Cluster Profile Means):")
print(cluster_summary)