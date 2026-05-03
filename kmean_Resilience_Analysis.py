import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
# Using the merged dataset which contains both mental health scores and lifestyle factors
df = pd.read_csv('mhp_dataset.csv')

# 2. Prepare data for Clustering
# Focus on Stress, Anxiety, and Depression z-scores to create mental health profiles
features = ['stress_z_score', 'anxiety_z_score', 'depression_z_score']
data_for_clustering = df.dropna(subset=features)

# Standardize features to ensure equal weighting in the KMeans algorithm
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_clustering[features])

# 3. Execute KMeans Clustering (k=3 based on initial exploratory analysis)
# Grouping students into 'Low Distress', 'Moderate Distress', and 'High Distress'
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data_for_clustering['Cluster'] = kmeans.fit_predict(scaled_data)

# 4. Resilience Analysis: Identifying High-Distress Top Performers
# Define 'Resilient Students' as those in Cluster 0 (High Distress) with top-tier CGPA
resilient_mask = (data_for_clustering['Cluster'] == 0) & (data_for_clustering['cgpa'] == '3.80-4.00')
data_for_clustering['Status'] = 'Normal'
data_for_clustering.loc[resilient_mask, 'Status'] = 'Resilient'

# 5. Comparative Visualization: Lifestyle Impact
# Check if lifestyle columns (study_hours_per_day, sleep_hours) exist for analysis
lifestyle_metrics = ['study_hours_per_day', 'sleep_hours']
available_metrics = [m for m in lifestyle_metrics if m in data_for_clustering.columns]

if available_metrics:
    # Compare lifestyle habits between 'Resilient' students and the rest of the High Distress group
    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(available_metrics, 1):
        plt.subplot(1, len(available_metrics), i)
        sns.boxplot(data=data_for_clustering[data_for_clustering['Cluster'] == 0],
                    x='Status', y=metric, palette='Set2')
        plt.title(f'{metric.replace("_", " ").title()} in High Distress Group')

    plt.tight_layout()
    plt.savefig('lifestyle_comparison.png')

# 6. Mental Health Profile Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_for_clustering, x='stress_z_score', y='depression_z_score',
                hue='Cluster', palette='viridis', alpha=0.6)
plt.title('Mental Health Clusters: Identifying Vulnerable vs Resilient Profiles')
plt.savefig('final_clusters_profile.png')

# Output cluster characteristics for the report's "4.3 K-Means Clustering" section
print("Cluster Profiles (Mean Z-Scores):")
print(data_for_clustering.groupby('Cluster')[features].mean())