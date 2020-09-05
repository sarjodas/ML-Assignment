import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/sarjodas/Desktop/data_edited_task2.csv')
data_matrix = data.as_matrix()
data_clustering = data_matrix[:, 3:-3]

stdScaler = StandardScaler()
data_scaled_clustering = stdScaler.fit_transform(data_clustering)
clusters = range(2,30)
sq_error = []
number_cluster = np.empty(28)

print('Shape of the scaled data')
print(data_scaled_clustering.shape)
for i in clusters:
    kmeans = KMeans(n_clusters=i, random_state=None, max_iter=1000)
    kmeans_labels = kmeans.fit_predict(data_clustering)
    sq_error.append(kmeans.inertia_)
    average_silhouette = silhouette_score(data_clustering, kmeans_labels)
    print('Average Silhouette for cluster {0} is  {1}'.format(i, average_silhouette))

figure_elbow = plt.figure(figsize=(10, 5))
plt.plot(clusters, sq_error)
plt.grid(True)
plt.xlabel('Number of Clusters')
plt.ylabel('Square Error within cluster')
plt.title('Elbow Curve Plot')


plt.show()