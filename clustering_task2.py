import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/sarjodas/Desktop/data_edited_task2.csv')

data_matrix = data.as_matrix()
group = data_matrix[:,-2]
data_clustering = data_matrix[:, 3:-3]

kmeans = KMeans(n_clusters=8, random_state=None, max_iter=1000)
kmeans_pred = kmeans.fit_predict(data_clustering)
kmean_label_group={}
for i in range(kmeans_pred.shape[0]):
    if not kmeans_pred[i] in kmean_label_group:
        kmean_label_group[kmeans_pred[i]] = [group[i]]
    else:
        kmean_label_group[kmeans_pred[i]].append(group[i])


print('KMeans Grouping:', kmean_label_group)


fig1, ax1=plt.subplots(1,2)

pd.DataFrame(data=kmean_label_group[0], columns=['grouping_appl'])['grouping_appl'].value_counts().plot(ax=ax1[0], kind='bar')
ax1[0].set_title('Cluster 0')

pd.DataFrame(data=kmean_label_group[1], columns=['grouping_appl'])['grouping_appl'].value_counts().plot(ax=ax1[1], kind='bar')
ax1[1].set_title('Cluster 1')

fig2, ax2=plt.subplots(1,2)

pd.DataFrame(data=kmean_label_group[2], columns=['grouping_appl'])['grouping_appl'].value_counts().plot(ax=ax2[0], kind='bar')
ax2[0].set_title('Cluster 2')

pd.DataFrame(data=kmean_label_group[3], columns=['grouping_appl'])['grouping_appl'].value_counts().plot(ax=ax2[1], kind='bar')
ax2[1].set_title('Cluster 3')

fig3, ax3=plt.subplots(1,2)

pd.DataFrame(data=kmean_label_group[4], columns=['grouping_appl'])['grouping_appl'].value_counts().plot(ax=ax3[0], kind='bar')
ax3[0].set_title('Cluster 4')

pd.DataFrame(data=kmean_label_group[5], columns=['grouping_appl'])['grouping_appl'].value_counts().plot(ax=ax3[1], kind='bar')
ax3[1].set_title('Cluster 5')

fig4, ax4=plt.subplots(1,2)

pd.DataFrame(data=kmean_label_group[6], columns=['grouping_appl'])['grouping_appl'].value_counts().plot(ax=ax4[0], kind='bar')
ax4[0].set_title('Cluster 6')

pd.DataFrame(data=kmean_label_group[7], columns=['grouping_appl'])['grouping_appl'].value_counts().plot(ax=ax4[1], kind='bar')
ax4[1].set_title('Cluster 7')


plt.show()

