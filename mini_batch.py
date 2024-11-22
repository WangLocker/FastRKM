import numpy as np
import pandas as pd


def mini_batch_kmeans(X, k, batch_size=100, max_iters=100):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    batch_count = n_samples // batch_size
    losses = []

    for _ in range(max_iters):
        batch_indices = np.random.choice(n_samples, batch_size, replace=True)

        for i in range(batch_count):
            batch = X[batch_indices]
            distances = np.linalg.norm(batch[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([batch[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centroids[j] for j in range(k)])

            # 更新聚类中心
            learning_rate = 0.05
            centroids = (1 - learning_rate) * centroids + learning_rate * new_centroids

        # 计算每次迭代的损失值
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        loss = np.sum((X - centroids[labels]) ** 2)
        losses.append(loss)

    return centroids, labels, losses


# 从CSV文件中读取数据
def read_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    return data.values

# 计算每个簇中样本点到中心的距离方差
def cluster_variances(X, centroids, labels):
    variances = []
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        variance = np.var(distances)
        variances.append(variance)
    return variances
# 设置CSV文件路径
csv_file_path = 'data/CleanData/bank.csv'

# 从CSV文件中读取数据
data = read_data_from_csv(csv_file_path)

# 调用Mini-Batch K-Means算法
k = 3
centroids, labels, losses = mini_batch_kmeans(data, k)

# 计算每个簇中样本点到中心的距离方差
variances = cluster_variances(data, centroids, labels)


# 打印结果
print("Mini-Batch K-Means聚类中心:\n", centroids)
print("每个样本所属的簇:\n", labels)

# 打印每次迭代的损失值
print("每次迭代的损失值:\n", losses)
# 打印每个簇中样本点到中心的距离方差
for i, variance in enumerate(variances):
    print(f"Cluster {i+1} Variance: {variance}")