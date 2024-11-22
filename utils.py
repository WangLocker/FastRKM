import pandas as pd
import numpy as np
# import torch
import random
from sklearn.cluster import KMeans

def setup_seed(seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def read_data(file_path):
    data = pd.read_csv(file_path)
    return data.values

def get_noised_data(data, noise_frac):
    # noise
    noise_size = int(noise_frac * len(data))
    noise = np.random.uniform(low=-5, high=5, size=(noise_size, data.shape[1]))
    noise_indices = list(range(len(data), len(data) + noise_size))

    data_with_noise = np.vstack((data, noise))
    num_noise_data = len(data_with_noise)
    indices = list(range(len(data_with_noise)))
    random.shuffle(indices)
    data_with_noise = data_with_noise[indices]

    shuffled_noise_indices = [indices.index(i) for i in noise_indices]

    return data_with_noise
def get_client_neighborhood(nr_points, all_pair_distances, l):
    radii = [0] * nr_points
    for p in range(nr_points):
        distances = np.array([all_pair_distances[p][q] for q in range(nr_points)])
        points_by_dist = np.argsort(distances)
        radii[p] = all_pair_distances[p][points_by_dist[l]]

    return radii


def get_center_assignments(nr_points, all_pair_distances, centers):
    closest = [-1] * nr_points
    second_closest = [-1] * nr_points
    dist_vec = [-1] * nr_points
    nr_centers = len(centers)
    for p in range(nr_points):
        # print("")
        # print("Clustering for point p = {}".format(p))
        center_distances = np.array([all_pair_distances[p][centers[c]] for c in range(nr_centers)])
        # print("Dist to centers = {}".format(center_distances))
        # List of centers INDICES sorted by increasing distance to p
        sorted_inds = np.argsort(center_distances)
        # print("Center indices sorted by inc distance: {}".format(sorted_inds))
        closest[p] = int(sorted_inds[0])
        dist_vec[p] = all_pair_distances[p][centers[closest[p]]]
        try:
            second_closest[p] = int(sorted_inds[1])
        except:
            second_closest[p] = None
    return closest, dist_vec, second_closest



# compute variance in each cluster
def cluster_variances(X, centroids, labels, nr_points, all_pair_distances, n_clusters):
    variances, largest_dist = [], []

    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        variance = np.var(distances)
        variances.append(variance)
        largest_dist.append(max(distances))
    return variances, largest_dist


def compute_sse(data, centroids):
    """
    Calculate K-means SSE (Sum of Squared Errors) using vectorized operations.

    Parameters:
    data (ndarray): 2D array of data points, shape (num_points, num_features)
    centroids (ndarray): 2D array of cluster centroids, shape (num_clusters, num_features)

    Returns:
    float: K-means SSE
    """
    # Expand dimensions of data and centroids to calculate distances between all points and all centroids
    # data[:, np.newaxis, :] has shape (num_points, 1, num_features)
    # centroids[np.newaxis, :, :] has shape (1, num_clusters, num_features)
    # The resulting diff has shape (num_points, num_clusters, num_features)
    diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]

    # Calculate squared distances between each data point and each centroid
    # np.sum(diff ** 2, axis=2) results in a matrix of shape (num_points, num_clusters)
    distances = np.sum(diff ** 2, axis=2)

    # Find the minimum squared distance for each data point to the nearest centroid
    min_distances = np.min(distances, axis=1)

    # SSE is the sum of all minimum squared distances
    sse = np.sum(min_distances)

    return sse
def compute_tilted_sse(X, centroids, labels, k, t, n_samples):
    distances_to_centroids = np.linalg.norm(X - centroids[labels], axis=1) ** 2
    phi = np.zeros((k,))
    for j in range(k):
        phi[j] = (np.logaddexp.reduce(t * distances_to_centroids*((labels == j).astype(int))) + np.log(1/n_samples))/t
    return sum(phi)

def exp_details(args):
    print('     Running ' + args.alg + 'on ' + args.dataset )
    print('\nParameter description')
    print(f'    Number of clusters : {args.num_clusters}')
    print(f'    Epoch size         : {args.num_epoch}')
    print(f'    Batch size         : {args.num_batch}')
    print(f'    Learning rate      : {args.lr}\n')
    print(f'    Maximum iterations : {args.maxIter}\n')

    return

def initialization(X, k, args):
    """
    Randomly initialize K-means cluster centroids and assign sample labels.

    Parameters:
    data: Input data, shape (n_samples, n_features)
    k: Number of clusters

    Returns:
    centroids: Initialized centroids
    labels: Labels for each sample
    """

    if args.init == 'random':
        # Randomly select k samples as initial centroids
        indices = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[indices]
        # Calculate the distance from each sample to each centroid and assign labels
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
    elif args.init == 'kmeans++':
        kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=1, max_iter=1000, tol=0.02).fit(X)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
    else:
        exit('Error: unrecognized initialization')





    return centroids, labels

def compute_tilted_sse_InEachCluster(X, centroids, labels, k, t):
    n_samples = X.shape[0]
    distances_to_centroids = np.linalg.norm(X - centroids[labels], axis=1) ** 2
    phi = np.zeros((k,))
    for j in range(k):
        phi[j] = (np.logaddexp.reduce(t * distances_to_centroids*((labels == j).astype(int))) + np.log(1/n_samples))/t
    return phi


