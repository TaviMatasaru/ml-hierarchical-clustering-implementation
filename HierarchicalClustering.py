import numpy as np
import pandas as pd

# Loading the dataset
# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
# data = pd.read_csv(url)

# I have loaded it locally for ease of work
path = ""
data = pd.read_csv(path)

print(f'The initial number of rows in the dataset is: {len(data)}')
print('\nThe first rows of the dataset are:')
print(data.head(10))


# 1. Preprocessing

# a. Description of the dataset
print('Description of the dataset:')
print(data.info())


# b. Removing rows with NaN
data = data.dropna()
print("\nNumber of rows after removing NaN:", len(data))
print(data.head(10))
print("\nNumber of NaN values for each attribute:")
print(data.isna().sum())


# Resetting the index after deletion
data.reset_index(drop=True, inplace=True)


# c. Calculation of the mean and variance
print("\nMean for each numeric attribute:")
print(data.mean(numeric_only=True))
print("\nVariance for each numeric attribute:")
print(data.var(numeric_only=True))

# d. Removal of the target attribute 'species'
data.drop('species', axis=1, inplace=True)


# 2. Distance Calculation
# a. Conversion of the 'island' and 'sex' attributes to numeric attributes
mapping = {
    'island': {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2},
    'sex': {'Female': 0, 'Male': 1}
}
data['island'] = data['island'].map(mapping['island'])
data['sex'] = data['sex'].map(mapping['sex'])
print(f'\n{data.dtypes}')


# b. distance_points function that uses the Minkowski distance formula
def distance_points(point1, point2, p):
    return np.sum(np.abs(point1 - point2) ** p) ** (1/p)


# c. calculate_distance_matrix function creates the distance matrix using Euclidean distance by default
def calculate_distance_matrix(data, p=2):
    n = len(data)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            M[i, j] = distance_points(data.iloc[i], data.iloc[j], p)
    return M


# d. closest_points function finds the smallest distance between two clusters in the matrix
def closest_points(M):
    i, j = np.where(M == np.min(M[np.nonzero(M)]))
    return (i[0], j[0])

# 3. Hierarchical Clustering


# a. update_distance_matrix_single function
# Updates the distance matrix using single-linkage
def update_distance_matrix_single(M, i, j):

    n = len(M)
    for k in range(n):
        if k != i and k != j:
            M[i, k] = M[k, i] = min(M[i, k], M[j, k])
    M[j, :] = M[:, j] = np.inf  # Mark the merged cluster with infinity
    return M


# b. update_distance_matrix_complete function
# Updates the distance matrix using complete-linkage
def update_distance_matrix_complete(M, i, j):
    n = len(M)
    for k in range(n):
        if k != i and k != j:
            M[i, k] = M[k, i] = max(M[i, k], M[j, k])
    M[j, :] = M[:, j] = np.inf  # Mark the merged cluster with infinity
    return M


# c. update_distance_matrix_average function
# Updates the distance matrix using average-linkage
def update_distance_matrix_average(M, i, j):
    n = len(M)
    for k in range(n):
        if k != i and k != j:
            M[i, k] = M[k, i] = (M[i, k] + M[j, k]) / 2
    M[j, :] = M[:, j] = np.inf  # Mark the merged cluster with infinity
    return M


# d. calculate_dendogram_height function
# Calculates the dendrogram height of a cluster
def calculate_dendogram_height(M, i, j):
    return np.mean(M[i][M[i] != np.inf])


# e. calculate_dendogram_height_average function
# Calculates the dendrogram height of a cluster when using average-linkage
def calculate_dendogram_height_average(M, i, j):
    return np.mean(M[i][M[i] != np.inf])


# f. agglomerative_clustering function
# Performs hierarchical clustering
def agglomerative_clustering(data, nclusters, linkage, p=2):
    M = calculate_distance_matrix(data, p)
    dendogram_heights = {}
    points = np.arange(len(data))

    while len(np.unique(points)) > nclusters:
        i, j = closest_points(M)

        if linkage == 'single':
            M = update_distance_matrix_single(M, i, j)
        elif linkage == 'complete':
            M = update_distance_matrix_complete(M, i, j)
        elif linkage == 'average':
            M = update_distance_matrix_average(M, i, j)

        dendogram_height = calculate_dendogram_height(M, i, j)
        dendogram_heights[f'cluster_{i}'] = dendogram_height

        # Update the cluster membership
        points[points == j] = i

        # Adjust the distance matrix to no longer consider the absorbed cluster
        M[j, :] = np.inf
        M[:, j] = np.inf

    print(f'\nThe number of clusters is: {len(np.unique(points))}')

    return {
        "membership": points,
        "dendogram_heights": dendogram_heights
    }


# g. Example of use for average-linkage and 30 clusters
result = agglomerative_clustering(data, 30, 'average')
print(result)
