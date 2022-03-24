import csv
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
def load_data(filepath):
  with open(filepath) as f:
    list_dict = [{k: str(v) for k, v in row.items()}
      for row in csv.DictReader(f, skipinitialspace=True)]
  return list_dict

def calc_features(row):
  feature_vec = np.array([int(row['Attack']), int(row['Sp. Atk']), int(row['Speed']), int(row['Defense']), int(row['Sp. Def']), int(row['HP'])], dtype= 'int64')
  return feature_vec

def hac(features):
  n = len(features)
  final_clusters = np.array([[0.0, 0.0, 0.0, 0.0]]*(n-1))
  cluster_distance = np.array([[0.0]*(n)]*(n))
  cluster_dict = {}

  for i in range(n):
    cluster_dict[i] = [i]

  for row in range(len(cluster_distance)):
    for column in range(len(cluster_distance[0])):
      if row == column:
        cluster_distance[row][column] = np.nan
        cluster_distance[row][column] = np.nan
      else:
        cluster_distance[row][column] = np.linalg.norm(features[row] - features[column])

  for i in range(len(final_clusters)):
    min_arrays = np.where(cluster_distance == np.nanmin(cluster_distance))
    min_coordinates = list(zip(min_arrays[0], min_arrays[1]))

    for p in range(len(min_coordinates)):
      min_coordinates[p] = sorted(min_coordinates[p])
    min_coordinates = [list(tupl) for tupl in {tuple(item) for item in min_coordinates }]

    if len(min_coordinates) > 1:
      pair = tiebreaker(min_coordinates)
    else:
      pair = min_coordinates[0]

    final_clusters[i][0] = min(pair[0], pair[1])
    final_clusters[i][1] = max(pair[0],pair[1])
    final_clusters[i][2] = cluster_distance[pair[0]][pair[1]]
    new_index = n + i
    cluster_dict[new_index] = cluster_dict[pair[0]] + cluster_dict[pair[1]]
    del cluster_dict[pair[0]]
    del cluster_dict[pair[1]]
    final_clusters[i][3] = len(cluster_dict[new_index])

    #adding new row for new cluster
    cluster_distance = np.pad(cluster_distance,(0,1))
  
    #updating distance to new clusters
    for c in range(len(cluster_distance)):
      if c in cluster_dict.keys():
        cluster_distance[c][new_index] = max(cluster_distance[pair[0]][c], cluster_distance[pair[1]][c])
        cluster_distance[new_index][c] = cluster_distance[c][new_index]
      else:
        cluster_distance[c][new_index] = np.nan
        cluster_distance[new_index][c] = np.nan
    
    #clearing old cluster row and columns by setting to NaN
    cluster_distance[pair[0]] = [np.nan]*(new_index+1)
    cluster_distance[:, pair[0]] = [np.nan]*(new_index+1)
    cluster_distance[pair[1]] = [np.nan]*(new_index+1)
    cluster_distance[:, pair[1]] = [np.nan]*(new_index+1)
  return final_clusters

def tiebreaker(cluster_pairs):
  cluster_pairs.sort(key=lambda x: x[0])
  if cluster_pairs[0][0] < cluster_pairs[1][0]:
    return cluster_pairs[0]
  else:
    x1 = cluster_pairs[0][0]
    cluster_pairs.sort(key=lambda x: x[1])
    for p in cluster_pairs:
      if p[0] == x1:
        return p

def imshow_hac(Z):
  dn = hierarchy.dendrogram(Z)
  plt.show()
