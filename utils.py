import torch
import torch.nn as nn
import numpy as np

'''
Radius NMS function to rmove points that belong to the same box

boxes: a 2d numpy array of shape (num_boxes,2)
radius: float representing the radius of a cone

returns: a 2d numpy array of shape (reduced_num_boxes,2)
'''
def radius_nms_cp(boxes,radius):
  final_cones_idx = cp.arange(0,boxes.shape[0])
  boxes = cp.array(boxes)
  num_boxes = boxes.shape[0]

  for bi in range(num_boxes):
    if bi >= final_cones_idx.shape[0]:
      break
    b1_idx = final_cones_idx[bi]
    b = boxes[b1_idx].reshape(1,2)
    diff = boxes[final_cones_idx]-b
    diff_sq = cp.sum(diff*diff,axis=1)
    dist = cp.sqrt(diff_sq)
    final_cones_idx = final_cones_idx[cp.where(dist>radius)]
    arr_idx = cp.array(np.array([b1_idx]),dtype=cp.int)
    final_cones_idx = cp.concatenate((final_cones_idx,arr_idx),axis=0)
  return cp.asnumpy(final_cones_idx)



'''
K-means implementation used from https://github.com/subhadarship/kmeans_pytorch.git
'''
def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(X,num_clusters,cluster_centers = [],tol=1e-4,device=torch.device('cpu'),iters=1):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)
    if num_clusters==1:
      return _,torch.mean(X,dim=0).reshape(1,2)

    # initialize
    if type(cluster_centers) == list: #ToDo: make this less annoyingly weird
        initial_state = initialize(X, num_clusters)
    else:
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    for i in range(iters):

      dis = pairwise_distance(X, initial_state)
      choice_cluster = torch.argmin(dis, dim=1)

      initial_state_pre = initial_state.clone()

      for index in range(num_clusters):
          selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

          selected = torch.index_select(X, 0, selected)

          initial_state[index] = selected.mean(dim=0)

      center_shift = torch.sum(
          torch.sqrt(
              torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
          ))
      if center_shift ** 2 < tol:
          break

    return choice_cluster.cpu(), initial_state.cpu()

def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis
