import numpy as np
import torch

class MergeLayer0(torch.nn.Module): # 小型的MLP
  def __init__(self, dim1, dim2, dim3, dim4):# (n_node_features, n_node_features, n_node_features, 1)
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim4)
    self.fc2 = torch.nn.Linear(dim2, dim4)
    self.act = torch.nn.ReLU()

  def forward(self, x1, x2):
    h1 = self.fc1(x1)
    h2 = self.fc2(x2)
    # h3 = h1 + h2
    # x = torch.cat([x1, x2], dim=1)  # x (n_node_features, 2*n_node_features)
    h = self.act(h2 + h1)
    return h



class MergeLayer(torch.nn.Module): # 小型的MLP
  def __init__(self, dim1, dim2, dim3, dim4):# (n_node_features, n_node_features, n_node_features, 1)
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()
    # self.act1 = torch.sigmoid(dim4, dim4)

    torch.nn.init.xavier_normal_(self.fc1.weight) # 服从高斯分布的初始化
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2, dim = 1):
    x = torch.cat([x1, x2], dim=dim) # x (n_node_features, 2*n_node_features)
    h = self.act(self.fc1(x))
    y = self.fc2(h)
    return y # h (n_node_features, n_node_features), 最后结果(n_node_features, 1)

class MergeLayer_mix(torch.nn.Module): # 小型的MLP
  def __init__(self, dim1, dim2, dim3):# (n_node_features, n_node_features, n_node_features, 1)
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim3)#, bias=False
    self.fc2 = torch.nn.Linear(dim2, dim3)#, bias=False
    self.act = torch.nn.LeakyReLU(negative_slope=0.5, inplace=False)
    # self.act1 = torch.sigmoid(dim4, dim4)

    torch.nn.init.xavier_normal_(self.fc1.weight) # 服从高斯分布的初始化
    torch.nn.init.xavier_normal_(self.fc2.weight)


  def forward(self, x1, x2, dim = 1):
    x1 = self.fc1(x1) # x (n_node_features, 2*n_node_features)
    x2 = self.fc2(x2)
    h = self.act(x1+x2)
    return h

class MergeLayer_Mix(torch.nn.Module): # 小型的MLP
  def __init__(self, dim1, dim2, dim3):# (n_node_features, n_node_features, n_node_features, 1)
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim3)
    self.fc2 = torch.nn.Linear(dim1, dim3)
    self.act = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
    # self.act1 = torch.sigmoid(dim4, dim4)

    torch.nn.init.xavier_normal_(self.fc1.weight) # 服从高斯分布的初始化
    torch.nn.init.xavier_normal_(self.fc2.weight)


  def forward(self, x1, x2):
    x1 = self.fc1(x1) # x (n_node_features, 2*n_node_features)
    x2 = self.fc2(x2)
    h = self.act(x1+x2)
    return h

class Merge_0(torch.nn.Module): # 小型的MLP
  def __init__(self, dim1, dim2, dim4):# (n_node_features, n_node_features, n_node_features, 1)
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim4)
    self.fc2 = torch.nn.Linear(dim2, dim4)
    # self.fc3 = torch.nn.Linear(dim4, dim5)
    self.fc = torch.nn.GRUCell(input_size=dim4,
                                     hidden_size=dim4)
    self.act_0 = torch.nn.Tanh()
    self.act = torch.nn.ReLU()
    # self.act1 = torch.sigmoid(dim4, dim4)

    torch.nn.init.xavier_normal_(self.fc1.weight) # 服从高斯分布的初始化
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, gene, u):
    # g = self.act(self.fc1(gene))
    # u = self.act(self.fc2(u))
    h = self.fc(gene, u)
    # x = torch.cat([, x3], dim=1) # x (n_node_features, 2*n_node_features)
    # h_u = self.act_0(self.fc1(x))
    return h


class MergeLayer_1(torch.nn.Module): # 小型的MLP
  def __init__(self, dim1, dim2, dim3):# (n_node_features, n_node_features, n_node_features, 1)
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim2)
    self.fc2 = torch.nn.Linear(dim2, dim3)
    self.act = torch.nn.ReLU()
    # self.act1 = torch.sigmoid(dim4, dim4)

    torch.nn.init.xavier_normal_(self.fc1.weight) # 服从高斯分布的初始化
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1):
    # x = torch.cat([x1, x2], dim=1) # x (n_node_features, 2*n_node_features)
    h = self.act(self.fc1(x1))
    y = self.act(self.fc2(h))
    return y


class MergeLayer_t(torch.nn.Module): # 小型的MLP
  def __init__(self, dim1, dim2, dim3, dim4):# (n_node_features, n_node_features, n_node_features, 1)
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 ,dim1)
    self.fc2 = torch.nn.Linear(dim1, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight) # 服从高斯分布的初始化
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    h_1 = self.fc1(x1)
    h_2 = self.fc1(x2).t()
    h = torch.diag((torch.mm(self.act(h_1), self.act(h_2))))
    y = h.sigmoid()
    return h

class MergeLayer_0(torch.nn.Module): # 小型的MLP
  def __init__(self, dim1, dim2, dim3, dim4):# (n_node_features, n_node_features, n_node_features, 1)
    super().__init__()
    self.fc0 = torch.nn.Linear(dim1, dim3)
    self.fc1 = torch.nn.Linear(dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc0.weight) # 服从高斯分布的初始化
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    h_0 = self.fc0(x1)
    h_1 = self.fc1(x2)# x (n_node_features, 2*n_node_features)
    h = self.act(h_0*h_1)
    return self.fc2(h)


class MLP(torch.nn.Module): # 多层感知机
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object): # max_round = 5(default)
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    min_max = 20
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = max(self.epoch_count, min_max)
      if max(self.epoch_count, min_max) == min_max and self.epoch_count != min_max:
        self.last_best = 0.1
    elif self.epoch_count >= min_max:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object): #
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size) # 生成size大小的在0-len(self.src_list)内随机取直的张量
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:

      src_index = self.random_state.randint(0, len(self.src_list), size) # 返回[0, len(self.src_list)-1] 的随机整数
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx # 确定最大节点编号

  adj_list = [[] for _ in range(max_node_idx + 1)] # 本节点的邻接矩阵
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time
      # 获得cut time前的目标节点的所有交互得到邻节点、交互的边、时间戳
      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)
          # 采样
          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
          # cut time前记录的所有邻节点经过采样后，用作目标节点嵌入的学习，即即使目标节点在早于截止时间前便已经停止更新，仍通过其过去交互过的邻节点的现状对其做嵌入的学习
    return neighbors, edge_idxs, edge_times
