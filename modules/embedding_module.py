import torch
from torch import nn
import numpy as np
import math

from model.temporal_attention import TemporalAttentionLayer
from utils.utils import MergeLayer_1

# 只用于初始化各类信息
class EmbeddingModule(nn.Module):
  def __init__(self, node_features, node_gene, edge_features, memory, neighbor_finder, time_encoder,  n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout, Sem):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.Sem = Sem
    if Sem == True:
        self.node_gene=node_gene
        self.node_gene = torch.from_numpy(semantic_feature.astype(np.float32)).to(device)
        self.n_gene = self.node_gene.shape[1]
        self.n_node_gene = self.node_gene.shape[0]
    else:
        self.node_gene = None
        self.n_node_gene = 0
        self.n_gene = 0
    self.edge_features = edge_features
    # self.memory = memory
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    # self.time_encoder_n = time_encoder_n
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device
    # the parameters for Wr,pe is adjustable
    self.Wr = nn.Sequential(
        nn.Linear(self.n_gene, 100, bias=True),
        # nn.Dropout(p=0.1),
        nn.GELU(),
        # nn.LeakyReLU(),
        nn.Linear(100, 200, bias=True),
        # nn.Dropout(p=0.4)
    )
    self.pe = nn.Sequential(
        nn.Linear(400, 200, bias=False),
        # nn.Dropout(p=0.1),
        nn.GELU(),
        nn.Linear(200, 400, bias=False),
        # nn.Dropout(p=0.1)
        # nn.GELU(),
    )
    #nn.init.normal_(self.Wr[0].weight.data, mean=0, std=100 )
    #nn.init.normal_(self.Wr[0].bias.data, mean=0, std=100)
    nn.init.normal_(self.Wr[2].weight.data, mean=0, std= 4 ** -2 )
    # nn.init.normal_(self.Wr[2].bias.data, mean=0, std=100 ** -2)
    # nn.init.uniform_(self.Wr[3].bias.data, -0 * np.pi, 0.5 * np.pi)
    # nn.init.normal_(self.pe[0].weight.data, mean=0, std=500)
    # nn.init.normal_(self.pe[2].weight.data, mean=0, std=500)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    pass #


class IdentityEmbedding(EmbeddingModule): # 直接使用memory
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule): # 时间映射
  def __init__(self, node_features, node_gene, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1, Sem=True):
    super(TimeEmbedding, self).__init__(node_features, node_gene, edge_features, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout, Sem=Sem)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, node_gene, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, Sem=True):
    super(GraphEmbedding, self).__init__(node_features, node_gene, edge_features, memory,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout, Sem)

    self.use_memory = use_memory
    self.device = device
    if self.Sem == True:
        self.Merge = MergeLayer_1(self.n_gene, 172, 50)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

    assert (n_layers >= 0) # default, n_layers=2
    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1) # (row_size, 1, line_size)

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))
    temp = self.node_gene[source_nodes_torch, :]
    # temp = 0
    source_node_features = self.node_features[source_nodes_torch, :] #

    if self.use_memory: # 默认None
      source_node_features = memory[source_nodes, :] + source_node_features

    if n_layers == 0:
      return source_node_features
    else:
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
      temp_0 = self.node_gene[neighbors_torch, :]
      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times # 时间差

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten() # 以一个大列表的形式存

      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   # np.repeat(timestamps_temp, n_neighbors),
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1) # [batch ,n_neighbor, feature]
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      mask = neighbors_torch == 0
      if self.Sem == True:
          S_s = self.mix_intention_sum(source_nodes_torch)
          S_d = self.mix_intention_sum_d(neighbors_torch)
      source_node_features =  torch.cat([source_node_features,S_s], dim = 1)# torch.cat([self.node_features[source_nodes_torch, :], temp], 1)
      neighbor_embeddings =  torch.cat([neighbor_embeddings,S_d], dim = 2)
      # source_node_features = source_node_features+S_s  # torch.cat([self.node_features[source_nodes_torch, :], temp], 1)
      # neighbor_embeddings =  neighbor_embeddings+S_d

      source_embedding = self.aggregate(n_layers, source_node_features,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)

      return source_embedding

  def mix_intention_sum(self, destination_nodes):
      # des_nodes_torch = torch.from_numpy(destination_nodes).long().to(self.device)
      temp = self.node_gene[destination_nodes, :]
      max = torch.max(temp, 2)[0]
      min = torch.min(temp, 2)[0]
      # mask = np.array(torch.eq(max, min).cpu())
      temp_mask = torch.sign(abs(max) + abs(min)).unsqueeze(dim = 2).repeat(1,1,50)
      temp_mask_0 = torch.sign(torch.sum(temp_mask, dim = 1))
      temp_0 = torch.sum(self.Merge(temp), dim=1).mul(temp_mask_0)#
      # destination_memory = temp_0
      cosines = torch.cos(self.Wr(temp_0))
      sines = torch.sin(self.Wr(temp_0))
      F = 1 / np.sqrt(400) * torch.cat([cosines, sines], dim = -1)
      # F = 1 / np.sqrt(100) * cosines
      # temp_0 = F.mul(temp_mask)
      # temp_0 = self.Merge(temp).mul(temp_mask)
      destination_memory = self.pe(F)##.mul(temp_mask_0)self.pe()
      # a = F.cpu().detach().numpy()
      return destination_memory

  def mix_intention_sum_d(self, destination_nodes):
      # des_nodes_torch = torch.from_numpy(destination_nodes).long().to(self.device)
      temp = self.node_gene[destination_nodes, :]
      max = torch.max(temp, 3)[0]
      min = torch.min(temp, 3)[0]
      # mask = np.array(torch.eq(max, min).cpu())
      temp_mask = torch.sign(abs(max) + abs(min)).unsqueeze(dim = 3).repeat(1,1,1,50)
      temp_mask_0 = torch.sign(torch.sum(temp_mask, dim=2))
      temp_0 = torch.sum(self.Merge(temp), dim=2).mul(temp_mask_0)#
      # destination_memory = temp_0
      cosines = torch.cos(self.Wr(temp_0))
      sines = torch.sin(self.Wr(temp_0))
      F = 1 / np.sqrt(400) * torch.cat([cosines, sines], dim = -1)
      # F = 1 / np.sqrt(100) * cosines
      # temp_0 = F.mul(temp_mask)
      # temp_0 = self.Merge(temp).mul(temp_mask)
      destination_memory = self.pe(F)##.mul(temp_mask_0)self.pe()
      return destination_memory

  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    return None


class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, node_gene, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, Sem=True):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            node_gene=node_gene,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder,# time_encoder_n=time_encoder_n,
                                            n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory, Sem=Sem)
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                         n_edge_features, embedding_dimension)
                                         for _ in range(n_layers)])
    self.linear_2 = torch.nn.ModuleList(
      [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                       embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                   dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

    source_features = torch.cat([source_node_features,
                                 source_nodes_time_embedding.squeeze()], dim=1)
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, node_gene, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, Sem=True):
    super(GraphAttentionEmbedding, self).__init__(node_features, node_gene, edge_features, memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory, Sem)
    if Sem == True:
        n_gene_embed = 400
    else:
        n_gene_embed = 0
    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      n_gene = n_gene_embed,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout, # 0.1
      output_dimension=n_node_features)
      for _ in range(n_layers)]) # 根据layer层数建立模型

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models[n_layer - 1] # 获得第l层的模型

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask) # output, attn(α)

    return source_embedding # 输出embedding


def get_embedding_module(module_type, node_features, node_gene, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True, Sem=True):
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                   node_gene=node_gene,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,# time_encoder_n=time_encoder_n,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory, Sem=Sem)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                             node_gene=node_gene,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,# time_encoder_n=time_encoder_n,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory, Sem=Sem)

  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             node_gene=node_gene,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,# time_encoder_n=time_encoder_n,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout, Sem=Sem)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         node_gene=node_gene,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,# time_encoder_n=time_encoder_n,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors, Sem=Sem)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))


