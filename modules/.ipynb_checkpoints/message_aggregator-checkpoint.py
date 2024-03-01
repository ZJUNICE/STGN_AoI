from collections import defaultdict
import torch
import numpy as np
import torch.nn as nn
import math
from utils.utils import MergeLayer
from model.time_encoding import TimeEncode

class attention_node(nn.Module):
  def __init__(self, n_node_features):
    super(attention_node, self).__init__()
    self.d = n_node_features
    self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)
    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.d,
                                                   num_heads=2)

  def forward(self, messages, time_features, neighbors_padding_mask): # 原message中不包含时间信息？
    messages_unrolled = messages + time_features
    Q = torch.unsqueeze(messages_unrolled[-1]) # [1, 1, D]
    # out = torch.zeros_like(Q)
    # for item in messages:
    #   K = self.W_K(item)
    #   V = self.W_V(item)
    #   alpha = self.softmax(torch.mm(Q, K.transpose(0,1), dim=0)/math.sqrt(self.d)) # dim=-1
    #   out += alpha * V
    # output = self.fc(out)

    K = messages # [1, N, D]
    Q = Q.permute([1, 0, 2])
    K = K.permute([1, 0, 2])


    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
    neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False
    attn_output, _ = self.multi_head_target(query=Q, key=K, value=K,
                                                              key_padding_mask=neighbors_padding_mask)
    attn_output = attn_output.squeeze()
    attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    out = self.merger(attn_output, messages[-1])
    return out


    # src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)
    # query = torch.cat([src_node_features_unrolled, src_time_features], dim=2) # [B, 1, DF + DT] 拼接时间编码与原内容特征编码得到query （单个）
    # key = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2) # [B, N, DF + DE + DT] 拼接邻节点特征、边特征、邻节点时间编码得到key
    # key也作为value使用 edge_feature （N个）
    # query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features(DF+DT)] 对query tensor转置
    # key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features(DF+DE+DT)] 对key转置
    # invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True) # 确定需要mask处理的位置
    #
    # neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False # 得到mask矩阵
    #
    # attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,
    #                                                           key_padding_mask=neighbors_padding_mask) # 对mask后的key求attention
    # attn_output = attn_output.squeeze()
    # attn_output_weights = attn_output_weights.squeeze()
    # attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    # attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)
    # attn_output = self.merger(attn_output, src_node_features)


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device


  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""    
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []
    
    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps

# class GRUMessageAggregator(MessageAggregator):
#   def __init__(self, device):
#     super(GRUMessageAggregator, self).__init__(device)
#     self.message_updater = nn.GRUCell(input_size=100,
#                                  hidden_size=100)
#   def aggregate(self, node_ids, messages):
#     """Only keep the last message for each node"""
#     unique_node_ids = np.unique(node_ids)
#     unique_messages = []
#     unique_timestamps = []
#
#     to_update_node_ids = []
#     # print(message_dimension)
#     for node_id in unique_node_ids:
#       if len(messages[node_id]) > 0:
#         to_update_node_ids.append(node_id)
#         hidden_message = torch.zeros_like(messages[node_id][0])
#         for m in messages[node_id]:
#           message = m(0)
#           hidden_message = self.message_updater(hidden_message, message)
#         unique_messages.append(hidden_message)
#         unique_timestamps.append(messages[node_id][-1][1])
#
#     unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
#     unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []
#
#     return to_update_node_ids, unique_messages, unique_timestamps

class AttnMessageAggregator(MessageAggregator):
  def __init__(self, device, message_dimension):
    super(AttnMessageAggregator, self).__init__(device)
    self.n_head = 2
    self.feat_dim = message_dimension  # 一般等于n_neighbors_features
    self.time_encoder = TimeEncode(dimension=message_dimension)
    # self.n_layers = 2
    self.message_updater = nn.MultiheadAttention(embed_dim=self.feat_dim,
                                                   kdim=self.feat_dim,
                                                   vdim=self.feat_dim,
                                                   num_heads=2,
                                                   )

  def aggregate(self, node_ids, messages):
    unique_node_ids = np.unique(node_ids) # get the id of all unique node
    unique_messages = [] # a list to store the unique node message
    unique_timestamps = [] # a list to store the unique node message time

    to_update_node_ids = [] # update node id
    n_messages = 0

    for node_id in unique_node_ids:
      n_temp = len(messages[node_id])
      if n_temp > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        # time_embedding = self.time_encoder(torch.zeros_like(messages[node_id][-1][1]))
        edge_deltas = torch.stack(m[1] for m in messages[node_id]) - messages[node_id][-1][1]
        edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
        edge_time_embeddings = self.time_encoder(edge_deltas_torch)
        # node_torch = torch.stack([m[0] for m in messages[node_id]])
        neighbor_temp = torch.zeros((len(messages[node_id][0][0], 10))) # 只采用最近10次的事件作为attention的依据

        if n_temp>10:
          neighbor_temp = messages[node_id][-10:][0]
        else:

          neighbor_temp[-n_temp:] = messages[node_id][:][0]
        mask = neighbor_temp == 0
        aggregated_messages = self.message_updater(neighbor_temp, edge_time_embeddings, mask)

        unique_messages.append(aggregated_messages)
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps



def get_message_aggregator(aggregator_type, device, message_dimension):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  # elif aggregator_type == "gru":
  #   return GRUMessageAggregator(device=device)
  elif aggregator_type == "attn":
    return AttnMessageAggregator(device=device, message_dimension = message_dimension)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
