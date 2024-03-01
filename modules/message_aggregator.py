from collections import defaultdict, Counter

import torch
import numpy as np
import torch.nn as nn
import time
from utils.utils import MergeLayer, MLP, MergeLayer0
from operator import itemgetter
import math
from model.time_encoding import TimeEncode


class AGE_MLP(torch.nn.Module): # 多层感知机
  def __init__(self, dim, drop=0.1):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 2*(dim-1))
    self.fc_2 = torch.nn.Linear(2*(dim-1), 1)
    # self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    # self.act_0 = torch.nn.Sigmoid()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)
    # torch.nn.init.xavier_normal_(self.fc_1.weight)  # 服从高斯分布的初始化
    # torch.nn.init.xavier_normal_(self.fc_2.weight)

  def forward(self, x):
    # x =  self.act(x)
    x_0 = self.act(self.fc_1(x))
    # h = self.act(self.fc_2(x))
    x_1 = self.dropout(x_0)
    out = self.act(self.fc_2(x_0))
    return out.squeeze(dim=1)#*x[:,-1]

class time_attn(nn.Module):
  def __init__(self, n_mem_dim, n_node_features, n_time_d):
    super(attention_node, self).__init__()
    self.mem_d = n_mem_dim
    self.multi_head_target = nn.MultiheadAttention(embed_dim= self.t,# embed_dim=self.d + self.t
                                                   kdim= self.t,
                                                   vdim= self.t,
                                                   num_heads=2)

  def forward(self, time_batch, time_mask): # 原message中不包含时间信息？

    messages_unrolled = time_batch.unsqueeze(dim = 1) # [118,10, D]
    source = messages_unrolled[:,-1,:]
    # source = messages.squeeze(dim = 1)[:,-1,:] #0824 changed 不考虑memory 直接用聚合的信息即可得到很好效果
    Q = source # [batch_size, neighbor, features] # [1, B, D]

    K = messages_unrolled.permute([1, 0, 2]) # [N, 200, D]
    invalid_neighborhood_mask = time_mask.all(dim=1, keepdim=True)
    time_mask[invalid_neighborhood_mask.squeeze(), 0] = False
    attn_output, _ = self.multi_head_target(query=Q, key=K, value=K,key_padding_mask=time_mask)
    attn_output = attn_output.squeeze()

    attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    out = self.merger(attn_output, source)
    return out # , attn_output_weights

class attention_node(nn.Module):
  def __init__(self, n_mem_dim, n_node_features, n_time_d, n_gene):
    super(attention_node, self).__init__()
    self.mem_d = n_mem_dim
    self.d = n_node_features
    self.t = n_time_d
    self.n_gene = n_gene
    self.merger_s = MergeLayer0(self.d , self.mem_d, self.mem_d, self.mem_d)
    # self.merger_s = MergeLayer0(self.d, self.mem_d, self.mem_d, self.mem_d)
    self.merger = MergeLayer(self.d + self.t, self.d, self.d, self.d)
    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.d + self.t,# embed_dim=self.d + self.t
                                                   kdim=self.d + self.t,
                                                   vdim=self.d + self.t,
                                                   num_heads=2)

  def forward(self, source, messages, time_features, neighbors_padding_mask): # 原message中不包含时间信息？
    messages = messages.squeeze(dim=1)
    # time_features=time_features.squeeze(dim=1)
    messages_unrolled = torch.cat((messages, time_features), dim=2) # [118,10, D]
    source_m = self.merger_s(messages[:,-1,:].squeeze(dim=1), source)
    # source_m = source
    # add the attention module
    source = messages[:,-1,:]
    Q = torch.cat([source, time_features[:,-1,:]], 1) # [batch_size, neighbor, features]

    Q = Q.unsqueeze(dim=1).permute([1, 0, 2]) # [1, B, D]

    K = messages_unrolled.permute([1, 0, 2]) # [N, 200, D]
    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
    neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False
    attn_output, _ = self.multi_head_target(query=Q, key=K, value=K,key_padding_mask=neighbors_padding_mask)
    attn_output = attn_output.squeeze()

    attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    out = self.merger(attn_output, source)
    return out # , attn_output_weights


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

  def aggregate(self, node_ids, messages, memory):
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
    num_age = 0

    return to_update_node_ids, unique_messages, unique_timestamps, num_age

class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device, n_neighbor):
    super(MeanMessageAggregator, self).__init__(device)
    self.n_neighbor = n_neighbor

  def aggregate(self, node_ids, messages, memory):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0
    j = 0
    start_t = time.time()
    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        msg = [m[0] for m in messages[node_id]][-self.n_neighbor:]
        unique_messages.append(torch.mean(torch.stack(msg), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])
        j += 1

    delta_t = time.time()-start_t
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []
    num_age = 0

    return to_update_node_ids, unique_messages, unique_timestamps, num_age

class AttnMessageAggregator(MessageAggregator):
  def __init__(self, device, message_dimension, batch_size, n_neighbor, node_gene, n_time_d = 100, use_age = False, n_mem_dim = 172, time_encoder = None,  n_gene = 50):
    super(AttnMessageAggregator, self).__init__(device)
    self.n_head = 2
    self.n_mem_dim = n_mem_dim
    self.use_age = use_age
    self.feat_dim = message_dimension  # 一般等于n_neighbors_features
    self.time_encoder = TimeEncode(dimension=n_time_d)
    self.batch_size = batch_size
    self.n_neighbor = n_neighbor
    self.node_gene = node_gene
    self.n_gene = n_gene
    self.message_updater = attention_node(self.n_mem_dim, self.feat_dim, n_time_d, self.n_gene)
    # self.Wr = nn.Linear(50, 50, bias=False)
    # self.pe = nn.Sequential(
    #   nn.Linear(100, 140, bias=True),
    #   nn.GELU(),
    #   nn.Linear(140, 172, bias=True)
    # )
    # nn.init.normal_(self.Wr.weight.data, mean=0, std=4 ** -2)
    # self.M_m = MLP_M_m(self.n_mem_dim, self.feat_dim)
    # self.m_M = MLP_m_M(self.feat_dim, self.n_mem_dim)
    if self.use_age == True:
      self.mlp = AGE_MLP(self.n_neighbor).to(self.device)
      self.relu = torch.nn.ReLU(n_neighbor).to(self.device)


  # def mix_intention_sum(self, destination_nodes):
  #     des_nodes_torch = torch.from_numpy(destination_nodes).long().to(self.device)
  #     temp = self.node_gene[des_nodes_torch, :]
  #     max = torch.max(temp, 2)[0]
  #     min = torch.min(temp, 2)[0]
  #     mask = torch.eq(max, min)
  #     temp_mask = torch.sign(abs(max) + abs(min)).unsqueeze(dim = 2).repeat(1,1,172)
  #     cosines = torch.cos(self.Wr(temp))
  #     sines = torch.sin(self.Wr(temp))
  #     F = 1 / np.sqrt(100) * torch.cat([cosines, sines], dim=-1)
  #     semantic_temp = self.pe(F).mul(temp_mask)
  #     # np_m = np.array(torch.tensor(semantic_temp).cpu())
  #     destination_memory = torch.sum(semantic_temp, dim=1)
  #     # temp_0 = self.merge_1(temp).mul(temp_mask)
  #     # destination_memory = torch.sum(temp_0, dim = 1)
  #     # np_m = np.array(torch.tensor(memory).cpu())
  #     # np_d = np.array(torch.tensor(destination_memory).cpu())
  #     return destination_memory

  def aggregate(self, node_ids, messages, memory):
    # start_t_0 = time.time()
    # unique_node_ids = np.unique(node_ids)
    # unique_timestamps = []
    # to_update_node_ids = []
    # messages_temp = []
    # time_temp = []
    # # source_temp = []
    # n_messages = 0
    # j = 0
    # num_age = 0
    # msg_list = []
    # msg_t_list = []
    # for node_id in unique_node_ids:
    #   if len(messages[node_id]) > 0:
    #     unique_timestamps.append(messages[node_id][-1][1])  # 记录最近一次的time stamp
    #     to_update_node_ids.append(node_id) # 将独立节点序号记录在列表中
    #     messages_all = messages[node_id]
    #     # messages_inf = dict(messages[node_id])
    #     # msg_torch = torch.stack(tuple(messages_inf.keys())).unsqueeze(dim=0)
    #     for i in messages_all:
    #       msg_list.extend([i[0]])
    #     # time_tuple = tuple(messages_inf.values())
    #       msg_t_list.extend([i[1]])
    #     # edge_time_list = torch.stack(time_tuple).unsqueeze(dim=0)
    #     # messages_temp.append(msg_torch)
    #     # time_temp.append(edge_time_list)
    #     j += 1 # 确定有信息的节点数量
    #
    # # 数据预处理
    # # batch化聚合节点数据
    # batch_time = time.time() - start_t_0
    # start_t_1 = time.time()
    #
    # if j > 0:
    #   # mytest = messages[to_update_node_ids][-self.n_neighbor:]
    #   # source_batch = torch.cat(source_temp)
    #   # message_batch = torch.cat(messages_temp)
    #   message_batch = torch.stack(msg_list).reshape(-1,self.n_neighbor,self.feat_dim)
    #   # time_batch = torch.cat(time_temp)
    #   time_batch = torch.stack(msg_t_list).reshape(-1,self.n_neighbor)
    #   stack_time = time.time() - start_t_1
    #   start_t_2 = time.time()
    #   max_time = max(unique_timestamps)
    #   if self.use_age:
    #     age_batch = max_time - time_batch
    #     thre_t = self.relu(self.mlp(age_batch)).unsqueeze(0).permute(1, 0)
    #     # learning the action model of the users in the batches and get the general function to calculate the threshold of ages
    #     time_batch_mask = torch.sigmoid(100*(time_batch-thre_t))
    #     time_batch = time_batch.mul(time_batch_mask)
    #     mask_info = torch.cat((age_batch[:, -self.n_neighbor:-1], thre_t), 1)
    #     mask_batch = (mask_info > thre_t) & (mask_info > max_time)
    #   else:
    #     mask_batch = time_batch < 0
    #   source_time = time_batch[:, -1].clone().unsqueeze(dim = 1)
    #   delta_batch = time_batch - source_time
    #   source = memory[to_update_node_ids]
    #   time_embeddings = self.time_encoder(delta_batch)
    #   aggregated_messages = self.message_updater(source, message_batch, time_embeddings, mask_batch)
    #   all_mask = mask_batch.detach().reshape(-1).cpu().numpy().tolist()
    #   num_age = Counter(all_mask)[False]
    #   num_age = num_age/len(to_update_node_ids)
    #   num_age
    #   cal_time = time.time() - start_t_2
    unique_node_ids = np.unique(node_ids)
    unique_timestamps = []
    to_update_node_ids = []
    j = 0
    num_age = 0
    msg_list = []
    msg_t_list = []
    start_time = time.time()
    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        unique_timestamps.append(messages[node_id][-1][1])  # 记录最近一次的time stamp
        to_update_node_ids.append(node_id) # 将独立节点序号记录在列表中
        messages_all = messages[node_id]
        gene_all = self.node_gene[node_id]
        for i in messages_all:
          msg_list.extend([i[0]])
          msg_t_list.extend([i[1]])
        j += 1 # 确定有信息的节点数量

    # 数据预处理
    # batch化聚合节点数据

    if j > 0:
      message_batch = torch.stack(msg_list).reshape(-1,self.n_neighbor,self.feat_dim)
      time_batch = torch.stack(msg_t_list).reshape(-1,self.n_neighbor)
      max_time = max(unique_timestamps)
      if self.use_age:
        mask_0 = time_batch < 0
        age_batch = max_time - time_batch
        # ones_batch = max_time * torch.ones(age_batch.shape[0]).to(self.device).unsqueeze(1)
        # age_tbatch = torch.cat((age_batch[:,-self.n_neighbor:], ones_batch), 1)
        thre_t = max_time - (self.mlp(time_batch)).unsqueeze(0).permute(1, 0)
        # thre_t = torch.ones(age_batch.shape[0]).unsqueeze(1)*6000
        # thre_t = thre_t.to(self.device)
        # learning the action model of the users in the batches and get the general function to calculate the threshold of ages
        time_batch_mask = torch.sigmoid(100*(thre_t-age_batch))
        time_batch = time_batch.mul(time_batch_mask)
        mask_info = torch.cat((age_batch[:, -self.n_neighbor:-1], thre_t*0), 1)
        # mask_info_np = mask_info.detach().cpu().numpy()
        mask_batch_0 = (mask_info > thre_t)
        mask_batch_1 = (mask_info > max_time)
        mask_batch = (mask_batch_0 | mask_batch_1) | mask_0
        # mask_np_0 = mask_batch_0.cpu().numpy()
        # mask_np_1 = mask_batch_1.cpu().numpy()

      else:
        mask_batch = time_batch < 0
      mask_batch_0 = time_batch < 0
      source_time = time_batch[:, -1].clone().unsqueeze(dim = 1)
      delta_batch = time_batch - source_time
      source = memory[to_update_node_ids]
      time_embeddings = self.time_encoder(delta_batch)
      aggregated_messages = self.message_updater(source, message_batch, time_embeddings, mask_batch)
      # all_mask = mask_batch.detach().reshape(-1).cpu().numpy()
      num_age = 0 # Counter(all_mask.tolist())[False]
      # all_mask_0 = mask_batch_0.detach().reshape(-1).cpu().numpy()
      # num_age_0 = Counter(all_mask_0.tolist())[False]
      num_age = num_age# /len(to_update_node_ids)
    unique_messages = aggregated_messages if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []
    # d_time = time.time()-start_time

    return to_update_node_ids, unique_messages, unique_timestamps, num_age

def get_message_aggregator(aggregator_type, device, message_dimension, batch_size, n_neighbor, node_gene, use_age = False, time_encoder = None, n_gene = 50):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device, n_neighbor=n_neighbor)
  elif aggregator_type == "attn":
    return AttnMessageAggregator(device=device, message_dimension = message_dimension, batch_size = batch_size, n_neighbor = n_neighbor, use_age = use_age, time_encoder = time_encoder, node_gene = node_gene, n_gene = n_gene)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
