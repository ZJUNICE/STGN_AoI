import logging
import numpy as np
import torch
from collections import defaultdict


from utils.utils import MergeLayer_t, MergeLayer_1, MergeLayer, Merge_0, MergeLayer0, MergeLayer_mix, MergeLayer_Mix
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode, TimeEncode_1
import torch.nn as nn
import time

class MsgProcess(nn.Module):
  def __init__(self, n_neighbor, feat_dim, device):
    super(MsgProcess, self).__init__()
    self.n_neighbor = n_neighbor
    self.feat_dim = feat_dim
    self.device = device
    self.message_zeros = [(n, torch.tensor(-1).to(self.device)) for n in torch.zeros(self.n_neighbor, self.feat_dim).to(self.device)]
  def process_msg(self, messages):
    message_list = defaultdict(list)
    for key in messages:
        if len(messages[key])>0:
            message_temp_node = self.message_zeros + messages[key]

        else:
            message_temp_node = messages[key]
        message_list[key] = message_temp_node[-self.n_neighbor:]
    return message_list



class attention_node(nn.Module):
  def __init__(self, n_node_features, n_gene):
    super(attention_node, self).__init__()
    self.d = n_node_features
    self.n_gene = n_gene
    # self.merger_s = MergeLayer0(self.d, self.mem_d, self.mem_d, self.mem_d)
    self.merger = MergeLayer(self.d, self.d, self.d, self.d)
    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.d ,# embed_dim=self.d + self.t
                                                   kdim=self.n_gene,
                                                   vdim=self.n_gene,
                                                   num_heads=2)

  def forward(self, source, gene, neighbors_padding_mask, mask_n): # 原message中不包含时间信息？
    Q = source.unsqueeze(dim=1)# [batch_size, neighbor, features]

    Q = Q.permute([1, 0, 2]) # [1, B, D]
    K = gene.permute([1, 0, 2]) # [N, 200, D]
    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
    neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False
    attn_output, attn_weight = self.multi_head_target(query=Q, key=K, value=K,key_padding_mask=neighbors_padding_mask)
    attn_output = attn_output.squeeze()
    attn_weight = attn_weight.squeeze()

    attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    # out = self.merger(attn_output.mul(mask_n), source)  # des_embedding   self.merger(
    out = attn_output.mul(mask_n) + source
    return out, attn_weight # , attn_output_weights



class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, semantic_feature, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False, use_age = False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False,
               batch_size=900,
               n_neighbor = 6, Sem=False, mix='Attn'):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)

    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.Sem=Sem
    if Sem == True:
        self.node_gene = torch.from_numpy(semantic_feature.astype(np.float32)).to(device)
        self.n_gene = self.node_gene.shape[1]
        self.n_node_gene = self.node_gene.shape[0]
        self.mix = mix
    else:
        self.node_gene = None
        self.n_node_gene = 0
        self.n_gene = 0
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep
    self.batch_size = batch_size
    self.n_neighbor = n_neighbor # neighbor for message aggregating

    self.use_memory = use_memory
    self.use_age = use_age
    # self.time_encoder_s = TimeEncode_1(dimension=self.n_node_features, device = self.device) # control the time encoder functionn of the source of the TGAT module
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    # self.time_encoder_n = TimeEncode(dimension=self.n_node_features)
    # self.time_encoder_n = self.time_encoder
    self.memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_memory:
      self.aggregator_type = aggregator_type
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device,
                           n_neighbor = self.n_neighbor)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device,
                                                       message_dimension = message_dimension,
                                                       batch_size = self.batch_size,
                                                       n_neighbor = self.n_neighbor,
                                                       use_age = self.use_age,
                                                       time_encoder = self.time_encoder,
                                                       node_gene = self.node_gene,
                                                       n_gene = self.n_gene)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device,
                                               gene = self.node_gene)

    self.embedding_module_type = embedding_module_type
    self.MsgProcesss = MsgProcess(n_neighbor, message_dimension, device)

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 node_gene = self.node_gene,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 # time_encoder_n=self.time_encoder_n,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors,
                                                 Sem=self.Sem)

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features,
                                     1)
    self.merge = MergeLayer_mix(self.n_node_features,self.n_node_features,self.n_node_features)
    self.merge_0 = MergeLayer(self.n_gene, self.n_node_features, self.n_node_features, self.n_node_features)
    self.merge_1 = MergeLayer_1(self.n_gene, self.n_node_features, self.n_node_features)
    self.attention = attention_node(self.n_node_features, self.n_gene)

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, aggregator_type, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])

    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    # TODO: KG
    # KG = True
    # if KG == True:
    #     pass

    memory = None
    time_diffs = None
    age_num = 0
    flag = 0
    if self.use_memory:
      if self.memory_update_at_start:
        memory, last_update, num_age = self.get_updated_memory(list(range(self.n_nodes)),
                                                          self.memory.messages)
        flag += 1
        age_num += num_age

      else: # False
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update


      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0) # no use

    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             # timestamps_temp=timestamps_temp,
                                                             timestamps = timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs, Sem = self.Sem)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if self.use_memory:
      if self.memory_update_at_start: # Setting False
        self.update_memory(positives, self.memory.messages, aggregator_type)
        self.memory.clear_messages(positives)

      # messages in a batch for storing new messages
      # The semantics aggregations mixed with pre-processed vertexes embeddings after first round training
      # performs better in our experiments.
      if self.Sem==True:
          unique_sources, source_id_to_messages = self.get_raw_messages_s(source_nodes,
                                                                        source_node_embedding,
                                                                        destination_nodes,
                                                                        destination_node_embedding,
                                                                        edge_times, edge_idxs)
          unique_destinations, destination_id_to_messages = self.get_raw_messages_d(destination_nodes,
                                                                                  destination_node_embedding,
                                                                                  source_nodes,
                                                                                  source_node_embedding,
                                                                                  edge_times, edge_idxs)
      else:
          unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                        source_node_embedding,
                                                                        destination_nodes,
                                                                        destination_node_embedding,
                                                                        edge_times, edge_idxs)
          unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                                  destination_node_embedding,
                                                                                  source_nodes,
                                                                                  source_node_embedding,
                                                                                  edge_times, edge_idxs)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages, self.aggregator_type) # source_message
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages, self.aggregator_type) # destination message
      else: # False
        self.update_memory(unique_sources, source_id_to_messages, aggregator_type)
        self.update_memory(unique_destinations, destination_id_to_messages, aggregator_type)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]

    

    return source_node_embedding, destination_node_embedding, negative_node_embedding, age_num

  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, aggregator_type, n_neighbors=20):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    n_samples = len(source_nodes)
    source_node_embedding, destination_node_embedding, negative_node_embedding, age_num = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, aggregator_type, n_neighbors)

    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples].sigmoid()
    neg_score = score[n_samples:].sigmoid()

    return pos_score, neg_score, age_num

  def update_memory(self, nodes, messages, aggregator_type):

    unique_nodes, unique_messages, unique_timestamps, num_age = \
            self.message_aggregator.aggregate(
                                                    nodes,
                                                    messages,self.memory.memory.data.clone())

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps) # store updated memory

  def get_updated_memory(self, nodes, messages):# get the memory updated before

    unique_nodes, unique_messages, unique_timestamps, num_age = \
      self.message_aggregator.aggregate(nodes, messages, self.memory.memory.data.clone())

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update, num_age

  def mix_intention(self, source_node_embedding, destination_nodes): # ATTENTION
      des_nodes_torch = torch.from_numpy(destination_nodes).long().to(self.device)
      temp = self.node_gene[des_nodes_torch, :]
      max = torch.max(temp, 2)[0]
      min = torch.min(temp, 2)[0]
      mask = torch.eq(max, min)
      # mask_numpy = np.array(mask.cpu())
      # multiple a weight for enhance the influence
      mask_n = torch.count_nonzero(torch.sign(abs(max) + abs(min)).int(), dim=1).reshape(-1, 1)

      # s_embedding = source_node_embedding + destination_node_embedding
      # s_embedding = torch.cat([source_node_embedding, destination_node_embedding], dim=1).unsqueeze(dim=1)
      # temp_0 = self.merge_0(temp, s_embedding.unsqueeze(dim=1).repeat(1, 8, 1), dim=2)  # [200, 8, 50]
      attn_out, attn_weight = self.attention(source_node_embedding, temp, mask, mask_n)
      destination_memory = attn_out # .mul(mask_n)
      return destination_memory

  def mix_intention_sum(self, destination_nodes): # SUMMATION
      des_nodes_torch = torch.from_numpy(destination_nodes).long().to(self.device)
      temp = self.node_gene[des_nodes_torch, :]
      max = torch.max(temp, 2)[0]
      min = torch.min(temp, 2)[0]
      mask = torch.eq(max, min)
      temp_mask = torch.sign(abs(max) + abs(min)).unsqueeze(dim = 2).repeat(1,1,172)
      # cosines = torch.cos(self.Wr(temp))
      # sines = torch.sin(self.Wr(temp))
      # F = 1 / np.sqrt(172) * torch.cat([cosines, sines], dim=-1)
      # semantic_temp = F.mul(temp_mask)
      # semantic_temp = 1 / np.sqrt(172) * cosines.mul(temp_mask)
      # np_m = np.array(torch.tensor(semantic_temp).cpu())
      # destination_memory = torch.sum(semantic_temp, dim=1)
      temp_0 = self.merge_1(temp).mul(temp_mask)
      destination_memory = torch.sum(temp_0, dim = 1)
      # np_m = np.array(torch.tensor(memory).cpu())
      # np_d = np.array(torch.tensor(destination_memory).cpu())
      return destination_memory

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
        messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def get_raw_messages_s(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    # # add the semantic
    # des_nodes_torch = torch.from_numpy(destination_nodes).long().to(self.device)
    # temp = self.node_gene[des_nodes_torch, :]
    # destination_memory = self.merge(temp) + destination_memory
    # #############################################################

    # add the semantic
    d_embedding = self.merge(destination_node_embedding, source_node_embedding)
    if self.mix == 'Attn':
        semantic_temp = self.mix_intention(d_embedding, destination_nodes)
    else:
        semantic_temp = self.mix_intention_sum(destination_nodes)
    # it is better to add them directly, or we can use self.merge_0 as done in the paper
    destination_memory = semantic_temp + destination_memory
    #############################################################

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)


    for i in range(len(source_nodes)):
        messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def get_raw_messages_d(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding
    #############################################################
    s_embedding = self.merge(source_node_embedding, destination_node_embedding)
    if self.mix == 'Attn':
        semantic_temp = self.mix_intention(s_embedding, source_nodes)
    else:
        semantic_temp = self.mix_intention_sum(source_nodes)
    # it is better to add them directly, or we can use self.merge_0 as done in the paper
    source_memory = semantic_temp + source_memory
    #############################################################
    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
        messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages
  # def get_raw_messages_s(self, source_nodes, source_node_embedding, destination_nodes,
  #                      destination_node_embedding, edge_times, edge_idxs):
  #   edge_times = torch.from_numpy(edge_times).float().to(self.device)
  #   edge_features = self.edge_raw_features[edge_idxs]
  #
  #   source_memory = self.memory.get_memory(source_nodes) if not \
  #     self.use_source_embedding_in_message else source_node_embedding
  #   destination_memory = self.memory.get_memory(destination_nodes) if \
  #     not self.use_destination_embedding_in_message else destination_node_embedding
  #
  #   # # add the semantic
  #   # des_nodes_torch = torch.from_numpy(destination_nodes).long().to(self.device)
  #   # temp = self.node_gene[des_nodes_torch, :]
  #   # destination_memory = self.merge(temp) + destination_memory
  #   # #############################################################
  #
  #   # add the semantic
  #   # d_embedding = self.merge(destination_node_embedding, source_node_embedding)
  #   # semantic_temp = self.mix_intention(d_embedding, destination_nodes)
  #   semantic_temp = self.mix_intention_sum(destination_nodes)
  #   destination_memory = semantic_temp + destination_memory
  #   #############################################################
  #
  #   source_time_delta = edge_times - self.memory.last_update[source_nodes]
  #   source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
  #     source_nodes), -1)
  #
  #   source_message = torch.cat([source_memory, destination_memory, edge_features,
  #                               source_time_delta_encoding],
  #                              dim=1)
  #   messages = defaultdict(list)
  #   unique_sources = np.unique(source_nodes)
  #
  #
  #   for i in range(len(source_nodes)):
  #       messages[source_nodes[i]].append((source_message[i], edge_times[i]))
  #
  #   return unique_sources, messages
  #
  # def get_raw_messages_d(self, source_nodes, source_node_embedding, destination_nodes,
  #                      destination_node_embedding, edge_times, edge_idxs):
  #   edge_times = torch.from_numpy(edge_times).float().to(self.device)
  #   edge_features = self.edge_raw_features[edge_idxs]
  #
  #   source_memory = self.memory.get_memory(source_nodes) if not \
  #     self.use_source_embedding_in_message else source_node_embedding
  #   destination_memory = self.memory.get_memory(destination_nodes) if \
  #     not self.use_destination_embedding_in_message else destination_node_embedding
  #   #############################################################
  #   # s_embedding = self.merge(source_node_embedding, destination_node_embedding)
  #   # semantic_temp = self.mix_intention(s_embedding, source_nodes)
  #   semantic_temp = self.mix_intention_sum(source_nodes)
  #   source_memory = semantic_temp + source_memory
  #   #############################################################
  #   source_time_delta = edge_times - self.memory.last_update[source_nodes]
  #   source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
  #     source_nodes), -1)
  #
  #   source_message = torch.cat([source_memory, destination_memory, edge_features,
  #                               source_time_delta_encoding],
  #                              dim=1)
  #   messages = defaultdict(list)
  #   unique_sources = np.unique(source_nodes)
  #
  #   for i in range(len(source_nodes)):
  #       messages[source_nodes[i]].append((source_message[i], edge_times[i]))
  #
  #   return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
