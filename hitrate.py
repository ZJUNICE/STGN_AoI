import math
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from collections import defaultdict,Counter

import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
import csv
from pathlib import Path
from tqdm import tqdm

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
# from utils.hitrate import hit_prediction
from collections import defaultdict, Counter

torch.manual_seed(1)
np.random.seed(1)

### Argument and global variables

# data is the val_data or test data
def hit_prediction(model,use_memory, negative_edge_sampler, data, n_neighbors, aggregator , batch_size=200, use_age = False, pred_period = 6, pop_t = 0.5, period_u = 24, range_d = 30):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    # sources = np.unique(data.sources)
    start_t = data.timestamps[0]
    end_t = data.timestamps[-1]
    all_time_mask = np.logical_and(data.timestamps < (78 * 3600 + start_t),
                                   data.timestamps >= (range_d * 3600 + start_t))
    destinations = np.unique(data.destinations[all_time_mask])
    # destinations = np.unique(data.destinations)

    # Source_id, Source_idx = np.unique(data.sources[all_time_mask][::-1], return_index=True)
    # sources_p = np.unique(Source_id[np.argsort(Source_idx)][0:1000])
    # sources = np.unique(data.sources[all_time_mask])
    sources = np.array(sorted(Counter(data.sources[all_time_mask]).items(), key=lambda x: x[1]))[:,0][-50:]
    sources_0 = []
    sources_1 = []
    all_time_mask_0 = np.logical_and(data.timestamps < (78 * 3600 + start_t),
                                   data.timestamps >= (range_d-5 * 3600 + start_t))
    sour_0=np.array(sorted(Counter(data.sources[all_time_mask_0]).items(), key=lambda x: x[1]))[:, 0][-50:]
    unique_sources_0 = np.unique(sour_0)
    for i in range(len(unique_sources_0)):
        a = random.uniform(0,1)
        if a > 0.5:
            sources_0.append(unique_sources_0[i])
        else:
            sources_1.append(unique_sources_0[i])

    # num_test_instance = len(sources)
    num_test_instance = len(destinations)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE) # 取整操作


    hit_list_0 = []
    hit_list_1 = []
    hit_list_2 = []
    hit_list_3 = []
    hit_list_0_r = []
    hit_list_1_r = []
    hit_list_2_r = []
    hit_list_3_r = []
    Ener = []
    Ener_r = []
    Ener_all = []
    E_3 = [0]
    E_3_r = [0]
    E_all_list = [0]
    hit_list_lfu_0 = []
    hit_list_lfu_1 = []
    hit_list_lfu_2 = []
    hit_list_lfu_3 = []

    # hit_list_lfu = []
    # hit_list_lru = []
    list_lfu = []
    cache_size_0 = 5
    cache_size_1 = 7
    cache_size_2 = 8
    cache_size_3 = 20

    if use_memory:
      memory_backup = model.memory.backup_memory()
    for hour in range(50, 74):
      # if hour == 1:
      # if use_memory:
      #     model.memory.restore_memory(memory_backup)

      if use_memory:
      #     # memory_backup = model.memory.backup_memory()
          model.memory.restore_memory(memory_backup)
      ################################################
      if (hour) % period_u == 0:
          # if hour != 25:
          past_time_mask = np.logical_and(data.timestamps < ((hour) * 3600 + start_t),data.timestamps > ((hour - period_u) * 3600 + start_t))
          # else:
          #   past_time_mask = np.logical_and(data.timestamps < ((hour) * 3600 + start_t),data.timestamps > (24 * 3600 + start_t))
          sources_true = data.sources[past_time_mask]
          destinations_true = data.destinations[past_time_mask]
          time_stamps_true = data.timestamps[past_time_mask]
          edge_true = data.edge_idxs[past_time_mask]
          num_test_instance_true = len(sources_true)
          num_test_batch_true = math.ceil(num_test_instance_true / TEST_BATCH_SIZE)
          if use_memory:
              model.memory.restore_memory(memory_backup)

          for k_t in range(num_test_batch_true):  # calculate the embedding for all nodes at the time stamp
              # 得到测试batch所需的所有数据
              s_idx_t = k_t * TEST_BATCH_SIZE
              e_idx_t = min(num_test_instance_true, s_idx_t + TEST_BATCH_SIZE)

              sources_batch_t = sources_true[s_idx_t:e_idx_t]
              destinations_batch_t = destinations_true[s_idx_t:e_idx_t]
              timestamps_batch_t = time_stamps_true[s_idx_t:e_idx_t]
              edge_idxs_batch_t = edge_true[s_idx_t:e_idx_t]

              size_t = len(sources_batch_t)
              _, negative_samples_t = negative_edge_sampler.sample(size_t)

              pos_prob_t, neg_prob_t, num_age_t = model.compute_edge_probabilities(sources_batch_t, destinations_batch_t,
                                                negative_samples_t, timestamps_batch_t,
                                                edge_idxs_batch_t, aggregator, n_neighbors)
              if k_t == 0:
                  pos_prob_list = pos_prob_t
                  neg_prob_list = neg_prob_t
              else:
                  pos_prob_list = torch.cat((pos_prob_list, pos_prob_t), 0)
                  neg_prob_list = torch.cat((neg_prob_list, neg_prob_t), 0)
          pos_mean = torch.mean(pos_prob_list)
          neg_mean = torch.mean(neg_prob_list)
          pos_true_np = pos_prob_list.cpu().numpy()
          p_0 = np.sum(np.heaviside(pos_true_np-0.7, 0))/np.sum(np.heaviside(pos_true_np, 0))
          neg_true_np = neg_prob_list.cpu().numpy()
          np_0 = np.sum(np.heaviside(neg_true_np - 0.7, 0))/np.sum(np.heaviside(neg_true_np, 0))

          neg_mean = torch.mean(neg_prob_list)
              # pos_prob_t = pos_prob_t.sigmoid()
              # neg_prob_t = neg_prob_t.sigmoid()
          if use_memory:
              memory_backup = model.memory.backup_memory()
              model.memory.restore_memory(memory_backup)
      list_p = []
      list_p_0 = []
      list_p_1 = []
      time_mask = np.logical_and(data.timestamps < ((hour) * 3600 + start_t),
                                   data.timestamps > (max(24,hour-1) * 3600 + start_t))
      # time_mask_0 = np.logical_and(data.timestamps < ((hour) * 3600 + start_t),
                                 # data.timestamps > (max(24, hour - 12) * 3600 + start_t))
      # sources = np.array(sorted(Counter(data.sources[time_mask_0]).items(), key=lambda x: x[1]))[:,0][-50:]
      sources_p = np.unique(data.sources[time_mask])
      #
      # sources_p = np.array(sorted(Counter(data.sources[time_mask]).items(), key=lambda x:x[1]))[:,0][-80:]
      # Source_id, Source_idx = np.unique(data.sources[all_time_mask][::-1], return_index=True)
      # sources_p = np.unique(np.append(np.array(sorted(Counter(data.sources[time_mask]).items(), key=lambda x:x[1]))[:,0][-100:], sources))

      dict_destination = defaultdict(list)
      dict_destination_0 = defaultdict(list)
      dict_destination_1 = defaultdict(list)
      time_period = int(3600/pred_period)
      time_span = pred_period
      time_mask_p = np.logical_and(data.timestamps < ((hour+1) * 3600 + start_t),
                                   data.timestamps > ((hour) * 3600 + start_t))
      base_dict = defaultdict(list)
      base_list_dict = defaultdict(list)
      true_timestamps = data.timestamps[time_mask_p]
      past_timestamps = data.timestamps[time_mask]
      sources_true = np.unique(data.sources[time_mask_p])
      sources_p = np.unique(data.sources[time_mask_p])

      # for start_time in (range(3600/time_period)):
      #       time_stamp = (hour) * 3600 + start_time + start_t
      #       # for each destination to calculate the popularity of every second
      #       # for des_idx in range(len(destinations)):
      #       if use_memory:
      #           model.memory.restore_memory(memory_backup)
      #       if len(sources_p)<len(destinations):
      #           des_batch_temp = destinations
      #           source_batch_temp = np.tile(sources_p, int(len(destinations)/len(sources_p)+1))
      #       else:
      #           source_batch_temp = sources_p
      #           des_batch_temp = np.tile(destinations, int(len(sources_p)/len(destinations)+1))
      #       num_test_instance = min(len(source_batch_temp), len(des_batch_temp))
      #       num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
      #
      #       for k in range(num_test_batch): # calculate the embedding for all nodes at the time stamp
      #       # 得到测试batch所需的所有数据
      #           s_idx = k * TEST_BATCH_SIZE
      #           e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      #
      #           sources_batch = source_batch_temp[s_idx:e_idx]
      #           size = len(sources_batch)
      #           ones = np.ones(size).astype('int64')
      #           # s_d_idx = s_idx if s_idx < len(destinations) else 0
      #           # e_d_idx = min(s_d_idx+TEST_BATCH_SIZE, len(destinations)) if s_idx < len(destinations) else TEST_BATCH_SIZE
      #           destinations_batch = des_batch_temp[s_idx:e_idx]
      #           timestamps_batch = ones*(time_stamp)
      #           edge_idxs_batch = np.random.permutation(data.edge_idxs)[0: size] # in the dataset all the edge features are [0, 0, 0, 0]
      #
      #
      #           _,negative_samples = negative_edge_sampler.sample(size) # target is the neibors i.e. the content
      #
      #
      #
      #           source_node_embedding, destination_node_embedding, negative_node_embedding, _ = model.compute_temporal_embeddings(sources_batch, destinations_batch,
      #                                                             negative_samples, timestamps_batch,
      #                                                             edge_idxs_batch, aggregator,n_neighbors)
      #           if k == 0:
      #               source_node_embedding_temp = source_node_embedding
      #               all_source_embedding = source_node_embedding_temp
      #               all_des_embedding = destination_node_embedding # len(destination) < size
      #           else:
      #               source_node_embedding_temp = source_node_embedding
      #               all_source_embedding = torch.cat((all_source_embedding, source_node_embedding_temp), 0)
      #               all_des_embedding = torch.cat((all_des_embedding, destination_node_embedding), 0)
      #
      #       # all_source_embedding = torch.stack(all_source_embedding_list)
      #       all_source_embedding = all_source_embedding[:len(sources_p)]
      #       all_des_embedding = all_des_embedding[:len(destinations)]
      #       ones_temp = torch.ones_like(all_source_embedding)
      #       pos_prob_all_list = []
      #
      #       for des_idx in range(len(destinations)):
      #
      #           source_embedding_0 = ones_temp * all_des_embedding[des_idx]
      #           # if des_idx != len(destinations)-1:
      #           # source_embedding_1 = ones_temp * all_source_embedding[des_idx]
      #           # else:
      #           #     des_embedding_1 = des_embedding_0
      #           score = model.affinity_score(torch.cat([all_source_embedding,all_source_embedding], dim=0),
      #                                       torch.cat([source_embedding_0, source_embedding_0])).squeeze(dim=0)
      #           pos_prob = score[:len(sources_p)].sigmoid()
      #           pos_prob_np = pos_prob.cpu().numpy()
      #           # if des_idx == 0:
      #
      #           if len(base_list_dict[des_idx]) < 3600 and (time_stamp not in true_timestamps):
      #               base_list_dict[des_idx].append(pos_prob_np)
      #       for des_idx in range(len(destinations)):
      #           base_dict[des_idx] = np.min(np.array(base_list_dict[des_idx]), axis=0)*0
      if use_memory:
          model.memory.restore_memory(memory_backup)
      for start_time in tqdm(range(time_period)):
        time_stamp = hour * 3600 + start_time*time_span + start_t
        # for each destination to calculate the popularity of every second
        # for des_idx in range(len(destinations)):
        if use_memory:
            model.memory.restore_memory(memory_backup)
        if len(sources_p)<len(destinations):
            des_batch_temp = destinations
            source_batch_temp = np.tile(sources_p, int(len(destinations)/len(sources_p)+1))
        else:
            source_batch_temp = sources_p
            des_batch_temp = np.tile(destinations, int(len(sources_p)/len(destinations)+1))
        num_test_instance = min(len(source_batch_temp), len(des_batch_temp))
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        true_timestamps = data.timestamps[time_mask_p]
        for k in range(num_test_batch): # calculate the embedding for all nodes at the time stamp
        # 得到测试batch所需的所有数据
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

            sources_batch = source_batch_temp[s_idx:e_idx]
            size = len(sources_batch)
            ones = np.ones(size).astype('int64')
            # s_d_idx = s_idx if s_idx < len(destinations) else 0
            # e_d_idx = min(s_d_idx+TEST_BATCH_SIZE, len(destinations)) if s_idx < len(destinations) else TEST_BATCH_SIZE
            destinations_batch = des_batch_temp[s_idx:e_idx]
            timestamps_batch = ones*(time_stamp)
            edge_idxs_batch = np.random.permutation(data.edge_idxs)[0: size] # in the dataset all the edge features are [0, 0, 0, 0]


            _,negative_samples = negative_edge_sampler.sample(size) # target is the neibors i.e. the content



            source_node_embedding, destination_node_embedding, negative_node_embedding, _ = model.compute_temporal_embeddings(sources_batch, destinations_batch,
                                                              negative_samples, timestamps_batch,
                                                              edge_idxs_batch, aggregator,n_neighbors)
            if k == 0:
                source_node_embedding_temp = source_node_embedding
                all_source_embedding = source_node_embedding_temp
                all_des_embedding = destination_node_embedding # len(destination) < size
            else:
                source_node_embedding_temp = source_node_embedding
                all_source_embedding = torch.cat((all_source_embedding, source_node_embedding_temp), 0)
                all_des_embedding = torch.cat((all_des_embedding, destination_node_embedding), 0)

        # all_source_embedding = torch.stack(all_source_embedding_list)
        all_source_embedding = all_source_embedding[:len(sources_p)]
        all_des_embedding = all_des_embedding[:len(destinations)]
        ones_temp = torch.ones_like(all_source_embedding)
        pos_prob_all_list = []

        for des_idx in range(len(destinations)):

            source_embedding_0 = ones_temp * all_des_embedding[des_idx]
            score = model.affinity_score(torch.cat([all_source_embedding,all_source_embedding], dim=0),
                                        torch.cat([source_embedding_0, source_embedding_0])).squeeze(dim=0)
            pos_prob = score[:len(sources_p)].sigmoid()
            pos_prob_np = pos_prob.cpu().numpy()
            pos_prob_np_0 = pos_prob_np
            # if des_idx == 0:

            base_dict
            max_np = base_dict[des_idx]
            pos_prob_all_list.append(pos_prob_np)
            var_pos = np.var(pos_prob_np-np.mean(pos_prob_np))
            mean_pos = np.mean(pos_prob_np)
            id_pos_max = sources_p[np.argmax(pos_prob_np)]
            id_i_pos_max = destinations[des_idx]
            max_pos = np.max(pos_prob_np)
            if np.max(pos_prob_np)>pop_t and np.max(pos_prob_np)<1: # wiki 0.995<max<1
                # if np.mean(pos_prob_np) < 0.5:
                dict_destination[destinations[des_idx]].append(np.array(1))
                for id_source in source_batch_temp[:len(sources_p)]:
                    if id_source in sources_0:
                        dict_destination_0[destinations[des_idx]].append(np.array(1))
                    else:
                        dict_destination_1[destinations[des_idx]].append(np.array(1))
            # else:
            #     pos_prob_all = torch.cat((pos_prob_all, pos_prob), 1)
        #
        pos_prob_all_np = np.heaviside(np.squeeze(np.array(pos_prob_all_list), axis = 2), 0)*np.squeeze(np.array(pos_prob_all_list), axis = 2)
        # pos_prob_all_np = pos_prob_all.cpu().numpy()
        max_pos = np.max(pos_prob_all_np)
        pos_prob_sum = np.sum(pos_prob_all_np, axis= 1)
        # pos_temp = np.heaviside(pos_prob_all_np-0.85, 0)*np.heaviside(0.9575-pos_prob_all_np, 0)
        dict_destination[destinations[np.argmax(pos_prob_sum)]].append(np.array(0))
        # dict_destination_0[destinations[np.argmax(pos_prob_sum)]].append(np.array(0))
        # dict_destination_1[destinations[np.argmax(pos_prob_sum)]].append(np.array(0))
        max_id_0 = np.argmax(pos_prob_sum)
        pos_prob_sum = np.delete(pos_prob_sum, np.argmax(pos_prob_sum))
        dict_destination[destinations[np.argmax(pos_prob_sum)]].append(np.array(0))
        max_id_1 = np.argmax(pos_prob_sum)

      if hour > 0:

        time_mask_l = np.logical_and(data.timestamps < ((hour) * 3600 + start_t),
                                   data.timestamps > ((hour - 5) * 3600 + start_t))
        LRU_id, LRU_idx = np.unique(data.destinations[time_mask_l][::-1], return_index=True)
        LRU_id = LRU_id[np.argsort(LRU_idx)]
        in_d_sort = Counter(data.destinations[time_mask_l])
        sources_true = np.unique(data.sources[time_mask_p])
        des_true = np.unique(data.destinations[time_mask_p])
        for k_l in LRU_id:
            if k_l in in_d_sort.keys():
                in_d_sort[k_l] += 1 - (LRU_id.tolist().index(k_l) + 1) / (len(LRU_id) + 1)
        in_d_sort = np.array(sorted(in_d_sort.items(), key=lambda x: x[1]))
        past_list = in_d_sort[:, 0].tolist()
        past_dict = {}
        # U_LRU_id, U_LRU_idx = np.unique()
        # U_LRU_id = U_LRU_id[np.argsort(U_LRU_idx)]
        users = data.sources[time_mask_l][::-1]
        deses = data.destinations[time_mask_l][::-1]
        des_00 = []
        des_01 = []
        for i, x in enumerate(users):
            if x in sources_0:
                des_00.append(deses[i])
            else:
                des_01.append(deses[i])

        LRU_id_0, LRU_idx_0 = np.unique(des_00, return_index=True)
        LRU_id_0 = LRU_id_0[np.argsort(LRU_idx_0)]
        in_d_sort_0 = Counter(des_00)
        for k_l in LRU_id_0:
            if k_l in in_d_sort_0.keys():
                in_d_sort_0[k_l] += 1 - (LRU_id_0.tolist().index(k_l) + 1) / (len(LRU_id_0) + 1)
        in_d_sort_0 = np.array(sorted(in_d_sort_0.items(), key=lambda x: x[1]))
        past_list_0 = in_d_sort_0[:, 0].tolist()
        past_dict_0 = {}

        LRU_id_1, LRU_idx_1 = np.unique(des_01, return_index=True)
        LRU_id_1 = LRU_id_1[np.argsort(LRU_idx_1)]
        in_d_sort_1 = Counter(des_01)
        for k_l in LRU_id_1:
            if k_l in in_d_sort_1.keys():
                in_d_sort_1[k_l] += 1 - (LRU_id_1.tolist().index(k_l) + 1) / (len(LRU_id_1) + 1)
        in_d_sort_1 = np.array(sorted(in_d_sort_1.items(), key=lambda x: x[1]))
        past_list_1 = in_d_sort_1[:, 0].tolist()
        past_dict_1 = {}
        # past_array = np.array(sorted(Counter(data.destinations[time_mask]).items(), key=lambda x: x[1]))
        # past_list = past_array[:, 0].tolist()
        # past_dict = {}
        for k in past_list:
            past_dict[k] = (past_list.index(k) + 1) / (len(past_list) + 1)

        for k in past_list_0:
            past_dict_0[k] = (past_list_0.index(k) + 1) / (len(past_list_0) + 1)

        for k in past_list_1:
            past_dict_1[k] = (past_list_1.index(k) + 1) / (len(past_list_1) + 1)
        #
        for idx in range(len(destinations)):
            #   if hour == 1:
            if destinations[idx] not in in_d_sort:
                list_p.append([destinations[idx], np.sum(dict_destination[destinations[idx]])])

                list_p_0.append([destinations[idx], np.sum(dict_destination_0[destinations[idx]])])
                list_p_1.append([destinations[idx], np.sum(dict_destination_1[destinations[idx]])])
            else:
                list_p.append([destinations[idx], np.sum(dict_destination[destinations[idx]]) + past_dict[destinations[idx]]])
                if destinations[idx] in in_d_sort_0:
                    list_p_0.append([destinations[idx], np.sum(dict_destination_0[destinations[idx]])+ past_dict_0[destinations[idx]]])
                elif destinations[idx] not in in_d_sort_0:
                    list_p_0.append([destinations[idx], np.sum(dict_destination_0[destinations[idx]])])
                if destinations[idx] in in_d_sort_1:
                    list_p_1.append([destinations[idx], np.sum(dict_destination_1[destinations[idx]])+ past_dict_1[destinations[idx]]])
                elif destinations[idx] in in_d_sort_1:
                    list_p_1.append([destinations[idx], np.sum(dict_destination_1[destinations[idx]])])
            # if hour == 2:pop_list_3 = {list: 20}  ... Loading Value

          # if hour == 2:pop_list_3 = {list: 20}  ... Loading Value
          #   list_p_1.append([destinations[idx], np.mean(dict_destination[destinations[idx]])])
        pop_listsort = np.array(sorted(list_p, key = lambda x:x[1]))

        pop_listsort_0 = np.array(sorted(list_p_0, key=lambda x: x[1]))
        pop_listsort_1 = np.array(sorted(list_p_1, key=lambda x: x[1]))

        pop_list_3 = pop_listsort[:, 0].astype('int64').tolist()[-cache_size_3:][::-1]
        # pop_list_3_new = pop_listsort[:, 0].astype('int64').tolist()[-30:][::-1]
        pop_list_3_new = in_d_sort[:, 0].astype('int64').tolist()[-30:][::-1]
        pop_list_0 = pop_list_3[:cache_size_0]
        pop_list_1 = pop_list_3[cache_size_0:cache_size_0+cache_size_1]
        pop_list_2 = pop_list_3[cache_size_0+cache_size_1:cache_size_0+cache_size_1+cache_size_2]
        # pop_list_3_0 =
        # pop_list_3_0 = pop_listsort_0[:, 0].astype('int64').tolist()[::-1]
        # pop_list_3_1 = pop_listsort_1[:, 0].astype('int64').tolist()[::-1]

        pop_list_3_0 = in_d_sort_0[:, 0].astype('int64').tolist()[::-1]
        pop_list_3_1 = in_d_sort_1[:, 0].astype('int64').tolist()[::-1]

        pop_list_0_0 = pop_list_3_0[:5]
        pop_list_0_1 = pop_list_3_1[:5]

        pop_list_1_new = []
        pop_list_2_new = []
        for n in pop_list_3_new:
            if n not in pop_list_0_0 and n not in pop_list_0_1:
                if len(pop_list_1_new) < cache_size_1:
                    pop_list_1_new.append(n)
                elif len(pop_list_1_new) == cache_size_1 and len(pop_list_2_new)<cache_size_2:
                    pop_list_2_new.append(n)
                elif len(pop_list_2_new) == cache_size_2:
                    break
        # hit_list_lfu_3 = pop_list_3
        # pop_list_0 = hit_list_lfu_0
        # pop_list_1 = hit_list_lfu_1
        # pop_list_2 = hit_list_lfu_2
        # pop_list_3 = hit_list_lfu_3


        in_d = data.destinations[time_mask_p]
        des_past = data.destinations[time_mask]
        sour_p = data.sources[time_mask_p]
        id_0 = []
        id_1 = []
        for ixd, true_s in enumerate(sour_p):
            if true_s in sources_0:
                id_0.append(in_d[ixd])
            else:
                id_1.append(in_d[ixd])
        hops = []
        for ix in id_0:
            if ix in pop_list_0_0:
                hops.append(1)
                continue
            elif ix not in pop_list_0_0 and ix in pop_list_0_1:
                hops.append(2)
            elif ix in pop_list_1_new:
                hops.append(2)
            elif ix in pop_list_2_new:
                hops.append(3)
            else:
                hops.append(6)

        for ix in id_1:
            if ix in pop_list_0_1:
                hops.append(1)
                continue
            elif ix not in pop_list_0_1 and ix in pop_list_0_0:
                hops.append(2)
            elif ix in pop_list_1_new:
                hops.append(2)
            elif ix in pop_list_2_new:
                hops.append(3)
            else:
                hops.append(6)





        unique_in_d = np.array(sorted(Counter(in_d).items(), key=lambda x:x[1]))
        time_mask_0 = np.logical_and(data.timestamps < ((hour) * 3600 + start_t),
                                   data.timestamps > ((hour-1) * 3600 + start_t))
        # in_d_sort = np.array(sorted(Counter(data.destinations[time_mask_0]).items(), key=lambda x:x[1]))

        # in_s_sort = np.array(sorted(Counter(data.sources[time_mask_p]).items(), key=lambda x:x[1]))
        in_s_sort = np.unique(data.sources[time_mask_p])
        sources_0 = np.unique(data.sources[time_mask_0])
        # hit_0 = [1 if in_d[i] in pop_list_0 else 0 for i in range(len(in_d))]
        hit_0 = [1 if in_d[i] in pop_list_0 else 0 for i in range(len(in_d))]
        hit_0_0 = [1 if in_d[i] in pop_list_0 else 0 for i in range(len(in_d))]
        # hit_rate_0 = np.sum(hit_0)/len(hit_0)
        in_d_sort = in_d_sort[::-1]
        hit_rate_0 = np.sum(hit_0) / len(hit_0)
        hit_0_r = [1 if in_d[i] in in_d_sort[:cache_size_0] else 0 for i in range(len(in_d))]
        hit_rate_0_r = np.sum(hit_0_r) / len(hit_0_r)
        hit_list_0_r.append(hit_rate_0_r)
        hit_list_0.append(hit_rate_0)
        # if hour != 30:


            # hit_00 = [0 if pop_list_0[i] in past_0 else 1 for i in range(len(pop_list_0))]
            # if sum(hit_00) == cache_size_0:


        hit_1 = [1 if in_d[i] in pop_list_1 else 0 for i in range(len(in_d))]
        hit_rate_1 = np.sum(hit_1) / len(hit_1)
        hit_1_r = [1 if in_d[i] in in_d_sort[cache_size_0:cache_size_0+cache_size_1] else 0 for i in range(len(in_d))]
        hit_rate_1_r = np.sum(hit_1_r) / len(hit_1_r)
        hit_list_1_r.append(hit_rate_1_r)
        hit_list_1.append(hit_rate_1)


        hit_2 = [1 if in_d[i] in pop_list_2 else 0 for i in range(len(in_d))]
        hit_rate_2 = np.sum(hit_2) / len(hit_2)
        hit_2_r = [1 if in_d[i] in in_d_sort[cache_size_0+cache_size_1:cache_size_0+cache_size_1+cache_size_2] else 0 for i in range(len(in_d))]
        hit_rate_2_r = np.sum(hit_2_r) / len(hit_2_r)
        hit_list_2_r.append(hit_rate_2_r)
        hit_list_2.append(hit_rate_2)


        hit_3 = [1 if in_d[i] in pop_list_3 else 0 for i in range(len(in_d))]
        hit_rate_3 = np.sum(hit_3) / len(hit_3)
        hit_3_r = [1 if in_d[i] in in_d_sort[:cache_size_3] else 0 for i in range(len(in_d))]
        hit_rate_3_r = np.sum(hit_3_r) / len(hit_3_r)
        hit_list_3_r.append(hit_rate_3_r)
        hit_list_3.append(hit_rate_3)

        past_0 = pop_list_0
        past_1 = pop_list_1
        past_2 = pop_list_2
        past_0_r = in_d_sort[:cache_size_0]
        past_1_r = in_d_sort[:cache_size_1]
        past_2_r = in_d_sort[:cache_size_2]
        cache_size = 3
        # E_3 = []
        # E_3_r = []
        # E_all_list = []
        for patter_n in range(1, 2):
            # E_0 = np.sum(hit_0) * cache_size * 1.92 * 85.89934592 * patter_n + np.sum(hit_0) * cache_size * (
            #         0.821 + 2.63 + 1.7 * 2 + 28.1) * 85.89934592
            # E_0_r = np.sum(hit_0_r) * cache_size * 1.92 * 85.89934592 * patter_n + np.sum(hit_0_r) * cache_size * (
            #         0.821 + 2.63 + 1.7 * 2 + 28.1) * 85.89934592
            # E_1 = np.sum(hit_1) * cache_size * (1.92 + 0.821) * 85.89934592 * patter_n + np.sum(hit_1) * cache_size * (
            #             2.63 + 1.7 * 2 + 28.1) * 85.89934592
            # E_1_r = np.sum(hit_1_r) * cache_size * (1.92 + 0.821) * 85.89934592 * patter_n + np.sum(
            #     hit_1_r) * cache_size * (2.63 + 1.7 * 2 + 28.1) * 85.89934592
            # E_2 = np.sum(hit_2) * cache_size * (1.92 + 0.821 + 2.63) * 85.89934592 * patter_n + np.sum(
            #     hit_2) * cache_size * (1.7 * 2 + 28.1) * 85.89934592
            # E_2_r = np.sum(hit_2_r) * cache_size * (1.92 + 0.821 + 2.63) * 85.89934592 * patter_n + np.sum(
            #     hit_2_r) * cache_size * (1.7 * 2 + 28.1) * 85.89934592
            # E_b = (len(hit_3) - np.sum(hit_3)) * cache_size * (
            #             1.92 + 0.821 + 2.63 + 1.7 * 2 + 28.1) * 85.89934592 * patter_n
            # E_b_r = (len(hit_3_r) - np.sum(hit_3_r)) * cache_size * (
            #             1.92 + 0.821 + 2.63 + 1.7 * 2 + 28.1) * 85.89934592 * patter_n
            # E_3.append(E_0 + E_1 + E_2 + E_b)
            # E_3_r.append(E_0_r + E_1_r + E_2_r + E_b_r)
            # E_all = (len(hit_3_r)) * cache_size * (1.92 + 0.821 + 2.63 + 1.7 * 2 + 28.1) * 85.89934592 * patter_n
            # E_all_list.append(E_all)
            #### TRAFFIC ####
            # E_0 = np.sum(hit_0) * cache_size * patter_n * 1
            # E_0_r = np.sum(hit_0_r) * cache_size * 1 * patter_n
            # E_1 = np.sum(hit_1) * cache_size * patter_n * 2
            # E_1_r = np.sum(hit_1_r) * cache_size * patter_n * 2
            # E_2 = np.sum(hit_2) * cache_size * patter_n * 3
            # E_2_r = np.sum(hit_2_r) * cache_size * patter_n * 3
            # E_b = (len(hit_3) - np.sum(hit_3)) * cache_size * patter_n * 6
            # E_b_r = (len(hit_3_r) - np.sum(hit_3_r)) * cache_size * patter_n * 6
            # E_3.append(E_0 + E_1 + E_2 + E_b)
            # E_3_r.append(E_0_r + E_1_r + E_2_r + E_b_r)
            # E_all = (len(hit_3_r)) * cache_size * patter_n * 6
            # E_all_list.append(E_all)
            #### HOPS ####
            E_0 = np.sum(hit_0) * patter_n * 1
            E_0_r = np.sum(hit_0_r) * 1 * patter_n
            E_1 = np.sum(hit_1) * patter_n * 2
            E_1_r = np.sum(hit_1_r) * patter_n * 2
            E_2 = np.sum(hit_2) * patter_n * 3
            E_2_r = np.sum(hit_2_r) * patter_n * 3
            E_b = (len(hit_3) - np.sum(hit_3)) * patter_n * 6
            E_b_r = (len(hit_3_r) - np.sum(hit_3_r)) * patter_n * 6
            # E_3.append((E_0 + E_1 + E_2 + E_b)+E_3[-1])
            E_3.append(np.sum(hops)+E_3[-1])
            E_3_r.append((E_0_r + E_1_r + E_2_r + E_b_r)+E_3_r[-1])
            E_all = len(hit_3_r) * patter_n * 6
            E_all_list.append(E_all+E_all_list[-1])
            #### DELAY ####
            # E_0 = np.sum(hit_0) * patter_n * 1 * (1/10)
            # E_0_r = np.sum(hit_0_r) * 1 * patter_n * (1/10)
            # E_1 = np.sum(hit_1) * patter_n * (1/10+1/20)
            # E_1_r = np.sum(hit_1_r) * patter_n * (1/10+1/20)
            # E_2 = np.sum(hit_2) * patter_n * (1/10+1/20*2)
            # E_2_r = np.sum(hit_2_r) * patter_n * (1/10+1/20*2)
            # E_b = (len(hit_3) - np.sum(hit_3)) * patter_n * (1/10+1/20*2+1/25*3)
            # E_b_r = (len(hit_3_r) - np.sum(hit_3_r)) * patter_n * (1/10+1/20*2+1/25*3)
            # # E_3.append((E_0 + E_1 + E_2 + E_b))
            # E_3.append(np.sum(hops))
            # E_3_r.append((E_0_r + E_1_r + E_2_r + E_b_r))
            # E_all = len(hit_3_r) * patter_n * (1/10+1/20*2+1/25*3)
            # E_all_list.append(E_all)
        Ener.append(E_3)
        Ener_r.append(E_3_r)
        Ener_all.append(E_all_list)



    delta_list_0 = np.sum(np.heaviside(np.array(hit_list_0) - np.array(hit_list_0_r), 0))
    delta_list_1 = np.sum(np.heaviside(np.array(hit_list_1) - np.array(hit_list_1_r), 0))
    delta_list_2 = np.sum(np.heaviside(np.array(hit_list_2) - np.array(hit_list_2_r), 0))
    delta_list_3 = np.sum(np.heaviside(np.array(hit_list_3) - np.array(hit_list_3_r), 0))
    # mean_0 = np.mean(np.array(hit_list_0))
    mean_0 =np.mean(np.array(hit_list_0))
    mean_1 = np.mean(np.array(hit_list_1))
    mean_2 = np.mean(np.array(hit_list_2))
    mean_3 = np.mean(np.array(hit_list_3))
    mean_0_r = np.mean(np.array(hit_list_0_r))
    mean_1_r = np.mean(np.array(hit_list_1_r))
    mean_2_r = np.mean(np.array(hit_list_2_r))
    mean_3_r = np.mean(np.array(hit_list_3_r))
    mult = 1
    E_sum = np.sum(Ener, axis=0)
    E_sum_r = np.sum(Ener_r, axis=0)
    E_all_sum = np.sum(Ener_all, axis=0)
    hit_list_0.append(mean_0)
    hit_list_1.append(mean_1)
    hit_list_2.append(mean_2)
    hit_list_3.append(mean_3)

  return hit_list_0, hit_list_1, hit_list_2, hit_list_3 ,E_3[1:], E_3_r[1:], E_all_list[1:]# , E_sum, E_sum_r, E_all_sum
  # return hit_list_0_r, hit_list_1_r, hit_list_2_r, hit_list_3_r
  # return hit_list_lfu_0, hit_list_lfu_1, hit_list_lfu_2, hit_list_lfu_3


torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')

parser.add_argument('--batch_size', type=int, default=900, help='Batch_size')
parser.add_argument('--n_neighbor', type=int, default=6, help='n_neighbor')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory') # store_true就代表着一旦有这个参数，做出动作“将其值标为True”，
parser.add_argument('--use_age', action='store_true', help='Whether to use age information')

parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn", "mlp", "id"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--pred_period', type=int, default=24, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

batch_size = args.batch_size
n_neighbor = args.n_neighbor
BATCH_SIZE = args.bs # default = 200
NUM_NEIGHBORS = args.n_degree # default = 10 Number of neighbors to sample
NUM_EPOCH = args.n_epoch # default = 50
NUM_HEADS = args.n_head # default = 2
DROP_OUT = args.drop_out # default = 0.1
GPU = args.gpu
DATA = args.data # default='wikipedia'
NUM_LAYER = args.n_layer # default = 1 Number of network layers
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory # store true
USE_AGE = args.use_age
MESSAGE_DIM = args.message_dim # default = 100
MEMORY_DIM = args.memory_dim # default = 172
aggregator =  args.aggregator
pred_period = args.pred_period

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
# MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(args.prefix,str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data, node_gene = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)


train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

full_ngh_finder = get_neighbor_finder(full_data, args.uniform)
user_n = np.unique(full_data.sources)
content_n = np.unique(full_data.destinations)

train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
                                      seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
                                       new_node_test_data.destinations,
                                       seed=3)
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
edge_features = edge_features

mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
for i in range(2, 3):
    model_i = 0
    period_u = [100, 12] # 12, 8, 6, 4, 3, 2,
    pred_period = [10, 30, 60, 120, 300, 600]
    pop_t = [0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]
    range_d = [0, 10, 20, 30, 40, 50]
    model = TGN(neighbor_finder=train_ngh_finder, node_features=node_features, semantic_feature = node_gene,
                edge_features=edge_features, device=device,
                n_layers=NUM_LAYER,
                n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,use_age = USE_AGE,
                message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                memory_update_at_start=not args.memory_update_at_end,
                embedding_module_type=args.embedding_module, # default = grapg_attention
                message_function=args.message_function,
                aggregator_type=args.aggregator,
                memory_updater_type=args.memory_updater,
                n_neighbors=NUM_NEIGHBORS,
                mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                use_source_embedding_in_message=args.use_source_embedding_in_message,
                dyrep=args.dyrep,
                batch_size=batch_size,
                n_neighbor=n_neighbor)
# model_i = 0
    model_path = './saved_models/netflix_attn_g_10-netflix--0.pth'#.format(i)# test-11-sig-2-2d-2-mooc wiki-6-sig-2-2d-2-ns-wikipedia netflix_attn_0-netflix--{}
    model.load_state_dict(torch.load(model_path)) #mooc-6-sig-2-2d-2_t+b_for_1-mooc 5:test-6-sig-mooc-46.pth  test-11-sig-2-2d-2-mooc--5   15: test-11-sig-2-2d-2-mooc--0  mooc-11-sig-2-2d-2_t+b_0-mooc--3
    num_para = sum(model_para.numel() for model_para in model.parameters())
    model = model.to(device)
    model.eval()
    model.embedding_module.neighbor_finder = full_ngh_finder
    if USE_MEMORY:
      val_memory_backup = model.memory.backup_memory()
      model.memory.restore_memory(val_memory_backup)
    hit_list_0, hit_list_1, hit_list_2, hit_list_3, E_sum, E_sum_r, E_all_sum = hit_prediction(model=model, use_memory=USE_MEMORY,
                                                        negative_edge_sampler=test_rand_sampler,
                                                        data=test_data, aggregator=aggregator,
                                                        n_neighbors=NUM_NEIGHBORS, use_age=USE_AGE, pred_period = pred_period[i], pop_t = 0.995,period_u = period_u[model_i], range_d = 50)
    if USE_MEMORY:
      model.memory.restore_memory(val_memory_backup)
    nn_hit_list_0, nn_hit_list_1, nn_hit_list_2, nn_hit_list_3, nn_E_sum, nn_E_sum_r, nn_E_all_sum = hit_prediction(model=model, use_memory=USE_MEMORY,
                                                                 negative_edge_sampler=nn_test_rand_sampler,
                                                                 data=new_node_test_data, aggregator=aggregator,
                                                                 n_neighbors=NUM_NEIGHBORS, use_age=USE_AGE, pred_period = pred_period[i], pop_t=0.995, period_u = period_u[model_i], range_d = 50)
    # hitrate_path = "results/{}_{}_rr.csv".format(args.prefix, i)
    # Path("results/").mkdir(parents=True, exist_ok=True)
    # hit_dict = defaultdict(list)
    # hit_dict['hit_c5'] = hit_list_0
    # hit_dict['hit_c10'] = hit_list_1
    # hit_dict['hit_c15'] = hit_list_2
    # hit_dict['hit_c20'] = hit_list_3
    # hit_dict['nn_hit_c5'] = nn_hit_list_0
    # hit_dict['nn_hit_c10'] = nn_hit_list_1
    # hit_dict['nn_hit_c15'] = nn_hit_list_2
    # hit_dict['nn_hit_c20'] = nn_hit_list_3
    # with open(hitrate_path, 'w') as f:
    #   csv_write = csv.writer(f)
    #   csv_write.writerow(hit_dict.keys())
    #   csv_write.writerows(zip(*hit_dict.values()))
    Energy_path = "results/{}_{}_hops_2_55_w_LRU_rr_95.csv".format(args.prefix, i)
    ener_dict = defaultdict(list)
    ener_dict['D_sum'] = E_sum
    ener_dict['D_sum_r'] = E_sum_r
    ener_dict['D_all_sum'] = E_all_sum
    ener_dict['nn_D_sum'] = nn_E_sum
    ener_dict['nn_D_sum_r'] = nn_E_sum_r
    ener_dict['nn_D_all_sum'] = nn_E_all_sum
    with open(Energy_path, 'w') as f:
      csv_write = csv.writer(f)
      csv_write.writerow(ener_dict.keys())
      csv_write.writerows(zip(*ener_dict.values()))
    # hit_dict